"""Reward-hack an oracle while preserving naturalness of the proposals.

This example additionally requires the ``peft`` and  ``matplotlib`` packages.
Expected runtime is ~11 mins on an Nvidia A10G + hf model download time on first run.
"""

from collections.abc import Sequence
import pathlib
import time

import matplotlib  # ty: ignore[unresolved-import]
import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np
import peft  # ty: ignore[unresolved-import]
import polars as pl
from scipy import stats
import torch
from torch import nn
from torch.nn import functional as F
import transformers

from protuna import data
from protuna import inference
from protuna import optim
from protuna.data import datasets
from protuna.data import tokenizer
from protuna.inference import models
from protuna.optim import acquisitions
from protuna.optim import predictors
from protuna.optim import samplers
from protuna.optim import spaces
from protuna.optim import storage
from protuna.score import mlm
from protuna.utils import paths

matplotlib.use("Agg")


# docs-config-start
class SearchConfig:
    """Settings for the simulated protein engineering campaign."""

    n_rounds: int = 5
    n_parents: int = 4
    n_acquire: int = 96
    n_subspaces: int = 100
    max_mutations: int = 12
    max_samples: int = 10
    top_p: float = 0.9
    n_steps: int = 6
    model_batch_size: int = 128
    sampler_batch_size: int | None = None
    seed: int = 0
    oracle_epochs: int = 150
    predictor_steps: tuple[int, ...] = (100, 500, 500, 500)


_CFG = SearchConfig()
_MODEL_NAME = models.EsmcModels.ESMC_300M
_MODEL_EMB_DIMS = 960
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# docs-config-end


class Oracle:
    """Estimate fluorescence from protein sequence embeddings."""

    def __init__(self, dataset: datasets.MutantFitnessDataset, save_dir: pathlib.Path):
        """Initialize the oracle and its checkpoint location."""
        torch.manual_seed(_CFG.seed)
        self.dataset = dataset
        self.path = save_dir / "oracle.pt"
        self.encoder = models.EsmcInferenceModel(
            _MODEL_NAME, batch_size=_CFG.model_batch_size, trust_remote_code=True
        )
        self.head = nn.Sequential(
            nn.Linear(_MODEL_EMB_DIMS, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        ).to(_DEVICE)

    def embeddings(self, mutants: Sequence[str]) -> torch.Tensor:
        """Compute mean embeddings for the supplied sequences."""
        self.encoder.load(_DEVICE)
        config = inference.MLMOutputConfig(
            return_logits=False, return_mean_embeddings=True
        )
        outputs = self.encoder(
            [inference.MLMRequest(tokenizer(seq), config) for seq in mutants]
        )
        embeddings = []
        for output in outputs:
            assert output.mean_embeddings is not None
            embeddings.append(output.mean_embeddings)
        return torch.stack(embeddings).to(_DEVICE)

    def train(self) -> None:
        """Train the oracle or load its existing checkpoint."""
        if self.path.exists():
            self.load()
            return
        rows = (
            self.dataset.get_fitness_df(prop=self.dataset.properties[0])
            .group_by("mutant")
            .agg(pl.col("sequence").first(), pl.col("fitness").mean())
            .sort("mutant")
            .filter(pl.col("mutant").str.count_matches(":", literal=True) < 4)
        )
        features = self.embeddings(rows["sequence"].to_list())
        targets = rows["fitness"].to_torch().to(_DEVICE, torch.float32)
        self.feature_mean = features.mean(0)
        self.feature_std = features.std(0).clamp_min(1e-6)
        self.fitness_mean = targets.mean()
        self.fitness_std = targets.std().clamp_min(1e-6)
        features = (features - self.feature_mean) / self.feature_std
        targets = (targets - self.fitness_mean) / self.fitness_std
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(features, targets),
            batch_size=256,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=1e-3, weight_decay=1e-2
        )
        self.head.train()
        for _ in range(_CFG.oracle_epochs):
            for batch, fitness in loader:
                optimizer.zero_grad()
                F.mse_loss(self.head(batch).squeeze(-1), fitness).backward()
                optimizer.step()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head": self.head.state_dict(),
                "scales": (
                    self.feature_mean,
                    self.feature_std,
                    self.fitness_mean,
                    self.fitness_std,
                ),
            },
            self.path,
        )

    def load(self) -> None:
        """Load the oracle weights and normalization scales."""
        checkpoint = torch.load(self.path, map_location=_DEVICE, weights_only=True)
        self.head.load_state_dict(checkpoint["head"])
        (self.feature_mean, self.feature_std, self.fitness_mean, self.fitness_std) = (
            checkpoint["scales"]
        )

    @torch.inference_mode()
    def predict(self, mutants: Sequence[str]) -> list[float]:
        """Predict fluorescence for the supplied sequences."""
        self.head.eval()
        features = (self.embeddings(mutants) - self.feature_mean) / self.feature_std
        return (
            self.head(features).squeeze(-1) * self.fitness_std + self.fitness_mean
        ).tolist()


def train_predictor(
    seen_data: dict[str, list], wt_sequence: str, steps: int, save_dir: pathlib.Path
) -> predictors.WildTypeMarginalPredictor:
    """Fine-tune a wild-type marginal predictor on observed fitness values."""
    torch.manual_seed(_CFG.seed)
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        _MODEL_NAME, trust_remote_code=True
    )
    assert hf_tokenizer is not None
    wt_tokens = hf_tokenizer(
        wt_sequence, return_tensors="pt", return_attention_mask=False
    )["input_ids"].to(_DEVICE)
    tokens = hf_tokenizer(
        seen_data["mutants"], return_tensors="pt", return_attention_mask=False
    )["input_ids"].to(_DEVICE)
    targets = torch.tensor(seen_data["fitness"], device=_DEVICE)
    backbone = (
        transformers.AutoModelForMaskedLM.from_pretrained(
            _MODEL_NAME, trust_remote_code=True
        )
        .to(_DEVICE)
        .eval()
    )
    with torch.no_grad():
        base_logp = backbone(wt_tokens).logits.log_softmax(-1)
    model = peft.get_peft_model(
        backbone,
        peft.LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear"
        ),
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5, weight_decay=1e-3
    )
    differences = targets[:, None] - targets[None, :]
    ordered = differences != 0
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(wt_tokens).logits
        scores = mlm.wildtype_marginal_from_logits(logits, wt_tokens, tokens).score
        deltas = scores[:, None] - scores[None, :]
        rank_loss = (
            F.softplus(-differences[ordered].sign() * deltas[ordered]).mean()
            if ordered.any()
            else scores.sum() * 0
        )
        kl_loss = F.kl_div(
            logits.log_softmax(-1), base_logp, reduction="batchmean", log_target=True
        )
        (rank_loss + kl_loss).backward()
        optimizer.step()
    adapter_path = save_dir / "predictor"
    model.save_pretrained(str(adapter_path))
    return predictors.WildTypeMarginalPredictor(
        tokenizer(wt_sequence),
        model=models.EsmcLogitsModel(
            _MODEL_NAME,
            pretrained_path=adapter_path,
            precision="fp32",
            trust_remote_code=True,
        ),
    )


# docs-campaign-init-start
class Campaign:
    """Propose protein variants using fitness observed in previous rounds."""

    def __init__(self, start_sequence: str, save_dir: pathlib.Path):
        """Initialize the campaign from its starting sequence."""
        self.wt_sequence = start_sequence
        self.init_tokens = tokenizer(start_sequence)
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        model = models.EsmcLogitsModel(
            _MODEL_NAME,
            batch_size=_CFG.model_batch_size,
            precision="bf16",
            trust_remote_code=True,
        )
        self.sampler = samplers.MaskGITSampler(
            model,
            max_samples=_CFG.max_samples,
            top_p=_CFG.top_p,
            n_iter=_CFG.n_steps,
            seed=_CFG.seed,
        )
        self.seen_data = {"mutants": [], "fitness": []}

    # docs-campaign-init-end

    # docs-design-space-start
    def get_design_space(self) -> spaces.DesignSpace:
        """Build a design space around the best observed sequences or wild type."""
        if self.seen_data["fitness"]:
            _, indices = torch.tensor(self.seen_data["fitness"]).topk(k=_CFG.n_parents)
            parents = tokenizer([self.seen_data["mutants"][idx] for idx in indices])
        else:
            parents = [self.init_tokens]
        return spaces.DesignSpace.from_rules(
            rules=[],
            init_sequences=parents,
            space_sampler=spaces.RandomMaskSampler(
                n_subspaces=_CFG.n_subspaces,
                max_mutations=_CFG.max_mutations,
                seed=_CFG.seed,
            ),
        )

    # docs-design-space-end

    # docs-pipeline-start
    def get_pipeline(
        self, round: int, mutants: Sequence[str], fitness: Sequence[float]
    ) -> optim.Pipeline:
        """Build a round's optimization pipeline using newly observed fitness."""
        if mutants:
            self.seen_data["mutants"].extend(mutants)
            self.seen_data["fitness"].extend(fitness)
            objective = train_predictor(
                self.seen_data,
                self.wt_sequence,
                _CFG.predictor_steps[round - 1],
                self.save_dir,
            )
        else:
            objective = predictors.WildTypeMarginalPredictor(
                self.init_tokens, model=self.sampler.model
            )
        return optim.Pipeline(
            self.get_design_space(),
            self.sampler,
            objective=objective,
            acquisition=acquisitions.Greedy(_CFG.n_acquire),
        )

    # docs-pipeline-end


# docs-evaluator-start
class Evaluator:
    """Run and summarize a simulated campaign using an oracle."""

    def __init__(self):
        """Initialize the dataset, campaign, and oracle."""
        dataset = data.get_dataset("creiLOV", data.DatasetSource.SGPO)
        self.save_dir = paths.get_cache_dir() / pathlib.Path(__file__).stem
        self.campaign = Campaign(dataset.target_sequence, self.save_dir / "campaign")
        self.oracle = Oracle(dataset, self.save_dir)
        self._summaries: list[dict] = []

    def evaluate(self) -> None:
        """Run the campaign and evaluate each round's proposals with the oracle."""
        print("Preparing oracle (training or loading checkpoint)...")
        self.oracle.train()
        mutants, fitness = [], []
        for round in range(_CFG.n_rounds):
            print(f"Round {round + 1}/{_CFG.n_rounds}: building pipeline...")
            pipeline = self.campaign.get_pipeline(round, mutants, fitness)
            print(f"Round {round + 1}/{_CFG.n_rounds}: generating candidates...")
            store = storage.CampaignStore(
                protein_length=len(self.campaign.init_tokens),
                location=self.campaign.save_dir,
                task_id=f"round_{round}",
            )
            output = pipeline.run(store=store)
            mutants = data.tokenizer.decode(list(output.selection.token_ids))
            print(f"Round {round + 1}/{_CFG.n_rounds}: predicting oracle fitness...")
            fitness = self.oracle.predict(mutants)
            print(f"Round {round + 1}/{_CFG.n_rounds}: recording results...")
            self._record(round, fitness)
        print("Summarizing results and saving plots...")
        self.summarize()

    # docs-evaluator-end

    def _record(self, round: int, fitness: Sequence[float]) -> None:
        self._summaries.append(
            {
                "round": round,
                "mean": sum(fitness) / len(fitness),
                "max": max(fitness),
                "values": list(fitness),
            }
        )

    def summarize(self) -> None:
        """Print round statistics and save fitness evolution and density plots."""
        dataset = self.oracle.dataset
        print(f"{'round':>16} {'mean':>10} {'max':>10}")
        for summary in self._summaries:
            print(
                f"{summary['round']:16d} {summary['mean']:10.4f} {summary['max']:10.4f}"
            )
        rounds = [s["round"] for s in self._summaries]
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ("mean", "max"):
            ax.plot(rounds, [s[key] for s in self._summaries], marker="o", label=key)
        wt_fitness = self.oracle.predict([dataset.target_sequence])[0]
        ax.axhline(wt_fitness, linestyle="--", label="wild-type", color="k")
        ax.set(xlabel="round", ylabel="oracle fitness", xticks=rounds)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.save_dir / "fitness_evolution.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        for summary in self._summaries:
            values = np.asarray(summary["values"])
            density = stats.gaussian_kde(values)
            bandwidth = float(np.sqrt(density.covariance[0, 0]))
            grid = np.linspace(
                values.min() - 3 * bandwidth, values.max() + 3 * bandwidth, 256
            )
            (line,) = ax.plot(grid, density(grid), label=f"round {summary['round']}")
            ax.fill_between(grid, density(grid), color=line.get_color(), alpha=0.15)
        ax.axvline(wt_fitness, linestyle="--", label="wild-type", color="k")
        ax.set(xlabel="oracle fitness", ylabel="density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.save_dir / "fitness_densities.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    start = time.perf_counter()
    Evaluator().evaluate()
    print(f"total runtime: {time.perf_counter() - start:.2f} seconds")
