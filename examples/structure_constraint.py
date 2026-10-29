"""Predicted structure confidence based hard constraint."""

from collections.abc import Sequence
import tempfile
import time
from typing import Any

from jaxtyping import Bool
import torch
import transformers

from protuna import data
from protuna import inference
from protuna import optim
from protuna.inference import models
from protuna.optim import samplers
from protuna.optim import spaces
from protuna.typing import SequenceLike, TokenIDs
from protuna.utils import devices


# docs-constraint-init-start
class StructureConfidenceConstraint(
    inference.InferenceModel[TokenIDs, Bool[torch.Tensor, ""]]
):
    """Require both mean residue pLDDT and pTM to meet inclusive thresholds."""

    def __init__(
        self,
        plddt_threshold: float = 0.7,
        ptm_threshold: float = 0.5,
        *,
        seed: int = 0,
        batch_size: int | None = None,
    ) -> None:
        """Configure the confidence criteria.

        Args:
            plddt_threshold: Minimum mean residue pLDDT in [0, 1].
            ptm_threshold: Minimum pTM in [0, 1].
            seed: Random seed for each structure prediction.
            batch_size: number of samples sent to forward at once. Requests are always
                processed one at a time by the model for simplicity.
        """
        super().__init__(batch_size=batch_size)
        if not 0 <= plddt_threshold <= 1 or not 0 <= ptm_threshold <= 1:
            raise ValueError("pLDDT and pTM thresholds must lie in [0, 1].")
        self.plddt_threshold = plddt_threshold
        self.ptm_threshold = ptm_threshold
        self.seed = seed
        self._model: Any | None = None
        self._device: torch.device | None = None

    # docs-constraint-init-end

    # docs-constraint-load-start
    def __getstate__(self) -> dict[str, Any]:
        """Exclude loaded resources when serializing the constraint."""
        return {**self.__dict__, "_model": None, "_device": None}

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load ESMFold2 onto the first supplied or available device."""
        device = devices.resolve_devices() if device is None else device
        if not isinstance(device, torch.device):
            device = device[0]
        if self._model is not None and self._device == device:
            return
        self.release()
        self._model = transformers.AutoModel.from_pretrained(
            "Synthyra/ESMFold2",
            trust_remote_code=True,
            dtype=torch.float32,
            device_map=str(device),
            esmc_precision="bf16" if device.type == "cuda" else "fp32",
            attn_implementation="sdpa",
        ).eval()
        self._device = device

    # docs-constraint-load-end

    # docs-constraint-forward-start
    @torch.inference_mode()
    def forward(self, requests: SequenceLike[TokenIDs]) -> list[Bool[torch.Tensor, ""]]:
        """Return one scalar Boolean verdict per candidate, in input order."""
        if self._model is None:
            raise RuntimeError("Load the constraint before evaluating candidates.")
        verdicts = []
        for tokens in requests:
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                result = self._model.infer_protein(
                    data.tokenizer.decode(tokens),
                    num_loops=3,
                    num_sampling_steps=50,
                    num_diffusion_samples=1,
                    verbose=False,
                )
            if result.plddt is None or result.ptm is None:
                raise RuntimeError("ESMFold2 did not return pLDDT and pTM scores.")
            mean_plddt = result.plddt.float().mean()
            ptm = result.ptm.float().squeeze()
            verdicts.append(
                (mean_plddt >= self.plddt_threshold) & (ptm >= self.ptm_threshold)
            )
        return verdicts

    # docs-constraint-forward-end

    # docs-constraint-release-start
    def release(self) -> None:
        """Release model resources, allowing the constraint to be loaded again."""
        self._model = None
        self._device = None

    # docs-constraint-release-end


def main() -> None:
    """Generate up to 128 CreiLOV variants and report constraint survivors."""
    # docs-design-space-start
    dataset = data.get_dataset("creiLOV", source=data.DatasetSource.SGPO)
    tokens = data.tokenizer.encode(dataset.target_sequence)
    design_space = spaces.DesignSpace.from_rules(
        rules=[],
        init_sequences=[tokens],
        space_sampler=spaces.RandomMaskSampler(
            max_mutations=int(len(tokens) * 0.8), n_subspaces=1, seed=0
        ),
    )
    # docs-design-space-end
    # docs-sampler-start
    model = models.DplmLogitsModel(
        models.DplmModels.DPLM_150M, batch_size=128, trust_remote_code=True
    )
    sampler = samplers.ReMDMSampler(
        model=model, n_steps=16, max_samples=128, batch_size=None, seed=0
    )
    # docs-sampler-end
    # docs-pipeline-start
    constraint = StructureConfidenceConstraint()
    with tempfile.TemporaryDirectory() as tmp:
        store = optim.CampaignStore(protein_length=len(tokens), location=tmp)
        pipeline = optim.Pipeline(
            design_space=design_space,
            sampler=sampler,
            hard_constraints={"structure_confidence": constraint},
        )
        output = pipeline.run(store=store)
        total = sum(1 for _ in store.iter_samples(cycle_id=1, exclude_failed=False))
        print(
            f"{len(output.selection.token_ids)} / {total} generated candidates passed "
            f"mean pLDDT >= {constraint.plddt_threshold} and "
            f"pTM >= {constraint.ptm_threshold}."
        )
    # docs-pipeline-end


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"total runtime: {time.perf_counter() - start:.2f} seconds")
