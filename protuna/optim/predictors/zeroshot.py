"""Predictors that score fitness using evolutionary likelihood proxies.

These wrap a zero-shot scoring function from :mod:`protuna.score` behind the
:class:`~protuna.optim.predictors.base.PredictorLike` interface, so they can be used
as the objective of a campaign or as the model behind a
:class:`~protuna.optim.constraints.base.ModelBasedConstraint`.
"""

from collections.abc import Sequence
import dataclasses

from jaxtyping import Float
import torch

from protuna import data
from protuna import inference
from protuna.optim.predictors import base
from protuna.score import mlm
from protuna.score import pssm
from protuna.typing import BatchTokenIDs, SequenceLike, TokenIDs
from protuna.utils import devices


@dataclasses.dataclass
class _MLMLogitsOutput:
    logits: Float[torch.Tensor, "b l v"]


class _MLMLogitsWrapper(mlm.LogitsModel):
    def __init__(self, model: inference.MLMForLogitsLike):
        self.model = model

    def __call__(self, batch: BatchTokenIDs) -> _MLMLogitsOutput:
        outputs = self.model(
            [inference.MLMLogitsRequest(token_ids) for token_ids in batch]
        )
        logits = torch.stack(list(outputs))
        return _MLMLogitsOutput(logits)


class MaskedMarginalPredictor(
    inference.HandlerRegistryMixin,
    inference.InferenceModel[base.PredictionRequest, base.Prediction],
):
    """Score candidates by the masked-marginal heuristic.

    Uses :func:`~protuna.score.mlm.masked_marginal_inference`; see its
    documentation for details of the score computation.
    """

    def __init__(
        self,
        wt_tokens: TokenIDs,
        model: inference.MLMForLogitsLike,
        *,
        wt_temp: float = 1.0,
        mt_temp: float = 1.0,
        batch_size: int | None = None,
    ):
        """Initialize the masked-marginal predictor.

        Args:
            wt_tokens: The tokenized wild-type sequence to score against.
            model: The masked language model or its served handle.
            wt_temp: Temperature applied to the wild-type logits.
            mt_temp: Temperature applied to the mutant logits.
            batch_size: Candidates per forward pass; ``None`` scores all at once.
        """
        super().__init__(batch_size=batch_size)
        self.model = model
        self._model_wrapper = _MLMLogitsWrapper(model)
        self.wt_tokens = wt_tokens
        self.wt_temp = wt_temp
        self.mt_temp = mt_temp
        self.model_output_cfg = inference.MLMOutputConfig(return_logits=True)
        self._device = None

    def forward(
        self, requests: SequenceLike[base.PredictionRequest]
    ) -> list[base.Prediction]:
        """Score the candidates against the wild-type sequence.

        Args:
            requests: Candidate sequences to score.

        Returns:
            Predictions with scores in ``mean`` and no uncertainty.

        Raises:
            ValueError: A request asks for standard deviation or samples.
        """
        if any(request.return_std or request.return_samples for request in requests):
            raise ValueError("MaskedMarginalPredictor does not support std or samples.")
        mt_tokens = torch.stack([request.token_ids for request in requests])
        scores = mlm.masked_marginal_inference(
            model=self._model_wrapper,
            wt_tokens=self.wt_tokens[None, :],
            mt_tokens=mt_tokens,
            mask_token_id=data.tokenizer.mask_token_id,
            wt_temp=self.wt_temp,
            mt_temp=self.mt_temp,
            return_logits=False,
            batch_size=len(requests) if self.batch_size is None else self.batch_size,
        )
        return [base.Prediction(score) for score in scores.score]


class WildTypeMarginalPredictor(
    inference.HandlerRegistryMixin,
    inference.InferenceModel[base.PredictionRequest, base.Prediction],
):
    """Score candidates by the wild-type marginal heuristic.

    Uses :func:`~protuna.score.mlm.wildtype_marginal_from_logits`; see its
    documentation for details of the score computation.
    """

    def __init__(
        self,
        wt_tokens: TokenIDs,
        model: inference.MLMForLogitsLike,
        *,
        wt_temp: float = 1.0,
        mt_temp: float = 1.0,
        batch_size: int | None = None,
    ):
        """Initialize the wild-type marginal predictor.

        Args:
            wt_tokens: The tokenized wild-type sequence to score against.
            model: The masked language model or its served handle.
            wt_temp: Temperature applied to the wild-type logits.
            mt_temp: Temperature applied to the mutant logits.
            batch_size: Candidates per forward pass; ``None`` scores all at once.
        """
        super().__init__(batch_size=batch_size)
        self.model = model
        self.wt_tokens = wt_tokens
        self.wt_temp = wt_temp
        self.mt_temp = mt_temp
        self.model_output_cfg = inference.MLMOutputConfig(return_logits=True)
        self._wt_logits = None

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load the language model and discard cached wild-type logits.

        Args:
            device: Device or sequence of devices passed to the language model's
                ``load`` method. ``None`` uses the model's default device selection.
        """
        self._wt_logits = None
        super().load(device)

    def release(self) -> None:
        """Release the language model and cached wild-type GPU resources."""
        self._wt_logits = None
        self.wt_tokens = self.wt_tokens.cpu()
        super().release()

    def forward(
        self, requests: SequenceLike[base.PredictionRequest]
    ) -> list[base.Prediction]:
        """Score the candidates against the wild-type sequence.

        Args:
            requests: Candidate sequences to score.

        Returns:
            Predictions with scores in ``mean`` and no uncertainty.

        Raises:
            ValueError: A request asks for standard deviation or samples.
        """
        if any(request.return_std or request.return_samples for request in requests):
            raise ValueError(
                "WildTypeMarginalPredictor does not support std or samples."
            )
        if self._wt_logits is None:
            # l, v
            self._wt_logits = self.model(
                [inference.MLMLogitsRequest(self.wt_tokens, aa_tokens_only=True)]
            )[0]
            self.wt_tokens = self.wt_tokens.to(device=self._wt_logits.device)
        mt_tokens = torch.stack([request.token_ids for request in requests])
        mt_tokens = mt_tokens.to(device=self._wt_logits.device)
        scores = mlm.wildtype_marginal_from_logits(
            self._wt_logits[None],
            self.wt_tokens[None],
            mt_tokens,
            wt_temp=self.wt_temp,
            mt_temp=self.mt_temp,
        )
        return [base.Prediction(score) for score in scores.score]


class MutantMarginalPredictor(
    inference.HandlerRegistryMixin,
    inference.InferenceModel[base.PredictionRequest, base.Prediction],
):
    """Score candidates by the mutant-marginal heuristic.

    Uses :func:`~protuna.score.mlm.mutant_marginal`; see its documentation
    for details of the score computation.
    """

    def __init__(
        self,
        wt_tokens: TokenIDs,
        model: inference.MLMForLogitsLike,
        *,
        wt_temp: float = 1.0,
        mt_temp: float = 1.0,
        batch_size: int | None = None,
    ):
        """Initialize the mutant-marginal predictor.

        Args:
            wt_tokens: The tokenized wild-type sequence to score against.
            model: The masked language model or its served handle.
            wt_temp: Temperature applied to the wild-type logits.
            mt_temp: Temperature applied to the mutant logits.
            batch_size: Candidates per forward pass; ``None`` scores all at once.
        """
        super().__init__(batch_size=batch_size)
        self.model = model
        self._model_wrapper = _MLMLogitsWrapper(model)
        self.wt_tokens = wt_tokens
        self.wt_temp = wt_temp
        self.mt_temp = mt_temp
        self.model_output_cfg = inference.MLMOutputConfig(return_logits=True)
        self._device = None

    def forward(
        self, requests: SequenceLike[base.PredictionRequest]
    ) -> list[base.Prediction]:
        """Score the candidates against the wild-type sequence.

        Args:
            requests: Candidate sequences to score.

        Returns:
            Predictions with scores in ``mean`` and no uncertainty.

        Raises:
            ValueError: A request asks for standard deviation or samples.
        """
        if any(request.return_std or request.return_samples for request in requests):
            raise ValueError("MutantMarginalPredictor does not support std or samples.")
        mt_tokens = torch.stack([request.token_ids for request in requests])
        scores = mlm.mutant_marginal(
            model=self._model_wrapper,
            wt_tokens=self.wt_tokens[None, :],
            mt_tokens=mt_tokens,
            wt_temp=self.wt_temp,
            mt_temp=self.mt_temp,
        )
        return [base.Prediction(score) for score in scores.score]


class PSSMZeroshotPredictor(
    inference.InferenceModel[base.PredictionRequest, base.Prediction],
):
    """Score candidates against wild-type residue frequencies in an MSA.

    Uses :func:`~protuna.score.pssm.compute_frequency_matrix` and
    :func:`~protuna.score.pssm.compute_msa_frequency_score`; see their
    documentation for details of the frequency and score computations.
    """

    def __init__(
        self,
        wt_tokens: TokenIDs,
        msa_tokens: data.TokenizedMSA,
        *,
        use_sample_reweighting: bool = False,
        aa_cols_only: bool = True,
        laplace_smoothing: bool = False,
        pseudocount: float = 1e-5,
        softmax_transform: bool = False,
        batch_size: int | None = None,
    ):
        """Initialize the position specific scoring matrix predictor.

        Args:
            wt_tokens: Tokenized wild-type sequence to score against.
            msa_tokens: Alignment with columns matching the wild-type sequence.
            use_sample_reweighting: Weight aligned sequences by inverse neighbour
                counts when computing frequencies.
            aa_cols_only: Restrict frequency columns to the 20 standard amino acids.
            laplace_smoothing: Add a pseudocount before normalizing residue counts.
            pseudocount: Count added when ``laplace_smoothing`` is enabled.
            softmax_transform: Apply log-softmax to frequencies instead of log.
            batch_size: Candidates per forward pass; ``None`` scores all at once.
        """
        super().__init__(batch_size=batch_size)
        self.msa_tokens = msa_tokens
        self.wt_tokens = wt_tokens
        self.use_sample_reweighting = use_sample_reweighting
        self.aa_cols_only = aa_cols_only
        self.laplace_smoothing = laplace_smoothing
        self.pseudocount = pseudocount
        self.softmax_transform = softmax_transform
        self.freq_matrix = None

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Move the MSA and wild type onto the first selected device.

        Args:
            device: Device or sequence of devices. ``None`` selects an available
                device automatically.
        """
        if device is None:
            device = devices.resolve_devices()
        if not isinstance(device, torch.device):
            device = device[0]
        self.msa_tokens = self.msa_tokens.to(device)
        self.wt_tokens = self.wt_tokens.to(device)

    def release(self) -> None:
        """Return the MSA and wild type to CPU and discard cached frequencies."""
        self.msa_tokens = self.msa_tokens.to(torch.device("cpu"))
        self.wt_tokens = self.wt_tokens.to(torch.device("cpu"))
        self.freq_matrix = None

    def forward(
        self, requests: SequenceLike[base.PredictionRequest]
    ) -> list[base.Prediction]:
        """Score the candidates against the wild-type sequence.

        Args:
            requests: Candidate sequences to score.

        Returns:
            Predictions with scores in ``mean`` and no uncertainty.

        Raises:
            ValueError: A request asks for standard deviation or samples.
        """
        if any(request.return_std or request.return_samples for request in requests):
            raise ValueError("PSSMZeroshotPredictor does not support std or samples.")
        if self.freq_matrix is None:
            self.freq_matrix = pssm.compute_frequency_matrix(
                self.msa_tokens,
                use_sample_reweighting=self.use_sample_reweighting,
                aa_cols_only=self.aa_cols_only,
                laplace_smoothing=self.laplace_smoothing,
                pseudocount=self.pseudocount,
            )
        mt_tokens = torch.stack([request.token_ids for request in requests])
        mt_tokens = mt_tokens.to(self.freq_matrix.device)
        scores = pssm.compute_msa_frequency_score(
            self.wt_tokens[None],
            mt_tokens,
            self.freq_matrix,
            softmax_transform=self.softmax_transform,
        )
        return [base.Prediction(score) for score in scores]
