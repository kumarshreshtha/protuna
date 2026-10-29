"""Orchestration of candidate generation, evaluation and selection.

With each new round of wet lab evidence and updated predictors and samplers, a
:class:`Pipeline` takes an optimization step proposing the batch of candidates to
evaluate next:

1. It samples search spaces from :class:`~protuna.optim.spaces.design.DesignSpace`

2. Proposes candidates for each of them with a
   :class:`~protuna.optim.samplers.base.SamplerLike`

3. Evaluates their feasibility using
   :py:type:`~protuna.optim.constraints.base.ConstraintLike`

4. Optionally scores them on the primary objective with
   :py:type:`~protuna.optim.predictors.base.PredictorLike`

5. And returns the batch of sequences an
   :class:`~protuna.optim.acquisitions.base.Acquisition` picks.

With a ``recycler`` the proposal step repeats for several cycles, each cycle
sampling candidates to start over as initial sequences based on the acquisition
criteria of the recycler.

Everything generated in the process is stored on disk (or cloud storage) using
:class:`~protuna.optim.storage.CampaignStore`.
"""

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import hashlib
import itertools
import logging
import tempfile

from jaxtyping import Bool, Integer
import lmdb
import torch

from protuna.optim import acquisitions
from protuna.optim import candidates
from protuna.optim import constraints
from protuna.optim import predictors
from protuna.optim import samplers
from protuna.optim import spaces
from protuna.optim import storage
from protuna.typing import SequenceLike, TokenIDs
from protuna.utils import paths

logger = logging.getLogger(__name__)


def _tokens_fp(token_ids: TokenIDs) -> bytes:
    bytearr = token_ids.byte().cpu().contiguous().numpy()
    return hashlib.blake2b(memoryview(bytearr), digest_size=16).digest()


class _FingerprintIndex:
    def __init__(self):
        self.tmp_dir = tempfile.TemporaryDirectory(dir=paths.get_cache_dir())
        self.index_path = f"{self.tmp_dir.name}/fingerprints.lmdb"
        self.env = lmdb.open(self.index_path)

    def add(self, fingerprints: Sequence[bytes]) -> Bool[torch.Tensor, " n"]:
        while True:
            accepted = torch.empty(len(fingerprints), dtype=torch.bool)
            try:
                with self.env.begin(write=True) as transaction:
                    for index, fp in enumerate(fingerprints):
                        accepted[index] = transaction.put(fp, b"", overwrite=False)
                return accepted
            except lmdb.MapFullError:
                current_size = self.env.info()["map_size"]
                self.env.set_mapsize(current_size * 2)

    def close(self) -> None:
        self.env.close()
        self.tmp_dir.cleanup()


@dataclasses.dataclass(kw_only=True)
class _State:
    design_space: spaces.DesignSpace
    store: storage.CampaignStore
    fp_index: _FingerprintIndex
    ancestor_trail: dict[bytes, tuple[int, int]]
    ancestor_id_to_space: dict[int, spaces.SearchSpace]


@dataclasses.dataclass
class PipelineOutput:
    """Selected candidates and history from a successful pipeline run.

    Args:
        selection: Final batch of candidates selected by the pipeline run.
        store: History of the pipeline run containing all generated samples and their
            evaluations.
    """

    selection: candidates.CandidateBatch
    store: storage.CampaignStore


class Pipeline:
    """Propose the next experimental batch for an optimization campaign.

    Call :meth:`run` to launch the orchestration and get back the requested number of
    candidates for wet-lab validation.
    """

    def __init__(
        self,
        design_space: spaces.DesignSpace,
        sampler: samplers.SamplerLike,
        *,
        objective: predictors.PredictorLike | None = None,
        acquisition: acquisitions.Acquisition | None = None,
        hard_constraints: Mapping[str, constraints.HardConstraintLike] | None = None,
        soft_constraints: Mapping[str, constraints.SoftConstraintLike] | None = None,
        recycler: acquisitions.Acquisition | None = None,
        num_cycles: int = 1,
        release_handlers_after_use: bool = False,
        lock_mutations_on_recycle: bool = False,
        evaluated_mutants: SequenceLike[Integer[torch.Tensor, " l"]] | None = None,
    ):
        """Configure the campaign workflow.

        Args:
            design_space: The starting sequences and allowed mutations from which to
                sample search subspaces for candidate generation.
            sampler: The proposal generator to sample candidates from.
            objective: Predictor for the primary objective. ``None`` skips objective
                evaluation. The responsibility of evaluating and sampling from the
                objective (if needed) is then on the acquisition.
            acquisition: Selector for the final experimental batch from the generated
                candidates. Defaults to selecting all eligible candidates.
            hard_constraints: Hard requirements that the generated candidates must
                satisfy, else they are excluded from further evaluation.
            soft_constraints: Soft requirements whose scores act as feasibility for
                acquisition and recycling.
            recycler: An acquisition to select candidates to use as starting points for
                the next cycle of generations. Required when ``num_cycles > 1``.
            num_cycles: Number of rounds of generation and evaluation before
                selecting the experimental batch.
            release_handlers_after_use: Release each handler's resources after
                its stage completes.
            lock_mutations_on_recycle: Prevent further mutations at sites already
                mutated on sequences selected for recycling.
            evaluated_mutants: Mutants to exclude from sampler proposals.
        """
        if num_cycles > 1 and recycler is None:
            raise ValueError("Must provide a recycling acquisition for num_cycles > 1")
        self.design_space = design_space
        self.acquisition = (
            acquisition
            if acquisition is not None
            else acquisitions.IdentityAcquisition()
        )
        self.objective = objective
        self.sampler = sampler
        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints
        self.recycler = recycler
        self.num_cycles = num_cycles
        self.lock_mutations_on_recycle = lock_mutations_on_recycle
        self.release_handlers_after_use = release_handlers_after_use
        self.evaluated_mutants = evaluated_mutants
        self._requires_std = self.acquisition.requires_std
        self._requires_samples = self.acquisition.requires_samples
        if self.recycler is not None:
            self._requires_std = self._requires_std or self.recycler.requires_std
            self._requires_samples = (
                self._requires_samples or self.recycler.requires_samples
            )

    def _run_sampler(self, cycle_id: int, state: _State) -> bool:
        logger.info("Cycle %s: starting candidate sampling", cycle_id)
        state.design_space.load()
        self.sampler.load()
        num_samples = 0
        num_excluded = 0
        for space_id, space, samples in self.sampler.stream(
            enumerate(state.design_space.sample_search_spaces())
        ):
            samples = samples.cpu()
            if not space.is_valid_batch(samples).all():
                raise ValueError("Sampler produced invalid candidates.")
            fps = [_tokens_fp(sample) for sample in samples]
            added = state.fp_index.add(fps)
            n_added = int(added.sum().item())
            num_excluded += len(samples) - n_added
            if n_added == 0:
                logger.debug(
                    "Cycle %s: no new designs generated for search space %s",
                    cycle_id,
                    space_id,
                )
                continue
            if n_added != samples.shape[0]:
                samples = samples[added]
            parent_hash = _tokens_fp(space.init_sequence)
            ancestor_id, parent_id = state.ancestor_trail[parent_hash]
            state.store.store_samples(samples, parent_id, ancestor_id, cycle_id)
            num_samples += len(samples)
            logger.debug(
                "Cycle %s: stored %s candidates for search space %s parent %s",
                cycle_id,
                len(samples),
                space_id,
                parent_id,
            )

        if self.release_handlers_after_use:
            state.design_space.release()
            self.sampler.release()
        logger.info(
            "Cycle %s: sampling complete; stored %s candidates, excluded %s "
            "duplicate or previously evaluated candidates",
            cycle_id,
            num_samples,
            num_excluded,
        )
        return num_samples != 0

    def _run_constraints(self, cycle_id: int, store: storage.CampaignStore) -> None:
        hard_constraints: Mapping[str, constraints.HardConstraintLike] = (
            {} if self.hard_constraints is None else self.hard_constraints
        )
        soft_constraints: Mapping[str, constraints.SoftConstraintLike] = (
            {} if self.soft_constraints is None else self.soft_constraints
        )
        for name, constraint in itertools.chain(
            hard_constraints.items(), soft_constraints.items()
        ):
            logger.info("Cycle %s: evaluating constraint %r", cycle_id, name)
            constraint.load()
            num_results = 0
            for sample_id, _, result in constraint.stream(
                store.iter_samples(cycle_id=cycle_id)
            ):
                store.store_constraint(name, sample_id, result)
                num_results += 1
            if self.release_handlers_after_use:
                constraint.release()
            logger.info(
                "Cycle %s: constraint %r complete; evaluated %s candidates",
                cycle_id,
                name,
                num_results,
            )

    def _run_objective(self, cycle_id: int, store: storage.CampaignStore) -> None:
        if self.objective is None:
            logger.debug("Cycle %s: skipping objective evaluation", cycle_id)
            return
        logger.info("Cycle %s: starting objective evaluation", cycle_id)
        self.objective.load()

        def pred_request_iterator() -> Iterator[
            tuple[int, predictors.PredictionRequest]
        ]:
            for sample_id, token_ids in store.iter_samples(cycle_id=cycle_id):
                request = predictors.PredictionRequest(
                    token_ids,
                    return_std=self._requires_std,
                    return_samples=self._requires_samples,
                )
                yield sample_id, request

        num_predictions = 0
        for sample_id, _, prediction in self.objective.stream(pred_request_iterator()):
            store.store_objective(sample_id, prediction)
            num_predictions += 1

        if self.release_handlers_after_use:
            self.objective.release()
        logger.info(
            "Cycle %s: objective evaluation complete; scored %s candidates",
            cycle_id,
            num_predictions,
        )

    def _run_cycle(self, cycle_id: int, state: _State) -> bool:
        has_new_samples = self._run_sampler(cycle_id, state)
        if has_new_samples:
            self._run_constraints(cycle_id, state.store)
            self._run_objective(cycle_id, state.store)
        return has_new_samples

    def _set_recycle_spaces(self, cycle_id: int, state: _State) -> None:
        if self.recycler is None:
            return
        logger.info("Cycle %s: selecting parents for the next cycle", cycle_id)
        self.recycler.load()
        recycle_seq_ids = self.recycler.select(
            state.store.get_candidate_stream(cycle_id)
        )
        if self.release_handlers_after_use:
            self.recycler.release()
        recycle_samples = state.store.get_samples_unordered(
            recycle_seq_ids, columns=["seq_id", "ancestor_id", "token_ids"]
        )
        ancestor_trail = {}
        recycle_spaces = []
        for row in recycle_samples:
            assert row.ancestor_id is not None
            assert row.seq_id is not None
            assert row.token_ids is not None
            parent_hash = _tokens_fp(row.token_ids)
            ancestor_trail[parent_hash] = (row.ancestor_id, row.seq_id)
            ancestor_space = state.ancestor_id_to_space[row.ancestor_id]
            if self.lock_mutations_on_recycle:
                mutations = ancestor_space.init_sequence != row.token_ids
                allowed = ancestor_space.allowed & ~mutations[:, None]
                recycle_space = spaces.SearchSpace(
                    init_sequence=row.token_ids, allowed=allowed
                )
            else:
                recycle_space = dataclasses.replace(
                    ancestor_space, init_sequence=row.token_ids
                )
            recycle_spaces.append(recycle_space)
        logger.info(
            "Cycle %s: recycling complete; selected %s parents",
            cycle_id,
            len(recycle_spaces),
        )
        state.ancestor_trail = ancestor_trail
        state.design_space = spaces.DesignSpace(
            init_spaces=recycle_spaces, space_sampler=self.design_space.space_sampler
        )

    def _register_base_sequences(
        self, store: storage.CampaignStore, fp_index: _FingerprintIndex
    ) -> tuple[dict[bytes, tuple[int, int]], dict[int, spaces.SearchSpace]]:
        # store the starting sequences
        base_sequences = torch.stack(
            [space.init_sequence for space in self.design_space.init_spaces]
        )
        n_unique = base_sequences.unique(dim=0).shape[0]
        if n_unique != base_sequences.shape[0]:
            n_dup = base_sequences.shape[0] - n_unique
            raise ValueError(
                "Initial search spaces must contain unique starting points. "
                f"Found {n_dup}/{base_sequences.shape[0]} duplicates."
            )
        store.store_base_sequences(base_sequences)
        ancestor_trail = {}
        id_to_space = {}
        fps = []
        for (sample_id, sample), space in zip(
            store.iter_base_sequences(),
            self.design_space.init_spaces,
            strict=True,
        ):
            fp = _tokens_fp(sample)
            ancestor_trail[fp] = (sample_id, sample_id)
            id_to_space[sample_id] = space
            fps.append(fp)
        fp_index.add(fps)
        if self.evaluated_mutants is not None:
            fps = [_tokens_fp(mutant) for mutant in self.evaluated_mutants]
            fp_index.add(fps)

        return ancestor_trail, id_to_space

    def _get_default_store(self) -> storage.CampaignStore:
        soft_constraint_names = (
            tuple(self.soft_constraints) if self.soft_constraints is not None else None
        )
        store = storage.CampaignStore(
            protein_length=len(self.design_space.init_spaces[0].init_sequence),
            soft_constraint_names=soft_constraint_names,
        )
        return store

    def _get_new_state(self, store: storage.CampaignStore | None) -> _State:
        store = store if store is not None else self._get_default_store()
        fp_index = _FingerprintIndex()
        try:
            ancestor_trail, id_to_space = self._register_base_sequences(store, fp_index)
        except BaseException:
            fp_index.close()
            raise
        return _State(
            design_space=self.design_space,
            store=store,
            fp_index=fp_index,
            ancestor_trail=ancestor_trail,
            ancestor_id_to_space=id_to_space,
        )

    def run(self, *, store: storage.CampaignStore | None = None) -> PipelineOutput:
        """Run the campaign and return selected candidates and run history.

        Stops early when a cycle generates no new candidates and selects from the
        candidates accumulated in earlier cycles.

        Args:
            store: Storage responsible for handling disk offloads and retrieval of
                artifacts during the pipeline. Defaults to a new store under protuna's
                cache directory.

        Returns:
            The final batch of candidates and the run history.

        Raises:
            ValueError: If a sampler candidate violates its supplied search space.
        """
        logger.info("Starting pipeline with %s cycles", self.num_cycles)
        state = self._get_new_state(store)
        try:
            for cycle_count in range(self.num_cycles):
                cycle_id = state.store.get_next_cycle_id()
                logger.info("Starting cycle: %s / %s", cycle_count + 1, self.num_cycles)
                has_new_samples = self._run_cycle(cycle_id, state)
                if not has_new_samples:
                    logger.info(
                        "Early stopping at Cycle %s / %s. No new samples generated.",
                        cycle_id,
                        self.num_cycles,
                    )
                    break
                logger.info("Cycle %s / %s complete", cycle_id, self.num_cycles)
                if cycle_count + 1 != self.num_cycles:  # not last cycle
                    self._set_recycle_spaces(cycle_id, state)
            logger.info("Starting final candidate acquisition")
            candidate_stream = state.store.get_candidate_stream()
            self.acquisition.load()
            seq_ids = self.acquisition.select(candidate_stream)
            if self.release_handlers_after_use:
                self.acquisition.release()
            selection = candidate_stream.candidates_from_ids(seq_ids)
            logger.info(
                "Acquisition complete; selected %s sequences", len(selection.token_ids)
            )
            state.store.finalize()
            return PipelineOutput(store=state.store, selection=selection)
        finally:
            state.fp_index.close()
