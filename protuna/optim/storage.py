"""Storage management for candidates, constraint results, and evaluations.

:class:`CampaignStore` buffers proposed sequences, constraint results, and
objective predictions, writing them to Parquet in chunks, reading or streaming from
the offloaded results as needed.

Storage can use a local directory or a cloud URI. Records are written under
``location/pipelines/task_id/``, with a generated task id unless one is supplied.
If ``location`` is omitted, the protuna cache directory is used.

Cloud options and credential providers are passed to Polars; see the
`Polars cloud authentication guide
<https://docs.pola.rs/user-guide/io/cloud-storage/#cloud-authentication>`_
for different authentication methods.

Examples:
    To store a campaign on S3, create a bucket and configure AWS credentials with
    access to it, for example through the ``AWS_ACCESS_KEY_ID`` and
    ``AWS_SECRET_ACCESS_KEY`` environment variables. Replace the bucket and region
    below with your own.

    >>> import torch
    >>> from protuna.optim import storage
    >>> store = storage.CampaignStore(  # doctest: +SKIP
    ...     protein_length=3,
    ...     location="s3://my-protuna-bucket/campaigns",
    ...     storage_options={"aws_region": "eu-west-1"},
    ... )
    >>> store.store_base_sequences(  # doctest: +SKIP
    ...     samples=torch.tensor([[1, 2, 3], [1, 4, 3]]),
    ... )
    >>> store.finalize()  # doctest: +SKIP
"""

import collections
from collections.abc import Iterable, Iterator, Sequence
import dataclasses
import itertools
import logging
import math
import os
import pathlib
import sys
from typing import Any, Literal
import uuid

from jaxtyping import Bool, Float
import numpy as np
import polars as pl
import torch

from protuna.optim import candidates
from protuna.optim import predictors
from protuna.typing import BatchTokenIDs, TokenIDs
from protuna.utils import devices
from protuna.utils import paths

logger = logging.getLogger(__name__)

BASE_SEQUENCE_CYCLE_ID = 0


def _estimate_value_size(value: object) -> int:
    size = sys.getsizeof(value)
    if isinstance(value, torch.Tensor) or (
        isinstance(value, np.ndarray) and not value.flags.owndata
    ):
        size += value.nbytes
    return size


class _StoreIO:
    def __init__(
        self,
        schema: dict[str, Any],
        prefix: str,
        max_size_bytes: float,
        storage_options: dict[str, Any] | None = None,
        credential_provider: pl.CredentialProvider | Literal["auto"] | None = "auto",
    ):
        self.max_size_bytes = max_size_bytes
        self._chunk_size_bytes = 0
        self.prefix = prefix
        self.schema = schema
        self._chunk_count = itertools.count(start=1)
        self._chunk = collections.defaultdict[str, list](list)
        self.storage_options = storage_options
        self.credential_provider = credential_provider

    def _buffer_frame(self) -> pl.DataFrame:
        tensors = [
            value
            for name in self.schema
            for value in self._chunk[name]
            if isinstance(value, torch.Tensor)
        ]
        moved = iter(devices.to_device_batch(tensors, torch.device("cpu")))
        columns = []
        for name in self.schema:
            column = []
            for value in self._chunk[name]:
                if isinstance(value, torch.Tensor):
                    tensor = next(moved)
                    value = tensor.item() if tensor.ndim == 0 else tensor.numpy()
                column.append(value)
            columns.append(column)
        return pl.DataFrame(
            zip(*columns, strict=True),
            schema=self.schema,
            orient="row",
        )

    def flush(self) -> None:
        if self._chunk:
            file_path = f"{self.prefix}part{next(self._chunk_count):05d}.parquet"
            num_rows = len(next(iter(self._chunk.values())))
            logger.debug("Writing %s buffered rows to %s", num_rows, file_path)
            self._buffer_frame().write_parquet(
                file_path,
                mkdir=True,
                storage_options=self.storage_options,
                credential_provider=self.credential_provider,
            )
            self._chunk = collections.defaultdict(list)
            self._chunk_size_bytes = 0
            logger.debug("Wrote %s rows to %s", num_rows, file_path)

    def store(self, batch: dict[str, Iterable], flush: bool = False) -> None:
        for key, values in batch.items():
            column = self._chunk[key]
            previous_size = sys.getsizeof(column)
            for value in values:
                column.append(value)
                self._chunk_size_bytes += _estimate_value_size(value)
            self._chunk_size_bytes += sys.getsizeof(column) - previous_size
        # Count shared values conservatively and reserve 2x headroom for writing.
        if 2 * self._chunk_size_bytes >= self.max_size_bytes or flush:
            self.flush()

    def get_lf(self) -> pl.LazyFrame:
        lf = pl.scan_parquet(
            f"{self.prefix}part*.parquet",
            schema=self.schema,
            storage_options=self.storage_options,
            credential_provider=self.credential_provider,
        )
        if self._chunk:
            last_lf = self._buffer_frame().lazy()
            lf = pl.concat([lf, last_lf])
        return lf


@dataclasses.dataclass(frozen=True)
class SampleRow:
    """A row of stored samples."""

    seq_id: int | None = None
    token_ids: TokenIDs | None = None
    parent_id: int | None = None
    ancestor_id: int | None = None
    cycle_id: int | None = None


class CandidateStreamFromStorage(candidates.CandidateStream):
    """Stream stored candidates with predictions and soft constraint probabilities.

    Iteration yields :class:`~protuna.optim.candidates.CandidateBatch` objects for the
    selected cycle, or all candidate cycles when ``cycle_id`` is ``None``, with at
    most ``scan_size`` candidates per batch. Only candidate cycles are included.

    Objective fields with no values in a batch are ``None``. Soft constraints with no
    results in a batch are omitted from``feasibility``, which is ``None`` when no soft
    constraint results are available.

    Missing values within populated evaluation columns are ``NaN``. Candidates
    with any failed hard constraint are excluded. :meth:`candidates_from_ids`
    retrieves stored candidates in the requested order with available evaluations.

    Both operations include buffered records and records already written to Parquet.
    """

    def __init__(
        self,
        scan_size: int,
        cycle_id: int | None,
        soft_constraint_names: list[str],
        samples_store: _StoreIO,
        objective_store: _StoreIO,
        hard_constraint_store: _StoreIO,
        soft_constraint_store: _StoreIO,
    ):
        """Configure candidate scans over campaign stores.

        Args:
            scan_size: Maximum rows yielded in each candidate batch.
            cycle_id: Candidate cycle to scan; ``None`` includes all nonzero cycles.
            soft_constraint_names: Soft constraints to include as feasibility columns.
            samples_store: Store of sequence tokens and campaign identifiers.
            objective_store: Store of objective means, uncertainties and samples.
            hard_constraint_store: Store of pass/fail constraint results.
            soft_constraint_store: Store of soft constraint probabilities.

        Raises:
            ValueError: If ``cycle_id`` is ``BASE_SEQUENCE_CYCLE_ID``.
        """
        if cycle_id == BASE_SEQUENCE_CYCLE_ID:
            raise ValueError(
                f"Invalid cycle id {cycle_id}. Only candidate cycles"
                " are supported. Received base sequence cycle id."
            )
        self._cycle_id = cycle_id
        self._scan_size = scan_size
        self._soft_constraint_names = soft_constraint_names
        self._samples_store = samples_store
        self._objective_store = objective_store
        self._hard_constraint_store = hard_constraint_store
        self._soft_constraint_store = soft_constraint_store

    def _make_candidate_batch(
        self, batch: pl.DataFrame, constraint_columns: dict[str, str]
    ) -> candidates.CandidateBatch:
        objective_columns = ["objective_mean", "objective_std", "objective_samples"]
        objective_data = {}
        for col in objective_columns:
            if batch[col].null_count() == batch.height:
                data = None
            elif col == "objective_samples":
                width = len(batch["objective_samples"].drop_nulls()[0])
                data = batch[col].cast(pl.Array(pl.Float32, width)).to_torch()
            else:
                data = batch[col].to_torch()
            objective_data[col] = data
        constraints_data = {
            name: batch[column].to_torch()
            for name, column in constraint_columns.items()
            if batch[column].null_count() < batch.height
        } or None
        batch = batch.drop(list(constraint_columns.values()) + objective_columns)
        if batch.height:
            batch = batch.with_columns(
                pl.col("token_ids").cast(
                    pl.Array(pl.Int64, batch["token_ids"][0].shape)
                )
            )
        candidates_ = candidates.CandidateBatch(
            **batch.to_torch("dict"),
            **objective_data,
            feasibility=constraints_data,
        )
        return candidates_

    def __iter__(self) -> Iterator[candidates.CandidateBatch]:
        """Yield candidates that pass hard constraints.

        Yields:
            Batches of candidates with available evaluation results.
        """
        samples_lf = self._samples_store.get_lf()
        if self._cycle_id is not None:
            samples_lf = samples_lf.filter(pl.col("cycle_id") == self._cycle_id)
        else:
            samples_lf = samples_lf.filter(pl.col("cycle_id") != BASE_SEQUENCE_CYCLE_ID)
        hard_constraints_lf = (
            self._hard_constraint_store.get_lf()
            .filter(~pl.col("passed"))
            .select("seq_id")
        )
        samples_lf = samples_lf.join(hard_constraints_lf, on="seq_id", how="anti")
        objective_lf = self._objective_store.get_lf()
        samples_lf = samples_lf.join(objective_lf, on="seq_id", how="left")
        constraint_columns = {
            name: f"feasibility::{name}" for name in self._soft_constraint_names
        }
        if constraint_columns:
            constraints_lf = (
                self._soft_constraint_store.get_lf()
                .pivot(
                    on="constraint",
                    on_columns=list(self._soft_constraint_names),
                    index="seq_id",
                    values="feasibility",
                )
                .rename(constraint_columns)
            )
            samples_lf = samples_lf.join(constraints_lf, on="seq_id", how="left")

        for batch in samples_lf.collect_batches(chunk_size=self._scan_size):
            yield self._make_candidate_batch(batch, constraint_columns)

    def candidates_from_ids(self, seq_ids: Sequence[int]) -> candidates.CandidateBatch:
        """Return stored candidates and available evaluations in identifier order.

        Args:
            seq_ids: Sequence identifiers to retrieve.

        Returns:
            A batch of candidates with the requested ``seq_ids``.
        """
        samples_lf = (
            self._samples_store.get_lf()
            .filter(pl.col("cycle_id") != BASE_SEQUENCE_CYCLE_ID)
            .filter(pl.col("seq_id").is_in(seq_ids))
        )
        objective_lf = self._objective_store.get_lf().filter(
            pl.col("seq_id").is_in(seq_ids)
        )
        samples_lf = samples_lf.join(objective_lf, on="seq_id", how="left")
        constraint_columns = {
            name: f"feasibility::{name}" for name in self._soft_constraint_names
        }
        if constraint_columns:
            constraints_lf = (
                self._soft_constraint_store.get_lf()
                .filter(pl.col("seq_id").is_in(seq_ids))
                .pivot(
                    on="constraint",
                    on_columns=list(self._soft_constraint_names),
                    index="seq_id",
                    values="feasibility",
                )
                .rename(constraint_columns)
            )
            samples_lf = samples_lf.join(constraints_lf, on="seq_id", how="left")
        samples_df = samples_lf.collect()
        order = {seq_id: i for i, seq_id in enumerate(seq_ids)}
        samples_df = samples_df.sort(by=pl.col("seq_id").replace_strict(order))
        return self._make_candidate_batch(samples_df, constraint_columns)


class CampaignStore:
    """Manage generated samples, constraint results, and objective predictions.

    :meth:`store_samples` assigns unique sequence ids, and records parent, ancestor,
    and cycle ids. Missing parent and ancestor ids are stored as zero. The stored
    candidates with their associated sequence ids can be accessed using
    :meth:`iter_samples`; base sequences use :meth:`iter_base_sequences`.

    :meth:`store_constraint` is used to store constraints and :meth:`store_objective`
    for storing objective predictions. Constraints listed in ``soft_constraint_names``
    are stored as feasibility probabilities; all others are stored as pass/fail
    verdicts.

    Samples, objective predictions, hard constraints, and soft constraints have
    separate buffers, each governed by ``max_size_mb``. Records are written to
    Parquet as buffers fill. Call :meth:`finalize` to write all remaining records
    before discarding the store.

    :meth:`get_candidate_stream` returns a :class:`CandidateStreamFromStorage`
    over generated candidates with available evaluations, excluding candidates that
    failed a hard constraint.

    Starting sequences can be accessed through :meth:`iter_base_sequences`.

    :meth:`get_samples_unordered` returns a list of :class:`SampleRow` of requested
    sequence ids in arbitrary order.

    All Reading operations include buffered records as well as records already offloaded
    to Parquet.
    """

    def __init__(
        self,
        *,
        protein_length: int,
        location: str | pathlib.Path | None = None,
        max_size_mb: float = 500,
        scan_size: int = 100_000,
        storage_options: dict[str, Any] | None = None,
        soft_constraint_names: Sequence[str] | None = None,
        credential_provider: pl.CredentialProvider | Literal["auto"] | None = "auto",
        task_id: str | None = None,
    ):
        """Initialize campaign storage and its record buffers.

        Args:
            protein_length: Number of tokens in each stored sequence.
            location: Base directory or cloud URI for campaign records. Records
                are written under ``location/pipelines/task_id/``. ``None`` uses
                the protuna cache directory as the base.
            max_size_mb: Approximate memory budget in MiB for each of the four
                record buffers. Must be positive and finite. Checked after each batch,
                so a large batch can exceed the budget.
            scan_size: Maximum rows per candidate batch or sample read chunk
                during iteration.
            storage_options: Cloud connection options passed to Polars for both
                reads and writes. ``None`` uses Polars defaults.
            soft_constraint_names: Constraint names to store as feasibility
                probabilities. All other constraints are stored as pass/fail
                verdicts. ``None`` treats every constraint as hard.
            credential_provider: Cloud credential provider passed to Polars for
                both reads and writes. ``"auto"`` lets Polars infer a provider;
                ``None`` disables the credential provider.
            task_id: Campaign identifier used in the storage path. ``None``
                generates a new UUID in hexadecimal form.

        Raises:
            ValueError: If ``max_size_mb`` is not positive and finite.
        """
        if not math.isfinite(max_size_mb) or max_size_mb <= 0:
            raise ValueError("max_size_mb must be positive and finite")
        if task_id is None:
            task_id = uuid.uuid4().hex
            logger.debug("Generated new task id for store, task_id=%r", task_id)

        if location is None:
            location = str(paths.get_cache_dir() / f"pipelines/{task_id}")
        else:
            location = os.path.join(location, f"pipelines/{task_id}")

        self._soft_constraint_names = (
            set(soft_constraint_names) if soft_constraint_names is not None else set()
        )

        self._scan_size = scan_size
        self.max_size_mb = max_size_mb
        self._seq_id_counter = itertools.count(1)
        self._cycle_id_counter = itertools.count(1)

        sample_schema = {
            "seq_id": pl.UInt32,
            "token_ids": pl.Array(pl.UInt8, protein_length),
            "parent_id": pl.UInt32,
            "ancestor_id": pl.UInt32,
            "cycle_id": pl.UInt32,
        }
        objective_schema = {
            "seq_id": pl.UInt32,
            "objective_mean": pl.Float32,
            "objective_std": pl.Float32,
            "objective_samples": pl.List(pl.Float32),
        }
        hard_constraint_schema = {
            "seq_id": pl.UInt32,
            "constraint": pl.Categorical,
            "passed": pl.Boolean,
        }
        soft_constraint_schema = {
            "seq_id": pl.UInt32,
            "constraint": pl.Categorical,
            "feasibility": pl.Float32,
        }
        max_size_bytes = max_size_mb * 1024 * 1024

        self.samples_store = _StoreIO(
            sample_schema,
            f"{location}/samples_",
            max_size_bytes=max_size_bytes,
            storage_options=storage_options,
            credential_provider=credential_provider,
        )
        self.objective_store = _StoreIO(
            objective_schema,
            f"{location}/objective_",
            max_size_bytes=max_size_bytes,
            storage_options=storage_options,
            credential_provider=credential_provider,
        )
        self.hard_constraint_store = _StoreIO(
            hard_constraint_schema,
            f"{location}/hard_constraints_",
            max_size_bytes=max_size_bytes,
            storage_options=storage_options,
            credential_provider=credential_provider,
        )
        self.soft_constraint_store = _StoreIO(
            soft_constraint_schema,
            f"{location}/soft_constraints_",
            max_size_bytes=max_size_bytes,
            storage_options=storage_options,
            credential_provider=credential_provider,
        )
        logger.debug(
            "Campaign store initialized: task_id=%s, location=%s", task_id, location
        )

    def _prepare_samples(
        self, samples: BatchTokenIDs, parent_id: int, ancestor_id: int, cycle_id: int
    ) -> dict[str, Iterable]:
        batch = {}
        batch["seq_id"] = itertools.islice(self._seq_id_counter, samples.shape[0])
        batch["token_ids"] = samples.byte().cpu().contiguous().numpy()
        batch["parent_id"] = itertools.repeat(parent_id, samples.shape[0])
        batch["ancestor_id"] = itertools.repeat(ancestor_id, samples.shape[0])
        batch["cycle_id"] = itertools.repeat(cycle_id, samples.shape[0])
        return batch

    def get_next_cycle_id(self) -> int:
        """Return a new unique cycle id."""
        return next(self._cycle_id_counter)

    def store_base_sequences(
        self, sequences: BatchTokenIDs, flush: bool = False
    ) -> None:
        """Record a batch of starting sequences and assign each a unique sequence id.

        Args:
            sequences: Tokenized sequences to record.
            flush: Whether to write all buffered samples to Parquet immediately.
        """
        batch = self._prepare_samples(sequences, parent_id=0, ancestor_id=0, cycle_id=0)
        self.samples_store.store(batch, flush=flush)

    def store_samples(
        self,
        samples: BatchTokenIDs,
        parent_id: int,
        ancestor_id: int,
        cycle_id: int,
        flush: bool = False,
    ) -> None:
        """Record a batch of generated candidates and assign each a unique sequence id.

        Args:
            samples: Tokenized sequences to record.
            parent_id: Immediate parent sequence id from which the sample evolved.
            ancestor_id: Ancestor sequence id indicating the starting sequence from
                which the sample evolved.
            cycle_id: Campaign cycle id that generated the batch. New cycle ids
                should be generated using :meth:`get_next_cycle_id`.
            flush: Whether to write all buffered samples to Parquet immediately.
        """
        batch = self._prepare_samples(samples, parent_id, ancestor_id, cycle_id)
        self.samples_store.store(batch, flush=flush)

    def store_constraint(
        self,
        constraint_id: str,
        seq_id: int,
        value: Bool[torch.Tensor, ""] | Float[torch.Tensor, ""],
    ) -> None:
        """Record a constraint result for a sequence.

        Constraints listed in ``soft_constraint_names`` are stored as feasibility
        probabilities; all others are stored as pass/fail verdicts. Tensor values
        remain on their original device until read or flushed.

        Args:
            constraint_id: Constraint name used to distinguish results and
                determine whether the constraint is soft or hard.
            seq_id: Sequence id assigned by :meth:`store_samples`.
            value: Feasibility probability for a soft constraint, or a boolean
                value for a hard constraint, where ``True`` means passed.
        """
        if constraint_id in self._soft_constraint_names:
            self.soft_constraint_store.store(
                {
                    "seq_id": [seq_id],
                    "constraint": [constraint_id],
                    "feasibility": [value],
                }
            )
        else:
            self.hard_constraint_store.store(
                {
                    "seq_id": [seq_id],
                    "constraint": [constraint_id],
                    "passed": [value],
                }
            )

    def store_objective(self, seq_id: int, prediction: predictors.Prediction) -> None:
        """Record an objective prediction for a sequence.

        Prediction tensors remain on their original device until read or flushed.

        Args:
            seq_id: Sequence id assigned by :meth:`store_samples`.
            prediction: Objective mean and any available standard deviation or
                objective samples. Missing optional values are stored as nulls.
        """
        self.objective_store.store(
            {
                "seq_id": [seq_id],
                "objective_mean": [prediction.mean],
                "objective_std": [
                    prediction.std if prediction.std is not None else None
                ],
                "objective_samples": [
                    prediction.samples if prediction.samples is not None else None
                ],
            }
        )

    def get_candidate_stream(
        self, cycle_id: int | None = None
    ) -> CandidateStreamFromStorage:
        """Return candidates with available evaluations that pass hard constraints.

        Args:
            cycle_id: Candidate cycle to scan; ``None`` includes all candidate cycles.

        Returns:
            A :class:`protuna.optim.candidates.CandidateStream` from the underlying
            disk storage.

        Raises:
            ValueError: If ``cycle_id`` is zero.
        """
        return CandidateStreamFromStorage(
            scan_size=self._scan_size,
            cycle_id=cycle_id,
            soft_constraint_names=list(self._soft_constraint_names),
            samples_store=self.samples_store,
            objective_store=self.objective_store,
            hard_constraint_store=self.hard_constraint_store,
            soft_constraint_store=self.soft_constraint_store,
        )

    def iter_samples(
        self, cycle_id: int | None = None, exclude_failed: bool = True
    ) -> Iterator[tuple[int, TokenIDs]]:
        """Yield stored candidate sequence ids and tokens.

        Samples are read in chunks of up to ``scan_size`` rows and yielded one
        at a time. No ordering is guaranteed.

        Args:
            cycle_id: Candidate cycle to scan; ``None`` includes all candidate cycles.
            exclude_failed: Whether to skip sequences with any failed hard
                constraint. Sequences without constraint results are included.

        Yields:
            A sequence id and its token tensor on the CPU.

        Raises:
            ValueError: If ``cycle_id`` is zero.
        """
        if cycle_id == BASE_SEQUENCE_CYCLE_ID:
            raise ValueError(
                "use iter_base_sequences() to retrieve starting sequences."
            )
        lf = self.samples_store.get_lf()
        if cycle_id is not None:
            lf = lf.filter(pl.col("cycle_id") == cycle_id)
        else:
            lf = lf.filter(pl.col("cycle_id") != BASE_SEQUENCE_CYCLE_ID)
        if exclude_failed:
            lf_constraint = (
                self.hard_constraint_store.get_lf()
                .filter(~pl.col("passed"))
                .select("seq_id")
            )
            lf = lf.join(lf_constraint, on="seq_id", how="anti")
        yield from self._iter_sample_rows(lf)

    def iter_base_sequences(self) -> Iterator[tuple[int, TokenIDs]]:
        """Yield starting sequence sequence ids and CPU tokens in registration order.

        Yields:
            A base sequence id and its token tensor on the CPU.
        """
        lf = (
            self.samples_store.get_lf()
            .filter(pl.col("cycle_id") == BASE_SEQUENCE_CYCLE_ID)
            .sort("seq_id")
        )
        yield from self._iter_sample_rows(lf)

    def _iter_sample_rows(self, lf: pl.LazyFrame) -> Iterator[tuple[int, TokenIDs]]:
        lf = lf.select("seq_id", "token_ids")
        for batch in lf.collect_batches(chunk_size=self._scan_size):
            batch_token_ids = batch["token_ids"].to_torch().long()
            yield from zip(batch["seq_id"], batch_token_ids, strict=True)

    def get_samples_unordered(
        self, seq_ids: Sequence[int], columns: Sequence[str] = ("token_ids",)
    ) -> list[SampleRow]:
        """Retrieve selected sample fields for the requested sequence ids.

        Results have no guaranteed order and are not filtered by constraints or
        availability of predictions.

        Args:
            seq_ids: Sequence ids to look up. Unknown ids are omitted.
            columns: Fields to populate in each :class:`SampleRow`: ``seq_id``,
                ``token_ids``, ``parent_id``, ``ancestor_id``, or ``cycle_id``.
                Defaults to tokens only; include ``seq_id`` to identify rows.

        Returns:
            Matching sample rows with the requested fields populated. Token
            tensors are on the CPU.
        """
        df = (
            self.samples_store.get_lf()
            .filter(pl.col("seq_id").is_in(seq_ids))
            .select(*columns)
            .collect()
        )
        return [
            SampleRow(
                seq_id=row.get("seq_id"),
                token_ids=torch.tensor(row["token_ids"], dtype=torch.long)
                if "token_ids" in row
                else None,
                parent_id=row.get("parent_id"),
                ancestor_id=row.get("ancestor_id"),
                cycle_id=row.get("cycle_id"),
            )
            for row in df.iter_rows(named=True)
        ]

    def finalize(self) -> None:
        """Write all buffered samples, predictions, and constraints to Parquet."""
        self.samples_store.flush()
        self.soft_constraint_store.flush()
        self.hard_constraint_store.flush()
        self.objective_store.flush()
