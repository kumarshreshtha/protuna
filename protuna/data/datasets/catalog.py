"""Access protein fitness datasets through a common source catalog.

Use :func:`list_datasets` to discover names within a :class:`DatasetSource`,
then :func:`get_dataset` to retrieve a dataset from that source.
"""

import enum

from protuna.data.datasets import align_bio
from protuna.data.datasets import base
from protuna.data.datasets import flip
from protuna.data.datasets import proteinguide
from protuna.data.datasets import proteingym
from protuna.data.datasets import sgpo


class DatasetSource(enum.StrEnum):
    """Data collection from where the dataset was sourced."""

    PROTEINGYM = "proteingym"
    ALIGN = "align_bio"
    FLIP = "flip"
    SGPO = "sgpo"
    PROTEINGUIDE = "proteinguide"


def list_datasets(source: DatasetSource) -> list[str]:
    """List the names of available mutant fitness datasets.

    Args:
        source: Collection whose dataset names to list.

    Returns:
        Dataset names suitable for passing to :func:`get_dataset`.

    Raises:
        ValueError: If ``source`` is invalid.
    """
    match source:
        case DatasetSource.PROTEINGYM:
            return proteingym.list_datasets()
        case DatasetSource.ALIGN:
            return align_bio.list_datasets()
        case DatasetSource.FLIP:
            return flip.list_datasets()
        case DatasetSource.SGPO:
            return sgpo.list_datasets()
        case DatasetSource.PROTEINGUIDE:
            return proteinguide.list_datasets()
        case _:
            raise ValueError(f"Invalid dataset source {source}")


def get_dataset(name: str, source: DatasetSource) -> base.MutantFitnessDataset:
    """Get a mutant fitness dataset by name.

    Args:
        name: Dataset name within ``source``, as returned by :func:`list_datasets`.
        source: Collection containing the dataset.

    Returns:
        The :class:`~protuna.data.datasets.base.MutantFitnessDataset` for the
        requested dataset.

    Raises:
        ValueError: If ``name`` is not an available dataset or ``source`` is invalid.
    """
    match source:
        case DatasetSource.PROTEINGYM:
            return proteingym.get_dataset(name)
        case DatasetSource.ALIGN:
            return align_bio.get_dataset(name)
        case DatasetSource.FLIP:
            return flip.get_dataset(name)
        case DatasetSource.SGPO:
            return sgpo.get_dataset(name)
        case DatasetSource.PROTEINGUIDE:
            return proteinguide.get_dataset(name)
        case _:
            raise ValueError(f"Invalid dataset source {source}")
