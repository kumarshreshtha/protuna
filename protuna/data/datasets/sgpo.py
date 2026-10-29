"""CreiLOV fluorescence fitness dataset from SGPO :cite:p:`SGPO`.

:func:`get_dataset` loads the processed assay from the `SGPO repository
<https://github.com/jsunn-y/SGPO>`_.

Original dataset sources:

* ``creiLOV``: deep mutational scanning of the oxygen-independent fluorescent
  protein CreiLOV by Chen et al. :cite:p:`creiLOV`.
"""

import dataclasses
import pathlib

import huggingface_hub
import polars as pl

from protuna.data.datasets import base
from protuna.utils import download
from protuna.utils import paths

DATA_URL = (
    "https://github.com/jsunn-y/SGPO/raw/refs/heads/main/data/CreiLOV/fitness.csv"
)

WT_SEQ = (
    "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGT"
    "PFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
)

_DATASET_NAME = "creiLOV"


def _load_raw_dataset() -> pl.DataFrame:
    data_path = paths.get_cache_dir() / "creiLOV/fitness.csv"
    if not data_path.exists():
        download.download_from_url(DATA_URL, data_path)
    return pl.read_csv(
        data_path,
        schema_overrides=dict.fromkeys(
            ["log_rep1", "log_rep2", "log_rep3"], pl.Float64
        ),
    )


def _download_from_hub(filename: str) -> str:
    cache_dir = paths.get_cache_dir() / "sgpo/hf_hub"
    cache_dir.mkdir(exist_ok=True, parents=True)
    return huggingface_hub.hf_hub_download(
        "protuna/mutant_fitness",
        filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


_STRUCTURE_PATH_TEMPLATE = "structures/sgpo/{source}/{filename}"
_MSA_PATH_TEMPLATE = "msa/sgpo/{protocol}/{filename}"


@dataclasses.dataclass(frozen=True, repr=False)
class CreiLOVDataset(base.MutantFitnessDataset):
    """Mutant fitness dataset for CreiLOV fluorescence from SGPO :cite:p:`SGPO`.

    Use :func:`get_dataset` to construct an instance.

    Args:
        dataset_name: Name identifying the dataset within its source collection.
        target_sequence: Wild-type amino acid sequence used to identify mutations.
        properties: Names of the measured properties.
    """

    def get_structure_path(self, source: base.StructureSource) -> pathlib.Path:
        """Return the path to the wild-type structure.

        Args:
            source: Source of the structure. Only ``AlphaFold2`` is supported.

        Returns:
            Path to the wild-type ``.pdb`` file.

        Raises:
            NotImplementedError: If ``source`` is not ``AlphaFold2``.
        """
        if source != base.StructureSource.AlphaFold2:
            raise NotImplementedError(
                "Only alphafold2 structures are available for now."
            )
        return pathlib.Path(
            _download_from_hub(
                _STRUCTURE_PATH_TEMPLATE.format(
                    filename=f"{self.dataset_name}.pdb", source=source
                )
            )
        )

    def get_msa_path(self, protocol: base.MSAProtocol) -> pathlib.Path:
        """Return the path to the wild-type multiple sequence alignment.

        Args:
            protocol: Protocol used to construct the MSA. Only ``ColabFold``
                is supported.

        Returns:
            Path to the MSA file.

        Raises:
            NotImplementedError: If ``protocol`` is not ``ColabFold``.
        """
        if protocol != base.MSAProtocol.ColabFold:
            raise NotImplementedError(
                "Only colabfold msa protocol is available for now."
            )
        return pathlib.Path(
            _download_from_hub(
                _MSA_PATH_TEMPLATE.format(
                    filename=f"{self.dataset_name}.a3m", protocol=protocol
                )
            )
        )

    def get_fitness_df(
        self, *, prop: str | None = None, drop_wt: bool = False
    ) -> pl.DataFrame:
        """Return experimental fitness as a Polars dataframe.

        Args:
            prop: Property to include, or ``None`` to include all properties.
            drop_wt: Whether to exclude rows matching the wild-type sequence.
                Defaults to ``False``.

        Returns:
            A dataframe with columns ``mutant``, ``sequence``, ``prop``, ``fitness``
            and ``split``, following
            :class:`~protuna.data.datasets.base.FitnessDFSchema`.

        Raises:
            ValueError: If ``prop`` is not a property of this dataset.
        """
        if prop is not None and prop != self.properties[0]:
            raise ValueError(
                f"{prop=} is not available for dataset {self.dataset_name}."
            )
        raw_df = _load_raw_dataset()
        raw_df = raw_df.select(
            "mut", "Combo", "split", "log_rep1", "log_rep2", "log_rep3"
        )
        raw_df = raw_df.unpivot(
            index=["mut", "Combo", "split"],
            on=["log_rep1", "log_rep2", "log_rep3"],
            value_name="fitness",
        ).drop("variable")
        raw_df = raw_df.with_columns(
            pl.lit(self.properties[0]).alias("prop"),
            pl.col("split").replace({"other": "test", "validation": "train"}),
        )
        fitness_df = raw_df.rename({"Combo": "sequence", "mut": "mutant"})
        if drop_wt:
            fitness_df = fitness_df.filter(pl.col("sequence") != self.target_sequence)
        fitness_df = fitness_df.select("mutant", "sequence", "prop", "fitness", "split")
        return base.FitnessDFSchema.validate(fitness_df)


def get_dataset(name: str) -> CreiLOVDataset:
    """Get a mutant fitness dataset by name.

    Args:
        name: Dataset name; only ``"creiLOV"`` is available.

    Returns:
        The :class:`CreiLOVDataset` for the requested dataset.

    Raises:
        ValueError: If ``name`` is not an available dataset.
    """
    if name != _DATASET_NAME:
        raise ValueError(f"Invalid dataset {name=}")
    return CreiLOVDataset(_DATASET_NAME, WT_SEQ, ["fluorescence"])


def list_datasets() -> list[str]:
    """List the names of available mutant fitness datasets.

    Returns:
        Dataset names suitable for passing to :func:`get_dataset`.
    """
    return [_DATASET_NAME]
