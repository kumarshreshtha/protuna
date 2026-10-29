"""PABP, UBE4B, and PbrR fitness datasets from ProteinGuide :cite:p:`ProteinGuide`.

Original dataset sources:

* ``PABP``: deep mutational scanning of the yeast Pab1 RRM2 domain by Melamed
  et al. :cite:p:`PABPLandscape`.
* ``UBE4B``: the Ube4b U-box auto-ubiquitination assay of Starita et al.
  :cite:p:`UBE4BLandscape`.
* ``PbrR``: the initial library from Wang et al.'s cell-free lead biosensor
  campaign :cite:p:`PbrRBiosensors`.

ProteinGuide uses the PABP and UBE4B versions processed by Hsu et al.
:cite:p:`AssayLabeledFitness`.
"""

import dataclasses
import pathlib

import huggingface_hub
import polars as pl

from protuna.data.datasets import base
from protuna.utils import paths


@dataclasses.dataclass
class _DatasetInfo:
    name: str
    properties: list[str]


_DATASETS = {
    "PABP": _DatasetInfo(name="PABP", properties=["binding"]),
    "PbrR": _DatasetInfo(name="PbrR", properties=["pb_affinity", "zn_affinity"]),
    "UBE4B": _DatasetInfo(name="UBE4B", properties=["activity"]),
}


def _download_from_hub(filename: str) -> str:
    cache_dir = paths.get_cache_dir() / "proteinguide"
    cache_dir.mkdir(exist_ok=True, parents=True)
    return huggingface_hub.hf_hub_download(
        "protuna/mutant_fitness",
        filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


_STRUCTURE_PATH_TEMPLATE = "structures/proteinguide/{source}/{filename}"
_MSA_PATH_TEMPLATE = "msa/proteinguide/{protocol}/{filename}"


def _load_raw_df(name: str) -> pl.DataFrame:
    path = _download_from_hub(f"raw/{name}.csv")
    measurement_columns = ["1uM Pb", "30uM Zn"] if name == "PbrR" else ["fitness"]
    raw_df = pl.read_csv(
        path,
        schema_overrides=dict.fromkeys(measurement_columns, pl.Float64),
    )
    raw_df = raw_df.rename(
        {
            "seq": "sequence",
            "1uM Pb": "pb_affinity",
            "30uM Zn": "zn_affinity",
            "fitness": _DATASETS[name].properties[0],
        },
        strict=False,
    )
    raw_df = raw_df.drop(["num_mutations_from_wt", "Mutant"], strict=False)
    return raw_df


@dataclasses.dataclass(frozen=True, repr=False)
class ProteinGuideDataset(base.MutantFitnessDataset):
    """Mutant fitness dataset from ProteinGuide :cite:p:`ProteinGuide`.

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
        """
        raw_df = _load_raw_df(self.dataset_name)
        if prop is not None:
            raw_df = raw_df.select("sequence", prop)
            value_vars = [prop]
        else:
            value_vars = self.properties
        fitness_df = raw_df.unpivot(
            index=[column for column in raw_df.columns if column not in value_vars],
            on=value_vars,
            value_name="fitness",
            variable_name="prop",
        )
        mutants = base.create_mutant_uuids(fitness_df["sequence"], self.target_sequence)
        seq_to_mutant = {seq: mutant for mutant, seq in mutants.items()}
        fitness_df = fitness_df.with_columns(
            pl.col("sequence")
            .replace_strict(seq_to_mutant, return_dtype=pl.String)
            .alias("mutant"),
            pl.lit("train").alias("split"),
        )
        if drop_wt:
            fitness_df = fitness_df.filter(pl.col("sequence") != self.target_sequence)
        fitness_df = fitness_df.select("mutant", "sequence", "prop", "fitness", "split")
        return base.FitnessDFSchema.validate(fitness_df)


def get_dataset(name: str) -> ProteinGuideDataset:
    """Get a mutant fitness dataset by name.

    Args:
        name: Dataset name, as returned by :func:`list_datasets`.

    Returns:
        The :class:`ProteinGuideDataset` for the requested dataset.

    Raises:
        ValueError: If ``name`` is not an available dataset.
    """
    if name not in _DATASETS:
        raise ValueError(f"Invalid dataset {name=}")
    raw_df = _load_raw_df(name)
    wt_sequence = base.wt_from_mutants(raw_df["sequence"].to_list())
    return ProteinGuideDataset(
        dataset_name=name,
        target_sequence=wt_sequence,
        properties=_DATASETS[name].properties,
    )


def list_datasets() -> list[str]:
    """List the names of available mutant fitness datasets.

    Returns:
        Dataset names suitable for passing to :func:`get_dataset`.
    """
    return list(_DATASETS.keys())
