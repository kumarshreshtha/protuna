"""Protein fitness datasets curated by FLIP :cite:p:`FLIP` and FLIP2 :cite:p:`FLIP2`.

:func:`get_dataset` loads selected benchmark splits from FLIP and FLIP2.

Original dataset sources:

* ``gb1``: the four-site IgG-binding landscape of Wu et al.
  :cite:p:`GB1Landscape`, using FLIP's ``one_vs_rest`` split.
* ``amylase``: the alpha-amylase stain-removal dataset of van der Flier et al.
  :cite:p:`AmylasePredictability`, using FLIP2's ``one_to_many`` split.
* ``ired``: the microdroplet-screened imine reductase landscape of Gantz et al.
  :cite:p:`IREDLandscape`, using FLIP2's ``two_to_many`` split. This is a
  different source from the Codexis imine reductase assay in
  :mod:`~protuna.data.datasets.align_bio`.
* ``nucb``: the nuclease activity dataset of Thomas et al.
  :cite:p:`NucBLandscape`, using FLIP2's ``two_to_many`` split with activity
  binned into four levels.
* ``trpb``: the combinatorial tryptophan synthase landscapes of Johnston et al.
  :cite:p:`TrpBLandscape`, using FLIP2's ``one_to_many`` split.
"""

import dataclasses
import pathlib

import huggingface_hub
import polars as pl

from protuna.data.datasets import base
from protuna.utils import download
from protuna.utils import paths

URL_TEMPLATE = (
    "https://zenodo.org/records/18433203/files/{dataset}/{filename}?download=1"
)


@dataclasses.dataclass
class _DatasetInfo:
    """Download and process FLIP protein fitness dataset."""

    name: str
    filename: str
    prop: str


GB1_WT = (
    "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTELEVLFQGPLDPNSMATYEVLCEVAR"
    "KLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRDLLHLDPRFLERHLAGTMSYFSP"
    "YQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLDLLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLH"
    "LSGPSHSQHYRHTPLEHHHHHH"
)

_FLIP_DATASETS = {
    "gb1": _DatasetInfo(name="gb1", filename="one_vs_rest.csv.gz", prop="binding"),
    "amylase": _DatasetInfo(
        name="amylase", filename="one_to_many.csv.gz", prop="activity"
    ),
    "ired": _DatasetInfo(name="ired", filename="two_to_many.csv.gz", prop="activity"),
    "nucb": _DatasetInfo(name="nucb", filename="two_to_many.csv.gz", prop="activity"),
    "trpb": _DatasetInfo(name="trpb", filename="one_to_many.csv.gz", prop="activity"),
}


def _download_from_hub(filename: str) -> str:
    cache_dir = paths.get_cache_dir() / "flip/hf_hub"
    cache_dir.mkdir(exist_ok=True, parents=True)
    return huggingface_hub.hf_hub_download(
        "protuna/mutant_fitness", filename, repo_type="dataset", cache_dir=cache_dir
    )


_STRUCTURE_PATH_TEMPLATE = "structures/flip/{source}/{filename}"
_MSA_PATH_TEMPLATE = "msa/flip/{protocol}/{filename}"


def _load_raw_dataset(name: str) -> pl.DataFrame:
    info = _FLIP_DATASETS[name]
    url = URL_TEMPLATE.format(dataset=info.name, filename=info.filename)
    data_path = paths.get_cache_dir() / f"flip/{info.name}/{info.filename}"
    if not data_path.exists():
        download.download_from_url(url, data_path)
    data_df = pl.read_csv(
        data_path,
        schema_overrides={"target": pl.Int64 if name == "nucb" else pl.Float64},
    )
    return data_df


@dataclasses.dataclass(frozen=True, repr=False)
class FLIPDataset(base.MutantFitnessDataset):
    """Mutant fitness dataset from FLIP or FLIP2 :cite:p:`FLIP,FLIP2`.

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
            :class:`~protuna.data.datasets.base.FitnessDFSchema`. The
            ``nucb`` dataset follows
            :class:`~protuna.data.datasets.base.BinnedFitnessDFSchema` instead.

        Raises:
            ValueError: If ``prop`` is not a property of this dataset.
        """
        if prop is not None and prop != self.properties[0]:
            raise ValueError(
                f"{prop=} is not available for dataset {self.dataset_name}."
            )
        raw_df = _load_raw_dataset(self.dataset_name)
        fitness_df = raw_df.rename({"target": "fitness", "set": "split"})
        fitness_df = fitness_df.with_columns(pl.lit(self.properties[0]).alias("prop"))
        fitness_df = fitness_df.drop("validation")
        uuids = base.create_mutant_uuids(fitness_df["sequence"], self.target_sequence)
        seq_to_uuid = {seq: uuid for uuid, seq in uuids.items()}
        fitness_df = fitness_df.with_columns(
            pl.col("sequence")
            .replace_strict(seq_to_uuid, return_dtype=pl.String)
            .alias("mutant")
        )
        if drop_wt:
            fitness_df = fitness_df.filter(pl.col("sequence") != self.target_sequence)
        fitness_df = fitness_df.select("mutant", "sequence", "prop", "fitness", "split")
        if self.dataset_name == "nucb":
            return base.BinnedFitnessDFSchema.validate(fitness_df)
        return base.FitnessDFSchema.validate(fitness_df)


def get_dataset(name: str) -> FLIPDataset:
    """Get a mutant fitness dataset by name.

    Args:
        name: Dataset name, as returned by :func:`list_datasets`.

    Returns:
        The :class:`FLIPDataset` for the requested dataset.

    Raises:
        ValueError: If ``name`` is not an available dataset.
    """
    info = _FLIP_DATASETS.get(name)
    if info is None:
        raise ValueError(f"No dataset with {name=} found in FLIP datasets.")
    if name == "gb1":
        wt_seq = GB1_WT
    else:
        raw_df = _load_raw_dataset(name)
        wt_seq = base.wt_from_mutants(raw_df["sequence"].to_list())
    return FLIPDataset(name, target_sequence=wt_seq, properties=[info.prop])


def list_datasets() -> list[str]:
    """List the names of available mutant fitness datasets.

    Returns:
        Dataset names suitable for passing to :func:`get_dataset`.
    """
    return list(_FLIP_DATASETS.keys())
