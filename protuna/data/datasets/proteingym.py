"""ProteinGym :cite:p:`ProteinGym` deep mutational scanning datasets."""

import dataclasses
import pathlib
import tarfile
import zipfile

import huggingface_hub
import polars as pl

from protuna.data.datasets import base
from protuna.utils import download
from protuna.utils import paths

BASE_URL = "https://zenodo.org/records/15293562/files"
PROTEINGYM_DIR = paths.get_cache_dir() / "proteingym"

DMS_SUB_METADATA_FILE = "DMS_substitutions.csv"
DMS_SUB_METADATA_URL = f"{BASE_URL}/{DMS_SUB_METADATA_FILE}?download=1"
DMS_SUB_METADATA_PATH = PROTEINGYM_DIR / DMS_SUB_METADATA_FILE

DMS_INDEL_METADATA_FILE = "DMS_indels.csv"
DMS_INDEL_METADATA_URL = f"{BASE_URL}/{DMS_INDEL_METADATA_FILE}?download=1"
DMS_INDEL_METADATA_PATH = PROTEINGYM_DIR / DMS_INDEL_METADATA_FILE

DMS_SUB_ZIPFILE = "DMS_ProteinGym_substitutions.zip"
DMS_SUB_URL = f"{BASE_URL}/{DMS_SUB_ZIPFILE}?download=1"
DMS_SUB_ZIPPATH = PROTEINGYM_DIR / DMS_SUB_ZIPFILE
DMS_SUB_DIR = DMS_SUB_ZIPPATH.with_suffix("")

DMS_SUB_SINGLE_SPLIT_ZIPFILE = "cv_folds_singles_substitutions.zip"
DMS_SUB_SINGLE_SPLIT_URL = f"{BASE_URL}/{DMS_SUB_SINGLE_SPLIT_ZIPFILE}?download=1"
DMS_SUB_SINGLE_SPLIT_ZIPPATH = PROTEINGYM_DIR / DMS_SUB_SINGLE_SPLIT_ZIPFILE
DMS_SUB_SINGLE_SPLIT_DIR = DMS_SUB_SINGLE_SPLIT_ZIPPATH.with_suffix("")

DMS_SUB_MULTIPLE_SPLIT_ZIPFILE = "cv_folds_multiples_substitutions.zip"
DMS_SUB_MULTIPLE_SPLIT_URL = f"{BASE_URL}/{DMS_SUB_MULTIPLE_SPLIT_ZIPFILE}?download=1"
DMS_SUB_MULTIPLE_SPLIT_ZIPPATH = PROTEINGYM_DIR / DMS_SUB_MULTIPLE_SPLIT_ZIPFILE
DMS_SUB_MULTIPLE_SPLIT_DIR = DMS_SUB_MULTIPLE_SPLIT_ZIPPATH.with_suffix("")

DMS_INDEL_ZIPFILE = "DMS_ProteinGym_indels.zip"
DMS_INDEL_URL = f"{BASE_URL}/{DMS_INDEL_ZIPFILE}?download=1"
DMS_INDEL_ZIPPATH = PROTEINGYM_DIR / DMS_INDEL_ZIPFILE
DMS_INDEL_DIR = DMS_INDEL_ZIPPATH.with_suffix("")

DMS_INDEL_SPLIT_ZIPFILE = "cv_folds_indels.zip"
DMS_INDEL_SPLIT_URL = f"{BASE_URL}/{DMS_INDEL_SPLIT_ZIPFILE}?download=1"
DMS_INDEL_SPLIT_ZIPPATH = PROTEINGYM_DIR / DMS_INDEL_SPLIT_ZIPFILE
DMS_INDEL_SPLIT_DIR = DMS_INDEL_SPLIT_ZIPPATH.with_suffix("")

AF_STRUCTURE_ZIPFILE = "ProteinGym_AF2_structures.zip"
AF_STRUCTURE_URL = f"{BASE_URL}/{AF_STRUCTURE_ZIPFILE}?download=1"
AF_STRUCTURE_ZIPPATH = PROTEINGYM_DIR / AF_STRUCTURE_ZIPFILE
AF_STRUCTURE_DIR = AF_STRUCTURE_ZIPPATH.with_suffix("")

MSA_ZIPFILE = "DMS_msa_files.zip"
MSA_URL = f"{BASE_URL}/{MSA_ZIPFILE}?download=1"
MSA_ZIPPATH = PROTEINGYM_DIR / MSA_ZIPFILE
MSA_DIR = MSA_ZIPPATH.with_suffix("")
COLABFOLD_MSA_DIR = PROTEINGYM_DIR / "colabfold_msa"


@dataclasses.dataclass
class ProteinGymMetadata:
    """Metadata for a single ProteinGym assay.

    The fields mirror the columns of the ``DMS_substitutions.csv`` reference file
    published with ProteinGym :cite:p:`ProteinGym` and are populated by
    :func:`get_dataset`. They describe the assayed protein (e.g. ``target_seq``,
    ``UniProt_ID``), the associated files (``DMS_filename``, ``MSA_filename``,
    ``pdb_file``), and statistics of the curated MSA.
    """

    DMS_id: str
    DMS_index: str
    DMS_filename: str
    UniProt_ID: str
    taxon: str
    source_organism: str
    target_seq: str
    seq_len: int
    DMS_binarization_cutoff: float
    DMS_binarization_method: str
    first_author: str
    title: str
    year: int
    jo: str
    region_mutated: str
    molecule_name: str
    selection_assay: str
    selection_type: str
    MSA_filename: str
    MSA_start: int
    MSA_end: int
    MSA_len: int
    MSA_bitscore: float
    MSA_theta: float
    MSA_num_seqs: int
    MSA_perc_cov: float
    MSA_num_cov: int
    MSA_N_eff: float
    MSA_Neff_L: float
    MSA_Neff_L_category: str
    MSA_num_significant: int
    MSA_num_significant_L: float
    raw_DMS_filename: str
    raw_DMS_phenotype_name: str
    raw_DMS_directionality: int
    raw_DMS_mutant_column: str
    weight_file_name: str
    pdb_file: str
    pdb_range: str
    ProteinGym_version: float
    coarse_selection_type: str
    includes_multiple_mutants: bool
    DMS_total_number_mutants: int
    DMS_number_single_mutants: int | None = None
    DMS_number_multiple_mutants: int | None = None
    raw_mut_offset: int | None = None


@dataclasses.dataclass(frozen=True, repr=False)
class ProteinGymDataset(base.MutantFitnessDataset):
    """Mutant fitness dataset from ProteinGym :cite:p:`ProteinGym`.

    Use :func:`get_dataset` to construct an instance. Data files are downloaded
    and cached on first access.

    Args:
        dataset_name: Name identifying the dataset within its source collection.
        target_sequence: Wild-type amino acid sequence used to identify mutations.
        properties: Names of the measured properties.
        metadata: The :class:`ProteinGymMetadata` describing this assay.
    """

    metadata: ProteinGymMetadata

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
                "only alphafold2 structures are available for proteingym datasets."
            )
        struct_path = AF_STRUCTURE_DIR / self.metadata.pdb_file
        if not struct_path.exists():
            zip_path = download.download_from_url(
                AF_STRUCTURE_URL, AF_STRUCTURE_ZIPPATH
            )
            with zipfile.ZipFile(zip_path) as zip:
                zip.extractall(PROTEINGYM_DIR)
        return struct_path

    def get_msa_path(self, protocol: base.MSAProtocol) -> pathlib.Path:
        """Return the path to the wild-type multiple sequence alignment.

        Args:
            protocol: Protocol used to construct the MSA. Supports ``ColabFold``
                and ``EVCoupling``.

        Returns:
            Path to the MSA file.
        """
        if protocol == base.MSAProtocol.ColabFold:
            msa_path = COLABFOLD_MSA_DIR / f"aa_seq_aln_a3m/{self.dataset_name}.a3m"
            if not msa_path.exists():
                tar_path = huggingface_hub.hf_hub_download(
                    "AI4Protein/VenusREM",
                    filename="aa_seq_aln_a3m.tar.gz",
                    repo_type="dataset",
                    cache_dir=PROTEINGYM_DIR,
                )
                COLABFOLD_MSA_DIR.mkdir(exist_ok=True, parents=True)
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=COLABFOLD_MSA_DIR, filter="data")
            return msa_path

        msa_path = MSA_DIR / self.metadata.MSA_filename
        if not msa_path.exists():
            zip_path = download.download_from_url(MSA_URL, MSA_ZIPPATH)
            with zipfile.ZipFile(zip_path) as zip:
                zip.extractall(PROTEINGYM_DIR)
        return msa_path

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
        data_path = DMS_SUB_DIR / self.metadata.DMS_filename
        if not data_path.exists():
            zip_path = download.download_from_url(DMS_SUB_URL, DMS_SUB_ZIPPATH)
            with zipfile.ZipFile(zip_path) as zip:
                zip.extractall(PROTEINGYM_DIR)

        fitness_df = pl.read_csv(
            data_path,
            schema_overrides={"DMS_score": pl.Float64},
            null_values={"DMS_score": "NA"},
        )
        fitness_df = fitness_df.select("mutant", "mutated_sequence", "DMS_score")
        fitness_df = fitness_df.rename(
            {"DMS_score": "fitness", "mutated_sequence": "sequence"}
        )
        if prop is not None:
            fitness_df = fitness_df.drop_nulls().drop_nans()
            fitness_df = fitness_df.with_columns(pl.col("fitness").cast(pl.Float64))
        if drop_wt:
            fitness_df = fitness_df.filter(pl.col("sequence") != self.target_sequence)
        fitness_df = fitness_df.with_columns(
            pl.lit(self.properties[0]).alias("prop"), pl.lit("train").alias("split")
        )
        fitness_df = fitness_df.select("mutant", "sequence", "prop", "fitness", "split")
        return base.FitnessDFSchema.validate(fitness_df)


def _get_substitution_datasets_metadata() -> pl.DataFrame:
    if not DMS_SUB_METADATA_PATH.exists():
        download.download_from_url(DMS_SUB_METADATA_URL, DMS_SUB_METADATA_PATH)
    return pl.read_csv(DMS_SUB_METADATA_PATH, schema_overrides={"DMS_index": pl.String})


def _get_indel_datasets_metadata() -> pl.DataFrame:
    if not DMS_INDEL_METADATA_PATH.exists():
        download.download_from_url(DMS_INDEL_METADATA_URL, DMS_INDEL_METADATA_PATH)
    return pl.read_csv(
        DMS_INDEL_METADATA_PATH, schema_overrides={"DMS_index": pl.String}
    )


def _get_metadata_by_name(name: str) -> ProteinGymMetadata:
    subs_metadata = _get_substitution_datasets_metadata()
    if name not in subs_metadata["DMS_id"]:
        raise ValueError(
            f"Dataset {name} not available in proteingym. check available datasets "
            "using `proteingym.list_datasets()`"
        )
    metadata = subs_metadata.filter(pl.col("DMS_id") == name).row(0, named=True)
    return ProteinGymMetadata(**metadata)


def list_datasets() -> list[str]:
    """List the names of available mutant fitness datasets.

    Downloads the substitution metadata file on first access.

    Returns:
        Dataset names suitable for passing to :func:`get_dataset`.
    """
    subs_metadata = _get_substitution_datasets_metadata()
    return subs_metadata["DMS_id"].to_list()


def get_dataset(name: str) -> ProteinGymDataset:
    """Get a mutant fitness dataset by name.

    Args:
        name: Assay ``DMS_id``, as returned by :func:`list_datasets`.

    Returns:
        The :class:`ProteinGymDataset` for the requested dataset.

    Raises:
        ValueError: If ``name`` is not an available dataset.
    """
    metadata = _get_metadata_by_name(name)
    return ProteinGymDataset(
        dataset_name=name,
        properties=[metadata.coarse_selection_type],
        target_sequence=metadata.target_seq,
        metadata=metadata,
    )
