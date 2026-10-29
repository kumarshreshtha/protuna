"""Enzyme fitness datasets from Align Foundation's 2023 Protein Engineering Tournament.

The tournament :cite:p:`Align2023` curated existing assays for its in silico
tracks and measured new designs in vitro. :func:`get_dataset` downloads assays
and metadata from the `tournament repository
<https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023>`_.

Original dataset sources:

* ``alpha_amylase`` and ``alpha_amylase_zeroshot``: the dataset of van der Flier
  et al. :cite:p:`AmylasePredictability` and the underlying patent data
  :cite:p:`AmylaseEP2698434,AmylaseUS8323945,AmylaseUS8975056,AmylaseUS9090887`.
* ``alpha_amylase_invitro``: training data from the same amylase sources;
  newly measured test designs from the tournament :cite:p:`Align2023`.
* ``alkaline_phosphatase``: Markin et al. :cite:p:`PafA`,
  The original data are deposited on `OSF <https://doi.org/10.17605/OSF.IO/QRN3C>`_.
* ``beta_glucosidase``: UC Davis Design to Data (D2D) database :cite:p:`D2DDatabase`
* ``aminotransferase_zeroshot``: Marian J. Menke and Uwe T. Bornscheuer
  :cite:p:`MenkeTransaminaseData`.
* ``imine_reductase``: van der Flier et al. :cite:p:`AmylasePredictability`
"""

import ast
import collections
from collections.abc import Sequence
import dataclasses
import json
import pathlib
import re
from typing import Any, Literal

import huggingface_hub
import polars as pl

from protuna.data.datasets import base
from protuna.utils import download
from protuna.utils import paths


def _get_cache_dir(dataset_name: str) -> pathlib.Path:
    return paths.get_cache_dir() / f"align_bio/{dataset_name}"


def _download_from_hub(filename: str) -> str:
    cache_dir = paths.get_cache_dir() / "align_bio/hf_hub"
    cache_dir.mkdir(exist_ok=True, parents=True)
    return huggingface_hub.hf_hub_download(
        "protuna/mutant_fitness", filename, repo_type="dataset", cache_dir=cache_dir
    )


_STRUCTURE_PATH_TEMPLATE = "structures/align_bio/{source}/{filename}"
_MSA_PATH_TEMPLATE = "msa/align_bio/{protocol}/{filename}"


@dataclasses.dataclass
class _DatasetInfo:
    """Download and process align foundation 2023 protein fitness dataset."""

    base_url: str
    name: str
    properties: list[str]
    test_file: str
    metadata_file: str
    structure_file: str
    msa_file: str
    train_file: str | None = None


_dataset_info = {
    "alpha_amylase_zeroshot": _DatasetInfo(
        base_url="https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023/raw/refs/heads/main/in_silico_zero-shot/input/Alpha-Amylase%20(in%20silico_%20Zero-Shot)/",
        test_file="Alpha-Amylase%20(in%20silico_%20Zero%20Shot).csv",
        metadata_file="Alpha-Amylase_in_silico_zero_shot.metadata.json",
        name="alpha_amylase_zeroshot",
        properties=["expression", "thermostability", "specific activity"],
        structure_file="alpha_amylase.pdb",
        msa_file="alpha_amylase.a3m",
    ),
    "aminotransferase_zeroshot": _DatasetInfo(
        base_url="https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023/raw/refs/heads/main/in_silico_zero-shot/input/Aminotransferase%20(in%20silico_%20Zero-Shot)/",
        test_file="Aminotransferase%20(Zero-shot).csv",
        metadata_file="Aminotransferase_in_silico_zero_shot.metadata.json",
        name="aminotransferase_zeroshot",
        properties=[
            "Phenylethylamine Activity",
            "(4-Chlorophenyl)phenylmethanamine Activity",
            "1_3-Diphenyl-propane-1-amine Activity",
        ],
        structure_file="aminotransferase.pdb",
        msa_file="aminotransferase.a3m",
    ),
    "alkaline_phosphatase": _DatasetInfo(
        base_url="https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023/raw/refs/heads/main/in_silico_supervised/input/Alkaline%20phosphatase%20PafA%20%20(In%20Silico_%20Supervised)/",
        train_file="train.csv",
        test_file="test%20(with%20values).csv",
        metadata_file="Alkaline_phosphatase_PafA_in_silico_supervised.metadata.json",
        name="alkaline_phosphatase",
        properties=[
            "Carboxy 4-methylumbelliferyl phosphate ester (cMUP)",
            "methyl phosphate (MeP)",
            "methyl phosphodiester (MecMUP)",
        ],
        structure_file="alkaline_phosphatase.pdb",
        msa_file="alkaline_phosphatase.a3m",
    ),
    "alpha_amylase": _DatasetInfo(
        base_url="https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023/raw/refs/heads/main/in_silico_supervised/input/Alpha-Amylase%20(In%20Silico_%20Supervised)/",
        train_file="train.csv",
        test_file="test%20(with%20values).csv",
        metadata_file="Alpha-Amylase_in_silico_supervised.metadata.json",
        name="alpha_amylase",
        properties=["expression", "thermostability", "specific activity"],
        structure_file="alpha_amylase.pdb",
        msa_file="alpha_amylase.a3m",
    ),
    "imine_reductase": _DatasetInfo(
        base_url="https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023/raw/refs/heads/main/in_silico_supervised/input/Imine%20reductase%20(In%20Silico_%20Supervised)/",
        train_file="train.csv",
        test_file="test%20(with%20values).csv",
        metadata_file="Imine_reductase_in_silico_supervised.metadata.json",
        name="imine_reductase",
        properties=["FIOP(fold improvement over positive control)"],
        structure_file="imine_reductase.pdb",
        msa_file="imine_reductase.a3m",
    ),
    "beta_glucosidase": _DatasetInfo(
        base_url="https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023/raw/refs/heads/main/in_silico_supervised/input/beta-glucosidase%20B%20%20(In%20Silico_%20Supervised)/",
        train_file="train.csv",
        test_file="test%20(with%20values).csv",
        metadata_file="beta-glucosidase_B_in_silico_supervised.metadata.json",
        name="beta_glucosidase",
        properties=["melting point", "activity", "expression"],
        structure_file="beta_glucosidase.pdb",
        msa_file="beta_glucosidase.a3m",
    ),
    "alpha_amylase_invitro": _DatasetInfo(
        base_url="https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023/raw/refs/heads/main/in_vitro/",
        train_file="input/alpha-amylase-training-data.csv",
        test_file="output/invitro-all.csv.gz",
        metadata_file="input/Alpha-Amylase_in_vitro.metadata.json",
        name="alpha_amylase_invitro",
        properties=["expression", "stability", "specific_activity"],
        structure_file="alpha_amylase.pdb",
        msa_file="alpha_amylase.a3m",
    ),
}


@dataclasses.dataclass
class AlignMetadata:
    """Metadata for a single Align tournament assay.

    The fields are parsed from the assay's ``*.metadata.json`` file published with
    the tournament :cite:p:`Align2023` and describe the assayed wild-type protein.

    Args:
        sequence_length: Length of the wild-type ``target_sequence``.
        classification: Enzyme classification (e.g. ``"hydrolase"``).
        ec_number: Enzyme Commission number.
        pdb_id: PDB identifier of the crystal structure, if any.
        target_sequence: The wild-type amino acid sequence.
        expression_system: Host used to express the protein.
        organism: Source organism of the wild-type protein.
        activity_substrate: Substrate used in the specific activity measurement.
    """

    sequence_length: int
    classification: str
    ec_number: str
    pdb_id: str
    target_sequence: str | None
    expression_system: str
    organism: str
    activity_substrate: str


def _process_json_for_metadata(file_path: pathlib.Path) -> dict[str, Any]:
    with open(file_path) as f:
        try:
            return json.load(f)["data"]
        # beta_glucosidase json file is malformed
        except json.JSONDecodeError as e:
            f.seek(0)
            metadata = f.read()
            match = re.search(r"\"data\":\s+(?P<data>\{.*?\})", metadata, re.DOTALL)
            if match is None:
                raise TypeError("metadata file corrupted beyond repair.") from e
            return ast.literal_eval(match.group("data"))


def _load_metadata(name: str) -> AlignMetadata:
    """Loads dataset's metadata."""
    info = _dataset_info[name]
    url = info.base_url + info.metadata_file
    metadata_path = _get_cache_dir(name) / info.metadata_file
    if not metadata_path.exists():
        download.download_from_url(url, metadata_path)
    metadata = _process_json_for_metadata(metadata_path)
    # Mistake in the dataset for alpha amylase
    pdb_id = (
        "1UA7"
        if metadata["pdb xtal structure"] == "IUA7"
        else None
        if metadata["pdb xtal structure"] == "Not given"
        else metadata["pdb xtal structure"]
    )
    ec_number = None if metadata["ec number"] == "Not given" else metadata["ec number"]
    target_sequence = (
        None
        if metadata["target sequence"] == "Not given"
        else metadata["target sequence"]
    )
    # ty: ignore[invalid-argument-type]
    return AlignMetadata(
        sequence_length=metadata.get("sequence length"),
        classification=metadata.get("classification"),
        ec_number=ec_number,
        pdb_id=pdb_id,
        target_sequence=target_sequence,
        expression_system=metadata.get("expression system"),
        organism=metadata.get("organism"),
        activity_substrate=metadata.get(
            "substrate used in specific activity measurement"
        ),
    )


def _infer_target_sequence(mutated_sequences: Sequence[str]) -> str | None:
    subs_only = all(
        [len(seq) == len(mutated_sequences[0]) for seq in mutated_sequences]
    )
    if not subs_only:
        return None
    target_sequence = [
        collections.Counter(amino_acids).most_common(1)[0][0]
        for amino_acids in zip(*list(mutated_sequences), strict=True)
    ]
    return "".join(target_sequence)


ALPHA_AMYLASE_SEQ_LEN = 425


def _process_invitro_study(
    data_df: pl.DataFrame, split: Literal["train", "test"]
) -> pl.DataFrame:
    if split == "test":
        # marks lab submissions have an additional "M" in the beginning
        data_df = data_df.with_columns(
            pl.when(pl.col("team") == "marks")
            .then(pl.col("sequence").str.slice(1))
            .otherwise(pl.col("sequence"))
            .alias("sequence")
        )
        # Nimbus submissions are longer than the wild-type
        data_df = data_df.filter(
            pl.col("sequence").str.len_chars() == ALPHA_AMYLASE_SEQ_LEN
        )
        return (
            data_df.drop_nulls(["objectid", "sequence", "target"])
            .group_by("objectid", "sequence", "target")
            .agg(pl.col("output").fill_nan(None).mean())
            .sort("objectid", "sequence", "target")
            .pivot(
                index=["objectid", "sequence"],
                on="target",
                values="output",
                sort_columns=True,
            )
            .drop("objectid")
            .rename({"sequence": "mutated_sequence"})
        )
    data_df = data_df.rename(
        {
            "stability_temperature": "stability",
            "activity_dp7": "specific_activity",
        }
    )
    data_df = data_df.select(
        ["mutated_sequence", "expression", "stability", "specific_activity"]
    )
    return data_df


def _load_raw_dataset(name: str, split: Literal["train", "test"]) -> pl.DataFrame:
    info = _dataset_info[name]
    if split == "train" and info.train_file is None:
        raise ValueError(f"Not training file exists for dataset {name}")
    src_filename: str = info.train_file if split == "train" else info.test_file  # ty: ignore[invalid-assignment]
    url = info.base_url + src_filename
    data_path = _get_cache_dir(name) / src_filename
    if not data_path.exists():
        download.download_from_url(url, data_path)
    measurement_columns = info.properties
    if name == "alpha_amylase_invitro":
        measurement_columns = (
            ["output"]
            if split == "test"
            else ["expression", "stability_temperature", "activity_dp7"]
        )
    data_df = pl.read_csv(
        data_path,
        schema_overrides=dict.fromkeys(measurement_columns, pl.Float64),
        null_values=dict.fromkeys(measurement_columns, "NA"),
    )
    if name == "alpha_amylase_invitro":
        data_df = _process_invitro_study(data_df, split)
    return data_df


def _standardize_dataset(dataset_df: pl.DataFrame) -> pl.DataFrame:
    """Loads and standardizes the dataset dataframe."""
    dataset_df = dataset_df.drop(
        [
            "mutant",
            "Design id",
            "Design",
            "mutation",
            "dataset",
            "Mutation",
            "Name",
            "DNA",
        ],
        strict=False,
    )
    if "AA" in dataset_df.columns:
        dataset_df = dataset_df.rename({"AA": "mutated_sequence"})
    # "*" indicates an early stop codon mutation
    dataset_df = dataset_df.filter(
        ~pl.col("mutated_sequence").str.contains("*", literal=True)
    )
    dataset_df = dataset_df.rename({"mutated_sequence": "sequence"})
    return dataset_df


def _remove_train_test_overlap(df: pl.DataFrame) -> pl.DataFrame:
    test_mutants = df.filter(pl.col("split") == "test")["mutant"].unique()
    mask = (pl.col("split") == "train") & pl.col("mutant").is_in(test_mutants.implode())
    return df.filter(~mask)


@dataclasses.dataclass(frozen=True, repr=False)
class AlignFoundationDataset(base.MutantFitnessDataset):
    """Mutant fitness dataset from the Align tournament :cite:p:`Align2023`.

    Instances are created by :func:`get_dataset`. Each accessor lazily downloads and
    caches the file it needs as requested.

    When the metadata does not provide a wild-type sequence (the
    ``imine_reductase`` assay), it is inferred at construction time from the test
    sequences as the most common amino acid at each position.

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
        filename = _dataset_info[self.dataset_name].structure_file
        return pathlib.Path(
            _download_from_hub(
                _STRUCTURE_PATH_TEMPLATE.format(filename=filename, source=source)
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
        filename = _dataset_info[self.dataset_name].msa_file
        return pathlib.Path(
            _download_from_hub(
                _MSA_PATH_TEMPLATE.format(filename=filename, protocol=protocol)
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
        info = _dataset_info[self.dataset_name]
        dfs = []
        for filename, split in [(info.train_file, "train"), (info.test_file, "test")]:
            if filename is None:
                continue
            df = _load_raw_dataset(self.dataset_name, split)  # ty: ignore[invalid-argument-type]
            df = _standardize_dataset(df)
            df = df.with_columns(pl.lit(split).alias("split"))
            dfs.append(df)
        fitness_df = pl.concat(dfs, how="diagonal_relaxed")
        if drop_wt:
            fitness_df = fitness_df.filter(pl.col("sequence") != self.target_sequence)
        if prop is not None:
            fitness_df = (
                fitness_df.select("sequence", prop, "split").drop_nulls().drop_nans()
            )
            fitness_df = fitness_df.with_columns(pl.col(prop).cast(pl.Float64))
        mutants = base.create_mutant_uuids(fitness_df["sequence"], self.target_sequence)
        seq_to_mutant = {seq: mutant for mutant, seq in mutants.items()}
        fitness_df = fitness_df.with_columns(
            pl.col("sequence")
            .replace_strict(seq_to_mutant, return_dtype=pl.String)
            .alias("mutant")
        )
        value_vars = self.properties if prop is None else [prop]
        fitness_df = fitness_df.unpivot(
            index=[column for column in fitness_df.columns if column not in value_vars],
            on=value_vars,
            value_name="fitness",
            variable_name="prop",
        )
        fitness_df = fitness_df.drop_nulls("fitness").drop_nans("fitness")
        fitness_df = fitness_df.select("mutant", "sequence", "prop", "fitness", "split")
        fitness_df = _remove_train_test_overlap(fitness_df)
        return base.FitnessDFSchema.validate(fitness_df)


def list_datasets() -> list[str]:
    """List the names of available mutant fitness datasets.

    Returns:
        Dataset names suitable for passing to :func:`get_dataset`.
    """
    return list(_dataset_info.keys())


def get_dataset(name: str) -> AlignFoundationDataset:
    """Get a mutant fitness dataset by name.

    Args:
        name: Dataset name, as returned by :func:`list_datasets`.

    Returns:
        The :class:`AlignFoundationDataset` for the requested dataset.

    Raises:
        ValueError: If ``name`` is not an available dataset.
    """
    if name not in _dataset_info:
        raise ValueError(
            f"Dataset {name} not available in align datasets. check available datasets "
            "using `align_bio.list_datasets()`"
        )
    metadata = _load_metadata(name)
    if metadata.target_sequence is None:
        # imine reductase
        test_df = _load_raw_dataset(name, "test")
        metadata.target_sequence = base.wt_from_mutants(test_df["AA"].to_list())

    return AlignFoundationDataset(
        dataset_name=name,
        target_sequence=metadata.target_sequence,
        properties=_dataset_info[name].properties,
    )
