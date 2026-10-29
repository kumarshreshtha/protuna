import gzip
import pathlib
from unittest import mock

import polars as pl
import pytest

from protuna.data.datasets import align_bio
from protuna.data.datasets import base
from protuna.data.datasets import flip
from protuna.data.datasets import proteinguide
from protuna.data.datasets import proteingym
from protuna.data.datasets import sgpo


@pytest.mark.parametrize("missing_measurement", ["", "NA", "NaN", "nan"])
def test_align_preserves_property_order_and_removes_training_overlap(
    dataset_cache: pathlib.Path,
    missing_measurement: str,
) -> None:
    """Keep test replicates, omit stop codons and drop missing measurements."""
    cache = dataset_cache / "align_bio/beta_glucosidase"
    cache.mkdir(parents=True)
    (cache / "train.csv").write_text(
        "mutated_sequence,melting point,activity,expression\n"
        "AC,1.0,10.0,100.0\n"
        "GC,2.0,20.0,200.0\n"
        "A*,3.0,30.0,300.0\n"
        "NA,4,40,400\n"
    )
    (cache / "test%20(with%20values).csv").write_text(
        "AA,melting point,activity,expression\n"
        f"GC,4.0,40.0,{missing_measurement}\n"
        f"GC,6.0,60.0,{missing_measurement}\n"
        f"AT,{missing_measurement},80.0,{missing_measurement}\n"
    )
    dataset = align_bio.AlignFoundationDataset(
        "beta_glucosidase", "AC", ["melting point", "activity", "expression"]
    )

    result = dataset.get_fitness_df()

    assert result.schema == base.FitnessDFSchema.schema
    assert result.rows() == [
        ("WT", "AC", "melting point", 1.0, "train"),
        ("A1N:C2A", "NA", "melting point", 4.0, "train"),
        ("A1G", "GC", "melting point", 4.0, "test"),
        ("A1G", "GC", "melting point", 6.0, "test"),
        ("WT", "AC", "activity", 10.0, "train"),
        ("A1N:C2A", "NA", "activity", 40.0, "train"),
        ("A1G", "GC", "activity", 40.0, "test"),
        ("A1G", "GC", "activity", 60.0, "test"),
        ("C2T", "AT", "activity", 80.0, "test"),
        ("WT", "AC", "expression", 100.0, "train"),
        ("A1N:C2A", "NA", "expression", 400.0, "train"),
    ]


def test_align_filters_a_property_before_removing_overlap(
    dataset_cache: pathlib.Path,
) -> None:
    """An absent test measurement must not remove a training measurement."""
    cache = dataset_cache / "align_bio/beta_glucosidase"
    cache.mkdir(parents=True)
    (cache / "train.csv").write_text(
        "mutated_sequence,melting point,activity,expression\n"
        "AC,,1.0,10.0\nGC,,2.0,20.0\n"
    )
    (cache / "test%20(with%20values).csv").write_text(
        "AA,melting point,activity,expression\nGC,,,30.0\n"
    )
    dataset = align_bio.AlignFoundationDataset(
        "beta_glucosidase", "AC", ["activity", "expression"]
    )

    result = dataset.get_fitness_df(prop="activity", drop_wt=True)

    assert result.rows() == [("A1G", "GC", "activity", 2.0, "train")]


def test_align_invitro_averages_replicates_and_sorts_pivot_rows(
    dataset_cache: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trim the marks prefix and omit sequences of the wrong length."""
    monkeypatch.setattr(align_bio, "ALPHA_AMYLASE_SEQ_LEN", 2)
    cache = dataset_cache / "align_bio/alpha_amylase_invitro"
    (cache / "input").mkdir(parents=True)
    (cache / "output").mkdir()
    (cache / "input/alpha-amylase-training-data.csv").write_text(
        "mutated_sequence,expression,stability_temperature,activity_dp7\nAC,1.0,2.0,3.0\n"
    )
    with gzip.open(cache / "output/invitro-all.csv.gz", "wt") as stream:
        stream.write(
            "objectid,team,sequence,target,output\n"
            "b,marks,MGC,expression,2.0\n"
            "b,marks,MGC,expression,6.0\n"
            "a,other,AT,expression,8.0\n"
            "c,nimbus,ACD,expression,99.0\n"
        )
    dataset = align_bio.AlignFoundationDataset(
        "alpha_amylase_invitro", "AC", ["expression", "stability", "specific_activity"]
    )

    result = dataset.get_fitness_df(prop="expression", drop_wt=True)

    assert result.rows() == [
        ("C2T", "AT", "expression", 8.0, "test"),
        ("A1G", "GC", "expression", 4.0, "test"),
    ]


@pytest.mark.parametrize(
    ("name", "filename", "values", "dtype", "expected"),
    [
        ("amylase", "one_to_many.csv.gz", "1.5\nGC,2.5", pl.Float64, 2.5),
        ("amylase", "one_to_many.csv.gz", "1\nGC,2", pl.Float64, 2.0),
        ("nucb", "two_to_many.csv.gz", "1\nGC,2", pl.Int64, 2),
    ],
)
def test_flip_preserves_fitness_dtype(
    dataset_cache: pathlib.Path,
    name: str,
    filename: str,
    values: str,
    dtype: type[pl.DataType],
    expected: float | int,
) -> None:
    """Read compressed data and retain integer bins for nucb."""
    cache = dataset_cache / "flip" / name
    cache.mkdir(parents=True)
    raw = pl.read_csv(f"sequence,target\nAC,{values}\n".encode()).with_columns(
        pl.lit("train").alias("set"), pl.lit(False).alias("validation")
    )
    with gzip.open(cache / filename, "wt") as stream:
        stream.write(raw.write_csv())
    dataset = flip.FLIPDataset(name, "AC", ["activity"])

    result = dataset.get_fitness_df(prop="activity", drop_wt=True)

    assert result["mutant"].to_list() == ["A1G"]
    assert result["fitness"].dtype == dtype
    assert result["fitness"].to_list() == [expected]


def test_sgpo_preserves_replicate_order_and_split_mapping(
    dataset_cache: pathlib.Path,
) -> None:
    """Unpivot replicates without mixing their row order or using fitness twice."""
    cache = dataset_cache / "creiLOV"
    cache.mkdir()
    (cache / "fitness.csv").write_text(
        "mut,Combo,split,log_rep1,log_rep2,log_rep3,fitness\n"
        "WT,AC,train,0.0,1.0,2.0,999\n"
        "A1G,GC,validation,3.0,4.0,5.0,999\n"
        "C2T,AT,other,6.0,7.0,8.0,999\n"
    )
    dataset = sgpo.CreiLOVDataset("creiLOV", "AC", ["fluorescence"])

    result = dataset.get_fitness_df(drop_wt=True)

    assert result.rows() == [
        ("A1G", "GC", "fluorescence", 3.0, "train"),
        ("C2T", "AT", "fluorescence", 6.0, "test"),
        ("A1G", "GC", "fluorescence", 4.0, "train"),
        ("C2T", "AT", "fluorescence", 7.0, "test"),
        ("A1G", "GC", "fluorescence", 5.0, "train"),
        ("C2T", "AT", "fluorescence", 8.0, "test"),
    ]


def test_proteinguide_infers_wild_type_and_preserves_property_order(
    dataset_cache: pathlib.Path,
) -> None:
    """Infer wild type from sequences and return properties in dataset order."""
    raw_path = dataset_cache / "PbrR.csv"
    raw_path.write_text("seq,1uM Pb,30uM Zn\nAC,1,10\nAC,2,20\nGC,3,30\n")
    with mock.patch.object(
        proteinguide.huggingface_hub,
        "hf_hub_download",
        autospec=True,
        return_value=str(raw_path),
    ):
        dataset = proteinguide.get_dataset("PbrR")
        result = dataset.get_fitness_df(drop_wt=True)

    assert dataset.target_sequence == "AC"
    assert result.rows() == [
        ("A1G", "GC", "pb_affinity", 3.0, "train"),
        ("A1G", "GC", "zn_affinity", 30.0, "train"),
    ]


def test_proteingym_reads_metadata_and_filters_missing_fitness(
    dataset_cache: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve an assay by name and preserve the explicit-property NA filter."""
    metadata_path = dataset_cache / "metadata.csv"
    monkeypatch.setattr(proteingym, "DMS_SUB_METADATA_PATH", metadata_path)
    monkeypatch.setattr(proteingym, "DMS_SUB_DIR", dataset_cache)
    pl.DataFrame(
        {
            "DMS_id": ["toy"],
            "DMS_index": ["1"],
            "DMS_filename": ["toy.csv"],
            "UniProt_ID": ["toy"],
            "taxon": ["test"],
            "source_organism": ["test"],
            "target_seq": ["AC"],
            "seq_len": [2],
            "DMS_binarization_cutoff": [0.0],
            "DMS_binarization_method": ["manual"],
            "first_author": ["test"],
            "title": ["test"],
            "year": [2020],
            "jo": ["test"],
            "region_mutated": ["1-2"],
            "molecule_name": ["toy"],
            "selection_assay": ["test"],
            "selection_type": ["Activity"],
            "MSA_filename": ["toy.a3m"],
            "MSA_start": [1],
            "MSA_end": [2],
            "MSA_len": [2],
            "MSA_bitscore": [1.0],
            "MSA_theta": [0.2],
            "MSA_num_seqs": [2],
            "MSA_perc_cov": [1.0],
            "MSA_num_cov": [2],
            "MSA_N_eff": [1.0],
            "MSA_Neff_L": [0.5],
            "MSA_Neff_L_category": ["low"],
            "MSA_num_significant": [1],
            "MSA_num_significant_L": [0.5],
            "raw_DMS_filename": ["toy.csv"],
            "raw_DMS_phenotype_name": ["Activity"],
            "raw_DMS_directionality": [1],
            "raw_DMS_mutant_column": ["mutant"],
            "weight_file_name": ["toy.npy"],
            "pdb_file": ["toy.pdb"],
            "pdb_range": ["1-2"],
            "ProteinGym_version": [1.0],
            "coarse_selection_type": ["Activity"],
            "includes_multiple_mutants": [False],
            "DMS_total_number_mutants": [3],
        }
    ).write_csv(metadata_path)
    (dataset_cache / "toy.csv").write_text(
        "mutant,mutated_sequence,DMS_score\nWT,AC,0.0\nA1G,GC,1.5\nC2T,AT,NA\n"
    )

    dataset = proteingym.get_dataset("toy")
    result = dataset.get_fitness_df(prop="Activity", drop_wt=True)

    assert proteingym.list_datasets() == ["toy"]
    assert dataset.target_sequence == "AC"
    assert result.rows() == [("A1G", "GC", "Activity", 1.5, "train")]
