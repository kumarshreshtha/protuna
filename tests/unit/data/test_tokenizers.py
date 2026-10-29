from __future__ import annotations

import numpy as np
import pytest
import torch

from protuna.data import tokenizers

# _ALPHABET = "ACDEFGHIKLMNPQRSTVWY-_^*?"
# A=0  C=1  D=2  E=3  F=4  G=5  H=6  I=7  K=8  L=9
# M=10 N=11 P=12 Q=13 R=14 S=15 T=16 V=17 W=18 Y=19
# gap=20  mask=21  cls=22  pad=23  unk=24

_TOKENIZER = tokenizers.ProteinTokenizer()

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------


def test_module_exposes_singleton_tokenizer() -> None:
    """The module exposes a ready-made tokenizer instance."""
    assert isinstance(tokenizers.tokenizer, tokenizers.ProteinTokenizer)


# ---------------------------------------------------------------------------
# Special token id properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_id"),
    [
        ("gap_token_id", 20),
        ("mask_token_id", 21),
        ("cls_token_id", 22),
        ("pad_token_id", 23),
        ("unk_token_id", 24),
    ],
)
def test_special_token_ids(name: str, expected_id: int) -> None:
    """Each special token property returns its position in the alphabet."""
    assert getattr(_TOKENIZER, name) == expected_id


@pytest.mark.parametrize(
    "name",
    ["gap_token_id", "mask_token_id", "cls_token_id", "pad_token_id", "unk_token_id"],
)
def test_special_token_ids_are_builtin_ints(name: str) -> None:
    """Special token properties return builtin ints, not numpy scalars."""
    assert isinstance(getattr(_TOKENIZER, name), int)


def test_special_token_ids_are_distinct() -> None:
    """gap, mask, cls, pad, and unk token ids are all different."""
    ids = {
        _TOKENIZER.gap_token_id,
        _TOKENIZER.mask_token_id,
        _TOKENIZER.cls_token_id,
        _TOKENIZER.pad_token_id,
        _TOKENIZER.unk_token_id,
    }
    assert len(ids) == 5


def test_special_token_ids_follow_amino_acids() -> None:
    """Every non-amino-acid token id sits above the 20 amino acid ids."""
    specials = [
        _TOKENIZER.gap_token_id,
        _TOKENIZER.mask_token_id,
        _TOKENIZER.cls_token_id,
        _TOKENIZER.pad_token_id,
        _TOKENIZER.unk_token_id,
    ]
    assert min(specials) >= len(_TOKENIZER.aa_token_ids)


# ---------------------------------------------------------------------------
# Vocabulary properties
# ---------------------------------------------------------------------------


def test_aa_token_ids_length() -> None:
    """aa_token_ids has exactly 20 entries."""
    assert len(_TOKENIZER.aa_token_ids) == 20


def test_aa_token_ids_excludes_specials() -> None:
    """aa_token_ids excludes gap and every special token id."""
    aa = _TOKENIZER.aa_token_ids
    assert _TOKENIZER.gap_token_id not in aa
    assert _TOKENIZER.mask_token_id not in aa
    assert _TOKENIZER.cls_token_id not in aa
    assert _TOKENIZER.pad_token_id not in aa
    assert _TOKENIZER.unk_token_id not in aa


def test_aa_token_ids_match_encoded_amino_acids() -> None:
    """aa_token_ids equals the ids the encoder assigns to the 20 amino acids."""
    assert _TOKENIZER.aa_token_ids == _TOKENIZER.encode(_AMINO_ACIDS).tolist()


def test_all_token_ids_length() -> None:
    """all_token_ids has exactly 25 entries."""
    assert len(_TOKENIZER.all_token_ids) == 25


def test_all_token_ids_contains_gap_and_specials() -> None:
    """all_token_ids includes gap and every special token id."""
    all_ids = _TOKENIZER.all_token_ids
    assert _TOKENIZER.gap_token_id in all_ids
    assert _TOKENIZER.mask_token_id in all_ids
    assert _TOKENIZER.cls_token_id in all_ids
    assert _TOKENIZER.pad_token_id in all_ids
    assert _TOKENIZER.unk_token_id in all_ids


def test_all_token_ids_is_contiguous_from_zero() -> None:
    """all_token_ids is a gap-free range starting at zero."""
    assert _TOKENIZER.all_token_ids == list(range(len(_TOKENIZER.all_token_ids)))


def test_ordered_tokens_length_matches_all_token_ids() -> None:
    """ordered_tokens has one symbol per token id."""
    assert len(_TOKENIZER.ordered_tokens) == len(_TOKENIZER.all_token_ids)


def test_ordered_tokens_starts_with_amino_acids() -> None:
    """The first 20 ordered tokens are the standard amino acids."""
    assert _TOKENIZER.ordered_tokens[:20] == tuple(_AMINO_ACIDS)


@pytest.mark.parametrize(
    ("name", "symbol"),
    [
        ("gap_token_id", "-"),
        ("mask_token_id", "_"),
        ("cls_token_id", "^"),
        ("pad_token_id", "*"),
        ("unk_token_id", "?"),
    ],
)
def test_ordered_tokens_names_special_tokens(name: str, symbol: str) -> None:
    """Special tokens are reported as their literal alphabet characters."""
    assert _TOKENIZER.ordered_tokens[getattr(_TOKENIZER, name)] == symbol


def test_ordered_tokens_is_index_aligned_with_id_to_token() -> None:
    """ordered_tokens is ordered by token id."""
    for token_id, symbol in enumerate(_TOKENIZER.ordered_tokens):
        assert _TOKENIZER.id_to_token(token_id) == symbol


# ---------------------------------------------------------------------------
# token_to_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("char", "expected_id"),
    [
        ("A", 0),
        ("C", 1),
        ("D", 2),
        ("W", 18),
        ("Y", 19),
        ("-", 20),
        ("_", 21),
        ("^", 22),
        ("*", 23),
        ("?", 24),
    ],
)
def test_token_to_id_known_chars(char: str, expected_id: int) -> None:
    """Known alphabet characters map to their expected integer ids."""
    assert _TOKENIZER.token_to_id(char) == expected_id


def test_token_to_id_returns_builtin_int() -> None:
    """token_to_id returns a builtin int, not a numpy scalar."""
    assert isinstance(_TOKENIZER.token_to_id("A"), int)


@pytest.mark.parametrize("char", ["X", "B", "J", "Z", "a", "1", " "])
def test_token_to_id_unknown_char_returns_unk(char: str) -> None:
    """A character outside the alphabet maps to unk_token_id."""
    assert _TOKENIZER.token_to_id(char) == _TOKENIZER.unk_token_id


@pytest.mark.parametrize("token", ["", "AC", "Ω", "🧬"])
def test_token_to_id_non_alphabet_input_returns_unk(token: str) -> None:
    """Empty, multi-character, and non-latin1 input falls back to unk_token_id."""
    assert _TOKENIZER.token_to_id(token) == _TOKENIZER.unk_token_id


# ---------------------------------------------------------------------------
# id_to_token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token_id", "expected_char"),
    [
        (0, "A"),
        (1, "C"),
        (19, "Y"),
        (20, "-"),
        (21, "_"),
        (22, "^"),
        (23, "*"),
        (24, "?"),
    ],
)
def test_id_to_token_known_ids(token_id: int, expected_char: str) -> None:
    """Known integer ids map to their expected character."""
    assert _TOKENIZER.id_to_token(token_id) == expected_char


def test_id_to_token_unk_id_returns_question_mark() -> None:
    """The unk token id maps to '?'."""
    assert _TOKENIZER.id_to_token(_TOKENIZER.unk_token_id) == "?"


@pytest.mark.parametrize("token_id", [25, 999, -1])
def test_id_to_token_unrecognized_id_returns_question_mark(token_id: int) -> None:
    """An id outside the alphabet maps to '?' rather than raising."""
    assert _TOKENIZER.id_to_token(token_id) == "?"


def test_id_to_token_roundtrips_token_to_id() -> None:
    """id_to_token inverts token_to_id for every alphabet character."""
    for token_id in _TOKENIZER.all_token_ids:
        char = _TOKENIZER.id_to_token(token_id)
        assert _TOKENIZER.token_to_id(char) == token_id


# ---------------------------------------------------------------------------
# encode — return shape
# ---------------------------------------------------------------------------


def test_encode_single_string_returns_bare_tensor() -> None:
    """A single string input returns a 1-D tensor, not a list."""
    result = _TOKENIZER.encode("ACDE")
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 1


def test_encode_batch_returns_list_of_correct_length() -> None:
    """A batch of N strings returns a list of length N."""
    result = _TOKENIZER.encode(["ACDE", "IKL"])
    assert isinstance(result, list)
    assert len(result) == 2


def test_encode_single_element_batch_returns_list_of_one() -> None:
    """A one-element sequence returns a list, not the bare tensor."""
    result = _TOKENIZER.encode(["ACDE"])
    assert isinstance(result, list)
    assert len(result) == 1


def test_encode_output_length_matches_sequence_length() -> None:
    """Each output tensor length matches its input sequence length."""
    seqs = ["ACDE", "IKL", "W"]
    tokens = _TOKENIZER.encode(seqs)
    for seq, tok in zip(seqs, tokens, strict=True):
        assert tok.shape[0] == len(seq)


def test_encode_empty_string_returns_empty_tensor() -> None:
    """An empty string tokenizes to an empty tensor."""
    assert _TOKENIZER.encode("").shape == (0,)


def test_encode_empty_batch_returns_empty_list() -> None:
    """An empty batch tokenizes to an empty list."""
    assert _TOKENIZER.encode([]) == []


def test_call_is_an_alias_for_encode() -> None:
    """Calling the tokenizer directly is equivalent to calling encode."""
    assert torch.equal(_TOKENIZER("ACDE"), _TOKENIZER.encode("ACDE"))


# ---------------------------------------------------------------------------
# encode — dtype
# ---------------------------------------------------------------------------


def test_encode_default_dtype_is_long() -> None:
    """Tokenized output defaults to dtype torch.long."""
    assert _TOKENIZER.encode("ACDE").dtype == torch.long


def test_encode_empty_string_default_dtype_is_long() -> None:
    """The empty-sequence path honours the default dtype."""
    assert _TOKENIZER.encode("").dtype == torch.long


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.long])
def test_encode_honours_requested_dtype(dtype: torch.dtype) -> None:
    """The dtype argument controls the dtype of the returned tensors."""
    assert _TOKENIZER.encode("ACDE", dtype=dtype).dtype == dtype


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.long])
def test_encode_honours_requested_dtype_for_empty_string(dtype: torch.dtype) -> None:
    """The empty-sequence path honours the requested dtype."""
    assert _TOKENIZER.encode("", dtype=dtype).dtype == dtype


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.long])
def test_encode_dtype_does_not_change_token_values(dtype: torch.dtype) -> None:
    """Changing dtype does not change the token ids produced."""
    expected = _TOKENIZER.encode("AY-_^*").tolist()
    assert _TOKENIZER.encode("AY-_^*", dtype=dtype).tolist() == expected


def test_encode_batch_honours_requested_dtype() -> None:
    """Every tensor in a batch gets the requested dtype."""
    tokens = _TOKENIZER.encode(["ACDE", "IKL"], dtype=torch.uint8)
    assert all(tok.dtype == torch.uint8 for tok in tokens)


def test_encode_dtype_is_keyword_only() -> None:
    """The dtype argument cannot be passed positionally."""
    with pytest.raises(TypeError):
        _TOKENIZER.encode("ACDE", torch.uint8)  # ty: ignore[no-matching-overload]


def test_encode_output_is_writable() -> None:
    """Encoded tensors do not alias a read-only buffer."""
    tokens = _TOKENIZER.encode("ACDE", dtype=torch.uint8)
    tokens[0] = _TOKENIZER.mask_token_id
    assert tokens[0].item() == _TOKENIZER.mask_token_id


# ---------------------------------------------------------------------------
# encode — token values
# ---------------------------------------------------------------------------


def test_encode_known_sequence_values() -> None:
    """ACDEFG tokenizes to [0, 1, 2, 3, 4, 5]."""
    assert _TOKENIZER.encode("ACDEFG").tolist() == [0, 1, 2, 3, 4, 5]


def test_encode_all_amino_acids_yields_contiguous_ids() -> None:
    """The 20 amino acids tokenize to ids 0..19 in alphabet order."""
    assert _TOKENIZER.encode(_AMINO_ACIDS).tolist() == list(range(20))


def test_encode_gap_char_maps_to_gap_token_id() -> None:
    """Gap character maps to gap_token_id."""
    assert _TOKENIZER.encode("A-C")[1].item() == _TOKENIZER.gap_token_id


@pytest.mark.parametrize(
    ("char", "name"),
    [("_", "mask_token_id"), ("^", "cls_token_id"), ("*", "pad_token_id")],
)
def test_encode_special_chars_map_to_special_token_ids(char: str, name: str) -> None:
    """Special characters map to their corresponding token ids."""
    assert _TOKENIZER.encode(f"A{char}C")[1].item() == getattr(_TOKENIZER, name)


@pytest.mark.parametrize("char", ["X", "B", "J", "Z", "a"])
def test_encode_unknown_char_maps_to_unk_token_id(char: str) -> None:
    """A character outside the alphabet maps to unk_token_id."""
    assert _TOKENIZER.encode(f"A{char}C")[1].item() == _TOKENIZER.unk_token_id


def test_encode_agrees_with_token_to_id() -> None:
    """Encoding a sequence agrees with per-character token_to_id lookups."""
    seq = "ACDEXY-_^*"
    expected = [_TOKENIZER.token_to_id(c) for c in seq]
    assert _TOKENIZER.encode(seq).tolist() == expected


def test_encode_batch_matches_individual_encodes() -> None:
    """Batched encoding matches encoding each sequence on its own."""
    seqs = ["ACDE", "IKL", "MWVY"]
    batched = _TOKENIZER.encode(seqs)
    for seq, tok in zip(seqs, batched, strict=True):
        assert torch.equal(tok, _TOKENIZER.encode(seq))


# ---------------------------------------------------------------------------
# encode — downstream index compatibility
# ---------------------------------------------------------------------------


def test_encode_output_is_usable_as_gather_index() -> None:
    """Default encoder output is an accepted index dtype for torch.gather."""
    tokens = _TOKENIZER.encode("ACDE")
    log_p = torch.randn(1, len(tokens), len(_TOKENIZER.all_token_ids))
    gathered = log_p.gather(dim=2, index=tokens[None, :, None]).squeeze(-1)
    assert gathered.shape == (1, len(tokens))


def test_encode_output_is_usable_as_embedding_index() -> None:
    """Default encoder output is an accepted index dtype for nn.Embedding."""
    tokens = _TOKENIZER.encode("ACDE")
    embedding = torch.nn.Embedding(len(_TOKENIZER.all_token_ids), 4)
    assert embedding(tokens).shape == (len(tokens), 4)


def test_encode_output_indexes_positionally_not_as_mask() -> None:
    """Encoded tokens index a vocabulary axis by position, not as a bool mask."""
    tokens = _TOKENIZER.encode("CA")
    vocab = torch.arange(len(_TOKENIZER.all_token_ids)) * 10
    assert vocab[tokens].tolist() == [10, 0]


# ---------------------------------------------------------------------------
# decode — return shape
# ---------------------------------------------------------------------------


def test_decode_single_tensor_returns_bare_string() -> None:
    """A single tensor input returns a string, not a list."""
    result = _TOKENIZER.decode(_TOKENIZER.encode("ACDE"))
    assert result == "ACDE"


def test_decode_batch_returns_list_of_correct_length() -> None:
    """A batch of N tensors returns a list of length N."""
    tokens = _TOKENIZER.encode(["ACDE", "IKL"])
    result = _TOKENIZER.decode(tokens)
    assert isinstance(result, list)
    assert len(result) == 2


def test_decode_single_element_list_returns_list_of_one() -> None:
    """A one-element list returns a list, not the bare string."""
    result = _TOKENIZER.decode([_TOKENIZER.encode("ACDE")])
    assert result == ["ACDE"]


def test_decode_empty_list_returns_empty_list() -> None:
    """An empty list decodes to an empty list."""
    assert _TOKENIZER.decode([]) == []


def test_decode_empty_tensor_returns_empty_string() -> None:
    """An empty tensor decodes to an empty string."""
    assert _TOKENIZER.decode(_TOKENIZER.encode("")) == ""


def test_decode_flags_are_keyword_only() -> None:
    """keep_gaps cannot be passed positionally."""
    with pytest.raises(TypeError):
        _TOKENIZER.decode(_TOKENIZER.encode("A-C"), False)  # ty: ignore[no-matching-overload]


# ---------------------------------------------------------------------------
# decode — round trips
# ---------------------------------------------------------------------------


def test_decode_roundtrip_single() -> None:
    """Encode then decode recovers the original sequence."""
    seq = _AMINO_ACIDS
    assert _TOKENIZER.decode(_TOKENIZER.encode(seq)) == seq


def test_decode_roundtrip_batch() -> None:
    """Encode then decode recovers all sequences in a batch."""
    seqs = ["ACDE", "IKL", "MWVY"]
    assert _TOKENIZER.decode(_TOKENIZER.encode(seqs)) == seqs


def test_decode_roundtrip_preserves_special_chars() -> None:
    """Gap and special characters survive an encode/decode round trip."""
    seq = "AC-DE_F^G*H"
    assert _TOKENIZER.decode(_TOKENIZER.encode(seq)) == seq


def test_decode_roundtrip_maps_unknown_chars_to_question_mark() -> None:
    """Characters outside the alphabet come back as '?'."""
    assert _TOKENIZER.decode(_TOKENIZER.encode("AXC")) == "A?C"


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.long])
def test_decode_accepts_any_integer_dtype(dtype: torch.dtype) -> None:
    """Round trips work regardless of the dtype tokens were encoded with."""
    assert _TOKENIZER.decode(_TOKENIZER.encode("AC-DE", dtype=dtype)) == "AC-DE"


def test_decode_accepts_numpy_backed_tensor_of_unrecognized_ids() -> None:
    """Ids outside the alphabet decode to '?' rather than raising."""
    tokens = torch.tensor([0, len(_TOKENIZER.all_token_ids), 1])
    assert _TOKENIZER.decode(tokens) == "A?C"


# ---------------------------------------------------------------------------
# decode — keep_gaps / keep_unk
# ---------------------------------------------------------------------------


def test_decode_keep_gaps_true_preserves_gap_chars() -> None:
    """keep_gaps=True preserves gap characters in the output."""
    assert _TOKENIZER.decode(_TOKENIZER.encode("A-C"), keep_gaps=True) == "A-C"


def test_decode_keep_gaps_false_removes_gap_chars() -> None:
    """keep_gaps=False removes gap characters from the output."""
    assert _TOKENIZER.decode(_TOKENIZER.encode("A-C"), keep_gaps=False) == "AC"


def test_decode_keep_gaps_false_removes_every_gap() -> None:
    """keep_gaps=False removes gaps at the start, middle, and end."""
    assert _TOKENIZER.decode(_TOKENIZER.encode("--A-C--"), keep_gaps=False) == "AC"


def test_decode_keep_unk_true_preserves_unk_as_question_mark() -> None:
    """keep_unk=True preserves unknown tokens as '?'."""
    result = _TOKENIZER.decode(_TOKENIZER.encode("AXC"), keep_unk=True)
    assert result == "A?C"


def test_decode_keep_unk_false_removes_unk_positions() -> None:
    """keep_unk=False removes unknown-token positions."""
    assert _TOKENIZER.decode(_TOKENIZER.encode("AXC"), keep_unk=False) == "AC"


def test_decode_keep_gaps_false_retains_unk() -> None:
    """keep_gaps=False leaves unknown tokens in place."""
    result = _TOKENIZER.decode(_TOKENIZER.encode("A-XC"), keep_gaps=False)
    assert result == "A?C"


def test_decode_keep_unk_false_retains_gaps() -> None:
    """keep_unk=False leaves gap characters in place."""
    result = _TOKENIZER.decode(_TOKENIZER.encode("A-XC"), keep_unk=False)
    assert result == "A-C"


def test_decode_both_flags_false_strips_gaps_and_unk() -> None:
    """Disabling both flags strips gaps and unknown tokens together."""
    result = _TOKENIZER.decode(
        _TOKENIZER.encode("A-XC"), keep_gaps=False, keep_unk=False
    )
    assert result == "AC"


def test_decode_flags_apply_per_sequence_in_a_batch() -> None:
    """Filtering flags are applied independently to each sequence in a batch."""
    tokens = _TOKENIZER.encode(["A-C", "IK", "-X-"])
    result = _TOKENIZER.decode(tokens, keep_gaps=False, keep_unk=False)
    assert result == ["AC", "IK", ""]


def test_decode_does_not_mutate_input_tensor() -> None:
    """Filtering does not modify the tensor passed in."""
    tokens = _TOKENIZER.encode("A-XC")
    before = tokens.clone()
    _TOKENIZER.decode(tokens, keep_gaps=False, keep_unk=False)
    assert torch.equal(tokens, before)


# ---------------------------------------------------------------------------
# Lookup table invariants
# ---------------------------------------------------------------------------


def test_alphabet_has_no_duplicate_chars() -> None:
    """Every alphabet character is unique, so ids are unambiguous."""
    alphabet = _TOKENIZER.ordered_tokens
    assert len(set(alphabet)) == len(alphabet)


def test_byte_tables_cover_the_full_byte_range() -> None:
    """Both lookup tables are 256 entries wide, one per possible byte."""
    assert _TOKENIZER._bytes_to_token_id.shape == (256,)
    assert _TOKENIZER._token_id_to_bytes.shape == (256,)


def test_unmapped_bytes_fall_back_to_unk() -> None:
    """Every byte outside the alphabet maps to unk_token_id."""
    alphabet_bytes = {ord(c) for c in "".join(_AMINO_ACIDS) + "-_^*?"}
    table = _TOKENIZER._bytes_to_token_id
    unmapped = [b for b in range(256) if b not in alphabet_bytes]
    assert np.all(table[unmapped] == _TOKENIZER.unk_token_id)
