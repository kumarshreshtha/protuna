"""Tokenizers for protein sequences.

Provides :class:`ProteinTokenizer`, a byte-level tokenizer for the standard
amino acid alphabet. All modules in protuna use this singleton tokenizer and expect
tokenized inputs from this tokenizer.

Examples:
    >>> from protuna.data import tokenizer

    Tokenize a single sequence:

    >>> tokens = tokenizer.encode("ACDEFG")
    >>> tokens.tolist()
    [0, 1, 2, 3, 4, 5]

    Tokenize a batch and decode back to strings:

    >>> batch = tokenizer.encode(["ACDEFG", "IKL"])
    >>> tokenizer.decode(batch)
    ['ACDEFG', 'IKL']

    Strip gap characters on detokenization:

    >>> tokenizer.decode(tokenizer.encode("A-C"), keep_gaps=False)
    'AC'
"""

from collections.abc import Sequence
from typing import final, overload

from jaxtyping import Integer
import numpy as np
import torch

from protuna import constants


@final
class ProteinTokenizer:
    """Byte-level tokenizer for protein sequences.

    Maps each character of the standard 20-amino-acid alphabet, the gap character
    ``"-"``, and the special tokens (mask, cls, pad, unk) to a unique integer ID.
    IDs are assigned in alphabet order, so the 20 amino acids occupy ``0..19`` and
    every non-amino-acid token follows them. Characters outside the alphabet are
    mapped to :attr:`unk_token_id`.

    The full vocabulary in ID order can be recovered from :attr:`ordered_tokens`,
    which reports every token as its literal alphabet character. The special token
    symbols are:

    - ``-``: Gap in an aligned sequence.
    - ``_``: Masked amino acid position.
    - ``^``: Classification (CLS) token for sequence-level representations.
    - ``*``: Padding token for matching sequence lengths in a batch.
    - ``?``: Unknown character outside the supported alphabet.
    """

    _ALPHABET = constants.PROTEIN_ALPHABET + tuple("-_^*?")

    def __init__(self) -> None:
        """Initialize lookup tables for encoding and decoding."""
        self._token_to_id = {aa: i for i, aa in enumerate(self._ALPHABET)}
        self._id_to_token = {i: aa for i, aa in enumerate(self._ALPHABET)}
        token_ids = np.arange(len(self._ALPHABET), dtype=np.uint8)
        token_bytes = np.frombuffer("".join(self._ALPHABET).encode(), dtype=np.uint8)
        bytes_to_token_id = np.full((256,), self._token_to_id["?"], dtype=np.uint8)
        bytes_to_token_id[token_bytes] = token_ids  # everything else maps to unk
        token_id_to_bytes = np.full((256,), ord("?"), dtype=np.uint8)
        token_id_to_bytes[token_ids] = token_bytes
        self._bytes_to_token_id = bytes_to_token_id
        self._token_id_to_bytes = token_id_to_bytes

    @property
    def ordered_tokens(self) -> tuple[str, ...]:
        """All token symbols in order."""
        return self._ALPHABET

    @property
    def all_token_ids(self) -> list[int]:
        """All valid token IDs, including the gap and special tokens."""
        return list(range(len(self._ALPHABET)))

    @property
    def aa_token_ids(self) -> list[int]:
        """Token IDs for the 20 standard amino acids (excludes gap and specials)."""
        return list(range(20))

    @property
    def gap_token_id(self) -> int:
        """Token ID for the gap character ``"-"``."""
        return self._token_to_id["-"]

    @property
    def mask_token_id(self) -> int:
        """Token ID for the mask character ``"_"``."""
        return self._token_to_id["_"]

    @property
    def cls_token_id(self) -> int:
        """Token ID for the cls character ``"^"``."""
        return self._token_to_id["^"]

    @property
    def pad_token_id(self) -> int:
        """Token ID for the pad character ``"*"``."""
        return self._token_to_id["*"]

    @property
    def unk_token_id(self) -> int:
        """Token ID for any character outside the alphabet (e.g. B, J, X, Z)."""
        return self._token_to_id["?"]

    def token_to_id(self, token: str) -> int:
        """Return the ID for the given token character.

        Args:
            token: A single character.

        Returns:
            The integer token ID, or :attr:`unk_token_id` if *token* is not
            in the alphabet.
        """
        return self._token_to_id.get(token, self.unk_token_id)

    def id_to_token(self, token_id: int) -> str:
        """Return the token character for the given ID.

        Args:
            token_id: An integer token ID.

        Returns:
            The corresponding character, or ``"?"`` if *token_id* is not
            recognized.
        """
        return self._id_to_token.get(token_id, "?")

    @overload
    def encode(
        self, input: str, /, *, dtype: torch.dtype = torch.long
    ) -> Integer[torch.Tensor, " l"]: ...

    @overload
    def encode(
        self, input: Sequence[str], /, *, dtype: torch.dtype = torch.long
    ) -> list[Integer[torch.Tensor, " l"]]: ...

    def encode(
        self, input: Sequence[str] | str, /, *, dtype: torch.dtype = torch.long
    ) -> list[Integer[torch.Tensor, " l"]] | Integer[torch.Tensor, " l"]:
        """Tokenize one or more protein sequences.

        Characters not in alphabet are mapped to :attr:`unk_token_id`.

        Args:
            input: A single sequence string or a sequence of strings.
            dtype: Integer dtype of the returned tensors. Defaults to ``torch.long``.

        Returns:
            A single 1-D tensor if *input* is a string, otherwise a list of 1-D
            tensors, one per input sequence. Each tensor has length equal to the
            corresponding sequence length.
        """
        is_single = isinstance(input, str)
        sequences = [input] if is_single else input
        tokens = []
        for seq in sequences:
            seq_bytes = np.frombuffer(seq.encode(), dtype=np.uint8)
            tokens.append(
                torch.from_numpy(self._bytes_to_token_id[seq_bytes]).to(dtype)
            )
        if is_single:
            return tokens[0]
        return tokens

    __call__ = encode

    @overload
    def decode(
        self,
        token_ids: Integer[torch.Tensor, " l"],
        /,
        *,
        keep_gaps: bool = True,
        keep_unk: bool = True,
    ) -> str: ...

    @overload
    def decode(
        self,
        token_ids: list[Integer[torch.Tensor, " l"]],
        /,
        *,
        keep_gaps: bool = True,
        keep_unk: bool = True,
    ) -> list[str]: ...

    def decode(
        self,
        token_ids: list[Integer[torch.Tensor, " l"]] | Integer[torch.Tensor, " l"],
        /,
        *,
        keep_gaps: bool = True,
        keep_unk: bool = True,
    ) -> list[str] | str:
        """Convert token ID tensors back to amino acid strings.

        Args:
            token_ids: A single 1-D tensor or a list of 1-D tensors of token IDs.
            keep_gaps: If ``False``, positions with :attr:`gap_token_id` are
                removed before decoding.
            keep_unk: If ``False``, positions with :attr:`unk_token_id` are
                removed before decoding.

        Returns:
            A single decoded string if *token_ids* is a tensor, otherwise a list
            of decoded amino acid strings, one per input tensor.
        """
        is_single = isinstance(token_ids, torch.Tensor)
        batch = [token_ids] if is_single else token_ids
        sequences = []
        for tokens in batch:
            tok_array = tokens.cpu().numpy()
            if not keep_gaps:
                tok_array = tok_array[tok_array != self.gap_token_id]
            if not keep_unk:
                tok_array = tok_array[tok_array != self.unk_token_id]
            seq = bytes(self._token_id_to_bytes[tok_array].tolist()).decode()
            sequences.append(seq)
        if is_single:
            return sequences[0]
        return sequences


tokenizer = ProteinTokenizer()  # create once at module import
