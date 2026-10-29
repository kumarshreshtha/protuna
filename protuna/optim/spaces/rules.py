"""Composable rules that decide the output space constraints for mutations.

Each rule takes a :py:type:`~protuna.optim.spaces.search.SearchMask` — an ``[L, 20]``
boolean mask where ``True`` at ``(site, aa)`` means amino acid ``aa`` is allowed at that
site — and returns a narrowed one.

Applying them in order composes a restricted
:class:`~protuna.optim.spaces.search.SearchSpace`; order matters, since a later
rule can allow substitutions at sites an earlier one fixed.

Examples:
    Restrict mutations to the two middle residues of ``MKTA`` and exclude cysteine:

    >>> from protuna.optim import spaces, rules
    >>> space = spaces.SearchSpace.from_rules(
    ...     "MKTA",
    ...     [rules.AllowOnly(positions=[1, 2]), rules.DenyResidues(amino_acids=["C"])],
    ... )
    >>> # The first and last residues remain fixed.
    >>> space.mutable_positions.tolist()
    [False, True, True, False]
    >>> # Position 1 allows every amino acid except cysteine.
    >>> space.available_aa_at_pos(1)
    'ADEFGHIKLMNPQRSTVWY'
"""

import dataclasses

from jaxtyping import Bool
import torch

from protuna import constants
from protuna import data
from protuna.data.msa.core import TokenizedMSA
from protuna.score import pssm
from protuna.typing import AminoAcid

type SearchMask = Bool[torch.Tensor, "l v"]


@dataclasses.dataclass
class AllowOnly:
    """Only allow these positions to mutate.

    Args:
        positions: Zero-based sites at which to retain the currently allowed
            substitutions. All other sites are fixed.
    """

    positions: list[int]

    def __call__(self, allowed: SearchMask) -> SearchMask:
        """Allow substitutions only at ``positions``.

        Args:
            allowed: Current mask of allowed residues.

        Returns:
            A new mask preserving the allowed residues at ``positions`` and
            disallowing all residues at every other site.
        """
        restricted = torch.zeros_like(allowed)
        restricted[self.positions] = allowed[self.positions]
        return restricted


@dataclasses.dataclass
class Allow:
    """Allow these positions to mutate to these amino acids.

    Args:
        positions: Zero-based sites at which to allow substitution with
            ``amino_acids``.
        amino_acids: Residues to allow. Defaults to all 20 standard amino acids.
    """

    positions: list[int]
    amino_acids: tuple[AminoAcid, ...] = constants.PROTEIN_ALPHABET

    def __call__(self, allowed: SearchMask) -> SearchMask:
        """Allow ``amino_acids`` at ``positions``.

        Args:
            allowed: Current mask of allowed residues, updated in place.

        Returns:
            The updated mask.
        """
        positions = torch.tensor(
            self.positions, dtype=torch.long, device=allowed.device
        )
        amino_acids = data.tokenizer.encode("".join(self.amino_acids))
        allowed[positions[:, None], amino_acids] = True
        return allowed


@dataclasses.dataclass
class Deny:
    """Deny these positions to mutate to these amino acids.

    Args:
        positions: Zero-based sites at which to exclude ``amino_acids`` from
            substitutions.
        amino_acids: Residues to exclude. Defaults to all 20 standard amino acids.
    """

    positions: list[int]
    amino_acids: tuple[AminoAcid, ...] = constants.PROTEIN_ALPHABET

    def __call__(self, allowed: SearchMask) -> SearchMask:
        """Deny ``amino_acids`` at ``positions``.

        Args:
            allowed: Current mask of allowed residues, updated in place.

        Returns:
            The updated mask.
        """
        positions = torch.tensor(
            self.positions, dtype=torch.long, device=allowed.device
        )
        amino_acids = data.tokenizer.encode("".join(self.amino_acids))
        allowed[positions[:, None], amino_acids] = False
        return allowed


@dataclasses.dataclass
class FixSites:
    """Protect key positions from being mutated.

    Args:
        positions: Zero-based sites to keep fixed.
    """

    positions: list[int]

    def __call__(self, allowed: SearchMask) -> SearchMask:
        """Keep ``positions`` fixed.

        Args:
            allowed: Current mask of allowed residues, updated in place.

        Returns:
            The updated mask.
        """
        positions = torch.tensor(
            self.positions, dtype=torch.long, device=allowed.device
        )
        allowed[positions] = False
        return allowed


@dataclasses.dataclass
class DenyResidues:
    """Deny specific residues from appearing in mutation.

    Args:
        amino_acids: Residues to exclude from substitutions at every site.
    """

    amino_acids: list[AminoAcid]

    def __call__(self, allowed: SearchMask) -> SearchMask:
        """Deny ``amino_acids`` at every site.

        Args:
            allowed: Current mask of allowed residues, updated in place.

        Returns:
            The updated mask.
        """
        amino_acids = data.tokenizer.encode("".join(self.amino_acids))
        allowed[:, amino_acids] = False
        return allowed


@dataclasses.dataclass
class Preserve:
    """Don't mutate any occurrences of these amino acids.

    Args:
        amino_acids: Residues whose occurrences in ``target_sequence`` identify
            sites to keep fixed.
        target_sequence: Sequence used to locate protected sites, with one
            residue per site in the mask.
    """

    amino_acids: list[AminoAcid]
    target_sequence: str

    def __call__(self, allowed: SearchMask) -> SearchMask:
        """Keep sites holding ``amino_acids`` in the target fixed.

        Args:
            allowed: Current mask of allowed residues, updated in place.

        Returns:
            The updated mask.
        """
        seq_tokens = data.tokenizer.encode(self.target_sequence)
        amino_acids = data.tokenizer.encode("".join(self.amino_acids))
        preserve_mask = (seq_tokens[:, None] == amino_acids).any(dim=1)
        allowed[preserve_mask] = False

        return allowed


@dataclasses.dataclass
class MSAConserved:
    """Do not mutate highly conserved positions.

    Args:
        tokenized_msa: Alignment used to estimate residue frequencies, with its
            query sequence in the first row and columns matching the mask sites.
        threshold: Minimum Laplace-smoothed frequency of the query residue for a
            site to be kept fixed. Expressed as a fraction from 0 to 1.
    """

    tokenized_msa: TokenizedMSA
    threshold: float

    def __call__(self, allowed: SearchMask) -> SearchMask:
        """Keep sites with MSA frequency at or above ``threshold`` fixed.

        Args:
            allowed: Current mask of allowed residues, updated in place.

        Returns:
            The updated mask.
        """
        freq_mat = pssm.compute_frequency_matrix(
            self.tokenized_msa, laplace_smoothing=True
        )
        conserve_mask = (
            freq_mat[torch.arange(freq_mat.shape[0]), self.tokenized_msa.msa_tokens[0]]
            >= self.threshold
        )
        allowed[conserve_mask] = False
        return allowed


@dataclasses.dataclass
class MSACovered:
    """Do not mutate positions with low coverage.

    Args:
        tokenized_msa: Alignment used to measure coverage, with columns matching
            the mask sites.
        threshold: Minimum fraction of non-gap residues required for substitutions
            to remain allowed at a site. Sites below this value are kept fixed.
    """

    tokenized_msa: TokenizedMSA
    threshold: float

    def __call__(self, allowed: SearchMask) -> SearchMask:
        """Keep sites with MSA coverage below ``threshold`` fixed.

        Args:
            allowed: Current mask of allowed residues, updated in place.

        Returns:
            The updated mask.
        """
        msa_tokens = self.tokenized_msa.msa_tokens
        non_gap_positions = msa_tokens != data.tokenizer.gap_token_id
        site_coverage = non_gap_positions.float().mean(dim=0)
        allowed[site_coverage < self.threshold] = False
        return allowed
