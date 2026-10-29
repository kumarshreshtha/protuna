# Spaces

A **search space** pairs a starting protein sequence with the mutations allowed
on it. A **design space** brings together one or more of these starting points
and a strategy for choosing smaller regions to explore. Together, they define
where a campaign can search before a
{py:class}`~protuna.optim.samplers.base.Sampler` proposes any candidates.

This separates the boundaries of the search from the method used to explore it.
The same space can be searched by random sampling, site saturation, or a model
that favors promising sequences. Changing the
{py:class}`~protuna.optim.samplers.base.Sampler` changes how candidates are
proposed within those boundaries.

## Discrete sequence spaces

For a protein of length `L`, choosing one of the 20 standard amino acids at each
site gives `20 ** L` possible sequences. Protuna represents regions of this
discrete space through the residues allowed at each position. These spaces
describe substitutions at a fixed sequence length; insertions and deletions
are not supported by the current representation but are planned for future
releases.

A {py:class}`~protuna.optim.spaces.search.SearchSpace` contains two parts:

| Attribute | Meaning |
| --- | --- |
| `init_sequence` | The tokenized starting sequence, with one residue per site. |
| `allowed` | A boolean mask of shape `[L, 20]`, where `True` at `(site, aa)` means that amino acid is allowed at that site. |

A site is **mutable** when it allows at least one alternative to its starting
residue. A fixed site allows only the starting residue. The mask describes all
combinations of the allowed residues, so the size of a space is the product of
the number of choices at each site. Two sites with three choices each and all
other sites fixed give nine possible sequences, including the starting one.

The starting sequence is always inside its search space: constructing a
{py:class}`~protuna.optim.spaces.search.SearchSpace` sets the entries for its
starting residues to `True`. This also
means that allowing a site to mutate does not require a candidate to change it.
The starting point may be a wild-type sequence, an engineered variant, or a
candidate selected during an earlier cycle.

## Defining allowed mutations

You can construct a space directly from a tokenized sequence and an `allowed`
mask. More commonly,
{py:meth}`~protuna.optim.spaces.search.SearchSpace.from_rules` builds it from a
sequence and a list of {py:type}`~protuna.optim.spaces.search.SearchRule`s.
A rule is a callable that transforms the mask. Construction starts with all
20 amino acids allowed at every site, then applies the rules in order.

For example, this space allows substitutions at the two middle sites of `MKTA`
and excludes the introduction of cysteine:

```python
from protuna.optim import rules
from protuna.optim import spaces

space = spaces.SearchSpace.from_rules(
    "MKTA",
    [
        rules.AllowOnly(positions=[1, 2]),
        rules.DenyResidues(amino_acids=["C"]),
    ],
)

assert space.mutable_positions.tolist() == [False, True, True, False]
assert space.available_aa_at_pos(1) == "ADEFGHIKLMNPQRSTVWY"
assert space.size() == 19 * 19
assert space.is_valid("MATA")
assert not space.is_valid("MCTA")
```

Site indices are zero-based. Here, the first and last sites remain fixed, and
each middle site has 19 choices. Passing an empty rule list would leave every
site open to all 20 amino acids.

Rules can express experimental restrictions or incorporate biological evidence.
For example, {py:class}`~protuna.optim.spaces.rules.FixSites` protects specified
positions, {py:class}`~protuna.optim.spaces.rules.Preserve` fixes sites occupied
by particular residues in a reference sequence, and
{py:class}`~protuna.optim.spaces.rules.MSAConserved` fixes highly conserved sites
identified from an alignment. The [mutation rules reference](../api/optim/spaces.rst#mutation-rules)
lists the available rules.

You can also create your own rules: any callable that accepts an `[L, 20]`
boolean search mask and returns an updated mask of the same shape satisfies
the {py:type}`~protuna.optim.spaces.search.SearchRule` contract. It can use
experimental data or other evidence to decide which entries to allow.

Order matters because rules can reopen choices as well as remove them.
{py:class}`~protuna.optim.spaces.rules.AllowOnly` keeps the currently allowed
residues at its chosen sites and clears all other rows. It preserves earlier
residue restrictions at the selected sites. In contrast,
{py:class}`~protuna.optim.spaces.rules.Allow` adds choices to the current mask
and can reopen sites or residues that earlier rules excluded. Apply restrictions
such as {py:class}`~protuna.optim.spaces.rules.DenyResidues` after
{py:class}`~protuna.optim.spaces.rules.Allow` if they should remain in effect.

After all rules have run, the search space restores any starting residues
that the rules masked out.

## Sampling subspaces

Even a restricted search space can contain too many combinations to explore
at once. A **subspace** keeps the same starting sequence while narrowing the
allowed mutation sites. For example, a campaign might permit changes at 50 sites,
but explore only five of those sites together in each proposal request.

Subspaces also distribute exploration across parallel workers. For example,
100 subspaces can give replicas of a
{py:class}`~protuna.optim.samplers.base.Sampler` different regions to search
concurrently. Giving every replica the same 50 mutable sites can duplicate search
effort, especially when their proposal strategy favors the same sequences.

A {py:class}`~protuna.optim.spaces.subspaces.SearchSpaceSampler` takes one
search space and yields an iterator over narrower search spaces. The built-in
strategies select a subset of mutable sites, retain their allowed residues, and fix all
other sites to the starting sequence. Some built-in strategies include:

| Strategy | How it works |
| --- | --- |
| {py:class}`~protuna.optim.spaces.subspaces.PartitionSampler` | Splits the ordered mutable sites into disjoint groups. |
| {py:class}`~protuna.optim.spaces.subspaces.RandomMaskSampler` | Draws groups of mutable sites uniformly at random. |
| {py:class}`~protuna.optim.spaces.subspaces.CouplingMaskSampler` | Grows groups by favoring sites with stronger pairwise couplings. |

The [subspace sampling reference](../api/optim/spaces.rst#subspace-sampling)
lists all built-in strategies and their parameters.

The choice affects which combinations of mutations can be proposed. Partitioning
two mutable sites into separate groups lets each site be explored independently,
but neither subspace contains a sequence mutated at both sites. Randomly sampled
groups can overlap, allowing a site to be explored alongside
different partners.

### Choosing sites to explore together

In masked diffusion sampling, each step predicts residues at masked sites from
the visible sequence, then unmasks some of them. When residues are drawn
independently from the predictions of one forward pass, choosing sites that are
approximately independent given the visible sequence reduces the risk of
incompatible combinations. Strongly coupled sites, such as residues that
interact in the folded protein, may need one another's resolved values to make
compatible choices.

Choosing a search subspace poses the reverse problem. If one member of a coupled
pair is fixed, conditioning on that residue can make changes at its partner
unlikely. Keeping both sites open gives the
{py:class}`~protuna.optim.samplers.base.Sampler` room to explore coordinated
changes using its own unmasking strategy.
{py:class}`~protuna.optim.spaces.subspaces.CouplingMaskSampler` favors such groups
using pairwise coupling scores.

Uncertainty can also guide where to search: sites with several plausible
residues may merit more exploration than sites dominated by one residue.
{py:class}`~protuna.optim.spaces.subspaces.MSAEntropyMaskSampler` uses residue
entropy in an alignment, while
{py:class}`~protuna.optim.spaces.subspaces.LogitsEntropyMaskSampler` uses entropy
in a model's predictions. These scores alone do not capture dependencies, but
uncertainty and coupling can work together. For example, a custom `site_selector`
can seed each group in
{py:class}`~protuna.optim.spaces.subspaces.CouplingMaskSampler` with an uncertain mutable site,
then let coupling scores guide the group's growth.

## Bringing starting points together

A {py:class}`~protuna.optim.spaces.design.DesignSpace` holds the starting search
spaces and an optional search space sampler. Each starting space has its own
sequence and allowed mask, so a campaign can explore several parent sequences
with shared or different mutation restrictions.

Continuing the example above, suppose a second parent already has alanine at
site 1. Both parents can be explored using the same mutation rules, with one
mutable site per subspace:

```python
second_space = spaces.SearchSpace.from_rules(
    "MATA",
    [
        rules.AllowOnly(positions=[1, 2]),
        rules.DenyResidues(amino_acids=["C"]),
    ],
)
design_space = spaces.DesignSpace(
    [space, second_space],
    space_sampler=spaces.PartitionSampler(n_partitions=2),
)

subspaces = list(design_space.sample_search_spaces())
assert len(subspaces) == 4
assert subspaces[0].mutable_positions.tolist() == [False, True, False, False]
assert subspaces[1].mutable_positions.tolist() == [False, False, True, False]
```

{py:meth}`~protuna.optim.spaces.design.DesignSpace.sample_search_spaces` applies
the sub-sampling strategy to each starting space in turn. With no `space_sampler`,
it yields the starting spaces unchanged. Within each starting space, it drops repeated
mutable-site masks, so random sampling may yield fewer subspaces than requested.

## Spaces in the pipeline

The {py:class}`~protuna.optim.pipeline.Pipeline` receives a
{py:class}`~protuna.optim.spaces.design.DesignSpace` and a
{py:class}`~protuna.optim.samplers.base.Sampler`. Each generation cycle draws
subspaces from the design space and streams them to the
{py:class}`~protuna.optim.samplers.base.Sampler` as proposal requests. The
resulting candidates are evaluated by the configured constraints and optional
objective, then an acquisition selects the experimental batch.

The allowed mask expresses choices that can be decided before generation, such
as keeping a catalytic residue fixed. Constraints evaluate complete candidates:
for example, limiting the overall fraction of hydrophobic amino acids, or
requiring a predicted fold to exceed a mean pLDDT or pTM confidence threshold.

With recycling enabled, selected candidates become the starting sequences for
another generation cycle. Their new spaces inherit the allowed mask from their
original ancestor's search space, and the same subspace sampling strategy is
applied again. This lets mutations accumulate across cycles even when each
subspace opens only a few sites. Setting `lock_mutations_on_recycle=True` fixes
sites that differ from the original ancestor, preserving those changes during
subsequent cycles.

See [Pipeline](pipeline.md) for the full generation, evaluation, recycling, and
selection workflow.
