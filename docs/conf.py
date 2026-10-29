"""Sphinx configuration for the Protuna documentation."""

from collections.abc import MutableMapping
import dataclasses
import datetime
import inspect
import pathlib
import re
import sys

from docutils import nodes
from pybtex import database
from pybtex import plugin
from pybtex.style import template
from pybtex.style.formatting import unsrt
from sphinx import addnodes
from sphinx import application
from sphinx import environment
from sphinx.ext import intersphinx
from sphinx.ext.napoleon import docstring

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

project = "Protuna"
copyright = f"{datetime.datetime.now(tz=datetime.UTC).year}, Protuna Contributors"
author = "Kumar Shreshtha"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autosummary",
]

# Resolve Markdown fragment links to page titles and section headings.
myst_heading_anchors = 2


class _SingleLinkStyle(unsrt.Style):
    """Format references with one link, preferring DOI, URL, then arXiv."""

    def format_web_refs(self, e: database.Entry) -> template.Node:
        """Show the first available link in DOI, URL, arXiv order."""
        return template.sentence[
            template.first_of[
                template.optional[self.format_doi(e)],
                template.optional[self.format_url(e)],
                template.optional[self.format_eprint(e)],
            ]
        ]


plugin.register_plugin("pybtex.style.formatting", "single_link", _SingleLinkStyle)

bibtex_bibfiles = ["references/references.bib", "references/datasets.bib"]
bibtex_default_style = "single_link"
bibtex_reference_style = "author_year"

html_theme = "furo"
html_title = "Protuna Documentation"

# Keep the documentation self-contained and the reading area generous.
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
html_favicon = "_static/favicon-32x32.png"

# Furo keeps its accessible navigation and mobile drawers; custom.css adjusts
# the desktop proportions without replacing the theme templates.
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": True,
    "light_logo": "protuna-wordmark-light.svg",
    "dark_logo": "protuna-wordmark-dark.svg",
    "light_css_variables": {
        "font-stack": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "font-stack--monospace": "'SFMono-Regular', Consolas, monospace",
        # Darken the logo gray (#8C8C82) for readable text on light surfaces.
        "color-brand-primary": "#62625B",
        "color-brand-content": "#3A3A35",
        "color-link": "var(--color-brand-primary)",
        "color-link--hover": "#3A3A35",
        "color-link-underline": "#8C8C82",
        "color-link-underline--visited": "#8C8C82",
        "color-link-underline--hover": "#3A3A35",
        "color-link-underline--visited--hover": "#3A3A35",
        "color-background-primary": "#FAFAF8",
        "color-sidebar-background": "#F0F0EC",
        "color-sidebar-search-background": "#F0F0EC",
        "color-background-secondary": "#F3F3EF",
        "color-background-hover": "#ECECE6",
        "color-background-border": "#E1E1D9",
        "color-foreground-primary": "#1A1A16",
        "color-foreground-secondary": "#6B6B62",
        "color-foreground-muted": "#8C8C82",
        "color-code-background": "#F3F3EF",
        "color-highlight-on-target": "#ECECE6",
        "color-foreground-border": "#D4D4CC",
        "color-code-foreground": "#3A3A35",
    },
    "dark_css_variables": {
        "font-stack": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "font-stack--monospace": "'SFMono-Regular', Consolas, monospace",
        "color-brand-primary": "#C8E600",
        "color-brand-content": "#C8E600",
        "color-link": "var(--color-brand-content)",
        "color-link--hover": "var(--color-brand-content)",
        "color-link-underline": "var(--color-background-border)",
        "color-link-underline--visited": "var(--color-background-border)",
        "color-link-underline--hover": "var(--color-foreground-border)",
        "color-link-underline--visited--hover": "var(--color-foreground-border)",
        "color-background-primary": "#1A1A16",
        "color-sidebar-background": "#211F1A",
        "color-sidebar-search-background": "#211F1A",
        "color-background-secondary": "#211F1A",
        "color-background-hover": "#2E2E28",
        "color-background-border": "#2E2E28",
        "color-foreground-primary": "#F0F0EC",
        "color-foreground-secondary": "#8C8C82",
        "color-foreground-muted": "#6B6B62",
        "color-code-background": "#211F1A",
        "color-highlight-on-target": "#2E2E28",
        "color-foreground-border": "#3A3A33",
        "color-code-foreground": "#D8D8CF",
    },
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_returns = True
napoleon_use_rtype = True
# Preserve example boundaries when Sphinx combines class and constructor docs.
# The doctree hook below restores the usual rubric presentation.
napoleon_use_admonition_for_examples = True
# Render ``Attributes:`` sections as a field list instead of standalone
# ``.. attribute::`` directives, so they do not collide with the same
# attributes picked up by autodoc's ``:undoc-members:``.
napoleon_use_ivar = True
napoleon_preprocess_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

autodoc_member_order = "bysource"
# Show callable and iterable interfaces on both direct and autosummary class pages.
autodoc_default_options = {"special-members": "__call__, __iter__"}
autodoc_typehints = "description"
# Add types only to parameters with descriptions in their docstrings.
autodoc_typehints_description_target = "documented_params"
# Include constructor parameter descriptions alongside every class docstring.
autoclass_content = "both"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = False
autosummary_generate_overwrite = True

# Keep generated pages concise in navigation and signatures readable in the article.
autosummary_short_signatures = True
python_maximum_signature_line_length = 88
toc_object_entries = True
toc_object_entries_show_parents = "hide"


def _clear_autosummary_stubs(app: application.Sphinx) -> None:
    """Remove generated API stubs before autosummary recreates current pages."""
    # This ignored directory contains only generated pages. Autosummary overwrites
    # current stubs but leaves pages for removed or renamed APIs behind.
    for stub in (pathlib.Path(app.srcdir) / "api" / "generated").glob("*.rst"):
        stub.unlink()
    # Incremental builds initially reuse the cached source list. Refresh it so
    # autosummary also discovers new pages before generating their API stubs.
    app.env.find_files(app.config, app.builder)


def _exclude_documented_dataclass_fields(
    app: application.Sphinx,
    what: str,
    name: str,
    obj: object,
    options: object,
    lines: list[str],
) -> None:
    """Deduplicate dataclass descriptions while preserving attribute targets."""
    if what != "class" or not dataclasses.is_dataclass(obj):
        return
    documented = {
        line.split(":", 2)[1].split()[-1] for line in lines if line.startswith(":ivar ")
    }
    # Autodoc processes the class and constructor docstrings separately.
    constructor_lines = docstring.GoogleDocstring(
        inspect.getdoc(obj.__init__) or "", config=app.config
    ).lines()
    is_constructor_docstring = (
        "\n".join(lines).strip() == "\n".join(constructor_lines).strip()
    )
    constructor_parameters = {
        line.split(":", 2)[1].split()[-1]
        for line in constructor_lines
        if line.startswith(":param ")
    }
    parameters = constructor_parameters | {
        line.split(":", 2)[1].split()[-1]
        for line in lines
        if line.startswith(":param ")
    }
    # Explicit constructor descriptions take precedence over class summaries.
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            line.startswith((":ivar ", ":vartype "))
            and line.split(":", 2)[1].split()[-1] in parameters
        ) or (
            line.startswith((":param ", ":type "))
            and line.split(":", 2)[1].split()[-1] in constructor_parameters
            and not is_constructor_docstring
        ):
            del lines[index]
            while index < len(lines) and lines[index].startswith(" "):
                del lines[index]
        else:
            index += 1
    # Parameter descriptions do not create attribute targets. Keep those fields
    # available to autodoc, including fields whose redundant ivar text was removed.
    repeated = documented - parameters
    # Sphinx 9 uses option attributes; older versions use a mutable mapping.
    if isinstance(options, MutableMapping):
        excluded = set(options.get("exclude-members") or ())
        options["exclude-members"] = excluded | repeated
    else:
        excluded = set(getattr(options, "exclude_members", None) or ())
        vars(options)["exclude_members"] = excluded | repeated


def _resolve_tensor_reference(
    app: application.Sphinx,
    env: environment.BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.TextElement,
) -> nodes.reference | None:
    """Link jaxtyping tensor annotations to the PyTorch Tensor documentation."""
    if node.get("refdomain") != "py" or not intersphinx.inventory_exists(env, "torch"):
        return None
    target = node.get("reftarget", "")
    # Sphinx can split an annotation into dtype and unqualified Tensor references,
    # or retain the entire dynamically generated jaxtyping class name.
    if target != "Tensor" and not re.fullmatch(
        r"jaxtyping\.\w+\[(?:torch\.)?Tensor,\s*(['\"]).*\1\]", target
    ):
        return None
    tensor_node = node.copy()
    tensor_node["reftarget"] = "torch.Tensor"
    tensor_node["reftype"] = "class"
    return intersphinx.resolve_reference_in_inventory(
        env, "torch", tensor_node, contnode
    )


def _resolve_external_type_reference(
    app: application.Sphinx,
    env: environment.BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.TextElement,
) -> nodes.reference | None:
    """Link external types missing from their inventories to overview pages."""
    if node.get("refdomain") != "py":
        return None
    targets = {
        "polars.DataFrame": ("polars", "reference/dataframe/index"),
        "polars.dataframe.frame.DataFrame": ("polars", "reference/dataframe/index"),
        "polars.LazyFrame": ("polars", "reference/lazyframe/index"),
        "polars.lazyframe.frame.LazyFrame": ("polars", "reference/lazyframe/index"),
    }
    target = targets.get(node.get("reftarget", ""))
    if target is None:
        return None
    inventory, page = target
    if not intersphinx.inventory_exists(env, inventory):
        return None
    page_node = node.copy()
    page_node["refdomain"] = "std"
    page_node["reftype"] = "doc"
    page_node["reftarget"] = page
    page_node["refexplicit"] = True
    return intersphinx.resolve_reference_in_inventory(
        env, inventory, page_node, contnode
    )


def _move_examples_after_descriptions(
    app: application.Sphinx,
    doctree: nodes.document,
) -> None:
    """Place examples after API descriptions and before nested members."""
    example_classes = {"admonition-example", "admonition-examples"}
    for description in doctree.findall(addnodes.desc):
        if description.get("domain") != "py" or description.get("objtype") not in {
            "class",
            "function",
            "method",
        }:
            continue
        for content in description.children:
            if not isinstance(content, addnodes.desc_content):
                continue
            examples: list[nodes.admonition] = []
            for child in list(content.children):
                if isinstance(child, nodes.admonition) and example_classes.intersection(
                    child["classes"]
                ):
                    content.remove(child)
                    examples.append(child)
            index = next(
                (
                    index
                    for index, child in enumerate(content.children)
                    if isinstance(child, addnodes.desc)
                ),
                len(content.children),
            )
            for offset, example in enumerate(examples):
                content.insert(index + offset, example)
    # Keep the original rubric styling for examples.
    for example in list(doctree.findall(nodes.admonition)):
        if example_classes.intersection(example["classes"]):
            title, *body = example.children
            heading = nodes.rubric("", title.astext())
            heading.source = title.source
            heading.line = title.line
            example.replace_self([heading, *body])


def setup(app: application.Sphinx) -> None:
    """Register docstring filters and external reference resolvers."""
    app.connect("builder-inited", _clear_autosummary_stubs, priority=100)
    app.connect("autodoc-process-docstring", _exclude_documented_dataclass_fields)
    app.connect("missing-reference", _resolve_tensor_reference)
    app.connect("missing-reference", _resolve_external_type_reference, priority=600)
    app.connect("doctree-read", _move_examples_after_descriptions)
