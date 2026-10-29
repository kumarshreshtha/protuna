---
name: documentation
description: Use it when asked to write docstrings for modules, functions and classes. Do not use it when writing pages in docs/ such as concepts, guides or examples.
---

## General instructions

- Use plain language. Don't be too verbose; clear and succinct communication is important.
- Make sure docstrings are compatible with google python style guidelines.
- Docstrings should be written in rst compatible format for automatic documentation generation with sphinx.
- Docstrings may already exist containing hints/instructions from the user in form of rough notes and citations, build on top of these notes and instructions when present.
- When writing examples in the docstring to illustrate the API, provide a short description of what the example demonstrates then follow with the code snippet.

---

## Module docstring

- Start module docstrings with a short noun phrase naming the module’s purpose or responsibility, such as “Storage management for candidates and evaluations.”
- Include doctest compatible usage examples on how to interact with the module when feasible – when small illustrative input and outputs are possible.
- Keep these examples faithful to the module usage rather than creating obscure examples as a work around for doctest feasibility.
- When doctests are infeasible – example leads to heavy computation/download etc. – mark the example to skip doctests, they'll just act as illustrations for the user.
- Examples should be simple and with the intent of clarifying the API.
- When talking about specific functions/classes in the module always provide links to them that sphinx can parse. For example, :func:`func_name`
- The module level docstring should act as the central thesis for the module - a short reference guide.

---

## Class and function docstring

- Only document functions/classes that are public.
- When describing a function's behavior always describe what a function does and not how it does it. The only exception is when the function is a specific algorithm or mathematical calculation so explaining the steps clarify the function behavior.
- When a function/method/class is just a wrapper around another then only write detailed documentation for one and cross-reference to it from the others.
- When both a functional and object-oriented API to the the same functionality are provided document the class and cross-reference to it from the function.
- Prefer math equations to describe the function behavior when the function is a specific formula or calculation.
- Use .. note:: , .. warning:: and .. seealso:: in documentations where appropriate.
- Include doctest compatible usage examples with docstrings (with the same caveats as those mentioned for module level) for non-trivial classes and functions.
- When hinted in comments that a implementation follows from a paper, cite the appropriate paper from `references.bib`. If the citation is not found in the file add the appropriate citation there (don't do this from memory, fetch the bibtex online) and cite it. Inform the user when you do this.
- Document attributes of dataclasses using "Args:" instead of "Attributes:" – required for proper autodoc parsing.

### Parameter descriptions

- Describe what each argument supplies and what it is used for in the enclosing API. Assume readers already understand the underlying abstraction; do not repeat its general definition.
- For component arguments, prefer a short noun phrase identifying their role, followed by context where needed. For example: `sampler: The proposal generator to sample candidates from.`
- Explain defaults and special values defined by this API when their meaning affects usage. Do not describe conversions performed by another API as defaults here.
- Keep straightforward descriptions short. Add detail where it clarifies a meaningful behavior or configuration requirement.
- Avoid assumptions about an input's origin or purpose that the API does not require.
