# Protuna skills plugin

Skills for agents coding with [Protuna](https://github.com/kumarshreshtha/protuna)

| Skill | Use it to |
| --- | --- |
| `protuna-pipeline` | Assemble, run, and troubleshoot optimization pipelines, including batching, recycling, Ray serving, and storage. |
| `protuna-components` | Implement custom samplers, selectors, adapters, predictors, constraints, and search rules. |

References and examples are bundled with each skill. The plugin supplies instructions;
executing campaigns requires Protuna, applicable model dependencies, and compute
in the agent's execution environment. Installing the plugin does not install the Python
packages.

## Installation

### Install from GitHub

#### Codex

```sh
codex plugin marketplace add kumarshreshtha/protuna
codex plugin add protuna-skills@protuna
```

Start a new session, or use `/plugins` to browse the configured marketplace.
These commands require a Codex CLI version supporting plugins.

#### Claude Code

Run these commands in your terminal:

```sh
claude plugin marketplace add kumarshreshtha/protuna
claude plugin install protuna-skills@protuna
```

Alternatively, inside Claude Code, run `/plugin marketplace add kumarshreshtha/protuna`
and then `/plugin install protuna-skills@protuna`.
Start a new session in your own project and invoke
`/protuna-skills:protuna-pipeline` or `/protuna-skills:protuna-components`.
Installation defaults to your user account; add `--scope project` to the terminal
install command to share the plugin configuration with a project.
See [Claude Code's marketplace installation documentation](https://code.claude.com/docs/en/discover-plugins).

#### Claude Science

In Claude Science, open **Settings > Skills > Add skill > Import from GitHub**
and enter `https://github.com/kumarshreshtha/protuna`. Select
`protuna-pipeline` and `protuna-components` from `protuna-skills/skills/`.

GitHub imports do not update
automatically; use **Settings > Skills > Imported**, open the repository's
three-dot menu, and choose **Check for updates…** when needed.
See [Claude Science's connectors and skills documentation](https://claude.com/docs/claude-science/connectors-and-skills).


### Manual copy

For other tools, copy the two directories inside `protuna-skills/skills/` into
your agent's skills directory, preserving each `SKILL.md` and `references/`
folder. Each skill must sit directly under the destination skills directory,
for example `<skills-directory>/protuna-pipeline/SKILL.md`.
No build step is required. Ensure the downloaded skills match your Protuna version.
