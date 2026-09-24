# DedeuceRL

**Benchmark LLMs on hidden-system exploration and control**: probe an unknown
system, infer its behavior, and use it to achieve a task-defined objective.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/AashVed/DedeuceRL/actions/workflows/ci.yml/badge.svg)](https://github.com/AashVed/DedeuceRL/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dedeucerl.svg)](https://pypi.org/project/dedeucerl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Architecture

DedeuceRL is split into four layers:

| Layer | Responsibility |
|---|---|
| `dedeucerl.kernel` | Pure hidden-system semantics. No LLMs, prompts, datasets, provider adapters, or Verifiers dependency. |
| `dedeucerl.ir` | Typed objectives, action spaces, hypothesis/equivalence checks, observations, resources, feedback, generators, and renderers. |
| `dedeucerl.runtime` | Budget, turns, traps, tool execution, structured events, submissions, and replay. |
| `dedeucerl.surface` | Prompt/tool-schema/dataset/Verifiers/MCP/CLI compilers. |

The extension point is `TaskIR`: pair a small pure `SystemKernel` with executable
contracts, then the engine provides the runtime and surfaces.

The `mealy` task asks for a transition table; `mealy_palindrome` asks for an action
sequence that produces a palindrome, or a correct impossibility claim, on the same kernel. Custom objectives work
with other kernels too, as the [workflow example](examples/custom_objective.py)
demonstrates. Protocol/APIEnv/ExprPolicy remain planned domains.

## Installation

```bash
pip install dedeucerl
pip install "dedeucerl[openai]"
pip install "dedeucerl[all]"
```

Requirements: Python 3.10+, `verifiers>=0.1.14,<0.2`, `datasets>=3.0,<4.7.0`,
`mcp>=2.1.1,<3`, `pydantic>=2.10,<3`, and `jsonschema>=4,<5`.

## Quickstart

```bash
dedeucerl-generate --skin mealy --seeds 0-9 --budget 25 --n-states 3 -o tasks.json
dedeucerl-eval --skin mealy --split tasks.json --model heuristic:none --out results.jsonl
dedeucerl-aggregate results.jsonl --format markdown
```

`heuristic:none` is an offline smoke baseline and does not require API keys.

For an objective that uses the machine to produce an output:

```bash
dedeucerl-mcp serve --task mealy_palindrome --seed 7 --budget 25 --feedback
```

The agent explores with `act` and submits a typed answer with `submit_answer`. Each plan
runs from the initial state in isolation and costs its submission fee plus executed
actions. Failed attempts can be revised until the shared budget runs out. The task
chooses its goal and feedback; it does not require a fixed output string. An
`impossible` answer costs one unit and is final whether correct or wrong. Generation
keeps natural impossible cases and skips possible-but-unaffordable machines. Default
reward is 1 for either correct answer and 0 for failure; costs are reported separately.
See the [answer schemas and authoring guide](docs/OBJECTIVES.md).

## MCP Mode

Run one benchmark episode as a provider-neutral MCP STDIO server:

```bash
dedeucerl-mcp serve --task mealy --seed 42
```

Configure any MCP host to launch that command. For example, in hosts that use a
JSON server map:

```json
{
  "mcpServers": {
    "dedeucerl": {
      "command": "dedeucerl-mcp",
      "args": ["serve", "--task", "mealy", "--seed", "42"]
    }
  }
}
```

The host receives the episode instructions and current tool schemas during MCP
discovery. It does not need a pasted run specification, a scoring tool, or a
provider-specific adapter. Each server process owns exactly one stateful episode.
Terminal tool calls return the final score/reward and are persisted immediately;
an unfinished episode is finalized when the host disconnects. Discovery-only
server processes create no artifacts.

By default, artifacts are written to
`.dedeucerl/runs/<run-id>/{result.json,trace.jsonl}`. Use `--out`, `--trace-out`,
or `--runs-dir` to change paths, and `--no-persist` for ephemeral sessions. See
[docs/MCP.md](docs/MCP.md) for the lifecycle and complete options.

## Programmatic Use

```python
import json

from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.runtime import EpisodeRuntime

entry = TASK_REGISTRY["mealy"]
instance = entry.ir.generator.sample(seed=0, budget=25, n_states=3, trap=True)
runtime = EpisodeRuntime(entry.ir, instance, feedback=True)

print(runtime.call_tool("act", {"symbol": "A"}).output)
print(runtime.call_tool("submit_table", {"table_json": json.dumps(instance.private["table"])}).output)
```

## Prime / Verifiers

Install Prime separately for RL workflows:

```bash
uv tool install prime
prime lab setup
```

Use `dedeucerl.vf_env` as the stable Verifiers entrypoint:

```python
import verifiers as vf

env = vf.load_environment(
    "dedeucerl.vf_env",
    skin="mealy",
    seeds="0-9",
    budget=25,
    n_states=3,
    feedback=True,
)
```

## Creating Tasks

See [docs/KERNELS.md](docs/KERNELS.md). A kernel provides:

- `initial_state(instance)`
- `call(instance, state, tool_name, action)`

A `TaskIR[State]` wraps that kernel with action spaces, observations, an
`ObjectiveContract[State]`, resources, a generator, and renderers. Define a typed
candidate, evaluator, and optional feedback with `Objective`, or adapt an existing
hypothesis contract with `HypothesisObjective`. The runtime handles validation,
budget, isolated execution, errors, traps, and replay across all surfaces.
See [the custom-objective guide](docs/OBJECTIVES.md) for the author API, examples,
and migration from 1.1.0.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest -q
python -m build
```

Before pushing, update the changelog, make the minimal appropriate version bump,
and keep docs in sync with public behavior.

## Citation

If you use DedeuceRL in research, cite the repository and the Zenodo DOI in
`CITATION.cff`.
