# DedeuceRL

**Benchmark LLMs on active hidden-system identification**: probe an unknown
system, infer its behavior, and submit a hypothesis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/AashVed/DedeuceRL/actions/workflows/ci.yml/badge.svg)](https://github.com/AashVed/DedeuceRL/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dedeucerl.svg)](https://pypi.org/project/dedeucerl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Architecture

DedeuceRL is split into three layers:

| Layer | Responsibility |
|---|---|
| `dedeucerl.kernel` | Pure hidden-system semantics. No LLMs, prompts, datasets, provider adapters, or Verifiers dependency. |
| `dedeucerl.ir` | Executable task contracts: action spaces, observations, hypothesis checks, resources, feedback, generators, and renderers. |
| `dedeucerl.runtime` | Budget, turns, traps, tool execution, structured events, submissions, and replay. |
| `dedeucerl.surface` | Prompt/tool-schema/dataset/Verifiers/CLI compilers. |

The extension point is `TaskIR`: pair a small pure `SystemKernel` with executable
contracts, then the engine provides the runtime and surfaces.

`mealy` is the current reference kernel. Protocol/APIEnv/ExprPolicy will return
as kernels after the architecture stabilizes.

## Installation

```bash
pip install dedeucerl
pip install "dedeucerl[openai]"
pip install "dedeucerl[all]"
```

Requirements: Python 3.10+, `verifiers>=0.1.12,<0.2`, `datasets>=3.0,<4.7.0`.

## Quickstart

```bash
dedeucerl-generate --skin mealy --seeds 0-9 --budget 25 --n-states 3 -o tasks.json
dedeucerl-eval --skin mealy --split tasks.json --model heuristic:none --out results.jsonl
dedeucerl-aggregate results.jsonl --format markdown
```

`heuristic:none` is an offline smoke baseline and does not require API keys.

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

A `TaskIR` wraps that kernel with an executable `ToolActionSpace`, observation
models, hypothesis contracts, resources, feedback, generators, and renderers.
The runtime canonicalizes raw tool arguments into domain actions, handles budget,
errors, trap state, submissions, event logs, and replay. Surfaces compile the
same IR into prompts, provider tool schemas, datasets, Verifiers envs, and CLI
workflows.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
python -m build
```

Before pushing, update the changelog, make the minimal appropriate version bump,
and keep docs in sync with public behavior.

## Citation

If you use DedeuceRL in research, cite the repository and the Zenodo DOI in
`CITATION.cff`.
