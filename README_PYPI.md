# DedeuceRL

Benchmark LLMs on active hidden-system identification.

DedeuceRL is organized around three layers:

- `dedeucerl.kernel`: pure hidden-system semantics
- `dedeucerl.ir`: executable task contracts and renderers
- `dedeucerl.runtime`: budget, traps, events, tool dispatch, and replay
- `dedeucerl.surface`: prompts, provider tool schemas, datasets, CLIs, and Verifiers

## Install

```bash
pip install dedeucerl
pip install "dedeucerl[openai]"
pip install "dedeucerl[all]"
```

## Quickstart

```bash
dedeucerl-generate --skin mealy --seeds 0-9 --budget 25 --n-states 3 -o tasks.json
dedeucerl-eval --skin mealy --split tasks.json --model heuristic:none --out results.jsonl
dedeucerl-aggregate results.jsonl --format markdown
```

`mealy` is the current reference kernel. Protocol/APIEnv/ExprPolicy are planned
to return as kernels after the architecture stabilizes.

## Prime / Verifiers

```bash
uv tool install prime
prime lab setup
```

Use `dedeucerl.vf_env` as the Verifiers entrypoint.

License: MIT.
