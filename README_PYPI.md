# DedeuceRL

Benchmark LLMs on active hidden-system identification.

DedeuceRL is organized around four layers:

- `dedeucerl.kernel`: pure hidden-system semantics
- `dedeucerl.ir`: executable action spaces, hypothesis/equivalence contracts, task contracts, and renderers
- `dedeucerl.runtime`: budget, traps, events, tool dispatch, and replay
- `dedeucerl.surface`: prompts, provider tool schemas, datasets, CLIs, Verifiers, and MCP

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

## MCP Mode

Expose one stateful episode to any MCP host over STDIO:

```bash
dedeucerl-mcp serve --task mealy --seed 42
```

The MCP initialization supplies the task instructions and tools. DedeuceRL
automatically records the semantic tool trace, scores terminal episodes, and
writes `result.json` plus `trace.jsonl` under `.dedeucerl/runs/<run-id>/`.
Terminal responses include the score, while discovery-only processes create no
artifacts. No provider adapter, manual run-spec tool, or manual scoring step is
required.

## Prime / Verifiers

```bash
uv tool install prime
prime lab setup
```

Use `dedeucerl.vf_env` as the Verifiers entrypoint.

License: MIT.
