# DedeuceRL

Benchmark LLMs on hidden-system exploration and control.

DedeuceRL is organized around four layers:

- `dedeucerl.kernel`: pure hidden-system semantics
- `dedeucerl.ir`: typed objectives, action spaces, hypothesis/equivalence contracts, and renderers
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

The `mealy` task asks for a transition table. `mealy_palindrome` asks for an action
sequence that produces a palindrome. Each submission executes from initial state
in isolation; exploration, submission fees, and executed actions share one budget.
Agents can retry failed plans while budget remains, or submit a final claim that
no qualifying safe sequence exists. Generation keeps natural impossible cases and
skips possible-but-unaffordable ones. Either correct answer earns reward 1; failure
earns 0, with budget usage reported separately.

Authors can define their own typed candidates, evaluators, and optional feedback
for any kernel through `Objective` and `TaskIR`. See the
[custom-objective guide](https://github.com/AashVed/DedeuceRL/blob/dev/docs/OBJECTIVES.md)
and its non-Mealy workflow example.

## MCP Mode

Expose one stateful episode to any MCP host over STDIO:

```bash
dedeucerl-mcp serve --task mealy --seed 42
dedeucerl-mcp serve --task mealy_palindrome --seed 7 --budget 25 --feedback
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
