# MCP Episode Server

DedeuceRL can expose a benchmark episode directly to any Model Context Protocol
host. The interface is deliberately agent-agnostic: DedeuceRL owns the hidden
task, state transitions, budget, trace, and score; the host owns model selection,
conversation orchestration, and sampling settings.

## Run an episode

```bash
dedeucerl-mcp serve --task mealy --seed 42
```

This command speaks MCP over STDIO, so it is normally launched by an MCP host
rather than typed into an interactive terminal. Add it to the host's MCP server
configuration, start a new host session, and let the agent use the discovered
tools. If a host does not begin from server instructions alone, the only user
message needed is a generic one such as: “Solve the DedeuceRL episode using its
MCP tools.” The benchmark specification itself does not need to be pasted.

Each server process owns one episode. Launch a fresh process for every seed or
rollout; this keeps state, attribution, and failure handling unambiguous.

## Lifecycle

1. On connection, MCP discovery returns task instructions and the current
   runtime-generated JSON schemas.
2. Every tool call goes directly through `EpisodeRuntime`, including validation,
   charging, traps, submissions, and structured errors.
3. The first semantic call starts artifact persistence, and each call is appended
   to `trace.jsonl` as it happens. Discovery-only processes create no artifacts.
4. A correct submission, budget exhaustion, or task-defined terminal trap causes
   immediate finalization and scoring.
5. If the host disconnects first, the incomplete episode is finalized with
   `termination_reason: "disconnected"` and score `0.0`.

Terminal tool responses include `score`, `reward`, and `termination_reason`, so
the agent can observe the finalized outcome without a separate scoring tool.
A correct submission after an earlier nonterminal trap is labeled
`solved_with_trap`; it remains unsuccessful and receives the trap-adjusted score.

There are intentionally no `run_spec` or `score` tools. Those are benchmark
control-plane operations, not actions the evaluated agent should choose. The
server handles them automatically.

## Artifacts

The default paths are:

```text
.dedeucerl/runs/<run-id>/result.json
.dedeucerl/runs/<run-id>/trace.jsonl
```

`result.json` contains the task identity, public generation parameters,
termination reason, benchmark score, budget accounting, and timestamps. It never
contains the hidden instance. `trace.jsonl` contains an `episode_start` record,
one record per tool call, and an `episode_end` record.

Useful options:

```bash
dedeucerl-mcp serve \
  --task mealy \
  --seed 42 \
  --budget 25 \
  --feedback \
  --param n_states=4 \
  --runs-dir benchmark-runs
```

- `--out PATH` overrides the final result path.
- `--trace-out PATH` overrides the trace path.
- `--run-id ID` supplies a filesystem-safe run identifier.
- `--no-persist` disables both artifacts.
- `--no-trap` and `--n-states` are convenient Mealy options; `--param KEY=VALUE`
  is the generic task-parameter interface.

Existing artifact paths are rejected instead of overwritten.

## MCP contract

The server uses the stable MCP Python SDK v2 line and STDIO transport. Tools are
compiled from the same `ToolActionContract` objects used by the evaluator and
Verifiers surface. Inputs use each contract's exact JSON schema; outputs use
structured MCP content and advertise schemas that cover both semantic success
and DedeuceRL's charged error envelope.
