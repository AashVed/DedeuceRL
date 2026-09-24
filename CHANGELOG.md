# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
During current development, release numbers do not guarantee backward compatibility
for task-author APIs. Breaking changes and migrations are documented per release.

## [1.1.1] - 2026-09-22

This release intentionally removes superseded task-author APIs. It is not a
backward-compatible patch under Semantic Versioning. See
[the migration guide](docs/OBJECTIVES.md#migration-from-110).

### Added
- Added general `Objective` contracts with typed candidates, private evaluation evidence, optional programmable feedback, and author-controlled terminal failures.
- Linked kernel state, task, runtime, and evaluation context types; added strict JSON candidate decoding, generated feedback schemas, a `py.typed` marker, and authoring API type checks in CI.
- Added isolated initial-state plan execution with per-action charging, retryable submissions, and `evaluation_steps` accounting across MCP, CLI, and Verifiers.
- Added the registered `mealy_palindrome` task with configurable minimum length/variety and exact safe-solution existence/affordability checks, plus a non-Mealy workflow example.
- Added custom-objective authoring and migration documentation, schema regressions, negative type tests, and cross-surface integration tests.
- Added exact uniform strongly connected transition sampling and a reusable connectivity check, with structural audits and installed-MCP model comparisons.

### Changed
- Palindrome `submit_answer` accepts a sequence or a final impossibility claim. Its generator retains naturally impossible machines and skips possible-but-unaffordable candidates, without balancing classes.
- Default reward is binary correctness for all tasks; valid plans and correct impossibility claims earn equally, with efficiency reported separately. Historical efficiency-weighted rewards are not directly comparable.
- `TaskIR.objective` replaces `hypothesis_contract` and `feedback_model`; existing identification contracts are composed through `HypothesisObjective`.
- Removed `FeedbackModel`, the old `TaskIR.submit` path, and unused hypothesis judgment enrichment fields. See `docs/OBJECTIVES.md` for migration.
- Verifiers forwards original JSON arguments to runtime validation, preserving explicit nulls, omitted fields, and aliases.
- Mealy task 2.3 and palindrome task 1.2 replace the fixed A cycle with a broader strongly connected, minimal-machine distribution; traps preserve safe recovery from every state. Existing saved instances retain their behavior.
- Removed the unused first-action backbone helpers. See `docs/GENERATION.md` for sampling guarantees, scaling limits, and migration.

### Fixed
- Preserved terminal evaluator decisions when optional feedback serialization fails.
- Stopped Verifiers on terminal tool results without requesting an extra model response.
- Reported `done=true` on calls consuming the final budget unit.
- Accounted for actually consumed budget when a tool's fee exceeds the remaining budget.
- Validated feedback after serialization without exposing private validation input in error messages.
- Kept nested feedback schema references valid in MCP response envelopes.
- MCP records the served instance's task version when replaying older datasets.

## [1.1.0] - 2026-09-18

### Added
- Added `dedeucerl-mcp serve`, a provider-neutral, one-episode MCP STDIO server built on the stable MCP Python SDK v2 API.
- Added automatic per-call JSONL tracing and automatic result scoring/persistence on terminal state or client disconnect.
- Added MCP instructions and structured input/output schemas compiled directly from each task's `TaskIR` contracts.
- Added final score, reward, and termination reason to terminal MCP tool responses.

### Changed
- Extracted pure identification scoring so non-Verifiers surfaces use the same benchmark reward semantics.
- Raised the Verifiers floor to 0.1.14 for a dependency set compatible with MCP SDK v2.
- Deferred MCP artifact creation until the first semantic tool call, preventing discovery-only server processes from leaving empty runs.

### Fixed
- Preserved TaskIR types, enums, and descriptions in Verifiers tool schemas.
- Derived Verifiers turn limits from the largest train/evaluation task budget instead of capping runs at 64 turns.
- Allowed generated training data to be paired with an independently loaded evaluation split.
- Made `--resume` reject result files with missing or incompatible split/evaluation provenance.
- Kept MCP traces terminal after `episode_end` and rejected aliased or concurrently claimed artifact paths.
- Reported correct terminal submissions after an earlier nonterminal trap as `solved_with_trap`.

## [1.0.14] - 2026-06-28

### Fixed
- Made `EpisodeRuntime.tool_calls` count every charged tool call, including invalid arguments, malformed submissions, unknown tools, and kernel/hypothesis input errors.

## [1.0.13] - 2026-06-27

### Changed
- Made hypothesis and equivalence contracts first-class TaskIR concepts with explicit parse, validate, normalize, judge, counterexample, and distance phases.
- Moved submission judgment out of semantic kernels; kernels now only return hidden-system transitions.
- Generated Mealy `submit_table` from `FiniteTransducerIsomorphismContract` instead of a task-local submit implementation.

### Added
- Added `ExactJSONContract` and `FiniteTransducerIsomorphismContract` as real, tested built-in hypothesis contracts.
- Added pure finite-transducer table helpers for normalization, isomorphism, and distinguishing counterexamples.

## [1.0.12] - 2026-06-27

### Changed
- Replaced static tool contracts with executable `ToolActionSpace` and `ToolActionContract` definitions.
- Added composable action spaces for enum, product, union, sequence, JSONSchema-backed, and masked actions.
- Updated runtime, Mealy, prompts, provider schemas, Verifiers, CLI, docs, and tests to canonicalize raw tool arguments before kernel dispatch.

## [1.0.11] - 2026-06-27

### Changed
- Added executable `TaskIR` as the contract layer between semantic kernels and surfaces.
- Moved Mealy tool contracts, observation rendering, hypothesis judgment, feedback policy, resource policy, and generator metadata into the Mealy TaskIR.
- Updated runtime, dataset, prompt, Verifiers, CLI, and interactive surfaces to compile from TaskIR instead of kernel-specific contract methods.

## [1.0.10] - 2026-06-27

### Changed
- Rebuilt DedeuceRL around Kernel, Runtime, and Surface layers.
- Replaced skin-based extension with pure `SystemKernel` contracts, generic `EpisodeRuntime`, and surface compilers for datasets, prompts, tools, CLI eval, CLI game, and Verifiers.
- Rebuilt Mealy as the reference `MealyKernel`.

### Removed
- Removed the old `HiddenSystemEnv`, `TaskGenerator`, `DomainSpec`, and built-in skin environment classes.
- Removed Protocol, APIEnv, and ExprPolicy example skins until they can be remade on the new kernel architecture.

## [1.0.9] - 2026-04-22

### Changed
- DedeuceRL is now Prime-first for RL workflows. Hosted training uses `prime rl run`, and self-managed training uses `prime-rl`, both through `dedeucerl.vf_env` instead of the removed legacy trainer wrapper.
- Package dependency floors now target the current stable Prime ecosystem: `verifiers>=0.1.12,<0.2` and `datasets>=3.0,<4.7.0`.
- Training documentation now reflects the current Prime workflow for both hosted training and self-managed Prime-compatible development.

### Removed
- The `dedeucerl-train` CLI and its checked-in legacy training templates.
- The `dedeucerl[rl]` extra. Prime should be installed separately via `uv tool install prime`.

## [1.0.8] - 2026-04-05

### Added
- `dedeucerl-aggregate` now emits additive benchmark summaries including `eval_config_hash`, `max_complete_k`, `pass_at_1`, and `pass_at_3` when sufficient rollouts are available.

### Fixed
- `dedeucerl-eval-parallel --resume` now reconstructs shard-local resume state from the merged output, so completed work is not rerun after shard part files have been deleted.
- `dedeucerl-eval` now derives episode turn limits per problem, allowing mixed-difficulty split files to run correctly.

### Changed
- Eval result rows now include normalized evaluation provenance via `eval_config` and `eval_config_hash`, and resume/summary filtering respects that identity.

## [1.0.7] - 2026-04-03

### Changed
- `dedeucerl-aggregate` now groups results by model + skin + split hash, and reports both run count (`n_runs`) and unique episode count (`n_episodes`).
- Documentation now clarifies that `dedeucerl-eval` reports the benchmark reward, while RL training may use `reward_mode="train_dense"` or other training-oriented rubric settings.

## [1.0.6] - 2026-02-01

### Added
- `dedeucerl-eval --effort <level>`: pass provider reasoning/thinking effort for supported models.
- `dedeucerl-eval --no-effort-probe`: skip the cheap probe call used to validate `--effort`.

### Changed
- Gemini adapter now supports only the official `google-genai` SDK (no legacy `google-generativeai` fallback).
- `--effort` validation is now provider-driven: a cheap probe call catches unsupported settings early (instead of hardcoded model tables).

## [1.0.5] - 2026-02-01

### Added
- Verifiers-compatible environment entrypoint (`dedeucerl.vf_env`) to train any skin via a single `env.id`.
- Seed-based dataset generation for training without requiring split files.
- Sample legacy training configs.
- Legacy training CLI for config generation and launch.
- Training-friendly `reward_mode="train_dense"` rubric (generic, skin-agnostic).

### Changed

## [1.0.4] - 2026-01-31

### Added
- `dedeucerl-eval-parallel`: shard-parallel evaluation runner that merges per-shard JSONL outputs.
- Episode selection + sharding for `dedeucerl-eval` via `--episodes` and `--shard`.
- Split-aware resuming for `dedeucerl-eval` via `--resume` (and `--append` for explicit appends).
- Episode utilities: `parse_index_spec`, `parse_shard`, `apply_shard`, `compute_split_hash`.
- CLI tests covering selection, sharding, resume, and parallel parity.

### Changed
- `dedeucerl-eval` now streams results to JSONL (flush per episode/rollout) instead of buffering all results in memory.
- Updated smoke splits (`seeds/*_smoke.json`) to use a `dev` subset key and expanded episode items.

## [1.0.3] - 2026-01-29

### Added
- GitHub Actions CI workflow and PyPI publish workflow.
- `cliGame`: interactive CLI game for playing/exploring environments.
- `CONTRIBUTING.md` and skin documentation (`docs/SKINS.md`).

### Changed
- Documentation improvements and prompt clarity tweaks.

## [1.0.2] - 2026-01-22

### Added
- New `ExprPolicyEnv` skin (`exprpolicy`) modeling typed policy/DSL debugging with compiler + test feedback.
- Universal, schema-first skin contract test suite (`tests/test_universal_skins.py`) to centralize skin validation.

### Changed
- `TaskGenerator` no longer forwards `None` values into `domain_spec()` / `build_observation()` to prevent `null` fields in observations.
- Standardized submit tool payloads to always include `counterexample` (set to `null` when not applicable).

## [1.0.1] - 2026-01-17

### Changed
- Improve split generation docs and OpenRouter support.
