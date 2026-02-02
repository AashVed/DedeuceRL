"""dedeucerl-eval: CLI for running DedeuceRL evaluations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from dedeucerl.skins import SKIN_REGISTRY
from dedeucerl.core import TaskGenerator, make_rubric
from dedeucerl.adapters import get_adapter
from dedeucerl.adapters.base import decompose_model_spec
from dedeucerl.utils import (
    DedeuceError,
    error_invalid_argument,
    error_invalid_json,
    error_unknown_tool,
    parse_index_spec,
    parse_shard,
    apply_shard,
    compute_split_hash,
)


def _one_line(text: Any, *, limit: int = 220) -> str:
    s = "" if text is None else str(text)
    s = " ".join(s.split())
    if len(s) <= limit:
        return s
    return s[:limit] + f"... (len={len(s)})"


def _preview_args(args: Dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 200:
            parts.append(f"{k}=<str len={len(v)}> ")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _tool_error_envelope(
    state: Dict[str, Any], err: DedeuceError, *, extra: Optional[Dict[str, Any]] = None
) -> str:
    payload: Dict[str, Any] = {
        "error": err.to_dict(),
        "budget_left": int(state.get("budget", 0)),
        "queries_used": int(state.get("queries_used", 0)),
        "trap_hit": bool(state.get("trap_hit", False)),
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload)


def _consume_cli_budget(state: Dict[str, Any], amount: int = 1) -> None:
    """Consume budget for *invalid* tool calls handled in the CLI.

    Normally, budget is consumed inside the environment tool methods.
    If the model emits an invalid tool call (unknown tool / malformed args),
    we still charge budget to avoid free retries.
    """

    amount = int(max(0, amount))
    if amount == 0:
        return

    budget = int(state.get("budget", 0))
    if budget <= 0:
        state["budget"] = 0
        state["done"] = True
        state["ok"] = False
        return

    used = min(budget, amount)
    state["budget"] = budget - used
    state["queries_used"] = int(state.get("queries_used", 0)) + used

    if int(state.get("budget", 0)) <= 0:
        state["budget"] = 0
        state["done"] = True
        if not bool(state.get("ok", False)):
            state["ok"] = False


def _domain_spec_from_answer(SkinClass, answer_data: Dict[str, Any]):
    """Build a skin DomainSpec from the episode answer payload.

    Preferred path:
    - If the skin implements `domain_params_from_answer(answer_data)`, use it.

    Backward-compatible path:
    - Fall back to legacy heuristics for older skins.
    """
    budget = int(answer_data.get("budget", 25))

    # "trap" is not explicitly stored; infer from trap lists.
    trap = False
    if "trap_pairs" in answer_data:
        trap = bool(answer_data.get("trap_pairs"))
    elif "trap_calls" in answer_data:
        trap = bool(answer_data.get("trap_calls"))

    extractor = getattr(SkinClass, "domain_params_from_answer", None)
    if callable(extractor):
        try:
            raw_params = extractor(answer_data)
        except Exception:
            raw_params = None

        params: Dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}

        # Always prefer effective episode values for these.
        params["budget"] = budget
        params["trap"] = trap
        return SkinClass.domain_spec(**params)

    # ─────────────────────────────────────────────────────────────
    # Legacy heuristics (kept for compatibility)
    # ─────────────────────────────────────────────────────────────

    if "table" in answer_data:
        table = answer_data.get("table", {})
        n_states = int(table.get("n", 5))
        return SkinClass.domain_spec(n_states=n_states, budget=budget, trap=trap)

    if "spec" in answer_data:
        spec = answer_data.get("spec", {})
        n_states = int(spec.get("n_states", 3))
        transitions = spec.get("transitions", {})
        endpoints: List[str] = []
        if (
            isinstance(transitions, dict)
            and "0" in transitions
            and isinstance(transitions["0"], dict)
        ):
            endpoints = list(transitions["0"].keys())
        n_endpoints = len(endpoints) if endpoints else int(answer_data.get("n_endpoints", 3))
        return SkinClass.domain_spec(
            n_states=n_states,
            n_endpoints=n_endpoints,
            endpoints=endpoints or None,
            budget=budget,
            trap=trap,
        )

    return SkinClass.domain_spec(budget=budget, trap=trap)


def _load_dotenv(path: Path) -> None:
    """Load environment variables from a local .env file.

    Supports both `KEY=VALUE` and `export KEY=VALUE` lines. Existing environment
    variables are not overwritten.
    """
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if not key or key in os.environ:
            continue

        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="dedeucerl-eval",
        description="Run DedeuceRL evaluations on hidden system identification tasks.",
    )

    parser.add_argument(
        "--skin",
        required=True,
        choices=list(SKIN_REGISTRY.keys()),
        help="Skin to use (e.g., 'mealy').",
    )
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        help="Path to split JSON file.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        type=str,
        help="Subset name within split file (e.g., 'dev', 'test').",
    )
    parser.add_argument(
        "--model",
        default="openai:gpt-4o",
        type=str,
        help=(
            "Model spec (e.g., 'openai:gpt-4o', 'anthropic:claude-3-opus', 'gemini:gemini-1.5-pro', "
            "or 'heuristic:none' for an offline smoke baseline)."
        ),
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=1,
        help="Number of rollouts per episode.",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        type=str,
        help="Episode indices to run (e.g., '0-9,15,20-30'). Defaults to all.",
    )
    parser.add_argument(
        "--shard",
        default=None,
        type=str,
        help="Shard spec in 'i/N' format (e.g., '0/4').",
    )
    parser.add_argument(
        "--out",
        default="results.jsonl",
        type=str,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file (requires matching split_hash).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output JSONL instead of overwriting.",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Enable counterexample feedback on incorrect submissions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional; if omitted, provider default).",
    )
    parser.add_argument(
        "--effort",
        type=str,
        default=None,
        help=(
            "Reasoning/thinking effort level. "
            "OpenAI (reasoning.effort): none|minimal|low|medium|high|xhigh (model-dependent). "
            "Gemini 3 (thinkingLevel): minimal|low|medium|high (model-dependent). "
            "If set, dedeucerl-eval can perform a cheap 1-token probe call to validate support."
        ),
    )
    parser.add_argument(
        "--no-effort-probe",
        action="store_true",
        help="Skip the provider probe call for --effort (not recommended).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument(
        "--trace-out",
        default=None,
        type=str,
        help="Optional JSONL trace output (per-turn events) for debugging.",
    )

    return parser.parse_args()


async def run_episode(
    env,
    adapter,
    episode_idx: int,
    *,
    rollout: int = 0,
    temperature: Optional[float] = None,
    effort: Optional[str] = None,
    model_spec: Optional[str] = None,
    verbose: bool = False,
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single episode and return results."""

    # Get initial state
    item = env.dataset[episode_idx]
    state = {"prompt": item["prompt"], "answer": item["answer"]}
    state = await env.setup_state(state)

    # Ensure per-episode adapter state is clean (important for OpenAI Responses API).
    try:
        adapter.reset_conversation()
    except Exception:
        pass

    # verifiers expects a trajectory for max-turn stop conditions; we manage it here
    # since this runner bypasses the built-in stepping loop.
    state.setdefault("trajectory", [])

    messages = list(state["prompt"])
    tools = env._get_tools()

    # Build tool schemas from the skin DomainSpec (enums, structured args).
    answer_data = json.loads(state.get("answer", "{}"))
    spec = _domain_spec_from_answer(env.__class__, answer_data)
    tool_schemas = spec.get_tools()

    # Trace helper
    trace_ctx: Dict[str, Any] = dict(trace_base or {})
    trace_ctx.setdefault("episode_idx", episode_idx)
    trace_ctx.setdefault("rollout", rollout)
    trace_ctx.setdefault("seed", answer_data.get("seed", -1))

    def _trace(event: Dict[str, Any]) -> None:
        if trace_writer is None:
            return
        payload = dict(trace_ctx)
        payload.update(event)
        trace_writer(payload)

    # Metrics
    tool_calls_total = 0
    usage_prompt = 0
    usage_completion = 0
    usage_total = 0
    usage_seen = False

    _trace(
        {
            "event": "episode_start",
            "budget_init": state.get("budget_init", None),
            "max_turns": env.max_turns,
        }
    )

    # Evaluation loop
    turn = 0
    max_turns = env.max_turns

    while (
        not bool(state.get("done", False))
        and int(state.get("budget", 0)) > 0
        and len(state.get("trajectory", [])) < max_turns
    ):
        turn += 1

        try:
            # Get model response
            request_kwargs: Dict[str, Any] = {}
            if temperature is not None:
                request_kwargs["temperature"] = temperature

            # Provider-specific effort controls (validated in main).
            if effort is not None:
                if not model_spec:
                    raise RuntimeError("Internal error: model_spec missing for effort validation.")
                provider, model_id = decompose_model_spec(model_spec)
                if provider in ("openai", "openrouter"):
                    request_kwargs["reasoning"] = {"effort": effort}
                elif provider in ("gemini", "google"):
                    request_kwargs["thinking_level"] = effort
                else:
                    raise ValueError(f"--effort is not supported for provider '{provider}'.")

            print(f"  Turn {turn}: requesting model...")
            reply = adapter.chat(
                messages,
                tools=tool_schemas,
                **request_kwargs,
            )

            # Track usage + tool call counts
            if reply.usage:
                usage_seen = True
                usage_prompt += int(reply.usage.get("prompt_tokens") or 0)
                usage_completion += int(reply.usage.get("completion_tokens") or 0)
                usage_total += int(reply.usage.get("total_tokens") or 0)

            tool_calls_total += len(reply.tool_calls or [])

            _trace(
                {
                    "event": "model_reply",
                    "turn": turn,
                    "finish_reason": reply.finish_reason,
                    "usage": reply.usage,
                    "tool_calls": reply.tool_calls,
                    "content": reply.content,
                }
            )

            if reply.tool_calls:
                # Add one assistant message containing all tool calls (OpenAI-style)
                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": reply.tool_calls}
                )

                # Process tool calls
                for tc_idx, tc in enumerate(reply.tool_calls):
                    tool_call_id = tc.get("call_id") or tc.get("id", "") or f"call_{turn}_{tc_idx}"
                    func_name = (tc.get("function") or {}).get("name") or ""
                    raw_args = (tc.get("function") or {}).get("arguments")

                    try:
                        if isinstance(raw_args, str):
                            func_args = json.loads(raw_args)
                        elif isinstance(raw_args, dict):
                            func_args = raw_args
                        else:
                            func_args = {}
                    except Exception:
                        _consume_cli_budget(state, 1)
                        err = error_invalid_json(context=f"{func_name} arguments")
                        result = _tool_error_envelope(
                            state,
                            err,
                            extra={"ok": False, "tool": func_name, "raw_arguments": raw_args},
                        )
                        print(
                            f"  Turn {turn}: invalid tool args for {func_name} -> {_one_line(result)}"
                        )
                        _trace(
                            {
                                "event": "tool_result",
                                "turn": turn,
                                "tool": func_name,
                                "tool_call_id": tool_call_id,
                                "args": {"raw_arguments": raw_args},
                                "result": result,
                            }
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": tool_call_id, "content": result}
                        )
                        state["trajectory"].append(
                            {
                                "tool": func_name,
                                "args": {"raw_arguments": raw_args},
                                "result": result,
                            }
                        )
                        if bool(state.get("done", False)):
                            break
                        continue

                    if not isinstance(func_args, dict):
                        _consume_cli_budget(state, 1)
                        err = error_invalid_argument(
                            f"Tool arguments must be an object for '{func_name}'",
                            details={"tool": func_name, "raw_arguments": raw_args},
                        )
                        result = _tool_error_envelope(state, err, extra={"ok": False})
                        print(
                            f"  Turn {turn}: invalid tool args for {func_name} -> {_one_line(result)}"
                        )
                        _trace(
                            {
                                "event": "tool_result",
                                "turn": turn,
                                "tool": func_name,
                                "tool_call_id": tool_call_id,
                                "args": {"raw_arguments": raw_args},
                                "result": result,
                            }
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": tool_call_id, "content": result}
                        )
                        state["trajectory"].append(
                            {"tool": func_name, "args": func_args, "result": result}
                        )
                        if bool(state.get("done", False)):
                            break
                        continue

                    # Find the tool
                    tool_fn = None
                    for t in tools:
                        if t.__name__ == func_name:
                            tool_fn = t
                            break

                    if not tool_fn:
                        _consume_cli_budget(state, 1)
                        err = error_unknown_tool(func_name, [t.__name__ for t in tools])
                        result = _tool_error_envelope(state, err, extra={"ok": False})
                        print(f"  Turn {turn}: unknown tool {func_name} -> {_one_line(result)}")
                        _trace(
                            {
                                "event": "tool_result",
                                "turn": turn,
                                "tool": func_name,
                                "tool_call_id": tool_call_id,
                                "args": func_args,
                                "result": result,
                            }
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": tool_call_id, "content": result}
                        )
                        state["trajectory"].append(
                            {"tool": func_name, "args": func_args, "result": result}
                        )
                        if bool(state.get("done", False)):
                            break
                        continue

                    # Update tool args with state
                    env.update_tool_args(func_name, func_args, messages, state)
                    print(f"  Turn {turn}: {func_name}({_preview_args(func_args)})")

                    try:
                        result = tool_fn(**func_args)
                    except Exception as e:
                        # Treat tool crashes as invalid calls (and charge budget) so the model
                        # can recover instead of spinning or hard-crashing the runner.
                        _consume_cli_budget(state, 1)
                        err = error_invalid_argument(
                            f"Tool '{func_name}' raised exception",
                            details={"tool": func_name, "error": str(e), "args": func_args},
                        )
                        result = _tool_error_envelope(state, err, extra={"ok": False})

                    print(f"  Turn {turn}: {func_name} -> {_one_line(result)}")

                    if verbose:
                        # Helpful for debugging exact JSON payloads.
                        print(f"  Turn {turn}: raw result len={len(str(result))}")

                    _trace(
                        {
                            "event": "tool_result",
                            "turn": turn,
                            "tool": func_name,
                            "tool_call_id": tool_call_id,
                            "args": func_args,
                            "result": result,
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result,
                        }
                    )

                    # Track tool-call count for max_turns stopping.
                    state["trajectory"].append(
                        {
                            "tool": func_name,
                            "args": func_args,
                            "result": result,
                        }
                    )

                    if bool(state.get("done", False)):
                        break
            else:
                # No tool calls, possibly natural language or end
                print(
                    f"  Turn {turn}: No tool calls; finish={reply.finish_reason} content={_one_line(reply.content)}"
                )
                if reply.content:
                    messages.append({"role": "assistant", "content": reply.content})
                break

        except Exception as e:
            if verbose:
                print(f"  Error on turn {turn}: {e}")
            break

    # Compute score
    import inspect

    rubric = make_rubric()

    reward_funcs = getattr(rubric, "reward_funcs", None)
    if reward_funcs is None:
        reward_funcs = getattr(rubric, "funcs", [])

    parser = getattr(rubric, "parser", None)
    scores: List[float] = []
    for func in list(reward_funcs) or []:
        raw: Any = func(None, state.get("answer", ""), state, parser)
        if inspect.isawaitable(raw):
            raw = await raw
        if isinstance(raw, list):
            raw = raw[0] if raw else 0.0
        scores.append(float(raw))

    usage_prompt_tokens = usage_prompt if usage_seen else None
    usage_completion_tokens = usage_completion if usage_seen else None
    usage_total_tokens = usage_total if usage_seen else None
    tool_calls_processed = len(state.get("trajectory", []))

    _trace(
        {
            "event": "episode_end",
            "ok": state.get("ok", False),
            "trap_hit": state.get("trap_hit", False),
            "queries_used": state.get("queries_used", 0),
            "budget_remaining": state.get("budget", 0),
            "turns": turn,
            "tool_calls_total": tool_calls_total,
            "tool_calls_processed": tool_calls_processed,
            "usage_prompt_tokens": usage_prompt_tokens,
            "usage_completion_tokens": usage_completion_tokens,
            "usage_total_tokens": usage_total_tokens,
        }
    )

    return {
        "episode_idx": episode_idx,
        "seed": json.loads(state.get("answer", "{}")).get("seed", -1),
        "ok": state.get("ok", False),
        "trap_hit": state.get("trap_hit", False),
        "queries_used": state.get("queries_used", 0),
        "budget_remaining": state.get("budget", 0),
        "turns": turn,
        "reward": scores[0] if scores else 0.0,
        "tool_calls_total": tool_calls_total,
        "tool_calls_processed": tool_calls_processed,
        "usage_prompt_tokens": usage_prompt_tokens,
        "usage_completion_tokens": usage_completion_tokens,
        "usage_total_tokens": usage_total_tokens,
    }


async def main_async():
    """Async main entry point."""
    # Load local env vars if present (makes first-time benchmarking easier).
    _load_dotenv(Path.cwd() / ".env")

    args = parse_args()

    # Normalize --effort. We avoid hardcoding per-model capability tables here.
    # Instead, when --effort is provided we do:
    # - a small "typo guard" (value is in the known vocabulary for that provider)
    # - an optional cheap probe call to let the provider validate the setting
    effort: Optional[str] = None
    if args.effort is not None:
        effort = str(args.effort).strip().lower()
        if effort == "":
            effort = None

    if effort is not None:
        provider, model_id = decompose_model_spec(args.model)

        if provider in ("openai", "openrouter"):
            allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
            if effort not in allowed:
                print(
                    f"Error: invalid --effort {effort!r} for provider '{provider}'. "
                    f"Expected one of: {sorted(allowed)}",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif provider in ("gemini", "google"):
            allowed = {"minimal", "low", "medium", "high"}
            if effort not in allowed:
                print(
                    f"Error: invalid --effort {effort!r} for provider '{provider}'. "
                    f"Expected one of: {sorted(allowed)}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            print(f"Error: --effort is not supported for provider '{provider}'.", file=sys.stderr)
            sys.exit(1)

    # Load skin
    if args.skin not in SKIN_REGISTRY:
        print(f"Error: Unknown skin '{args.skin}'", file=sys.stderr)
        sys.exit(1)

    SkinClass = SKIN_REGISTRY[args.skin]

    # Build dataset
    generator = TaskGenerator(SkinClass)

    # Determine subset
    subset = args.subset
    if subset is None:
        # Try to auto-detect subset from split file
        with open(args.split, "r") as f:
            split_data = json.load(f)
        subsets = [k for k in split_data.keys() if k not in ("version", "metadata")]
        if len(subsets) == 1:
            subset = subsets[0]
        else:
            print(f"Error: Multiple subsets found: {subsets}. Use --subset.", file=sys.stderr)
            sys.exit(1)

    dataset = generator.build_dataset(args.split, subset, feedback=args.feedback)

    split_hash = compute_split_hash(args.split, subset, args.feedback)

    # Create environment
    rubric = make_rubric()

    first_answer = json.loads(dataset[0]["answer"])
    spec0 = _domain_spec_from_answer(SkinClass, first_answer)
    max_turns = generator.derive_max_turns(
        budget=int(first_answer.get("budget", 25)),
        n_states=int(getattr(spec0, "n_states", 5)),
        feedback=args.feedback,
    )

    env = SkinClass(
        dataset=dataset,
        rubric=rubric,
        feedback=args.feedback,
        max_turns=max_turns,
    )

    # Get adapter
    adapter_kwargs: Dict[str, Any] = {}
    if args.temperature is not None:
        adapter_kwargs["temperature"] = args.temperature

    adapter = get_adapter(
        args.model,
        **adapter_kwargs,
    )

    # When --effort is provided, validate it by doing a tiny probe call.
    # This avoids hardcoding per-model capability tables and prevents wasting
    # money on long eval runs with an invalid setting.
    if effort is not None and not bool(args.no_effort_probe):
        provider, model_id = decompose_model_spec(args.model)
        # Use a tiny output budget, but keep it above provider minimums.
        # OpenAI GPT-5* enforces a minimum max_output_tokens (currently 16).
        probe_kwargs: Dict[str, Any] = {"max_tokens": 16}
        if provider in ("openai", "openrouter"):
            probe_kwargs["reasoning"] = {"effort": effort}
        elif provider in ("gemini", "google"):
            probe_kwargs["thinking_level"] = effort

        try:
            adapter.chat(
                [{"role": "system", "content": "ping"}, {"role": "user", "content": "ping"}],
                tools=None,
                **probe_kwargs,
            )
        except Exception as e:
            print(
                f"Error: --effort {effort!r} was rejected for model '{model_id}' "
                f"(provider '{provider}').\n"
                f"Provider error: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Resolve episode indices + shard
    n_episodes = len(dataset)
    try:
        episode_indices = parse_index_spec(args.episodes, max_len=n_episodes)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    shard = None
    try:
        shard = parse_shard(args.shard)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if shard is not None:
        shard_index, shard_count = shard
        episode_indices = apply_shard(episode_indices, shard_index, shard_count)

    # Resume handling
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    existing_split_hashes = set()
    if args.resume and out_path.exists():
        with open(out_path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"Error: Malformed JSONL in {out_path} at line {line_no}: {e}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                existing_split_hashes.add(row.get("split_hash"))
                if (
                    row.get("split_hash") == split_hash
                    and row.get("model") == args.model
                    and row.get("skin", args.skin) == args.skin
                ):
                    try:
                        ep = int(row.get("episode_idx"))
                        ro = int(row.get("rollout", 0))
                    except (TypeError, ValueError):
                        continue
                    done.add((ep, ro))

        if None in existing_split_hashes:
            print(
                "Error: --resume requires existing results to include split_hash. "
                "Re-run without --resume or delete the output file.",
                file=sys.stderr,
            )
            sys.exit(1)

        if len(existing_split_hashes) > 1:
            print(
                "Error: Output file contains multiple split_hash values; cannot safely resume.",
                file=sys.stderr,
            )
            sys.exit(1)

        if existing_split_hashes and split_hash not in existing_split_hashes:
            print(
                "Error: split_hash mismatch; the split file or subset changed. "
                "Re-run without --resume or use a new output file.",
                file=sys.stderr,
            )
            sys.exit(1)

    mode = "a" if (args.append or (args.resume and out_path.exists())) else "w"

    trace_f = None
    trace_path = None
    if args.trace_out:
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_mode = "a" if (args.append or (args.resume and trace_path.exists())) else "w"
        trace_f = open(trace_path, trace_mode)

    def _write_trace(event: Dict[str, Any]) -> None:
        if trace_f is None:
            return
        trace_f.write(json.dumps(event) + "\n")
        trace_f.flush()

    total_selected = len(episode_indices)
    print(
        f"Running {total_selected} episodes with {args.rollouts} rollout(s) each "
        f"(total runs={total_selected * args.rollouts})..."
    )

    # Run evaluations (stream results)
    with open(out_path, mode) as f:
        for episode_idx in episode_indices:
            for rollout in range(args.rollouts):
                if (episode_idx, rollout) in done:
                    continue
                print(f"Episode {episode_idx + 1}/{n_episodes}, Rollout {rollout + 1}")

                result = await run_episode(
                    env,
                    adapter,
                    episode_idx,
                    rollout=rollout,
                    temperature=args.temperature,
                    effort=effort,
                    model_spec=args.model,
                    verbose=args.verbose,
                    trace_writer=_write_trace if trace_f is not None else None,
                    trace_base={
                        "model": args.model,
                        "skin": args.skin,
                        "split_hash": split_hash,
                    },
                )
                result["rollout"] = rollout
                result["model"] = args.model
                result["skin"] = args.skin
                result["split_hash"] = split_hash
                f.write(json.dumps(result) + "\n")
                f.flush()

    if trace_f is not None:
        trace_f.close()

    # Print summary (for this model + split)
    results = []
    with open(out_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping malformed JSONL in {out_path} at line {line_no}: {e}",
                    file=sys.stderr,
                )
                continue
            if (
                row.get("model") == args.model
                and row.get("split_hash") == split_hash
                and row.get("skin", args.skin) == args.skin
            ):
                results.append(row)

    n_success = sum(1 for r in results if r.get("ok"))
    n_trap = sum(1 for r in results if r.get("trap_hit"))
    avg_queries = sum(r.get("queries_used", 0) for r in results) / len(results) if results else 0
    avg_reward = sum(r.get("reward", 0) for r in results) / len(results) if results else 0

    print(f"\nResults written to {args.out}")
    if results:
        print(f"Success rate: {n_success}/{len(results)} ({100 * n_success / len(results):.1f}%)")
        print(f"Trap rate: {n_trap}/{len(results)} ({100 * n_trap / len(results):.1f}%)")
    else:
        print("Success rate: 0/0 (0.0%)")
        print("Trap rate: 0/0 (0.0%)")
    print(f"Avg queries: {avg_queries:.1f}")
    print(f"Avg reward: {avg_reward:.3f}")


def main():
    """Entry point for dedeucerl-eval CLI."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
