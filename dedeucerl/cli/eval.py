"""dedeucerl-eval: CLI for running DedeuceRL evaluations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dedeucerl.skins import SKIN_REGISTRY
from dedeucerl.core import TaskGenerator, make_rubric
from dedeucerl.adapters import get_adapter
from dedeucerl.utils import (
    DedeuceError,
    error_invalid_argument,
    error_invalid_json,
    error_unknown_tool,
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
        "--out",
        default="results.jsonl",
        type=str,
        help="Output JSONL file path.",
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
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )

    return parser.parse_args()


async def run_episode(
    env,
    adapter,
    episode_idx: int,
    *,
    temperature: Optional[float] = None,
    verbose: bool = False,
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

            print(f"  Turn {turn}: requesting model...")
            reply = adapter.chat(
                messages,
                tools=tool_schemas,
                **request_kwargs,
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

    return {
        "episode_idx": episode_idx,
        "seed": json.loads(state.get("answer", "{}")).get("seed", -1),
        "ok": state.get("ok", False),
        "trap_hit": state.get("trap_hit", False),
        "queries_used": state.get("queries_used", 0),
        "budget_remaining": state.get("budget", 0),
        "turns": turn,
        "reward": scores[0] if scores else 0.0,
    }


async def main_async():
    """Async main entry point."""
    # Load local env vars if present (makes first-time benchmarking easier).
    _load_dotenv(Path.cwd() / ".env")

    args = parse_args()

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

    # Run evaluations
    results = []
    n_episodes = len(dataset)

    print(f"Running {n_episodes} episodes with {args.rollouts} rollout(s) each...")

    for episode_idx in range(n_episodes):
        for rollout in range(args.rollouts):
            print(f"Episode {episode_idx + 1}/{n_episodes}, Rollout {rollout + 1}")

            result = await run_episode(
                env,
                adapter,
                episode_idx,
                temperature=args.temperature,
                verbose=args.verbose,
            )
            result["rollout"] = rollout
            result["model"] = args.model
            results.append(result)

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    n_success = sum(1 for r in results if r["ok"])
    n_trap = sum(1 for r in results if r["trap_hit"])
    avg_queries = sum(r["queries_used"] for r in results) / len(results) if results else 0
    avg_reward = sum(r["reward"] for r in results) / len(results) if results else 0

    print(f"\nResults written to {args.out}")
    print(f"Success rate: {n_success}/{len(results)} ({100 * n_success / len(results):.1f}%)")
    print(f"Trap rate: {n_trap}/{len(results)} ({100 * n_trap / len(results):.1f}%)")
    print(f"Avg queries: {avg_queries:.1f}")
    print(f"Avg reward: {avg_reward:.3f}")


def main():
    """Entry point for dedeucerl-eval CLI."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
