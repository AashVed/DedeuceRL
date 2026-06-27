"""dedeucerl-eval: run model evaluations against kernel runtimes."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable

from dedeucerl.adapters import get_adapter
from dedeucerl.adapters.base import decompose_model_spec
from dedeucerl.core import make_rubric
from dedeucerl.kernel import KERNEL_REGISTRY
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface.dataset import build_dataset_from_split, instance_from_dict, load_split
from dedeucerl.surface.tools import compile_tool_schemas
from dedeucerl.utils import (
    apply_shard,
    compute_eval_config_hash,
    compute_split_hash,
    normalize_eval_config,
    parse_index_spec,
    parse_shard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dedeucerl-eval",
        description="Run DedeuceRL evaluations.",
    )
    parser.add_argument("--skin", "--kernel", dest="kernel", required=True, choices=KERNEL_REGISTRY)
    parser.add_argument("--split", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--model", default="openai:gpt-4o")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--shard", default=None)
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--trace-out", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--no-effort-probe", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


async def run_episode(
    *,
    item: dict[str, Any],
    adapter,
    episode_idx: int,
    rollout: int,
    feedback: bool,
    temperature: float | None,
    effort: str | None,
    model_spec: str,
    verbose: bool,
    trace_writer: Callable[[dict[str, Any]], None] | None = None,
    trace_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    instance = instance_from_dict(json.loads(item["answer"]))
    entry = KERNEL_REGISTRY[instance.kernel_name]
    runtime = EpisodeRuntime(entry.kernel, instance, feedback=feedback)
    messages = list(item["prompt"])
    max_turns = _derive_max_turns(instance.budget, feedback=feedback)
    usage_prompt = 0
    usage_completion = 0
    usage_total = 0
    usage_seen = False
    tool_calls_total = 0
    turn = 0

    trace_ctx = dict(trace_base or {})
    trace_ctx.setdefault("episode_idx", episode_idx)
    trace_ctx.setdefault("rollout", rollout)
    trace_ctx.setdefault("seed", instance.seed)

    def _trace(event: dict[str, Any]) -> None:
        if trace_writer is None:
            return
        payload = dict(trace_ctx)
        payload.update(event)
        trace_writer(payload)

    _trace({"event": "episode_start", "budget_init": runtime.budget_init, "max_turns": max_turns})

    try:
        adapter.reset_conversation()
    except Exception:
        pass

    while not runtime.done and runtime.budget > 0 and runtime.tool_calls < max_turns:
        turn += 1
        contracts = runtime.contracts()
        tool_schemas = compile_tool_schemas(contracts)
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if effort is not None:
            provider, _ = decompose_model_spec(model_spec)
            if provider in ("openai", "openrouter"):
                kwargs["reasoning"] = {"effort": effort}
            elif provider in ("gemini", "google"):
                kwargs["thinking_level"] = effort

        print(f"  Turn {turn}: requesting model...")
        try:
            reply = adapter.chat(messages, tools=tool_schemas, **kwargs)
        except Exception as e:
            if verbose:
                print(f"  Adapter error: {e}")
            break

        if reply.usage:
            usage_seen = True
            usage_prompt += int(reply.usage.get("prompt_tokens") or 0)
            usage_completion += int(reply.usage.get("completion_tokens") or 0)
            usage_total += int(reply.usage.get("total_tokens") or 0)

        tool_calls = reply.tool_calls or []
        tool_calls_total += len(tool_calls)
        _trace(
            {
                "event": "model_reply",
                "turn": turn,
                "finish_reason": reply.finish_reason,
                "usage": reply.usage,
                "tool_calls": tool_calls,
                "content": reply.content,
            }
        )

        if not tool_calls:
            if reply.content:
                messages.append({"role": "assistant", "content": reply.content})
            print(
                f"  Turn {turn}: No tool calls; finish={reply.finish_reason} "
                f"content={_one_line(reply.content)}"
            )
            break

        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for tc_idx, tc in enumerate(tool_calls):
            tool_call_id = tc.get("call_id") or tc.get("id", "") or f"call_{turn}_{tc_idx}"
            function = tc.get("function") or {}
            tool_name = str(function.get("name") or "")
            raw_args = function.get("arguments")
            args = _parse_tool_args(raw_args)
            print(f"  Turn {turn}: {tool_name}({_preview_args(args)})")
            event = runtime.call_tool(tool_name, args)
            result = json.dumps(event.output)
            print(f"  Turn {turn}: {tool_name} -> {_one_line(result)}")

            _trace(
                {
                    "event": "tool_result",
                    "turn": turn,
                    "tool": tool_name,
                    "tool_call_id": tool_call_id,
                    "args": args,
                    "result": result,
                    "runtime_event": event.to_dict(),
                }
            )
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
            if runtime.done:
                break

    state = runtime.state_dict()
    reward = await _score(state, item["answer"])
    usage_prompt_tokens = usage_prompt if usage_seen else None
    usage_completion_tokens = usage_completion if usage_seen else None
    usage_total_tokens = usage_total if usage_seen else None
    _trace(
        {
            "event": "episode_end",
            "ok": runtime.ok,
            "trap_hit": runtime.trap_hit,
            "queries_used": runtime.queries_used,
            "budget_remaining": runtime.budget,
            "turns": turn,
            "tool_calls_total": tool_calls_total,
            "tool_calls_processed": runtime.tool_calls,
            "usage_prompt_tokens": usage_prompt_tokens,
            "usage_completion_tokens": usage_completion_tokens,
            "usage_total_tokens": usage_total_tokens,
        }
    )
    return {
        "episode_idx": episode_idx,
        "seed": instance.seed,
        "ok": runtime.ok,
        "trap_hit": runtime.trap_hit,
        "queries_used": runtime.queries_used,
        "budget_remaining": runtime.budget,
        "turns": turn,
        "reward": reward,
        "tool_calls_total": tool_calls_total,
        "tool_calls_processed": runtime.tool_calls,
        "usage_prompt_tokens": usage_prompt_tokens,
        "usage_completion_tokens": usage_completion_tokens,
        "usage_total_tokens": usage_total_tokens,
    }


async def main_async() -> None:
    _load_dotenv(Path.cwd() / ".env")
    args = parse_args()
    effort = _normalize_effort(args.effort, args.model)
    split_data = load_split(args.split)
    subset = args.subset or _infer_subset(split_data)
    dataset = build_dataset_from_split(split_data, subset, feedback=args.feedback)

    split_hash = compute_split_hash(args.split, subset, args.feedback)
    eval_config = normalize_eval_config(temperature=args.temperature, effort=effort)
    eval_config_hash = compute_eval_config_hash(eval_config)

    adapter_kwargs: dict[str, Any] = {}
    if args.temperature is not None:
        adapter_kwargs["temperature"] = args.temperature
    adapter = get_adapter(args.model, **adapter_kwargs)
    if effort is not None and not args.no_effort_probe:
        _probe_effort(adapter, args.model, effort)

    indices = parse_index_spec(args.episodes, max_len=len(dataset))
    shard = parse_shard(args.shard)
    if shard is not None:
        indices = apply_shard(indices, shard[0], shard[1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(
        out_path,
        resume=args.resume,
        model=args.model,
        kernel=args.kernel,
        split_hash=split_hash,
        eval_config_hash=eval_config_hash,
    )
    mode = "a" if (args.append or (args.resume and out_path.exists())) else "w"

    trace_f = None
    if args.trace_out:
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_mode = "a" if (args.append or (args.resume and trace_path.exists())) else "w"
        trace_f = open(trace_path, trace_mode, encoding="utf-8")

    def write_trace(event: dict[str, Any]) -> None:
        if trace_f is None:
            return
        trace_f.write(json.dumps(event) + "\n")
        trace_f.flush()

    print(
        f"Running {len(indices)} episodes with {args.rollouts} rollout(s) each "
        f"(total runs={len(indices) * args.rollouts})..."
    )
    with open(out_path, mode, encoding="utf-8") as f:
        for episode_idx in indices:
            for rollout in range(args.rollouts):
                if (episode_idx, rollout) in done:
                    continue
                print(f"Episode {episode_idx + 1}/{len(dataset)}, Rollout {rollout + 1}")
                result = await run_episode(
                    item=dataset[episode_idx],
                    adapter=adapter,
                    episode_idx=episode_idx,
                    rollout=rollout,
                    feedback=args.feedback,
                    temperature=args.temperature,
                    effort=effort,
                    model_spec=args.model,
                    verbose=args.verbose,
                    trace_writer=write_trace if trace_f is not None else None,
                    trace_base={
                        "model": args.model,
                        "skin": args.kernel,
                        "split_hash": split_hash,
                        "eval_config_hash": eval_config_hash,
                    },
                )
                result.update(
                    {
                        "rollout": rollout,
                        "model": args.model,
                        "skin": args.kernel,
                        "split_hash": split_hash,
                        "eval_config": eval_config,
                        "eval_config_hash": eval_config_hash,
                    }
                )
                f.write(json.dumps(result) + "\n")
                f.flush()

    if trace_f is not None:
        trace_f.close()
    _print_summary(out_path, args.model, args.kernel, split_hash, eval_config_hash)


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except Exception:
            return {"raw_arguments": raw}
        return decoded if isinstance(decoded, dict) else {"raw_arguments": decoded}
    return {}


async def _score(state: dict[str, Any], answer: str) -> float:
    rubric = make_rubric()
    funcs = getattr(rubric, "reward_funcs", None) or getattr(rubric, "funcs", [])
    parser = getattr(rubric, "parser", None)
    if not funcs:
        return 0.0
    raw = funcs[0](None, answer, state, parser)
    if inspect.isawaitable(raw):
        raw = await raw
    if isinstance(raw, list):
        raw = raw[0] if raw else 0.0
    return float(raw)


def _derive_max_turns(budget: int, *, feedback: bool) -> int:
    return int(budget) + (10 if feedback else 2)


def _infer_subset(split_data: dict[str, Any]) -> str:
    subsets = [k for k in split_data.keys() if k not in ("version", "metadata")]
    if len(subsets) != 1:
        raise SystemExit(f"Error: Multiple subsets found: {subsets}. Use --subset.")
    return subsets[0]


def _load_done(
    path: Path,
    *,
    resume: bool,
    model: str,
    kernel: str,
    split_hash: str,
    eval_config_hash: str,
) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not resume or not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Error: Malformed JSONL in {path} at line {line_no}: {e}") from e
            if (
                row.get("model") == model
                and row.get("skin") == kernel
                and row.get("split_hash") == split_hash
                and row.get("eval_config_hash") == eval_config_hash
            ):
                done.add((int(row.get("episode_idx")), int(row.get("rollout", 0))))
    return done


def _normalize_effort(raw: str | None, model_spec: str) -> str | None:
    if raw is None:
        return None
    effort = raw.strip().lower()
    if not effort:
        return None
    provider, _ = decompose_model_spec(model_spec)
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    if provider in ("gemini", "google"):
        allowed = {"minimal", "low", "medium", "high"}
    if effort not in allowed:
        raise SystemExit(
            f"Error: invalid --effort {effort!r} for provider '{provider}'. "
            f"Expected one of: {sorted(allowed)}"
        )
    if provider not in ("openai", "openrouter", "gemini", "google"):
        raise SystemExit(f"Error: --effort is not supported for provider '{provider}'.")
    return effort


def _probe_effort(adapter, model_spec: str, effort: str) -> None:
    provider, model_id = decompose_model_spec(model_spec)
    kwargs: dict[str, Any] = {"max_tokens": 16}
    if provider in ("openai", "openrouter"):
        kwargs["reasoning"] = {"effort": effort}
    elif provider in ("gemini", "google"):
        kwargs["thinking_level"] = effort
    try:
        adapter.chat(
            [{"role": "system", "content": "ping"}, {"role": "user", "content": "ping"}],
            tools=None,
            **kwargs,
        )
    except Exception as e:
        raise SystemExit(
            f"Error: --effort {effort!r} was rejected for model '{model_id}'.\n"
            f"Provider error: {e}"
        ) from e


def _print_summary(
    out_path: Path, model: str, kernel: str, split_hash: str, eval_config_hash: str
) -> None:
    rows = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("model") == model
                and row.get("skin") == kernel
                and row.get("split_hash") == split_hash
                and row.get("eval_config_hash") == eval_config_hash
            ):
                rows.append(row)
    n_success = sum(1 for row in rows if row.get("ok"))
    n_trap = sum(1 for row in rows if row.get("trap_hit"))
    avg_queries = sum(row.get("queries_used", 0) for row in rows) / len(rows) if rows else 0
    avg_reward = sum(row.get("reward", 0) for row in rows) / len(rows) if rows else 0
    print(f"\nResults written to {out_path}")
    print(f"Success rate: {n_success}/{len(rows)} ({100 * n_success / len(rows):.1f}%)" if rows else "Success rate: 0/0 (0.0%)")
    print(f"Trap rate: {n_trap}/{len(rows)} ({100 * n_trap / len(rows):.1f}%)" if rows else "Trap rate: 0/0 (0.0%)")
    print(f"Avg queries: {avg_queries:.1f}")
    print(f"Avg reward: {avg_reward:.3f}")


def _load_dotenv(path: Path) -> None:
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
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _one_line(text: Any, *, limit: int = 220) -> str:
    value = "" if text is None else str(text)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + f"... (len={len(value)})"


def _preview_args(args: dict[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 200:
            parts.append(f"{key}=<str len={len(value)}>")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
