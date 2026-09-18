"""CLI for the one-episode DedeuceRL MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.surface.mcp import MCPEpisodeServer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dedeucerl-mcp",
        description="Serve one self-contained DedeuceRL episode over MCP STDIO.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="serve one benchmark episode")
    serve.add_argument("--task", "--skin", "--kernel", dest="task", default="mealy", choices=TASK_REGISTRY)
    serve.add_argument("--seed", type=int, default=0)
    serve.add_argument("--budget", type=int, default=25)
    serve.add_argument("--feedback", action="store_true")
    serve.add_argument("--no-trap", action="store_true")
    serve.add_argument("--n-states", type=int, default=None)
    serve.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    serve.add_argument("--runs-dir", type=Path, default=Path(".dedeucerl/runs"))
    serve.add_argument("--run-id", default=None)
    serve.add_argument("--out", type=Path, default=None, help="override result.json path")
    serve.add_argument("--trace-out", type=Path, default=None, help="override trace.jsonl path")
    serve.add_argument("--no-persist", action="store_true")
    return parser.parse_args(argv)


def build_episode(args: argparse.Namespace) -> MCPEpisodeServer:
    entry = TASK_REGISTRY[args.task]
    params = {
        name: parameter.default
        for name, parameter in entry.ir.generator.params.items()
        if parameter.default is not None
    }
    if "trap" in params and args.no_trap:
        params["trap"] = False
    if args.n_states is not None:
        if "n_states" not in entry.ir.generator.params:
            raise ValueError(f"task '{args.task}' does not define an n_states parameter")
        params["n_states"] = args.n_states
    for raw in args.param:
        key, separator, value = raw.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--param must be KEY=VALUE, got: {raw}")
        key = key.strip()
        if key not in entry.ir.generator.params:
            available = ", ".join(sorted(entry.ir.generator.params)) or "<none>"
            raise ValueError(f"unknown parameter '{key}' for task '{args.task}'; available: {available}")
        params[key] = _parse_value(value)

    instance = entry.ir.generator.sample(seed=args.seed, budget=args.budget, **params)
    return MCPEpisodeServer(
        entry.ir,
        instance,
        feedback=args.feedback,
        persist=not args.no_persist,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        result_path=args.out,
        trace_path=args.trace_out,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        episode = build_episode(args)
        asyncio.run(episode.run_stdio())
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"dedeucerl-mcp: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    finally:
        if "episode" in locals() and episode.final_result is not None:
            path = episode.final_result["result_path"]
            if path is not None:
                print(f"DedeuceRL result: {path}", file=sys.stderr)


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    main()
