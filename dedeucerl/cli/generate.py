"""dedeucerl-generate: generate TaskIR-backed task splits."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.surface.dataset import generate_split, save_split


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dedeucerl-generate",
        description="Generate DedeuceRL task splits.",
    )
    parser.add_argument("--skin", "--kernel", "--task", dest="kernel", required=True, choices=TASK_REGISTRY)
    parser.add_argument("--show-task-params", "--show-skin-params", dest="show_params", action="store_true")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--subset", default="dev")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--no-trap", action="store_true")
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--task-kwargs", "--skin-kwargs", dest="task_kwargs", type=str, default=None)
    parser.add_argument("--n-states", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    entry = TASK_REGISTRY[args.kernel]

    if args.show_params:
        print(f"Task '{entry.name}' parameters:")
        for name, param in sorted(entry.ir.generator.params.items()):
            print(f"- {name} (default={param.default}): {param.description}")
        sys.exit(0)

    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as e:
        print(f"Error parsing seeds: {e}", file=sys.stderr)
        sys.exit(1)

    params: dict[str, Any] = {"trap": not args.no_trap}
    if args.n_states is not None:
        params["n_states"] = int(args.n_states)

    if args.task_kwargs:
        try:
            extra = json.loads(args.task_kwargs)
            if not isinstance(extra, dict):
                raise ValueError("--task-kwargs must be a JSON object")
            params.update(extra)
        except Exception as e:
            print(f"Error parsing --task-kwargs: {e}", file=sys.stderr)
            sys.exit(1)

    for kv in args.param:
        if "=" not in kv:
            print(f"Error: --param must be KEY=VALUE, got: {kv}", file=sys.stderr)
            sys.exit(1)
        key, value = kv.split("=", 1)
        params[key.strip()] = _parse_value(value)

    print(f"Generating {len(seeds)} episodes for kernel '{entry.name}'...")
    print(f"  Budget: {args.budget}")
    print(f"  Params: {params}")

    split = generate_split(
        entry.ir,
        seeds=seeds,
        budget=int(args.budget),
        subset_name=args.subset,
        **params,
    )
    out_path = args.out or f"dataset/{entry.name}_{args.subset}.json"
    save_split(split, out_path)

    print(f"\nSaved to: {out_path}")
    print(f"Episodes: {len(split[args.subset]['items'])}")
    if split[args.subset]["items"]:
        print("\nSample item:")
        print(json.dumps(split[args.subset]["items"][0], indent=2)[:500] + "...")


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return json.loads(value)
    except Exception:
        return value


if __name__ == "__main__":
    main()
