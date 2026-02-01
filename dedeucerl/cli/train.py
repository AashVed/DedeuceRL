"""dedeucerl-train: generate (and optionally run) vf-rl training configs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_seeds(spec: str) -> List[int]:
    spec = spec.strip()
    if not spec:
        return []
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            if start_raw == "" or end_raw == "":
                raise ValueError(f"Invalid seed range: '{part}'")
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid seed range: '{part}' (end < start)")
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return out


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return f'"{_toml_escape(v)}"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        items = []
        for k, val in v.items():
            items.append(f"{k} = {_toml_value(val)}")
        return "{ " + ", ".join(items) + " }"
    raise TypeError(f"Unsupported TOML value: {type(v).__name__}")


def _emit_table(lines: List[str], name: str, table: Dict[str, Any]) -> None:
    lines.append(f"[{name}]")
    for key, value in table.items():
        lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")


def build_config(args: argparse.Namespace) -> str:
    env_args: Dict[str, Any] = {}

    if args.skins:
        env_args["skins"] = args.skins
    else:
        env_args["skin"] = args.skin

    if args.split_path:
        env_args["split_path"] = args.split_path
    else:
        if not args.seeds:
            raise ValueError("Provide --split-path or --seeds.")
        env_args["seeds"] = _parse_seeds(args.seeds)
        if args.budget is not None:
            env_args["budget"] = args.budget

    if args.subset:
        env_args["subset"] = args.subset

    if args.feedback:
        env_args["feedback"] = True

    if args.reward_mode:
        env_args["reward_mode"] = args.reward_mode

    if args.skin_args:
        try:
            skin_args = json.loads(args.skin_args)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --skin-args JSON: {e}") from e
        if not isinstance(skin_args, dict):
            raise ValueError("--skin-args must be a JSON object.")
        env_args["skin_args"] = skin_args

    lines: List[str] = []
    lines.append(f"model = {_toml_value(args.model)}")
    lines.append("")

    _emit_table(lines, "env", {"id": "dedeucerl.vf_env"})
    _emit_table(lines, "env.args", env_args)

    _emit_table(lines, "inference", {"gpus": args.inference_gpus})
    _emit_table(lines, "trainer", {"gpus": args.trainer_gpus})

    trainer_args: Dict[str, Any] = {
        "run_name": args.run_name,
        "micro_batch_size": args.micro_batch_size,
        "rollouts_per_example": args.rollouts_per_example,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
    }
    _emit_table(lines, "trainer.args", trainer_args)

    return "\n".join(lines).strip() + "\n"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dedeucerl-train",
        description="Generate (and optionally run) a vf-rl config for DedeuceRL.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--skin", default="mealy")
    parser.add_argument(
        "--skins",
        type=str,
        default=None,
        help="Comma-separated list of skins for multi-skin training.",
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--subset", default="train")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--skin-args", default=None, help="JSON dict for skin kwargs.")
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--reward-mode", default="train_dense")
    parser.add_argument("--inference-gpus", type=int, default=1)
    parser.add_argument("--trainer-gpus", type=int, default=1)
    parser.add_argument("--run-name", default="dedeucerl-train")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--rollouts-per-example", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out", default="dedeucerl-train.toml")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run vf-rl after writing the config.",
    )
    parser.add_argument("--session", default=None, help="tmux session name for vf-rl.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.skins:
        args.skins = [s.strip() for s in args.skins.split(",") if s.strip()]

    if args.split_path and args.seeds:
        print("Error: Use --split-path or --seeds, not both.", file=sys.stderr)
        sys.exit(1)

    if args.split_path and (args.skin_args or args.budget):
        print("Error: --skin-args and --budget cannot be used with --split-path.", file=sys.stderr)
        sys.exit(1)

    try:
        config_text = build_config(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(config_text, encoding="utf-8")
    print(f"Wrote config to {out_path}")

    if not args.run:
        print("Run: vf-rl @", out_path)
        return

    uv = shutil.which("uv")
    cmd = ["vf-rl", "@", str(out_path)]
    if uv:
        cmd = ["uv", "run", "vf-rl", "@", str(out_path)]
    if args.session:
        cmd.extend(["-s", args.session])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
