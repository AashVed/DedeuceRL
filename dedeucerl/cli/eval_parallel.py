"""dedeucerl-eval-parallel: Run shard-parallel evaluations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _default_jobs() -> int:
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda:
        devices = [d.strip() for d in cuda.split(",") if d.strip() != ""]
        if devices:
            return len(devices)
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 8))


def _part_path(out_path: Path, idx: int) -> Path:
    suffix = out_path.suffix
    stem = out_path.stem if suffix else out_path.name
    return out_path.with_name(f"{stem}.part{idx}{suffix}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dedeucerl-eval-parallel",
        description="Run DedeuceRL evals in parallel by sharding episodes.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto).",
    )
    parser.add_argument(
        "--out",
        default="results.jsonl",
        type=str,
        help="Output JSONL file path (merged).",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep per-shard part files after merging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print subprocess commands without running.",
    )
    args, eval_args = parser.parse_known_args(argv)
    args.eval_args = eval_args
    return args


def _validate_eval_args(eval_args: List[str]) -> None:
    forbidden = {"--out", "--shard"}
    for token in eval_args:
        if token in forbidden:
            raise ValueError(
                f"Do not pass {token} to eval-parallel; it is managed here."
            )
        if token.startswith("--out=") or token.startswith("--shard="):
            raise ValueError(
                "Do not pass --out/--shard to eval-parallel; it is managed here."
            )


def main():
    args = parse_args()
    eval_args = list(args.eval_args)
    if eval_args and eval_args[0] == "--":
        eval_args = eval_args[1:]

    _validate_eval_args(eval_args)

    jobs = args.jobs if args.jobs is not None else _default_jobs()
    if jobs <= 0:
        print("Error: --jobs must be > 0", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = []
    if cuda:
        devices = [d.strip() for d in cuda.split(",") if d.strip() != ""]
        if devices and jobs > len(devices):
            jobs = len(devices)

    procs = []
    part_paths = []

    for i in range(jobs):
        part_path = _part_path(out_path, i)
        part_paths.append(part_path)

        cmd = [
            sys.executable,
            "-m",
            "dedeucerl.cli.eval",
            "--shard",
            f"{i}/{jobs}",
            "--out",
            str(part_path),
            *eval_args,
        ]

        env = os.environ.copy()
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[i]

        if args.dry_run:
            print(" ".join(cmd))
            continue

        procs.append(subprocess.Popen(cmd, env=env))

    if args.dry_run:
        return

    failed = False
    for p in procs:
        rc = p.wait()
        if rc != 0:
            failed = True

    if failed:
        print("Error: One or more shards failed.", file=sys.stderr)
        sys.exit(1)

    # Merge parts
    with open(out_path, "w") as out_f:
        for part_path in part_paths:
            if not part_path.exists():
                continue
            with open(part_path, "r") as pf:
                for line in pf:
                    out_f.write(line)

    if not args.keep_parts:
        for part_path in part_paths:
            if part_path.exists():
                part_path.unlink()

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
