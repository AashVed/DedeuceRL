"""dedeucerl-eval-parallel: Run shard-parallel evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from dedeucerl.utils import apply_shard


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


def _load_jsonl_rows(path: Path, *, kind: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed {kind} JSONL in {path} at line {line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Malformed {kind} JSONL in {path} at line {line_no}: row is not an object")
            rows.append(row)
    return rows


def _row_matches_shard(row: dict, *, shard_index: int, shard_count: int) -> bool:
    try:
        episode_idx = int(row["episode_idx"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Resume row is missing a valid episode_idx: {row!r}") from e
    return bool(apply_shard([episode_idx], shard_index, shard_count))


def _rebuild_resume_parts(
    merged_path: Path,
    part_paths: List[Path],
    *,
    kind: str,
) -> None:
    rows = _load_jsonl_rows(merged_path, kind=kind)
    shard_count = len(part_paths)
    for shard_index, part_path in enumerate(part_paths):
        with open(part_path, "w") as part_f:
            for row in rows:
                if _row_matches_shard(row, shard_index=shard_index, shard_count=shard_count):
                    part_f.write(json.dumps(row) + "\n")


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
        "--trace-out",
        default=None,
        type=str,
        help=(
            "Optional merged JSONL trace output (per-turn events). "
            "When set, each shard writes to a part trace file which is merged on success."
        ),
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
    args.resume = "--resume" in eval_args
    return args


def _validate_eval_args(eval_args: List[str]) -> None:
    # These flags are managed by this wrapper.
    forbidden = {"--out", "--shard", "--trace-out"}
    for token in eval_args:
        if token in forbidden:
            raise ValueError(f"Do not pass {token} to eval-parallel; it is managed here.")
        if token.startswith("--out=") or token.startswith("--shard=") or token.startswith(
            "--trace-out="
        ):
            raise ValueError(
                "Do not pass --out/--shard/--trace-out to eval-parallel; it is managed here."
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

    trace_out_path: Path | None = None
    if args.trace_out:
        trace_out_path = Path(args.trace_out)
        trace_out_path.parent.mkdir(parents=True, exist_ok=True)

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = []
    if cuda:
        devices = [d.strip() for d in cuda.split(",") if d.strip() != ""]
        if devices and jobs > len(devices):
            jobs = len(devices)

    procs = []
    part_paths = []
    trace_part_paths: List[Path] = []
    proc_specs: List[tuple[List[str], dict]] = []

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

        if trace_out_path is not None:
            trace_part_path = _part_path(trace_out_path, i)
            trace_part_paths.append(trace_part_path)
            cmd.extend(["--trace-out", str(trace_part_path)])

        env = os.environ.copy()
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[i]

        if args.dry_run:
            print(" ".join(cmd))
            continue

        proc_specs.append((cmd, env))

    if args.dry_run:
        return

    if args.resume and out_path.exists():
        try:
            _rebuild_resume_parts(out_path, part_paths, kind="results")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if trace_out_path is not None and trace_out_path.exists():
            try:
                _rebuild_resume_parts(trace_out_path, trace_part_paths, kind="trace")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

    for cmd, env in proc_specs:
        procs.append(subprocess.Popen(cmd, env=env))

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

    if trace_out_path is not None:
        with open(trace_out_path, "w") as trace_f:
            for trace_part in trace_part_paths:
                if not trace_part.exists():
                    continue
                with open(trace_part, "r") as pf:
                    for line in pf:
                        trace_f.write(line)

    if not args.keep_parts:
        for part_path in part_paths:
            if part_path.exists():
                part_path.unlink()
        for trace_part in trace_part_paths:
            if trace_part.exists():
                trace_part.unlink()

    print(f"Results written to {out_path}")
    if trace_out_path is not None:
        print(f"Trace written to {trace_out_path}")


if __name__ == "__main__":
    main()
