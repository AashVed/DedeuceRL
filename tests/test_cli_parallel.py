"""CLI tests for episode selection, sharding, resume, and parallel eval."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "seeds" / "mealy_smoke.json"


def _run_module(module: str, args: List[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"{module} failed\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


def _load_results(path: Path) -> List[Dict]:
    lines = path.read_text().splitlines() if path.exists() else []
    return [json.loads(line) for line in lines if line.strip()]


def _get_subset_and_count() -> Tuple[str, int]:
    split = json.loads(SPLIT_PATH.read_text())
    subsets = [k for k in split.keys() if k not in ("version", "metadata")]
    assert len(subsets) == 1
    subset = subsets[0]
    return subset, len(split[subset]["items"])


def _selection_spec(n_items: int) -> Tuple[str, List[int]]:
    if n_items <= 0:
        raise AssertionError("Split must contain at least one episode.")
    if n_items == 1:
        return "0", [0]
    if n_items == 2:
        return "0-1", [0, 1]
    return f"0-1,{n_items - 1}", [0, 1, n_items - 1]


def _index_results(rows: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    return {(r["episode_idx"], r.get("rollout", 0)): r for r in rows}


def test_episode_selection_and_resume(tmp_path: Path) -> None:
    _, n_items = _get_subset_and_count()
    spec, expected = _selection_spec(n_items)

    out_path = tmp_path / "selection.jsonl"
    args = [
        "--skin",
        "mealy",
        "--split",
        str(SPLIT_PATH),
        "--model",
        "heuristic:none",
        "--rollouts",
        "1",
        "--episodes",
        spec,
        "--out",
        str(out_path),
    ]
    _run_module("dedeucerl.cli.eval", args)

    rows = _load_results(out_path)
    assert {r["episode_idx"] for r in rows} == set(expected)
    split_hashes = {r.get("split_hash") for r in rows}
    assert len(split_hashes) == 1 and None not in split_hashes

    # Resume should not append duplicates.
    _run_module("dedeucerl.cli.eval", args + ["--resume"])
    rows_after = _load_results(out_path)
    assert len(rows_after) == len(rows)


def test_sharding_partitions_full_set(tmp_path: Path) -> None:
    _, n_items = _get_subset_and_count()

    out0 = tmp_path / "shard0.jsonl"
    out1 = tmp_path / "shard1.jsonl"

    base_args = [
        "--skin",
        "mealy",
        "--split",
        str(SPLIT_PATH),
        "--model",
        "heuristic:none",
        "--rollouts",
        "1",
    ]

    _run_module("dedeucerl.cli.eval", base_args + ["--shard", "0/2", "--out", str(out0)])
    _run_module("dedeucerl.cli.eval", base_args + ["--shard", "1/2", "--out", str(out1)])

    rows0 = _load_results(out0)
    rows1 = _load_results(out1)

    idx0 = {r["episode_idx"] for r in rows0}
    idx1 = {r["episode_idx"] for r in rows1}
    assert idx0.isdisjoint(idx1)
    assert idx0.union(idx1) == set(range(n_items))


def test_parallel_matches_sequential(tmp_path: Path) -> None:
    seq_out = tmp_path / "sequential.jsonl"
    par_out = tmp_path / "parallel.jsonl"

    base_args = [
        "--skin",
        "mealy",
        "--split",
        str(SPLIT_PATH),
        "--model",
        "heuristic:none",
        "--rollouts",
        "1",
    ]

    _run_module("dedeucerl.cli.eval", base_args + ["--out", str(seq_out)])
    _run_module(
        "dedeucerl.cli.eval_parallel",
        ["--jobs", "2", "--out", str(par_out), *base_args],
    )

    seq_rows = _load_results(seq_out)
    par_rows = _load_results(par_out)

    seq_map = _index_results(seq_rows)
    par_map = _index_results(par_rows)

    assert seq_map.keys() == par_map.keys()
    for k in seq_map.keys():
        assert seq_map[k] == par_map[k]
