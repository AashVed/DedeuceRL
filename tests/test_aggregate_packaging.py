from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_json_output(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "model": "heuristic:none",
            "skin": "mealy",
            "split_hash": "split",
            "eval_config_hash": "cfg",
            "episode_idx": 0,
            "rollout": 0,
            "ok": True,
            "trap_hit": False,
            "queries_used": 3,
            "reward": 0.9,
        }
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = subprocess.run(
        [sys.executable, "-m", "dedeucerl.cli.aggregate", str(path), "--format", "json"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data[0]["pass_at_1"] == 1.0


def test_removed_skin_or_train_modules_are_not_in_source() -> None:
    assert not (REPO_ROOT / "dedeucerl" / "cli" / "train.py").exists()
    assert not any((REPO_ROOT / "dedeucerl" / "skins").glob("*.py"))
