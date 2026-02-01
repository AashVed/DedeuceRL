"""CLI tests for dedeucerl-train config generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_writes_config(tmp_path: Path) -> None:
    out_path = tmp_path / "train.toml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dedeucerl.cli.train",
            "--skin",
            "mealy",
            "--seeds",
            "0-1",
            "--budget",
            "10",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"train failed\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    content = out_path.read_text()
    assert 'id = "dedeucerl.vf_env"' in content
    assert "seeds = [0, 1]" in content
    assert "budget = 10" in content
