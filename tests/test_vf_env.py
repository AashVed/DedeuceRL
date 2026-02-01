"""Tests for the verifiers-compatible DedeuceRL env entrypoint."""

from __future__ import annotations

from pathlib import Path

import verifiers as vf


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "seeds" / "mealy_smoke.json"


def test_vf_env_load_with_seeds() -> None:
    env = vf.load_environment(
        "dedeucerl.vf_env",
        skin="mealy",
        seeds=[0, 1],
        budget=10,
        subset="train",
        feedback=False,
    )
    assert env.dataset is not None
    assert len(env.dataset) == 2


def test_vf_env_load_with_split_path() -> None:
    env = vf.load_environment(
        "dedeucerl.vf_env",
        skin="mealy",
        split_path=str(SPLIT_PATH),
        subset="dev",
        feedback=False,
    )
    assert env.dataset is not None
    assert len(env.dataset) > 0


def test_vf_env_load_with_import_path() -> None:
    env = vf.load_environment(
        "dedeucerl.vf_env",
        skin="dedeucerl.skins.mealy:MealyEnv",
        seeds="0-1",
        budget=10,
        subset="train",
        feedback=False,
    )
    assert env.dataset is not None
    assert len(env.dataset) == 2
