from __future__ import annotations

import json
from pathlib import Path

import pytest

from dedeucerl.cli.eval import _load_done


def _write_result(path: Path, **overrides) -> None:
    row = {
        "model": "heuristic:none",
        "skin": "mealy",
        "split_hash": "split-a",
        "eval_config_hash": "config-a",
        "episode_idx": 3,
        "rollout": 1,
        **overrides,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _resume(path: Path) -> set[tuple[int, int]]:
    return _load_done(
        path,
        resume=True,
        model="heuristic:none",
        kernel="mealy",
        split_hash="split-a",
        eval_config_hash="config-a",
    )


def test_resume_loads_matching_results(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_result(path)

    assert _resume(path) == {(3, 1)}


def test_resume_rejects_multiple_split_hashes(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_result(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "model": "another-model",
                    "skin": "mealy",
                    "split_hash": "split-b",
                    "eval_config_hash": "config-b",
                    "episode_idx": 0,
                    "rollout": 0,
                }
            )
            + "\n"
        )

    with pytest.raises(SystemExit, match="multiple split_hash values"):
        _resume(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"split_hash": None}, "missing split_hash provenance"),
        ({"split_hash": "split-b"}, "split_hash mismatch"),
        ({"eval_config_hash": None}, "missing eval_config_hash provenance"),
        ({"eval_config_hash": "config-b"}, "eval_config_hash mismatch"),
    ],
)
def test_resume_rejects_incompatible_provenance(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    path = tmp_path / "results.jsonl"
    _write_result(path, **overrides)

    with pytest.raises(SystemExit, match=message):
        _resume(path)
