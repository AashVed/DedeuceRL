from __future__ import annotations

import json
import asyncio
import subprocess
import sys
from pathlib import Path

import verifiers as vf

from dedeucerl.kernel import KERNEL_REGISTRY
from dedeucerl.surface import (
    build_dataset_from_split,
    compile_prompt,
    compile_tool_schemas,
    generate_split,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )


def test_surface_compilers_roundtrip() -> None:
    entry = KERNEL_REGISTRY["mealy"]
    instance = entry.sampler.sample(seed=0, budget=5, n_states=2)
    contracts = entry.kernel.tool_contracts(instance, entry.kernel.initial_state(instance))

    prompt = compile_prompt(entry.kernel, instance, list(contracts), feedback=True)
    schemas = compile_tool_schemas(list(contracts))
    assert "OBSERVATION" in prompt[1]["content"]
    assert {schema["name"] for schema in schemas} == {"act", "submit_table"}

    split = generate_split(entry.sampler, seeds=[0, 1], budget=5, subset_name="dev", n_states=2)
    dataset = build_dataset_from_split(split, "dev", feedback=False)
    assert len(dataset) == 2
    assert json.loads(dataset[0]["answer"])["kernel_name"] == "mealy"


def test_vf_env_loads_with_seeds() -> None:
    env = vf.load_environment(
        "dedeucerl.vf_env",
        skin="mealy",
        seeds=[0, 1],
        budget=5,
        n_states=2,
        feedback=False,
    )
    assert env.dataset is not None
    assert len(env.dataset) == 2

    state = {"prompt": env.dataset[0]["prompt"], "answer": env.dataset[0]["answer"]}
    state = asyncio.run(env.setup_state(state))
    env.update_tool_args("act", {"symbol": "A"}, [], state)
    result = env.tools[0](symbol="A")
    assert json.loads(result)["queries_used"] == 1
    assert state["queries_used"] == 1


def test_generate_eval_parallel_and_selfcheck(tmp_path: Path) -> None:
    split = tmp_path / "mealy.json"
    out = tmp_path / "results.jsonl"
    par = tmp_path / "parallel.jsonl"
    trace = tmp_path / "trace.jsonl"

    gen = _run_module(
        "dedeucerl.cli.generate",
        ["--skin", "mealy", "--seeds", "0-2", "--budget", "5", "--n-states", "2", "-o", str(split)],
    )
    assert gen.returncode == 0, gen.stderr
    assert split.exists()

    eval_result = _run_module(
        "dedeucerl.cli.eval",
        [
            "--skin",
            "mealy",
            "--split",
            str(split),
            "--model",
            "heuristic:none",
            "--out",
            str(out),
            "--trace-out",
            str(trace),
        ],
    )
    assert eval_result.returncode == 0, eval_result.stderr
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert trace.exists() and trace.stat().st_size > 0

    parallel = _run_module(
        "dedeucerl.cli.eval_parallel",
        [
            "--jobs",
            "2",
            "--out",
            str(par),
            "--skin",
            "mealy",
            "--split",
            str(split),
            "--model",
            "heuristic:none",
        ],
    )
    assert parallel.returncode == 0, parallel.stderr
    par_rows = [json.loads(line) for line in par.read_text().splitlines() if line.strip()]
    assert {(row["episode_idx"], row["rollout"]) for row in rows} == {
        (row["episode_idx"], row["rollout"]) for row in par_rows
    }

    selfcheck = _run_module("dedeucerl.cli.selfcheck", [])
    assert selfcheck.returncode == 0, selfcheck.stderr
    assert "All checks passed" in selfcheck.stdout
