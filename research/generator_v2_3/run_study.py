"""Run a predefined installed-MCP comparison; retain every attempted model thread."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SNAPSHOT = """
import json, sys
from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.surface.dataset import instance_to_dict
ir=TASK_REGISTRY['mealy_palindrome'].ir
instance=ir.generator.sample(seed=int(sys.argv[1]),budget=200,n_states=12,
                             trap=False,min_length=24,min_distinct=3)
print(json.dumps(instance_to_dict(instance),indent=2))
"""
FINGERPRINT = """
import dedeucerl, hashlib, importlib.metadata, json
from pathlib import Path
root=Path(dedeucerl.__file__).parent
print(json.dumps({'versions':{n:importlib.metadata.version(n) for n in
 ['dedeucerl','mcp','verifiers','pydantic','jsonschema']},
 'sha256_by_file':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted(root.rglob('*.py'))}},indent=2))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.out_dir.resolve()
    root.mkdir(exist_ok=False)
    interpreters = {
        "cycle": args.baseline_python.absolute(),
        "uniform": args.candidate_python.absolute(),
    }
    launcher = Path(__file__).parents[1] / "custom_objectives_v1_1/run_models.py"
    jobs = []
    for seed in (2001, 2002):
        for model, distribution in (
            ("sol", "uniform"),
            ("luna", "cycle"),
            ("luna", "uniform"),
            ("sol", "cycle"),
        ):
            run_id = f"{model}-{distribution}-s{seed}"
            jobs.append(
                {
                    "run_id": run_id,
                    "model": f"gpt-5.6-{model}",
                    "seed": seed,
                    "distribution": distribution,
                }
            )
    manifest = {
        "task": "mealy_palindrome",
        "n_states": 12,
        "min_length": 24,
        "min_distinct": 3,
        "budget": 200,
        "feedback": True,
        "trap": False,
        "effort": "high",
        "host_timeout_seconds": 900,
        "workers": 4,
        "jobs": jobs,
        "scope": "Two seeds per model/distribution, one rollout per cell; diagnostic, not a pass-rate estimate.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for label, python in interpreters.items():
        fingerprint = subprocess.check_output([str(python), "-I", "-c", FINGERPRINT], text=True)
        (root / f"{label}_fingerprint.json").write_text(fingerprint)

    def run(job):
        python = str(interpreters[job["distribution"]])
        episode = root / "episodes" / job["run_id"]
        episode.mkdir(parents=True)
        # Controller-only snapshot for replay. The solving model has no file tool
        # and is explicitly prohibited from inspecting artifacts or hidden data.
        snapshot = subprocess.check_output(
            [python, "-I", "-c", SNAPSHOT, str(job["seed"])], text=True
        )
        (episode / "instance.json").write_text(snapshot)
        command = [
            sys.executable,
            str(launcher),
            "--out-dir",
            str(root),
            "--run-id",
            job["run_id"],
            "--server-python",
            python,
            "--model",
            job["model"],
            "--n-states",
            "12",
            "--min-length",
            "24",
            "--budget",
            "200",
            "--seed",
            str(job["seed"]),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        print(
            json.dumps(
                {
                    "run_id": job["run_id"],
                    "launcher_exit": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            ),
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, jobs))


if __name__ == "__main__":
    main()
