"""Launch one fresh Codex benchmark thread configured for MCP-only interaction.

Requires a ChatGPT-authenticated Codex CLI and a Python environment with the
DedeuceRL release installed. Artifacts stay outside the model's empty workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

PROMPT = """Solve the DedeuceRL benchmark episode exposed by the dedeucerl_benchmark MCP server. Use the server's MCP instructions and tools as your only source of task information and your only way to interact with the hidden system. Do not inspect repository source code, generated run artifacts, process state, MCP configuration, or hidden task data. Do not use shell, file, web, computer, or other tools to bypass the benchmark, and do not delegate. Continue using the benchmark tools until the episode reaches a terminal state. Then report whether it was solved, the score/reward, calls and budget used, termination reason, and MCP usability or benchmark-design issues you observed. Do not ask the user for help or reveal a plan without executing it."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--service-tier", choices=("default", "fast"), default="default")
    parser.add_argument("--task", default="mealy_palindrome")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--n-states", type=int, default=8)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--min-length", type=int, default=12)
    parser.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = root / args.run_id
    run.mkdir(exist_ok=False)
    # Codex exposes cwd in the model context. Keep condition/seed labels out of
    # it, even though the agent has no filesystem tools.
    workspace = Path(tempfile.mkdtemp(prefix="dedeucerl-episode-"))
    server = [
        "-m",
        "dedeucerl.cli.mcp",
        "serve",
        "--task",
        args.task,
        "--seed",
        str(args.seed),
        "--n-states",
        str(args.n_states),
        "--budget",
        str(args.budget),
        "--runs-dir",
        str(root / "episodes"),
        "--run-id",
        args.run_id,
    ]
    if args.task == "mealy_palindrome":
        server += ["--param", f"min_length={args.min_length}", "--param", "min_distinct=3"]
    if args.feedback:
        server += ["--feedback"]
    if not args.trap:
        server += ["--no-trap"]
    cmd = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--strict-config",
        "--sandbox",
        "read-only",
        "--model",
        args.model,
        "--json",
        "--cd",
        str(workspace),
        "--output-last-message",
        str(run / "final.txt"),
    ]
    config = {
        "model_reasoning_effort": "high",
        "approval_policy": "never",
        "web_search": "disabled",
        "project_doc_max_bytes": 0,
        "suppress_unstable_features_warning": True,
        "features.shell_tool": False,
        "features.view_image": False,
        "features.apps": False,
        "features.plugins": False,
        "features.hooks": False,
        "features.multi_agent": False,
        "features.browser_use": False,
        "features.computer_use": False,
        "features.image_generation": False,
        "features.skill_search": False,
        "features.memories": False,
        "mcp_servers.dedeucerl_benchmark.command": str(args.server_python.absolute()),
        "mcp_servers.dedeucerl_benchmark.args": server,
        "mcp_servers.dedeucerl_benchmark.startup_timeout_sec": 60,
        "mcp_servers.dedeucerl_benchmark.tool_timeout_sec": 60,
        "mcp_servers.dedeucerl_benchmark.required": True,
    }
    if args.service_tier == "fast":
        config.update({"service_tier": "fast", "features.fast_mode": True})
    for key, value in config.items():
        cmd += ["-c", f"{key}={json.dumps(value)}"]
    cmd += [PROMPT]
    metadata = {
        **vars(args),
        "out_dir": str(root),
        "server_python": str(args.server_python.absolute()),
        "effort": "high",
        "workspace": str(workspace),
        "neutral_workspace": True,
        "codex_version": subprocess.check_output(["codex", "--version"], text=True).strip(),
        "command": cmd,
        "started_at": time.time(),
    }
    (run / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (run / "events.jsonl").open("w") as out, (run / "stderr.log").open("w") as err:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True
        )
        print(json.dumps({"run_id": args.run_id, "pid": proc.pid, "model": args.model}), flush=True)
        try:
            status = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                status = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                status = proc.wait()
            metadata["infrastructure_timeout"] = True
    metadata.update(exit_code=status, finished_at=time.time())
    (run / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = root / "episodes" / args.run_id / "result.json"
    if result.is_file():
        result = json.loads(result.read_text())
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in [
                        "task",
                        "ok",
                        "score",
                        "tool_calls",
                        "evaluation_steps",
                        "queries_used",
                        "budget_remaining",
                        "termination_reason",
                    ]
                }
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "exit_code": status,
                    "result": "missing",
                    "stderr": (run / "stderr.log").read_text()[-3000:],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
