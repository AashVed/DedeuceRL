"""Audit completed model episodes against a fresh DedeuceRL runtime.

The controller runs this after episodes finish. Solving agents never receive
these checks or any access to the controller's artifacts or hidden instances.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from dedeucerl.core.rubric import score_identification
from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface.dataset import instance_from_dict
from dedeucerl.utils.errors import ErrorCode


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_run(root: Path, run: Path, sessions: Path) -> dict:
    launch = json.loads((run / "launch.json").read_text())
    # Session metadata exposes model/effort, but not the server's processing
    # tier. Verify the requested Fast configuration without claiming delivery.
    command = launch["command"]
    config = {
        key: json.loads(value)
        for index, arg in enumerate(command[:-1])
        if arg == "-c"
        for key, value in [command[index + 1].split("=", 1)]
    }
    service_tier = launch.get("service_tier", "default")
    if service_tier == "fast":
        assert config.get("service_tier") == "fast"
        assert config.get("features.fast_mode") is True
    host_timed_out = bool(launch.get("infrastructure_timeout"))
    assert "finished_at" in launch, f"running episode: {run.name}"
    events = read_jsonl(run / "events.jsonl")
    if not host_timed_out and launch.get("exit_code") != 0:
        # Keep pre-episode setup failures in the study record, without assigning
        # them a benchmark score or treating them as model failures.
        assert not (root / "episodes" / run.name / "trace.jsonl").exists()
        assert not any(event.get("item", {}).get("type") == "mcp_tool_call" for event in events)
        return {
            "run_id": run.name,
            "model": launch["model"],
            "effort": "high",
            "benchmark_started": False,
            "outcome": "launch_failed",
            "exit_code": launch["exit_code"],
        }
    thread_id = next(event["thread_id"] for event in events if event["type"] == "thread.started")
    result = json.loads((root / "episodes" / run.name / "result.json").read_text())
    # A host timeout during the final answer does not censor a terminal episode.
    censored = host_timed_out and not result["done"]
    trace = read_jsonl(root / "episodes" / run.name / "trace.jsonl")
    assert trace[0]["event"] == "episode_start" and trace[-1]["event"] == "episode_end"
    if censored:
        assert result["termination_reason"] == "disconnected" and not result["done"]
    else:
        assert result["termination_reason"] in {
            "solved",
            "solved_with_trap",
            "objective_failed",
            "budget_exhausted",
            "trap",
        }
    calls = [event for event in trace if event["event"] == "tool_result"]
    mcp_calls = [
        event["item"]
        for event in events
        if event["type"] == "item.completed"
        and event.get("item", {}).get("type") == "mcp_tool_call"
    ]
    assert len(calls) == result["tool_calls"] and len(mcp_calls) >= len(calls)
    post_terminal = mcp_calls[len(calls) :]
    for call in post_terminal:
        output = call["result"]["structured_content"]
        assert output["error"]["code"] == ErrorCode.EPISODE_FINISHED.value
        assert output["budget_left"] == result["budget_remaining"]
        assert output["queries_used"] == result["queries_used"]
    assert all(call["server"] == "dedeucerl_benchmark" for call in mcp_calls)
    assert all(
        event.get("item", {}).get("type") in {"mcp_tool_call", "agent_message", "reasoning"}
        for event in events
        if event["type"] == "item.completed"
    )
    for call, mcp_call in zip(calls, mcp_calls[: len(calls)], strict=True):
        assert call["tool_name"] == mcp_call["tool"]
        assert call["args"] == mcp_call["arguments"]
        assert call["cost"] == call["budget_before"] - call["budget_after"]
    assert sum(call["cost"] for call in calls) == result["queries_used"]

    # Only this post-episode auditor recreates private data, to independently
    # verify actual outputs, costs, traps, and terminal status against the trace.
    ir = TASK_REGISTRY[launch["task"]].ir
    params = {"n_states": launch["n_states"], "trap": launch["trap"]}
    if launch["task"] == "mealy_palindrome":
        params.update(min_length=launch["min_length"], min_distinct=3)
    snapshot = root / "episodes" / run.name / "instance.json"
    if snapshot.exists():
        instance = instance_from_dict(json.loads(snapshot.read_text()))
    else:
        assert result["task_version"] == ir.version, (
            "This trace needs its original instance snapshot or the original generator version."
        )
        instance = ir.generator.sample(seed=launch["seed"], budget=launch["budget"], **params)
    runtime = EpisodeRuntime(ir, instance, feedback=launch["feedback"])
    for expected in calls:
        actual = runtime.call_tool(expected["tool_name"], expected["args"]).to_dict()
        assert all(actual[key] == expected[key] for key in actual), (
            run.name,
            expected["tool_name"],
        )
    assert runtime.ok == result["ok"] and runtime.done == result["done"]
    assert runtime.budget == result["budget_remaining"]
    assert runtime.budget_init == result["budget_init"]
    assert runtime.queries_used == result["queries_used"]
    assert runtime.trap_hit == result["trap_hit"]
    assert runtime.evaluation_steps == result["evaluation_steps"]
    assert result["run_id"] == run.name
    assert result["task"] == ir.name and result["task_version"] == instance.kernel_version
    assert result["seed"] == instance.seed and result["params"] == instance.params
    assert result["feedback"] == launch["feedback"]
    assert all(trace[-1][key] == value for key, value in result.items())
    assert score_identification(runtime.state_dict()) == result["score"] == result["reward"]

    # Read only the exact new session's metadata and submitted tool code, never
    # export hidden reasoning or unrelated historical session contents.
    session_excerpt = run / "session_tools.jsonl"
    if not session_excerpt.exists():
        paths = list(sessions.glob(f"**/*{thread_id}.jsonl"))
        assert len(paths) == 1, f"missing/ambiguous session metadata for {thread_id}"
        excerpt = []
        for event in read_jsonl(paths[0]):
            payload = event.get("payload", {})
            if event["type"] == "turn_context":
                excerpt.append(
                    {
                        "type": "turn_context",
                        "payload": {key: payload.get(key) for key in ("model", "effort", "cwd")},
                    }
                )
            elif event["type"] == "response_item" and payload.get("type") in {
                "function_call",
                "custom_tool_call",
            }:
                excerpt.append(
                    {
                        "type": "response_item",
                        "payload": {
                            key: payload[key]
                            for key in ("type", "name", "input", "arguments")
                            if key in payload
                        },
                    }
                )
        session_excerpt.write_text("".join(json.dumps(event) + "\n" for event in excerpt))
    model_effort, workspaces, wrappers, referenced = set(), set(), set(), set()
    for event in read_jsonl(session_excerpt):
        payload = event.get("payload", {})
        if event["type"] == "turn_context":
            model_effort.add((payload.get("model"), payload.get("effort")))
            workspaces.add(payload.get("cwd"))
        if event["type"] == "response_item" and payload.get("type") in {
            "function_call",
            "custom_tool_call",
        }:
            wrappers.add(payload["name"])
            code = payload.get("input", payload.get("arguments", ""))
            referenced.update(re.findall(r"tools\.([A-Za-z0-9_]+)", code))
    assert model_effort == {(launch["model"], "high")}
    if launch.get("neutral_workspace"):
        workspace = Path(launch["workspace"])
        assert workspaces == {str(workspace)}
        assert launch["command"][launch["command"].index("--cd") + 1] == str(workspace)
        assert workspace.name.startswith("dedeucerl-episode-")
        assert not workspace.is_relative_to(Path(launch["out_dir"]))
    assert wrappers <= {"exec", "wait"}, wrappers
    assert referenced and all(name.startswith("mcp__dedeucerl_benchmark__") for name in referenced)
    usage = [event["usage"] for event in events if event["type"] == "turn.completed"]
    return {
        "run_id": run.name,
        "thread_id": thread_id,
        "model": launch["model"],
        "effort": "high",
        "host_time_limit_reached": host_timed_out,
        "benchmark_started": True,
        "outcome": "time_limited" if censored else result["termination_reason"],
        "n_states": launch["n_states"],
        "seed": launch["seed"],
        "min_length": launch["min_length"],
        "min_distinct": 3,
        "feedback": launch["feedback"],
        "trap": launch["trap"],
        "budget": launch["budget"],
        "duration_seconds": round(launch["finished_at"] - launch["started_at"], 2),
        **{
            key: result[key]
            for key in [
                "ok",
                "score",
                "reward",
                "termination_reason",
                "queries_used",
                "budget_remaining",
                "tool_calls",
                "evaluation_steps",
            ]
        },
        "calls_by_tool": dict(Counter(call["tool_name"] for call in calls)),
        "mcp_invocations": len(mcp_calls),
        "post_terminal_invocations": len(post_terminal),
        "actual_model_effort_verified": True,
        "service_tier_requested": service_tier,
        "fast_mode_config_verified": service_tier == "fast",
        "neutral_workspace_verified": bool(launch.get("neutral_workspace")),
        "runtime_replay_verified": True,
        "referenced_tools": sorted(referenced),
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--sessions", type=Path, default=Path.home() / ".codex/sessions")
    args = parser.parse_args()
    rows = [
        audit_run(args.out_dir, run, args.sessions)
        for run in sorted(args.out_dir.iterdir())
        if (run / "launch.json").is_file()
    ]
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(
            json.dumps(
                {
                    key: row.get(key)
                    for key in ["run_id", "outcome", "ok", "score", "tool_calls", "queries_used"]
                }
            )
        )


if __name__ == "__main__":
    main()
