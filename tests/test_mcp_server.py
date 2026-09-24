from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from dedeucerl.cli.mcp import build_episode, parse_args
from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.surface.mcp import MCPEpisodeServer
from dedeucerl.surface.dataset import instance_from_dict
from dedeucerl.utils.errors import ErrorCode


def _episode(
    *,
    seed: int = 42,
    budget: int = 5,
    persist: bool = False,
    runs_dir: Path | str = ".dedeucerl/runs",
    run_id: str | None = None,
) -> MCPEpisodeServer:
    ir = TASK_REGISTRY["mealy"].ir
    instance = ir.generator.sample(seed=seed, budget=budget, n_states=2, trap=False)
    return MCPEpisodeServer(
        ir,
        instance,
        persist=persist,
        runs_dir=runs_dir,
        run_id=run_id,
    )


def test_mcp_exposes_instructions_and_runtime_tools() -> None:
    async def exercise() -> None:
        episode = _episode()
        async with Client(episode.server, mode="legacy") as client:
            assert "hidden-system identification" in (client.instructions or "")
            assert json.dumps(episode.instance.private, sort_keys=True) not in client.instructions

            listed = await client.list_tools()
            assert [tool.name for tool in listed.tools] == ["act", "submit_table"]
            assert listed.tools[0].input_schema["properties"]["symbol"]["enum"] == [
                "A",
                "B",
                "C",
            ]
            assert listed.tools[0].output_schema is not None
            assert listed.tools[0].output_schema["properties"]["score"]["type"] == "number"
            assert listed.tools[0].annotations is not None
            assert listed.tools[0].annotations.open_world_hint is False

            invalid = await client.call_tool("act", {"symbol": "Z"})
            assert invalid.is_error is True
            assert invalid.structured_content["error"]["code"] == ErrorCode.INVALID_ARGUMENT.value
            assert invalid.structured_content["budget_left"] == 4

    asyncio.run(exercise())


def test_mcp_terminal_tool_call_auto_scores_and_persists(tmp_path: Path) -> None:
    async def exercise() -> MCPEpisodeServer:
        episode = _episode(persist=True, runs_dir=tmp_path, run_id="test-run")
        async with Client(episode.server, mode="legacy") as client:
            result = await client.call_tool(
                "submit_table",
                {"table_json": json.dumps(episode.instance.private["table"])},
            )
            assert result.is_error is False
            assert result.structured_content["ok"] is True
            assert result.structured_content["score"] == 1.0
            assert result.structured_content["reward"] == 1.0
            assert result.structured_content["termination_reason"] == "solved"
            assert episode.final_result is not None
        return episode

    episode = asyncio.run(exercise())
    result_path = tmp_path / "test-run" / "result.json"
    trace_path = tmp_path / "test-run" / "trace.jsonl"
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert saved["termination_reason"] == "solved"
    assert saved["score"] == 1.0
    assert saved["reward"] == saved["score"]
    assert saved["tool_calls"] == 1
    assert [event["event"] for event in trace] == [
        "episode_start",
        "tool_result",
        "episode_end",
    ]
    assert episode.finalize("disconnected") == episode.final_result
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 3


def test_saved_older_distribution_keeps_its_machine_and_version(tmp_path: Path) -> None:
    split = json.loads((Path(__file__).parents[1] / "dataset/smoke/mealy_smoke.json").read_text())
    instance = instance_from_dict(split["dev"]["items"][0]["instance"])
    ir = TASK_REGISTRY["mealy"].ir
    assert instance.kernel_version == "2.0" != ir.version

    async def exercise():
        episode = MCPEpisodeServer(ir, instance, runs_dir=tmp_path, run_id="saved")
        async with Client(episode.server, mode="legacy") as client:
            observed = await client.call_tool("act", {"symbol": "A"})
            assert (
                observed.structured_content["out"]
                == instance.private["table"]["trans"]["0"]["A"][1]
            )
            solved = await client.call_tool(
                "submit_table", {"table_json": json.dumps(instance.private["table"])}
            )
            assert solved.structured_content["ok"]
        assert episode.final_result["task_version"] == "2.0"

    asyncio.run(exercise())
    trace = [json.loads(line) for line in (tmp_path / "saved/trace.jsonl").read_text().splitlines()]
    assert trace[0]["task_version"] == trace[-1]["task_version"] == "2.0"


def test_mcp_rejects_calls_after_terminal_without_extending_trace(tmp_path: Path) -> None:
    async def exercise() -> MCPEpisodeServer:
        episode = _episode(persist=True, runs_dir=tmp_path, run_id="terminal-run")
        async with Client(episode.server, mode="legacy") as client:
            solved = await client.call_tool(
                "submit_table",
                {"table_json": json.dumps(episode.instance.private["table"])},
            )
            assert solved.is_error is False

            rejected = await client.call_tool("act", {"symbol": "A"})
            assert rejected.is_error is True
            assert rejected.structured_content["error"]["code"] == (
                ErrorCode.EPISODE_FINISHED.value
            )
            assert episode.runtime.tool_calls == 1
        return episode

    episode = asyncio.run(exercise())
    trace_path = tmp_path / "terminal-run" / "trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert [event["event"] for event in trace] == [
        "episode_start",
        "tool_result",
        "episode_end",
    ]
    assert episode.final_result is not None
    assert episode.final_result["tool_calls"] == 1


def test_mcp_budget_exhaustion_auto_finalizes(tmp_path: Path) -> None:
    async def exercise() -> None:
        episode = _episode(budget=1, persist=True, runs_dir=tmp_path, run_id="budget-run")
        async with Client(episode.server, mode="legacy") as client:
            result = await client.call_tool("act", {"symbol": "A"})
            assert result.structured_content["budget_left"] == 0
            assert result.structured_content["score"] == 0.0
            assert result.structured_content["termination_reason"] == "budget_exhausted"
            assert episode.final_result is not None

    asyncio.run(exercise())
    saved = json.loads((tmp_path / "budget-run" / "result.json").read_text(encoding="utf-8"))
    assert saved["termination_reason"] == "budget_exhausted"
    assert saved["done"] is True
    assert saved["score"] == 0.0


def test_mcp_reports_correct_submission_after_trap() -> None:
    async def exercise() -> None:
        ir = TASK_REGISTRY["mealy"].ir
        instance = ir.generator.sample(seed=0, budget=5, n_states=3, trap=True)
        # This tests terminal scoring, independent of generator seeds/topology.
        instance = replace(
            instance,
            private={
                "table": {
                    "n": 3,
                    "start": 0,
                    "trans": {str(s): {a: [(s + 1) % 3, s] for a in "ABC"} for s in range(3)},
                },
                "trap_pairs": [[1, "B"]],
            },
        )
        episode = MCPEpisodeServer(ir, instance, persist=False)
        async with Client(episode.server, mode="legacy") as client:
            await client.call_tool("act", {"symbol": "A"})
            trapped = await client.call_tool("act", {"symbol": "B"})
            assert trapped.structured_content["trap_hit"] is True

            result = await client.call_tool(
                "submit_table",
                {"table_json": json.dumps(instance.private["table"])},
            )
            assert result.structured_content["ok"] is False
            assert result.structured_content["score"] == 0.0
            assert result.structured_content["termination_reason"] == "solved_with_trap"

    asyncio.run(exercise())


def test_mcp_stdio_disconnect_finalizes_incomplete_episode(tmp_path: Path) -> None:
    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "dedeucerl.cli.mcp",
                "serve",
                "--seed",
                "7",
                "--no-trap",
                "--runs-dir",
                str(tmp_path),
                "--run-id",
                "stdio-run",
            ],
        )
        async with Client(parameters) as client:
            assert client.protocol_version == "2026-07-28"
            assert "hidden-system identification" in (client.instructions or "")
            result = await client.call_tool("act", {"symbol": "A"})
            assert result.is_error is False

    asyncio.run(exercise())
    saved = json.loads((tmp_path / "stdio-run" / "result.json").read_text(encoding="utf-8"))
    assert saved["termination_reason"] == "disconnected"
    assert saved["done"] is False
    assert saved["score"] == 0.0
    assert saved["tool_calls"] == 1


def test_mcp_idle_stdio_connection_creates_no_artifacts(tmp_path: Path) -> None:
    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "dedeucerl.cli.mcp",
                "serve",
                "--no-trap",
                "--runs-dir",
                str(tmp_path),
                "--run-id",
                "idle-run",
            ],
        )
        async with Client(parameters) as client:
            assert [tool.name for tool in (await client.list_tools()).tools] == [
                "act",
                "submit_table",
            ]

    asyncio.run(exercise())
    assert not (tmp_path / "idle-run").exists()


def test_mcp_cli_defaults_and_generic_params() -> None:
    args = parse_args(["serve", "--no-persist", "--param", "n_states=4"])
    episode = build_episode(args)

    assert args.task == "mealy"
    assert episode.instance.seed == 0
    assert episode.instance.budget == 25
    assert episode.instance.params == {"n_states": 4, "trap": True}
    assert episode.artifacts.result is None
    assert episode.artifacts.trace is None


def test_mcp_rejects_aliased_artifact_paths(tmp_path: Path) -> None:
    ir = TASK_REGISTRY["mealy"].ir
    instance = ir.generator.sample(seed=0, budget=5, n_states=2, trap=False)

    with pytest.raises(ValueError, match="result and trace paths must be different"):
        MCPEpisodeServer(
            ir,
            instance,
            result_path=tmp_path / "result.json",
            trace_path=tmp_path / "alias" / ".." / "result.json",
        )
