from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace

import jsonschema
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel, ConfigDict, Field

from dedeucerl.adapters.base import ModelReply
from dedeucerl.cli.eval import run_episode
from dedeucerl.core.rubric import score_identification
from dedeucerl.ir import (
    TASK_REGISTRY,
    Evaluation,
    Objective,
    TaskEntry,
    ToolActionContract,
    TypedSpace,
)
from dedeucerl.ir.palindrome import palindrome_witness
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface.dataset import (
    build_dataset_from_split,
    generate_split,
    instance_from_dict,
)
from dedeucerl.surface.mcp import MCPEpisodeServer
from dedeucerl.surface.vf import KernelToolEnv


def test_custom_objective_agrees_across_dataset_cli_verifiers_and_mcp():
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    split = generate_split(ir, seeds=[7], budget=50, trap=False, min_length=3)
    dataset = build_dataset_from_split(json.loads(json.dumps(split)), "dev", feedback=True)
    item = dataset[0]
    instance = instance_from_dict(json.loads(item["answer"]))
    plan = palindrome_witness(instance)
    assert plan is not None and len(plan) + 4 <= instance.budget
    assert "submit_answer" in str(item["prompt"]) and "palindrome" in str(item["prompt"])
    calls = [
        ("act", {"symbol": "C"}),
        ("submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}}),
        ("submit_answer", {"answer": {"kind": "sequence", "actions": plan}}),
    ]
    direct = EpisodeRuntime(ir, instance, feedback=True)
    expected = [direct.call_tool(name, args).output for name, args in calls]
    assert expected[1]["feedback"]["reason"] == "too_short" and direct.ok

    env = KernelToolEnv(dataset=dataset, feedback=True)
    state = asyncio.run(env.setup_state({"prompt": item["prompt"], "answer": item["answer"]}))
    vf_tools = {tool.__name__: tool for tool in env.tools}
    for (name, args), output in zip(calls, expected, strict=True):
        env.update_tool_args(name, args, [], state)
        assert json.loads(vf_tools[name](**args)) == output
    assert state["evaluation_steps"] == len(plan) + 1

    async def exercise_mcp():
        episode = MCPEpisodeServer(ir, instance, feedback=True, persist=False)
        async with Client(episode.server, mode="legacy") as client:
            schemas = {tool.name: tool.output_schema for tool in (await client.list_tools()).tools}
            for (name, args), output in zip(calls, expected, strict=True):
                result = await client.call_tool(name, args)
                actual = result.structured_content
                assert not result.is_error
                assert {key: actual[key] for key in output} == output
                jsonschema.validate(actual, schemas[name])
            assert actual["termination_reason"] == "solved"
            assert actual["reward"] == score_identification(direct.state_dict())
        assert episode.final_result["evaluation_steps"] == len(plan) + 1

    asyncio.run(exercise_mcp())

    class ScriptedAdapter:
        def reset_conversation(self):
            self.pending = iter(calls)

        def chat(self, messages, tools, **kwargs):
            name, args = next(self.pending)
            return ModelReply(
                tool_calls=[
                    {
                        "id": f"call_{len(messages)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ]
            )

    trace = []
    result = asyncio.run(
        run_episode(
            item=item,
            adapter=ScriptedAdapter(),
            episode_idx=0,
            rollout=0,
            feedback=True,
            temperature=None,
            effort=None,
            model_spec="scripted:test",
            verbose=False,
            trace_writer=trace.append,
        )
    )
    assert result["ok"] and result["queries_used"] == direct.queries_used
    assert result["evaluation_steps"] == state["evaluation_steps"]
    assert result["tool_calls_processed"] == direct.tool_calls == 3
    assert result["reward"] == score_identification(direct.state_dict())
    events = [record["runtime_event"] for record in trace if record["event"] == "tool_result"]
    assert direct.replay(events).ok
    assert trace[-1]["evaluation_steps"] == len(plan) + 1


def test_custom_objective_stdio_retry_and_persisted_accounting(tmp_path):
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    instance = ir.generator.sample(seed=7, budget=25, trap=False)
    plan = palindrome_witness(instance)
    assert plan is not None and len(plan) + 3 <= instance.budget

    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "dedeucerl.cli.mcp",
                "serve",
                "--task",
                "mealy_palindrome",
                "--seed",
                "7",
                "--budget",
                "25",
                "--no-trap",
                "--feedback",
                "--runs-dir",
                str(tmp_path),
                "--run-id",
                "palindrome",
            ],
        )
        async with Client(params) as client:
            assert "palindrome" in client.instructions
            assert {tool.name for tool in (await client.list_tools()).tools} == {
                "act",
                "submit_answer",
            }
            failed = await client.call_tool(
                "submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}}
            )
            assert failed.structured_content["ok"] is False and not failed.is_error
            assert failed.structured_content["feedback"]["reason"] == "too_short"
            solved = await client.call_tool(
                "submit_answer", {"answer": {"kind": "sequence", "actions": plan}}
            )
            assert solved.structured_content["ok"] is True
            assert solved.structured_content["termination_reason"] == "solved"

    asyncio.run(exercise())
    result = json.loads((tmp_path / "palindrome/result.json").read_text())
    assert result["tool_calls"] == 2 and result["evaluation_steps"] == len(plan) + 1
    assert result["queries_used"] == len(plan) + 3
    trace = [
        json.loads(line) for line in (tmp_path / "palindrome/trace.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in trace] == [
        "episode_start",
        "tool_result",
        "tool_result",
        "episode_end",
    ]


@pytest.mark.parametrize(
    "arguments, expected_ok", [({}, False), ({"from": None}, True), ({"from": 3}, False)]
)
def test_verifiers_preserves_null_omission_and_json_aliases(monkeypatch, arguments, expected_ok):
    class NullableCandidate(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int | None = Field(default=7, alias="from")

    base = TASK_REGISTRY["mealy"].ir
    objective = Objective(
        name="nullable",
        version="1",
        submission=ToolActionContract(
            name="submit_nullable",
            kind="submit",
            description="Submit explicit null.",
            action_space=TypedSpace("nullable", NullableCandidate),
            return_schema={},
        ),
        evaluator=lambda context, candidate: Evaluation(candidate.value is None, None),
    )
    ir = replace(base, objective=objective)
    monkeypatch.setitem(TASK_REGISTRY, "mealy", TaskEntry("mealy", ir))
    dataset = build_dataset_from_split(generate_split(ir, seeds=[0], budget=10), "dev")
    env = KernelToolEnv(dataset=dataset)

    async def exercise():
        state = await env.setup_state(
            {"prompt": dataset[0]["prompt"], "answer": dataset[0]["answer"]}
        )
        result = await env.call_tool("submit_nullable", arguments, "call_0")
        assert json.loads(result.content)["ok"] is expected_ok
        assert state["queries_used"] == 1
        assert state["_runtime"].events[0].args == arguments

    asyncio.run(exercise())
    schema = next(tool.parameters for tool in env.tool_defs if tool.name == "submit_nullable")
    assert schema == objective.submission.to_tool_schema()["parameters"]
