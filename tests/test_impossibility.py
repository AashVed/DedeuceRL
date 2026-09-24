from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import replace

import jsonschema
import pytest
import verifiers as vf
from mcp import Client
from verifiers.types import ResponseMessage

from dedeucerl.core.rubric import score_identification
from dedeucerl.ir import TASK_REGISTRY, Evaluation, FeedbackSpec, ToolCall
from dedeucerl.ir.objectives import ObjectiveInputError
from dedeucerl.ir.mealy import MealyGenerator
from dedeucerl.ir.palindrome import PalindromeFeedback, palindrome_witness
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface.dataset import build_dataset_from_split, generate_split
from dedeucerl.surface.mcp import MCPEpisodeServer
from dedeucerl.surface.vf import KernelToolEnv
from dedeucerl.utils.errors import ErrorCode
from examples.custom_objective import build_workflow_ir


CLAIM = {"answer": {"kind": "impossible"}}


def cycle_instance(outputs, *, budget=100, minimum=3, distinct=3):
    """All inputs advance the same cycle; output words can be checked analytically."""
    base = MealyGenerator().sample(seed=0, budget=budget, n_states=len(outputs), trap=False)
    return replace(
        base,
        kernel_name="mealy_palindrome",
        kernel_version=TASK_REGISTRY["mealy_palindrome"].ir.version,
        params={**base.params, "min_length": minimum, "min_distinct": distinct},
        private={
            "table": {
                "n": len(outputs),
                "start": 0,
                "trans": {
                    str(s): {a: [(s + 1) % len(outputs), out] for a in "ABC"}
                    for s, out in enumerate(outputs)
                },
            },
            "trap_pairs": [],
        },
    )


def test_impossible_despite_all_outputs_being_reachable():
    # Every output word is a prefix of (001122)*. Only 0 and 00 are palindromes:
    # a longer palindrome ending 0 either ends in 20 or in 2200, not its prefix.
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    instance = cycle_instance([0, 0, 1, 1, 2, 2], minimum=40)
    assert palindrome_witness(instance) is None
    runtime = EpisodeRuntime(ir, instance, feedback=True)
    runtime.call_tool("act", {"symbol": "A"})
    dirty = runtime.state
    event = runtime.call_tool("submit_answer", CLAIM)
    assert event.ok and event.done and event.cost == 1
    assert event.output["feedback"] is None
    assert runtime.state == dirty and runtime.evaluation_steps == 0
    assert score_identification(runtime.state_dict()) == 1
    assert runtime.replay(runtime.events).ok


@pytest.mark.parametrize("budget", [1, 3, 100])
@pytest.mark.parametrize("feedback", [False, True])
def test_wrong_claim_is_final_even_if_solution_exceeds_remaining_budget(budget, feedback):
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    instance = cycle_instance([0, 1, 0], budget=budget, distinct=2)
    assert len(palindrome_witness(instance)) == 3
    runtime = EpisodeRuntime(ir, instance, feedback=feedback)
    event = runtime.call_tool("submit_answer", CLAIM)
    assert not event.ok and event.done and runtime.terminal_failure
    assert event.output["done"] and event.cost == 1 and runtime.evaluation_steps == 0
    assert event.output["feedback"] == (
        {"reason": "solution_exists", "mismatches": []} if feedback else None
    )
    assert score_identification(runtime.state_dict()) == 0
    after = runtime.call_tool("act", {"symbol": "A"})
    assert after.error["code"] == ErrorCode.EPISODE_FINISHED.value
    assert after.cost == 0 and runtime.budget == budget - 1
    assert runtime.replay(runtime.events).ok


@pytest.mark.parametrize("seed,n,budget", [(0, 1, 1), (7202, 6, 100)])
def test_generator_retains_natural_impossible_machine(seed, n, budget):
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    base = MealyGenerator().sample(seed=seed, budget=budget, n_states=n, trap=False)
    instance = ir.generator.sample(
        seed=seed, budget=budget, n_states=n, trap=False, min_length=40, min_distinct=3
    )
    assert instance.private == base.private
    assert palindrome_witness(instance) is None
    assert EpisodeRuntime(ir, instance).call_tool("submit_answer", CLAIM).ok


def test_generator_skips_only_possible_but_unaffordable_candidates(monkeypatch):
    costly = cycle_instance([0, 0, 1, 0, 0], budget=4, distinct=2)
    affordable = cycle_instance([0, 1, 0, 0, 0], budget=4, distinct=2)
    assert len(palindrome_witness(costly)) == 5
    assert len(palindrome_witness(affordable)) == 3
    seeds = []

    def sample(self, *, seed, budget, **kwargs):
        seeds.append(seed)
        return costly if len(seeds) == 1 else affordable

    monkeypatch.setattr(MealyGenerator, "sample", sample)
    instance = TASK_REGISTRY["mealy_palindrome"].ir.generator.sample(
        seed=23, budget=4, n_states=5, trap=False, min_length=3, min_distinct=2
    )
    assert seeds == [23, 23 + 1_000_003]
    assert instance.private == affordable.private and instance.seed == 23


def test_one_state_solver_matches_analytical_shortest_lengths():
    instance = cycle_instance([0], distinct=2)
    for outputs in itertools.product(range(3), repeat=3):
        table = {"n": 1, "start": 0, "trans": {"0": dict(zip("ABC", ([0, o] for o in outputs)))}}
        for minimum in range(2, 12):
            for distinct in (2, 3):
                case = replace(
                    instance,
                    params={**instance.params, "min_length": minimum, "min_distinct": distinct},
                    private={"table": table, "trap_pairs": []},
                )
                plan = palindrome_witness(case)
                if len(set(outputs)) < distinct:
                    assert plan is None
                else:
                    assert plan is not None and len(plan) == max(minimum, 2 * distinct - 1)
                    word = [table["trans"]["0"][a][1] for a in plan]
                    assert word == word[::-1] and len(set(word)) >= distinct


def test_terminal_failure_is_generic_and_survives_broken_feedback():
    ir = build_workflow_ir()

    def broken(evidence):
        raise ValueError(evidence)

    ir = replace(
        ir,
        objective=replace(
            ir.objective,
            evaluator=lambda context, candidate: Evaluation(False, "secret-table", terminal=True),
            feedback=FeedbackSpec(PalindromeFeedback, broken),
        ),
    )
    runtime = EpisodeRuntime(ir, ir.generator.sample(seed=1, budget=20), feedback=True)
    event = runtime.call_tool("submit_workflow", {"operations": ["login"]})
    assert event.error and event.done and not event.ok and runtime.terminal_failure
    assert "secret-table" not in json.dumps(event.to_dict())
    assert runtime.replay(runtime.events).ok


def test_terminal_evaluation_keeps_a_caught_execution_error_final():
    ir = build_workflow_ir()

    def evaluate(context, candidate):
        try:
            context.run([ToolCall("unknown", {})])
        except ObjectiveInputError:
            return Evaluation(False, None, terminal=True)
        pytest.fail("invalid execution unexpectedly succeeded")

    ir = replace(ir, objective=replace(ir.objective, evaluator=evaluate, feedback=None))
    runtime = EpisodeRuntime(ir, ir.generator.sample(seed=1, budget=20))
    event = runtime.call_tool("submit_workflow", {"operations": ["login"]})
    assert event.error["code"] == ErrorCode.UNKNOWN_TOOL.value
    assert event.done and not event.ok and runtime.terminal_failure
    assert runtime.replay(runtime.events).ok


def test_mcp_terminal_claim_on_last_budget_unit():
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    instance = cycle_instance([0, 1, 0], budget=1, distinct=2)

    async def exercise():
        episode = MCPEpisodeServer(ir, instance, feedback=True, persist=False)
        async with Client(episode.server, mode="legacy") as client:
            schema = next(
                t.output_schema
                for t in (await client.list_tools()).tools
                if t.name == "submit_answer"
            )
            result = await client.call_tool("submit_answer", CLAIM)
            output = result.structured_content
            jsonschema.validate(output, schema)
            assert output["done"] and output["termination_reason"] == "objective_failed"
            assert output["score"] == output["reward"] == 0 and output["budget_left"] == 0
            again = (await client.call_tool("act", {"symbol": "A"})).structured_content
            assert again["done"] and again["error"]["code"] == ErrorCode.EPISODE_FINISHED.value

    asyncio.run(exercise())


@pytest.mark.parametrize("queries", [1, 50, 100])
@pytest.mark.parametrize("ok,trap,expected", [(True, False, 1), (False, False, 0), (True, True, 0)])
def test_correctness_reward_is_independent_of_efficiency(queries, ok, trap, expected):
    assert (
        score_identification(
            {"ok": ok, "trap_hit": trap, "queries_used": queries, "budget_init": 100}
        )
        == expected
    )


@pytest.mark.parametrize("answer_kind", ["impossible", "sequence", "zero_budget"])
def test_verifiers_rollout_stops_without_another_model_request(answer_kind):
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    dataset = build_dataset_from_split(generate_split(ir, seeds=[7], budget=50, trap=False), "dev")
    row = dict(dataset[0])
    instance = ir.generator.sample(seed=7, budget=50, trap=False)
    answer = (
        CLAIM
        if answer_kind == "impossible"
        else {"answer": {"kind": "sequence", "actions": palindrome_witness(instance)}}
    )
    if answer_kind == "zero_budget":
        saved = json.loads(row["answer"])
        saved["budget"] = 0
        row["answer"] = json.dumps(saved)

    class ScriptedClient(vf.Client):
        def __init__(self):
            super().__init__(None)
            self.request_count = 0

        def setup_client(self, config):
            return None

        async def to_native_tool(self, tool):
            return tool

        async def to_native_prompt(self, messages):
            return messages, {}

        async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
            self.request_count += 1
            assert self.request_count == 1, "model called after terminal tool"
            return vf.Response(
                id="scripted",
                created=0,
                model=model,
                message=ResponseMessage(
                    content="",
                    tool_calls=[
                        vf.ToolCall(id="answer", name="submit_answer", arguments=json.dumps(answer))
                    ],
                    finish_reason="tool_calls",
                    is_truncated=False,
                ),
            )

        async def raise_from_native_response(self, response):
            pass

        async def from_native_response(self, response):
            return response

        async def close(self):
            pass

    env, client = KernelToolEnv(dataset=dataset), ScriptedClient()
    state = asyncio.run(env.rollout({**row, "example_id": 0}, client, model="scripted"))
    assert state["done"] and state["is_completed"] and state["error"] is None
    assert state["ok"] == (answer_kind == "sequence")
    assert client.request_count == (0 if answer_kind == "zero_budget" else 1)
    if answer_kind != "zero_budget":
        assert [message.role for message in state["completion"]] == ["assistant", "tool"]
        assert json.loads(state["completion"][-1].content)["done"]
