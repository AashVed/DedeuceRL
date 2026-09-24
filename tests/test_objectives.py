from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, replace
from typing import Any

import jsonschema
import pytest
from pydantic import BaseModel, ConfigDict, Field, computed_field

from dedeucerl.ir import (
    TASK_REGISTRY,
    ActionValidationError,
    Evaluation,
    FeedbackSpec,
    Objective,
    ResourceModel,
    ToolActionContract,
    ToolCall,
    TypedSpace,
)
from dedeucerl.ir.palindrome import PalindromeAnswer, palindrome_witness
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface.mcp import _compile_mcp_tool
from dedeucerl.utils.errors import ErrorCode
from examples.custom_objective import WorkflowState, build_workflow_ir


def palindrome(budget: int = 20, *, feedback: bool = True, traps=(), terminal_trap=False):
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    instance = ir.generator.sample(seed=0, budget=20, trap=False, min_length=3)
    table = {
        "n": 2,
        "start": 0,
        "trans": {
            "0": {"A": [1, 0], "B": [0, 2], "C": [0, 0]},
            "1": {"A": [1, 1], "B": [0, 0], "C": [1, 2]},
        },
    }
    instance = replace(
        instance,
        private={"table": table, "trap_pairs": list(traps)},
        params={"n_states": 2, "trap": bool(traps), "min_length": 3, "min_distinct": 2},
        budget=budget,
    )
    ir = replace(ir, resource_model=ResourceModel(trap_ends_episode=terminal_trap))
    return EpisodeRuntime(ir, instance, feedback=feedback)


def test_initial_state_evaluation_retries_cost_and_replay():
    runtime = palindrome(budget=10)
    runtime.call_tool("act", {"symbol": "A"})
    assert runtime.state == 1
    failed = runtime.call_tool(
        "submit_answer", {"answer": {"kind": "sequence", "actions": ["A", "A", "A"]}}
    )
    assert failed.output["feedback"] == {"reason": "not_palindrome", "mismatches": [[0, 2]]}
    assert not runtime.done and runtime.state == 1
    assert failed.cost == 4 and runtime.tool_calls == 2
    runtime.call_tool("act", {"symbol": "C"})
    solved = runtime.call_tool(
        "submit_answer", {"answer": {"kind": "sequence", "actions": ["A", "A", "B"]}}
    )
    # AAB produces 010 initially, but 110 from dirty state 1.
    assert solved.output["ok"] and runtime.state == 1
    assert runtime.done and runtime.budget == 0
    assert runtime.tool_calls == 4 and runtime.evaluation_steps == 6
    assert runtime.queries_used == 10
    assert sum(event.cost for event in runtime.events) == 10
    assert solved.action == {"answer": {"kind": "sequence", "actions": ["A", "A", "B"]}}
    serialized = json.loads(json.dumps([event.to_dict() for event in runtime.events]))
    assert runtime.replay(serialized).ok


def test_partial_execution_cannot_pass_or_mutate_exploration():
    runtime = palindrome(budget=2)
    result = runtime.call_tool(
        "submit_answer", {"answer": {"kind": "sequence", "actions": ["A", "A", "B"]}}
    )
    assert result.error["code"] == ErrorCode.BUDGET_EXHAUSTED.value
    assert not runtime.ok and runtime.done and runtime.state == 0
    assert runtime.evaluation_steps == 1 and runtime.tool_calls == 1
    assert runtime.queries_used == result.cost == 2


def test_insufficient_submission_fee_accounts_for_remaining_budget():
    runtime = palindrome(budget=3)
    runtime.ir = replace(
        runtime.ir,
        resource_model=ResourceModel(
            cost_overrides={"submit_answer": 5},
        ),
    )
    result = runtime.call_tool(
        "submit_answer", {"answer": {"kind": "sequence", "actions": ["A", "A", "B"]}}
    )
    assert result.error and runtime.done
    assert result.cost == runtime.queries_used == 3
    assert runtime.evaluation_steps == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"answer": {"kind": "sequence", "actions": ["Z"]}},
        {"answer": {"kind": "sequence", "actions": []}},
        {"answer": {"kind": "sequence", "actions": [1]}},
        {"answer": {"kind": "sequence", "actions": ["A"], "extra": 1}},
        {"answer": {"kind": "impossible", "actions": ["A"]}},
        {"answer": {"kind": "unknown"}},
        {"answer": {"actions": ["A"]}},
        {"actions": ["A"]},
    ],
)
def test_typed_submission_rejects_invalid_inputs_before_execution(arguments):
    runtime = palindrome()
    result = runtime.call_tool("submit_answer", arguments)
    assert result.error["code"] == ErrorCode.INVALID_ARGUMENT.value
    assert runtime.queries_used == 1 and runtime.evaluation_steps == 0


def test_feedback_disabled_returns_only_verdict_and_accounting():
    runtime = palindrome(feedback=False)
    result = runtime.call_tool(
        "submit_answer", {"answer": {"kind": "sequence", "actions": ["A", "A", "A"]}}
    )
    assert result.output == {
        "ok": False,
        "done": False,
        "feedback": None,
        "budget_left": 16,
        "queries_used": 4,
        "trap_hit": False,
    }


@pytest.mark.parametrize("terminal", [False, True])
def test_evaluated_traps_keep_existing_episode_policy(terminal):
    runtime = palindrome(traps=[[1, "A"]], terminal_trap=terminal)
    result = runtime.call_tool(
        "submit_answer", {"answer": {"kind": "sequence", "actions": ["A", "A", "B"]}}
    )
    assert runtime.state == 0 and runtime.trap_hit and not runtime.ok and runtime.done
    assert runtime.evaluation_steps == (2 if terminal else 3)
    assert result.cost == (3 if terminal else 4)


def test_evaluation_cannot_call_submission_tools_recursively():
    runtime = palindrome()

    def evaluate(context, plan):
        context.run([ToolCall("submit_answer", plan.model_dump())])
        return Evaluation(True, None)

    runtime.ir = replace(runtime.ir, objective=replace(runtime.ir.objective, evaluator=evaluate))
    result = runtime.call_tool("submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}})
    assert result.error["code"] == ErrorCode.UNKNOWN_TOOL.value
    assert runtime.queries_used == 1 and not runtime.done and runtime.evaluation_steps == 0


def test_catching_execution_failure_does_not_turn_partial_evidence_into_success():
    runtime = palindrome(budget=2)

    def evaluate(context, plan):
        try:
            context.run([ToolCall("act", {"symbol": symbol}) for symbol in plan.answer.actions])
        except Exception:
            pass
        return Evaluation(True, None)

    runtime.ir = replace(runtime.ir, objective=replace(runtime.ir.objective, evaluator=evaluate))
    result = runtime.call_tool(
        "submit_answer", {"answer": {"kind": "sequence", "actions": ["A", "A", "B"]}}
    )
    assert result.error and runtime.done and not runtime.ok


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


class NestedFeedback(BaseModel):
    detail: Diagnostic


def test_nested_feedback_schema_is_valid_on_mcp_and_evidence_stays_private():
    runtime = palindrome()
    objective = replace(
        runtime.ir.objective,
        evaluator=lambda context, plan: Evaluation(False, "secret-ground-truth"),
        feedback=FeedbackSpec(
            NestedFeedback, lambda evidence: NestedFeedback(detail=Diagnostic(message="retry"))
        ),
    )
    runtime.ir = replace(runtime.ir, objective=objective)
    result = runtime.call_tool("submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}})
    assert result.output["feedback"] == {"detail": {"message": "retry"}}
    schema = _compile_mcp_tool(objective.tool_contracts()[0]).output_schema
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(result.output, schema)
    assert "secret-ground-truth" not in json.dumps(result.to_dict())


def test_invalid_feedback_is_validated_and_does_not_leak_private_validation_input():
    runtime = palindrome()
    objective = replace(
        runtime.ir.objective,
        evaluator=lambda context, plan: Evaluation(False, "secret-ground-truth"),
        feedback=FeedbackSpec(Diagnostic, lambda evidence: {"message": [evidence]}),
    )
    runtime.ir = replace(runtime.ir, objective=objective)
    result = runtime.call_tool("submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}})
    assert result.error and "secret-ground-truth" not in json.dumps(result.to_dict())


def test_disabled_feedback_formatter_is_not_executed():
    runtime = palindrome(feedback=False)

    def forbidden(evidence):
        raise AssertionError("must not compute private feedback")

    runtime.ir = replace(
        runtime.ir,
        objective=replace(
            runtime.ir.objective,
            feedback=FeedbackSpec(Diagnostic, forbidden),
        ),
    )
    result = runtime.call_tool("submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}})
    assert result.error is None and result.output["feedback"] is None


def test_custom_objective_on_non_mealy_kernel():
    ir = build_workflow_ir()
    runtime = EpisodeRuntime(ir, ir.generator.sample(seed=1, budget=20), feedback=True)
    runtime.call_tool("request", {"operation": "login"})
    dirty = runtime.state
    failed = runtime.call_tool("submit_workflow", {"operations": ["create", "grant_viewer"]})
    assert not failed.output["ok"] and runtime.state == dirty
    assert failed.output["feedback"]["missing"] == ["project", "viewer"]
    assert not runtime.done
    solved = runtime.call_tool(
        "submit_workflow",
        {
            "operations": ["login", "verify", "create", "grant_viewer"],
        },
    )
    assert solved.output["ok"] and runtime.state == WorkflowState(authenticated=True)
    assert runtime.evaluation_steps == 6 and runtime.queries_used == 9


def test_custom_non_execution_objective():
    class Number(BaseModel):
        value: int

    objective: Objective[int, Number, None, Any] = Objective(
        name="number",
        version="1.0",
        submission=ToolActionContract[Number](
            name="submit_number",
            kind="submit",
            description="Submit the requested number.",
            action_space=TypedSpace("number", Number),
            return_schema={},
        ),
        evaluator=lambda context, candidate: Evaluation(candidate.value == 42, None),
    )
    runtime = palindrome()
    runtime.ir = replace(runtime.ir, objective=objective)
    assert not runtime.call_tool("submit_number", {"value": 1}).output["ok"]
    assert runtime.call_tool("submit_number", {"value": 42}).output["ok"]
    assert runtime.evaluation_steps == 0 and runtime.queries_used == 2


def test_generated_palindrome_tasks_have_affordable_correct_answers():
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    for seed in range(20):
        instance = ir.generator.sample(
            seed=seed, budget=25, n_states=4, trap=True, min_length=5, min_distinct=3
        )
        plan = palindrome_witness(instance)
        assert plan is None or len(plan) + 1 <= instance.budget
        answer = {"kind": "impossible"} if plan is None else {"kind": "sequence", "actions": plan}
        runtime = EpisodeRuntime(ir, instance)
        assert runtime.call_tool("submit_answer", {"answer": answer}).output["ok"]
        assert not runtime.trap_hit
        assert (
            ir.generator.sample(
                seed=seed, budget=25, n_states=4, trap=True, min_length=5, min_distinct=3
            )
            == instance
        )


@pytest.mark.parametrize("budget", [0, -1])
def test_unaffordable_submission_fee_rejected_before_machine_generation(monkeypatch, budget):
    from dedeucerl.ir.mealy import MealyGenerator

    def unexpected_sample(*args, **kwargs):
        pytest.fail("an unaffordable submission must not sample hidden machines")

    monkeypatch.setattr(MealyGenerator, "sample", unexpected_sample)
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    with pytest.raises(ValueError, match="submission fee"):
        ir.generator.sample(seed=0, budget=budget, n_states=64)


@pytest.mark.parametrize("min_distinct,budget", [(2, 4), (3, 6)])
def test_palindrome_variety_minimum_budget_still_allows_a_solution(min_distinct, budget):
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    instance = ir.generator.sample(
        seed=0, budget=budget, n_states=4, trap=False, min_length=2, min_distinct=min_distinct
    )
    plan = palindrome_witness(instance)
    assert plan is not None
    runtime = EpisodeRuntime(ir, instance)
    result = runtime.call_tool("submit_answer", {"answer": {"kind": "sequence", "actions": plan}})
    assert result.output["ok"] and runtime.budget == 0


def test_typed_space_agrees_with_its_advertised_schema():
    space = TypedSpace("palindrome", PalindromeAnswer)
    for value in (
        {"answer": {"kind": "sequence", "actions": ["A", "B"]}},
        {"answer": {"kind": "impossible"}},
        {"answer": {"kind": "sequence", "actions": []}},
        {"answer": {"kind": "sequence", "actions": [3]}},
        {"answer": {"kind": "impossible", "actions": ["A"]}},
    ):
        assert space.contains(value) == jsonschema.Draft202012Validator(
            space.to_json_schema()
        ).is_valid(value)


@dataclass(frozen=True)
class TupleCandidate:
    actions: tuple[int, ...]


def test_dataclass_and_tuple_candidates_accept_json_but_keep_scalar_validation_strict():
    space = TypedSpace("tuple_candidate", TupleCandidate)
    candidate = {"actions": [1, 2]}
    jsonschema.validate(candidate, space.to_json_schema())
    assert space.canonicalize(candidate) == TupleCandidate((1, 2))
    assert space.canonicalize(TupleCandidate((1, 2))) == TupleCandidate((1, 2))
    for invalid in ({"actions": ["1"]}, {"actions": [True]}, {"actions": [1.2]}):
        with pytest.raises(ActionValidationError):
            space.canonicalize(invalid)


def test_trusted_python_candidates_preserve_model_codecs():
    class EncodedPlan(BaseModel):
        model_config = ConfigDict(ser_json_bytes="hex", val_json_bytes="hex")
        value: bytes

    space = TypedSpace("encoded", EncodedPlan)
    plan = EncodedPlan(value=b"AB")
    assert space.canonicalize(plan) is plan
    assert space.canonicalize({"value": "4142"}).value == b"AB"


def test_python_candidate_revalidation_follows_model_policy():
    class RevalidatedPlan(BaseModel):
        model_config = ConfigDict(revalidate_instances="always")
        actions: list[int] = Field(min_length=1)

    space = TypedSpace("revalidated", RevalidatedPlan)
    plan = RevalidatedPlan(actions=[1])
    plan.actions.clear()
    assert not space.contains(plan)
    assert not space.contains({"actions": []})


class AliasedFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(alias="explanation")

    @computed_field
    @property
    def length(self) -> int:
        return len(self.message)


def test_feedback_serialization_schema_matches_aliases_and_computed_fields():
    runtime = palindrome()
    objective = replace(
        runtime.ir.objective,
        feedback=FeedbackSpec(
            AliasedFeedback,
            lambda evidence: AliasedFeedback(explanation="retry"),
        ),
    )
    runtime.ir = replace(runtime.ir, objective=objective)
    result = runtime.call_tool("submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}})
    assert result.output["feedback"] == {"explanation": "retry", "length": 5}
    jsonschema.validate(
        result.output, _compile_mcp_tool(objective.tool_contracts()[0]).output_schema
    )


def test_each_execution_starts_fresh_and_private_instance_mutation_is_isolated():
    runtime = palindrome()
    original = json.dumps(runtime.instance.private, sort_keys=True)

    def evaluate(context, plan):
        first = context.run([ToolCall("act", {"symbol": "A"})])
        second = context.run([ToolCall("act", {"symbol": "A"})])
        assert first.steps[0].observation == second.steps[0].observation == {"out": 0}
        context.instance.private["table"]["trans"]["0"]["A"][0] = 999
        return Evaluation(False, None)

    runtime.ir = replace(
        runtime.ir, objective=replace(runtime.ir.objective, evaluator=evaluate, feedback=None)
    )
    result = runtime.call_tool("submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}})
    assert result.error is None and not runtime.done and runtime.state == 0
    assert result.cost == 3 and runtime.evaluation_steps == 2
    assert json.dumps(runtime.instance.private, sort_keys=True) == original


class RecursiveCandidate(BaseModel):
    action: str
    children: list[RecursiveCandidate] = Field(default_factory=list)


def test_recursive_object_candidates_compile_without_losing_nested_definitions():
    space = TypedSpace("tree", RecursiveCandidate)
    wire = {"action": "root", "children": [{"action": "leaf"}]}
    schema = space.to_tool_schema("submit_tree", "Submit a tree.")["parameters"]
    assert "properties" in schema and "$defs" in schema
    jsonschema.validate(wire, schema)
    assert space.canonicalize(wire).children[0].action == "leaf"


def test_feedback_wire_validation_catches_post_construction_constraint_violations():
    class BoundedFeedback(BaseModel):
        messages: list[str] = Field(max_length=1)

    invalid = BoundedFeedback(messages=["retry"])
    invalid.messages.append("private-evidence")
    runtime = palindrome()
    runtime.ir = replace(
        runtime.ir,
        objective=replace(
            runtime.ir.objective,
            feedback=FeedbackSpec(BoundedFeedback, lambda evidence: invalid),
        ),
    )
    result = runtime.call_tool("submit_answer", {"answer": {"kind": "sequence", "actions": ["A"]}})
    assert result.error is not None
    assert "private-evidence" not in json.dumps(result.to_dict())


def test_sequence_length_is_limited_by_budget_without_an_independent_cap():
    actions = ["A"] * 130 + ["B"]
    runtime = palindrome(budget=len(actions) + 1)
    result = runtime.call_tool(
        "submit_answer", {"answer": {"kind": "sequence", "actions": actions}}
    )
    assert result.output["ok"] and runtime.budget == 0
    assert runtime.evaluation_steps == len(actions)


def test_palindrome_feasibility_solver_agrees_with_exhaustive_search():
    runtime = palindrome(budget=7)
    for seed in range(20):
        rng = random.Random(seed)
        table = {
            "n": 2,
            "start": 0,
            "trans": {
                str(state): {symbol: [rng.randrange(2), rng.randrange(3)] for symbol in "ABC"}
                for state in range(2)
            },
        }
        traps = {(1, "B")} if seed % 2 else set()
        instance = replace(runtime.instance, private={"table": table, "trap_pairs": list(traps)})

        def satisfies(plan):
            state, outputs = 0, []
            for symbol in plan:
                if (state, symbol) in traps:
                    return False
                state, output = table["trans"][str(state)][symbol]
                outputs.append(output)
            return outputs == outputs[::-1] and len(set(outputs)) >= 2

        feasible = any(
            satisfies(plan)
            for length in range(3, instance.budget)
            for plan in itertools.product("ABC", repeat=length)
        )
        witness = palindrome_witness(instance)
        assert (witness is not None and len(witness) < instance.budget) == feasible
        if witness is not None:
            assert satisfies(witness)
