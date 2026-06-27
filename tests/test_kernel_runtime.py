from __future__ import annotations

import json

from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.utils.errors import ErrorCode


def _mealy(seed: int = 0, budget: int = 10, **kwargs):
    entry = TASK_REGISTRY["mealy"]
    params = {"n_states": 3, "trap": False}
    params.update(kwargs)
    instance = entry.ir.generator.sample(
        seed=seed,
        budget=budget,
        **params,
    )
    return entry.ir, instance


def test_mealy_generator_is_deterministic() -> None:
    entry = TASK_REGISTRY["mealy"]
    a = entry.ir.generator.sample(seed=123, budget=10, n_states=4, trap=True)
    b = entry.ir.generator.sample(seed=123, budget=10, n_states=4, trap=True)
    assert a.private == b.private
    assert a.kernel_name == "mealy"


def test_mealy_generator_rejects_invalid_parameters() -> None:
    entry = TASK_REGISTRY["mealy"]
    try:
        entry.ir.generator.sample(seed=0, budget=1, n_states=0)
    except ValueError as e:
        assert "n_states" in str(e)
    else:
        raise AssertionError("expected invalid n_states to fail")


def test_runtime_probe_and_correct_submit() -> None:
    ir, instance = _mealy()
    runtime = EpisodeRuntime(ir, instance, feedback=True)

    event = runtime.call_tool("act", {"symbol": "A"})
    assert "out" in event.output
    assert event.action == {"symbol": "A"}
    assert event.output["budget_left"] == 9
    assert runtime.queries_used == 1

    submit = runtime.call_tool("submit_table", {"table_json": json.dumps(instance.private["table"])})
    assert submit.output["ok"] is True
    assert runtime.done is True
    assert runtime.ok is True


def test_mealy_submit_tool_is_generated_by_hypothesis_contract() -> None:
    ir, instance = _mealy()
    context = ir.action_context(
        instance,
        ir.kernel.initial_state(instance),
        budget=instance.budget,
        queries_used=0,
        tool_calls=0,
        done=False,
    )

    assert ir.action_space.names() == ["act"]
    assert [contract.name for contract in ir.action_contracts(context)] == [
        "act",
        "submit_table",
    ]


def test_runtime_accepts_relabelled_mealy_submit() -> None:
    ir, instance = _mealy(seed=5, n_states=4)
    table = instance.private["table"]
    relabeled = _relabel_table(table, {0: 0, 1: 2, 2: 1, 3: 3})
    runtime = EpisodeRuntime(ir, instance, feedback=True)

    submit = runtime.call_tool("submit_table", {"table_json": json.dumps(relabeled)})

    assert submit.output["ok"] is True
    assert submit.output["counterexample"] is None
    assert runtime.done is True


def test_runtime_incorrect_submit_can_return_counterexample() -> None:
    ir, instance = _mealy()
    runtime = EpisodeRuntime(ir, instance, feedback=True)
    table = json.loads(json.dumps(instance.private["table"]))
    table["trans"]["0"]["A"][1] = (int(table["trans"]["0"]["A"][1]) + 1) % 3

    event = runtime.call_tool("submit_table", {"table_json": json.dumps(table)})
    assert event.output["ok"] is False
    assert event.output["counterexample"]
    assert runtime.done is False


def test_runtime_errors_are_structured_and_charge_budget() -> None:
    ir, instance = _mealy(budget=2)
    runtime = EpisodeRuntime(ir, instance)

    unknown = runtime.call_tool("missing", {})
    assert unknown.error is not None
    assert unknown.error["code"] == ErrorCode.UNKNOWN_TOOL.value
    assert runtime.budget == 1

    bad = runtime.call_tool("act", {"symbol": "Z"})
    assert bad.error is not None
    assert bad.error["code"] == ErrorCode.INVALID_ARGUMENT.value
    assert bad.action is None
    assert runtime.budget == 0
    assert runtime.done is True


def test_runtime_replay_matches_events() -> None:
    ir, instance = _mealy()
    runtime = EpisodeRuntime(ir, instance)
    runtime.call_tool("act", {"symbol": "A"})
    runtime.call_tool("act", {"symbol": "B"})

    result = runtime.replay(runtime.events)
    assert result.ok is True
    assert result.mismatch is None


def test_runtime_replay_detects_action_mismatch() -> None:
    ir, instance = _mealy()
    runtime = EpisodeRuntime(ir, instance)
    event = runtime.call_tool("act", {"symbol": "A"}).to_dict()
    event["action"] = {"symbol": "B"}

    result = runtime.replay([event])
    assert result.ok is False
    assert result.mismatch == "event 0: action mismatch"


def _relabel_table(table: dict, mapping: dict[int, int]) -> dict:
    inverse = {new: old for old, new in mapping.items()}
    return {
        "n": int(table["n"]),
        "start": mapping[int(table["start"])],
        "trans": {
            str(new_state): {
                symbol: [mapping[int(next_state)], output]
                for symbol, (next_state, output) in table["trans"][str(old_state)].items()
            }
            for new_state, old_state in inverse.items()
        },
    }
