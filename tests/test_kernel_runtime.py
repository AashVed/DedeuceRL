from __future__ import annotations

import json

from dedeucerl.kernel import KERNEL_REGISTRY
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.utils.errors import ErrorCode


def _mealy(seed: int = 0, budget: int = 10, **kwargs):
    entry = KERNEL_REGISTRY["mealy"]
    instance = entry.sampler.sample(seed=seed, budget=budget, n_states=3, trap=False, **kwargs)
    return entry.kernel, instance


def test_mealy_sampler_is_deterministic() -> None:
    entry = KERNEL_REGISTRY["mealy"]
    a = entry.sampler.sample(seed=123, budget=10, n_states=4, trap=True)
    b = entry.sampler.sample(seed=123, budget=10, n_states=4, trap=True)
    assert a.private == b.private
    assert a.kernel_name == "mealy"


def test_mealy_sampler_rejects_invalid_parameters() -> None:
    entry = KERNEL_REGISTRY["mealy"]
    try:
        entry.sampler.sample(seed=0, budget=1, n_states=0)
    except ValueError as e:
        assert "n_states" in str(e)
    else:
        raise AssertionError("expected invalid n_states to fail")


def test_runtime_probe_and_correct_submit() -> None:
    kernel, instance = _mealy()
    runtime = EpisodeRuntime(kernel, instance, feedback=True)

    event = runtime.call_tool("act", {"symbol": "A"})
    assert "out" in event.output
    assert event.output["budget_left"] == 9
    assert runtime.queries_used == 1

    submit = runtime.call_tool("submit_table", {"table_json": json.dumps(instance.private["table"])})
    assert submit.output["ok"] is True
    assert runtime.done is True
    assert runtime.ok is True


def test_runtime_incorrect_submit_can_return_counterexample() -> None:
    kernel, instance = _mealy()
    runtime = EpisodeRuntime(kernel, instance, feedback=True)
    table = json.loads(json.dumps(instance.private["table"]))
    table["trans"]["0"]["A"][1] = (int(table["trans"]["0"]["A"][1]) + 1) % 3

    event = runtime.call_tool("submit_table", {"table_json": json.dumps(table)})
    assert event.output["ok"] is False
    assert event.output["counterexample"]
    assert runtime.done is False


def test_runtime_errors_are_structured_and_charge_budget() -> None:
    kernel, instance = _mealy(budget=2)
    runtime = EpisodeRuntime(kernel, instance)

    unknown = runtime.call_tool("missing", {})
    assert unknown.error is not None
    assert unknown.error["code"] == ErrorCode.UNKNOWN_TOOL.value
    assert runtime.budget == 1

    bad = runtime.call_tool("act", {"symbol": "Z"})
    assert bad.error is not None
    assert bad.error["code"] == ErrorCode.INVALID_ARGUMENT.value
    assert runtime.budget == 0
    assert runtime.done is True


def test_runtime_replay_matches_events() -> None:
    kernel, instance = _mealy()
    runtime = EpisodeRuntime(kernel, instance)
    runtime.call_tool("act", {"symbol": "A"})
    runtime.call_tool("act", {"symbol": "B"})

    result = runtime.replay(runtime.events)
    assert result.ok is True
    assert result.mismatch is None
