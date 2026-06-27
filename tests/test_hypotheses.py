from __future__ import annotations

import json
from typing import Any

from dedeucerl.ir import ExactJSONContract, FiniteTransducerIsomorphismContract
from dedeucerl.kernel.types import TaskInstance
from dedeucerl.utils.errors import ErrorCode


def _instance(private: dict[str, Any]) -> TaskInstance:
    return TaskInstance(
        id="test-1",
        seed=1,
        kernel_name="test",
        kernel_version="0.1",
        private=private,
        params={},
        budget=5,
    )


def test_exact_json_contract_normalizes_and_judges() -> None:
    contract = ExactJSONContract(
        schema={
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["items"],
        },
        normalizer=lambda value: {"items": sorted(value["items"])},
    )
    instance = _instance({"answer": {"items": [1, 2, 3]}})

    parsed = contract.parse("submit", {"value": {"items": [3, 1, 2]}}).unwrap()
    contract.validate(instance, parsed).raise_for_error()
    normalized = contract.normalize(instance, parsed)
    judgment = contract.judge(instance, normalized)

    assert normalized == {"items": [1, 2, 3]}
    assert judgment.ok is True
    assert contract.counterexample(instance, normalized) is None
    assert contract.distance(instance, normalized) is None


def test_exact_json_contract_rejects_malformed_schema_value() -> None:
    contract = ExactJSONContract(schema={"type": "boolean"})
    instance = _instance({"answer": True})
    parsed = contract.parse("submit", {"value": "yes"}).unwrap()
    validation = contract.validate(instance, parsed)

    assert validation.ok is False
    assert validation.error is not None
    assert validation.error.code == ErrorCode.MALFORMED_HYPOTHESIS


def test_finite_transducer_contract_accepts_relabeling() -> None:
    truth = {
        "n": 3,
        "start": 0,
        "trans": {
            "0": {"A": [1, 0], "B": [2, 1]},
            "1": {"A": [1, 1], "B": [2, 0]},
            "2": {"A": [1, 2], "B": [2, 2]},
        },
    }
    relabeled = _relabel_table(truth, {0: 0, 1: 2, 2: 1})
    contract = FiniteTransducerIsomorphismContract(alphabet=["A", "B"], outputs=[0, 1, 2])
    instance = _instance({"table": truth})

    parsed = contract.parse("submit_table", {"table_json": json.dumps(relabeled)}).unwrap()
    contract.validate(instance, parsed).raise_for_error()
    normalized = contract.normalize(instance, parsed)
    judgment = contract.judge(instance, normalized)

    assert judgment.ok is True
    assert contract.counterexample(instance, normalized) is None
    assert contract.distance(instance, normalized) is None


def test_finite_transducer_contract_accepts_relabelled_start_when_generic() -> None:
    truth = {
        "n": 2,
        "start": 0,
        "trans": {
            "0": {"A": [0, "x"]},
            "1": {"A": [1, "y"]},
        },
    }
    relabeled = _relabel_table(truth, {0: 1, 1: 0})
    contract = FiniteTransducerIsomorphismContract(alphabet=["A"], outputs=["x", "y"])
    instance = _instance({"table": truth})

    parsed = contract.parse("submit_table", {"table_json": json.dumps(relabeled)}).unwrap()
    contract.validate(instance, parsed).raise_for_error()
    normalized = contract.normalize(instance, parsed)

    assert normalized["start"] == 1
    assert contract.judge(instance, normalized).ok is True


def test_finite_transducer_contract_rejects_invalid_transition_and_finds_counterexample() -> None:
    truth = {
        "n": 2,
        "start": 0,
        "trans": {
            "0": {"A": [1, 0], "B": [0, 1]},
            "1": {"A": [0, 1], "B": [1, 0]},
        },
    }
    contract = FiniteTransducerIsomorphismContract(alphabet=["A", "B"], outputs=[0, 1])
    instance = _instance({"table": truth})

    invalid = json.loads(json.dumps(truth))
    invalid["trans"]["0"]["A"][0] = 7
    parsed_invalid = contract.parse("submit_table", {"table_json": json.dumps(invalid)}).unwrap()
    invalid_validation = contract.validate(instance, parsed_invalid)
    assert invalid_validation.ok is False
    assert invalid_validation.error is not None
    assert invalid_validation.error.code == ErrorCode.MALFORMED_HYPOTHESIS

    incorrect = json.loads(json.dumps(truth))
    incorrect["trans"]["0"]["A"][1] = 1
    parsed = contract.parse("submit_table", {"table_json": json.dumps(incorrect)}).unwrap()
    contract.validate(instance, parsed).raise_for_error()
    normalized = contract.normalize(instance, parsed)

    judgment = contract.judge(instance, normalized)
    cex = contract.counterexample(instance, normalized)

    assert judgment.ok is False
    assert cex
    assert cex[0]["in"] == "A"


def _relabel_table(table: dict[str, Any], mapping: dict[int, int]) -> dict[str, Any]:
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
