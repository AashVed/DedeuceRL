from __future__ import annotations

import random

from dedeucerl.ir import (
    ActionContext,
    ActionValidationError,
    EnumSpace,
    JsonSchemaSpace,
    MaskedSpace,
    NonEnumerableActionSpace,
    ProductSpace,
    SequenceSpace,
    ToolActionContract,
    UnionSpace,
)
from dedeucerl.kernel.types import TaskInstance


def _context(*, budget: int = 5) -> ActionContext:
    return ActionContext(
        instance=TaskInstance(
            id="x",
            seed=1,
            kernel_name="fake",
            kernel_version="0.1",
            private={},
            params={},
            budget=budget,
        ),
        state=0,
        budget=budget,
        queries_used=0,
        tool_calls=0,
        done=False,
    )


def test_enum_space_sample_contains_canonicalize_schema_and_enumerate() -> None:
    space = EnumSpace("letter", ["A", "B"])

    assert space.sample(random.Random(0)) in {"A", "B"}
    assert space.contains("A")
    assert not space.contains("C")
    assert space.canonicalize({"value": "B"}) == "B"
    assert space.to_json_schema() == {"type": "string", "enum": ["A", "B"]}
    assert list(space.enumerate(limit=1)) == ["A"]


def test_product_space_rejects_missing_and_unknown_fields() -> None:
    space = ProductSpace(
        "api_call",
        {
            "method": EnumSpace("method", ["GET", "POST"]),
            "endpoint": EnumSpace("endpoint", ["/users", "/orgs"]),
        },
    )

    assert space.canonicalize({"method": "GET", "endpoint": "/users"}) == {
        "method": "GET",
        "endpoint": "/users",
    }
    assert space.to_json_schema()["additionalProperties"] is False
    assert list(space.enumerate(limit=2)) == [
        {"method": "GET", "endpoint": "/users"},
        {"method": "GET", "endpoint": "/orgs"},
    ]

    try:
        space.canonicalize({"method": "GET"})
    except ActionValidationError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected missing field to fail")

    try:
        space.canonicalize({"method": "GET", "endpoint": "/users", "extra": True})
    except ActionValidationError as e:
        assert "unknown field" in str(e)
    else:
        raise AssertionError("expected unknown field to fail")


def test_union_space_uses_tagged_canonical_actions() -> None:
    space = UnionSpace(
        "tool_action",
        {
            "query": ProductSpace("query", {"q": EnumSpace("q", ["x"])}),
            "submit": ProductSpace("submit", {"answer": EnumSpace("answer", [True, False])}),
        },
    )

    assert space.canonicalize({"type": "query", "value": {"q": "x"}}) == {
        "type": "query",
        "value": {"q": "x"},
    }
    assert list(space.enumerate(limit=2)) == [
        {"type": "query", "value": {"q": "x"}},
        {"type": "submit", "value": {"answer": True}},
    ]
    assert space.to_json_schema()["properties"]["type"]["enum"] == ["query", "submit"]

    try:
        space.canonicalize({"type": "missing", "value": {}})
    except ActionValidationError as e:
        assert "unknown variant" in str(e)
    else:
        raise AssertionError("expected invalid union tag to fail")


def test_sequence_space_validates_bounds_and_enumerates() -> None:
    space = SequenceSpace("word", EnumSpace("letter", ["A", "B"]), max_len=2, min_len=1)

    assert space.canonicalize({"items": ["A", "B"]}) == ["A", "B"]
    assert list(space.enumerate(limit=3)) == [["A"], ["B"], ["A", "A"]]
    assert space.to_json_schema()["maxItems"] == 2

    try:
        space.canonicalize([])
    except ActionValidationError as e:
        assert "at least" in str(e)
    else:
        raise AssertionError("expected short sequence to fail")


def test_json_schema_space_validates_and_reports_non_enumerable() -> None:
    space = JsonSchemaSpace("payload", {"type": "object", "required": ["x"]})

    assert space.canonicalize({"x": 1}) == {"x": 1}
    assert not space.contains({"y": 1})

    try:
        list(space.enumerate())
    except NonEnumerableActionSpace:
        pass
    else:
        raise AssertionError("expected schema-only space to be non-enumerable")


def test_json_schema_space_preserves_object_value_property() -> None:
    space = JsonSchemaSpace(
        "payload",
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    assert space.canonicalize({"value": "x"}) == {"value": "x"}


def test_masked_space_filters_contains_canonicalize_sample_and_enumerate() -> None:
    base = EnumSpace("symbol", ["A", "B", "C"])
    masked = MaskedSpace(
        "budgeted_symbol",
        base,
        lambda context, action: action != "C" and context.budget > 0,
    ).mask(_context(budget=1))

    assert masked.contains("A")
    assert not masked.contains("C")
    assert list(masked.enumerate(limit=None)) == ["A", "B"]
    assert masked.sample(random.Random(1)) in {"A", "B"}

    try:
        masked.canonicalize("C")
    except ActionValidationError as e:
        assert "masked out" in str(e)
    else:
        raise AssertionError("expected masked action to fail")


def test_tool_action_contract_generates_provider_neutral_schema() -> None:
    contract = ToolActionContract(
        name="act",
        kind="probe",
        description="Act.",
        action_space=ProductSpace("act_args", {"symbol": EnumSpace("symbol", ["A"])}),
        return_schema={"type": "object"},
    )

    schema = contract.to_tool_schema()
    assert schema["name"] == "act"
    assert schema["parameters"]["properties"]["symbol"]["enum"] == ["A"]
