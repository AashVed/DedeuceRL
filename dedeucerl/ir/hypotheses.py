"""Hypothesis and equivalence contracts for TaskIR tasks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Protocol, TypeVar

from dedeucerl.core.transducers import (
    normalize_transducer_table,
    transducer_counterexample,
    transducer_table_schema,
    transducer_tables_are_isomorphic,
)
from dedeucerl.ir.actions import JsonSchemaSpace, ProductSpace, ToolActionContract
from dedeucerl.kernel.types import TaskInstance
from dedeucerl.utils import (
    DedeuceError,
    error_invalid_json,
    error_malformed_hypothesis,
    error_unknown_tool,
)
from dedeucerl.utils.schema import validate_jsonschema


H = TypeVar("H")


@dataclass(frozen=True)
class HypothesisParseResult(Generic[H]):
    """Result of parsing canonical submit-tool action into a hypothesis."""

    hypothesis: H | None = None
    error: DedeuceError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def success(cls, hypothesis: H) -> HypothesisParseResult[H]:
        return cls(hypothesis=hypothesis)

    @classmethod
    def failure(cls, error: DedeuceError) -> HypothesisParseResult[H]:
        return cls(error=error)

    def unwrap(self) -> H:
        if self.error is not None:
            raise HypothesisInputError(self.error)
        return self.hypothesis  # type: ignore[return-value]


@dataclass(frozen=True)
class HypothesisValidationResult:
    """Result of semantic hypothesis validation."""

    ok: bool = True
    error: DedeuceError | None = None

    @classmethod
    def success(cls) -> HypothesisValidationResult:
        return cls(ok=True)

    @classmethod
    def failure(cls, error: DedeuceError) -> HypothesisValidationResult:
        return cls(ok=False, error=error)

    def raise_for_error(self) -> None:
        if not self.ok:
            error = self.error or error_malformed_hypothesis("validation failed")
            raise HypothesisInputError(error)


@dataclass(frozen=True)
class HypothesisJudgment:
    """Result of judging a normalized hypothesis."""

    ok: bool
    observation: Mapping[str, Any] = field(default_factory=dict)
    counterexample: Any | None = None
    distance: float | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


class HypothesisInputError(Exception):
    """User-facing hypothesis input error returned through runtime envelopes."""

    def __init__(self, error: DedeuceError):
        super().__init__(error.message)
        self.error = error


class HypothesisContract(Protocol, Generic[H]):
    """Submission tools and semantic correctness for a task."""

    name: str
    version: str

    def tool_contracts(self) -> Sequence[ToolActionContract[Any]]: ...

    def parse(self, tool_name: str, action: Any) -> HypothesisParseResult[H]: ...

    def validate(
        self,
        instance: TaskInstance,
        hypothesis: H,
    ) -> HypothesisValidationResult: ...

    def normalize(self, instance: TaskInstance, hypothesis: H) -> H: ...

    def judge(self, instance: TaskInstance, hypothesis: H) -> HypothesisJudgment: ...

    def counterexample(self, instance: TaskInstance, hypothesis: H) -> Any | None: ...

    def distance(self, instance: TaskInstance, hypothesis: H) -> float | None: ...


@dataclass(frozen=True)
class ExactJSONContract:
    """Exact JSON hypothesis equality against a private instance value."""

    name: str = "exact_json"
    version: str = "1.0"
    tool_name: str = "submit"
    description: str = "Submit a JSON value as the hypothesis."
    action_field: str = "value"
    expected_private_key: str = "answer"
    schema: Mapping[str, Any] = field(default_factory=dict)
    cost: int = 1
    normalizer: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", dict(self.schema))

    def tool_contracts(self) -> Sequence[ToolActionContract[Any]]:
        return (
            ToolActionContract(
                name=self.tool_name,
                kind="submit",
                description=self.description,
                action_space=ProductSpace(
                    name=f"{self.tool_name}_args",
                    fields={
                        self.action_field: JsonSchemaSpace(
                            name=self.action_field,
                            schema=self.schema,
                            description="Submitted hypothesis value.",
                        )
                    },
                ),
                return_schema=_submit_return_schema(),
                cost=self.cost,
            ),
        )

    def parse(self, tool_name: str, action: Any) -> HypothesisParseResult[Any]:
        if tool_name != self.tool_name:
            return HypothesisParseResult.failure(error_unknown_tool(tool_name, [self.tool_name]))
        if not isinstance(action, Mapping):
            return HypothesisParseResult.failure(
                error_malformed_hypothesis("submit action must be an object")
            )
        if self.action_field not in action:
            return HypothesisParseResult.failure(
                error_malformed_hypothesis(f"missing {self.action_field!r}")
            )
        return HypothesisParseResult.success(action[self.action_field])

    def validate(self, instance: TaskInstance, hypothesis: Any) -> HypothesisValidationResult:
        _ = instance
        reason = validate_jsonschema(hypothesis, dict(self.schema))
        if reason:
            return HypothesisValidationResult.failure(error_malformed_hypothesis(reason))
        return HypothesisValidationResult.success()

    def normalize(self, instance: TaskInstance, hypothesis: Any) -> Any:
        _ = instance
        value = self.normalizer(hypothesis) if self.normalizer is not None else hypothesis
        return _json_roundtrip(value)

    def judge(self, instance: TaskInstance, hypothesis: Any) -> HypothesisJudgment:
        expected = instance.private[self.expected_private_key]
        if self.normalizer is not None:
            expected = self.normalizer(expected)
        ok = hypothesis == _json_roundtrip(expected)
        return HypothesisJudgment(ok=ok, observation={"ok": ok})

    def counterexample(self, instance: TaskInstance, hypothesis: Any) -> Any | None:
        _ = (instance, hypothesis)
        return None

    def distance(self, instance: TaskInstance, hypothesis: Any) -> float | None:
        _ = (instance, hypothesis)
        return None


@dataclass(frozen=True)
class FiniteTransducerIsomorphismContract:
    """Complete-table finite-transducer isomorphism contract."""

    name: str = "finite_transducer_isomorphism"
    version: str = "1.0"
    tool_name: str = "submit_table"
    description: str = "Submit a complete finite-transducer transition table as JSON."
    action_field: str = "table_json"
    alphabet: Sequence[str] = field(default_factory=tuple)
    outputs: Sequence[Any] | None = None
    truth_private_key: str = "table"
    start_const: int | None = None
    cost: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "alphabet", tuple(str(symbol) for symbol in self.alphabet))
        if self.outputs is not None:
            object.__setattr__(self, "outputs", tuple(self.outputs))

    def tool_contracts(self) -> Sequence[ToolActionContract[Any]]:
        return (
            ToolActionContract(
                name=self.tool_name,
                kind="submit",
                description=self.description,
                action_space=ProductSpace(
                    name=f"{self.tool_name}_args",
                    fields={
                        self.action_field: JsonSchemaSpace(
                            name=self.action_field,
                            schema={"type": "string"},
                            description=(
                                'JSON: {"n": <int>, "start": 0, "trans": '
                                '{"0": {"A": [next_state, output], ...}, ...}}'
                            ),
                        )
                    },
                ),
                return_schema=_submit_return_schema(
                    counterexample_schema={"type": ["array", "null"]}
                ),
                cost=self.cost,
            ),
        )

    def parse(
        self,
        tool_name: str,
        action: Any,
    ) -> HypothesisParseResult[Mapping[str, Any]]:
        if tool_name != self.tool_name:
            return HypothesisParseResult.failure(error_unknown_tool(tool_name, [self.tool_name]))
        if not isinstance(action, Mapping):
            return HypothesisParseResult.failure(
                error_malformed_hypothesis("submit action must be an object")
            )

        raw = action.get(self.action_field)
        if not isinstance(raw, str):
            return HypothesisParseResult.failure(
                error_malformed_hypothesis(f"{self.action_field} must be a string")
            )

        try:
            table = json.loads(raw)
        except Exception:
            return HypothesisParseResult.failure(error_invalid_json(self.action_field))

        if not isinstance(table, Mapping):
            return HypothesisParseResult.failure(
                error_malformed_hypothesis(f"{self.action_field} must encode an object")
            )
        return HypothesisParseResult.success(table)

    def validate(
        self,
        instance: TaskInstance,
        hypothesis: Mapping[str, Any],
    ) -> HypothesisValidationResult:
        _ = instance
        try:
            normalize_transducer_table(
                hypothesis,
                self.alphabet,
                outputs=self.outputs,
                start_const=self.start_const,
            )
        except ValueError as e:
            return HypothesisValidationResult.failure(error_malformed_hypothesis(str(e)))
        return HypothesisValidationResult.success()

    def normalize(
        self,
        instance: TaskInstance,
        hypothesis: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        _ = instance
        return normalize_transducer_table(
            hypothesis,
            self.alphabet,
            outputs=self.outputs,
            start_const=self.start_const,
        )

    def judge(self, instance: TaskInstance, hypothesis: Mapping[str, Any]) -> HypothesisJudgment:
        truth = instance.private[self.truth_private_key]
        ok = transducer_tables_are_isomorphic(
            truth,
            hypothesis,
            self.alphabet,
            outputs=self.outputs,
            start_const=self.start_const,
        )
        return HypothesisJudgment(ok=ok, observation={"ok": ok})

    def counterexample(self, instance: TaskInstance, hypothesis: Mapping[str, Any]) -> Any | None:
        truth = instance.private[self.truth_private_key]
        return transducer_counterexample(
            truth,
            hypothesis,
            self.alphabet,
            outputs=self.outputs,
            start_const=self.start_const,
        )

    def distance(self, instance: TaskInstance, hypothesis: Mapping[str, Any]) -> float | None:
        _ = (instance, hypothesis)
        return None

    def schema(self) -> dict[str, Any]:
        return transducer_table_schema(
            self.alphabet,
            outputs=self.outputs,
            start_const=self.start_const,
        )


def enrich_judgment(
    judgment: HypothesisJudgment,
    *,
    counterexample: Any | None,
    distance: float | None,
) -> HypothesisJudgment:
    """Attach optional feedback fields computed outside `judge()`."""

    return replace(judgment, counterexample=counterexample, distance=distance)


def _submit_return_schema(
    *,
    counterexample_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "budget_left": {"type": "integer"},
            "queries_used": {"type": "integer"},
            "trap_hit": {"type": "boolean"},
            "counterexample": (
                {"type": ["object", "array", "null"]}
                if counterexample_schema is None
                else dict(counterexample_schema)
            ),
            "distance": {"type": ["number", "null"]},
        },
        "required": ["ok", "budget_left", "queries_used", "trap_hit"],
    }


def _json_roundtrip(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return value
