"""Typed submission objectives, independent of a particular hidden system.

Only ``ObjectiveResult.observation`` crosses the agent boundary. Evaluation
evidence is private and is exposed solely through an explicitly enabled formatter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

from dedeucerl.ir.actions import ToolActionContract
from dedeucerl.kernel.types import TaskInstance
from dedeucerl.utils.errors import DedeuceError, error_invalid_argument

Candidate = TypeVar("Candidate")
Evidence = TypeVar("Evidence")
Feedback = TypeVar("Feedback")
State = TypeVar("State")
State_co = TypeVar("State_co", covariant=True)
State_contra = TypeVar("State_contra", contravariant=True)


@dataclass(frozen=True)
class ToolCall:
    """One proposed system operation, with arguments validated by the runtime."""

    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionStep:
    tool_name: str
    action: Any
    observation: Mapping[str, Any]
    trap: bool


@dataclass(frozen=True)
class Execution(Generic[State_co]):
    """Private evidence from a complete, isolated execution from initial state."""

    steps: tuple[ExecutionStep, ...]
    final_state: State_co


class EvaluationContext(Protocol[State_co]):
    @property
    def instance(self) -> TaskInstance: ...

    def run(self, calls: Sequence[ToolCall]) -> Execution[State_co]:
        """Execute from initial state, charging the episode for every step.

        Raises on invalid calls, insufficient budget, or terminal traps. An
        incomplete execution is never returned as successful evidence.
        """
        ...


@dataclass(frozen=True)
class Evaluation(Generic[Evidence]):
    ok: bool
    evidence: Evidence
    terminal: bool = False


@dataclass(frozen=True)
class ObjectiveResult:
    ok: bool
    observation: Mapping[str, Any] = field(default_factory=dict)
    terminal: bool = False


class ObjectiveInputError(Exception):
    def __init__(self, error: DedeuceError, *, terminal: bool = False):
        super().__init__(error.message)
        self.error = error
        self.terminal = terminal


class ObjectiveContract(Protocol[State_contra]):
    """Runtime boundary; typed candidate/evidence handling stays in the objective."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def tool_contracts(self) -> Sequence[ToolActionContract[Any]]: ...

    def evaluate(
        self,
        context: EvaluationContext[State_contra],
        tool_name: str,
        candidate: Any,
        *,
        feedback: bool,
    ) -> ObjectiveResult: ...


@dataclass(frozen=True)
class FeedbackSpec(Generic[Evidence, Feedback]):
    """An author's formatter and its single source of validation/schema truth."""

    model: type[Feedback]
    render: Callable[[Evidence], Feedback]

    def schema(self) -> dict[str, Any]:
        return TypeAdapter(self.model).json_schema(mode="serialization")

    def encode(self, evidence: Evidence) -> Any:
        adapter: TypeAdapter[Feedback] = TypeAdapter(self.model)
        value = adapter.validate_python(self.render(evidence), strict=True)
        encoded = adapter.dump_python(value, mode="json", by_alias=True, warnings="error")
        # Models can contain mutable fields or custom serializers. Validate the
        # actual wire value, including constraints on fields changed after init.
        Draft202012Validator(self.schema()).validate(encoded)
        return encoded


@dataclass(frozen=True)
class Objective(Generic[State, Candidate, Evidence, Feedback]):
    """Define a custom objective with a typed tool, evaluator, and optional feedback.

    Failures are retryable unless the evaluator marks its decision terminal.
    Evaluation never receives the live exploration state. Use ``context.run``
    for budgeted system execution.
    """

    name: str
    version: str
    submission: ToolActionContract[Candidate]
    evaluator: Callable[[EvaluationContext[State], Candidate], Evaluation[Evidence]]
    feedback: FeedbackSpec[Evidence, Feedback] | None = None

    def __post_init__(self) -> None:
        if self.submission.kind != "submit":
            raise ValueError("objective tools must have kind='submit'")

    def tool_contracts(self) -> Sequence[ToolActionContract[Any]]:
        schema: dict[str, Any] = (
            self.feedback.schema() if self.feedback is not None else {"type": "null"}
        )
        definitions: dict[str, Any] = schema.pop("$defs", {})
        output_schema = submission_schema(
            {
                "feedback": {"anyOf": [schema, {"type": "null"}]}
                if self.feedback is not None
                else schema,
            }
        )
        if definitions:
            output_schema["$defs"] = definitions
        return (replace(self.submission, return_schema=output_schema),)

    def evaluate(
        self, context: EvaluationContext[State], tool_name: str, candidate: Any, *, feedback: bool
    ) -> ObjectiveResult:
        if tool_name != self.submission.name:
            raise ValueError(f"unknown objective tool {tool_name!r}")
        evaluation = self.evaluator(context, candidate)
        visible = None
        if feedback and not evaluation.ok and self.feedback is not None:
            try:
                visible = self.feedback.encode(evaluation.evidence)
            except Exception as exc:
                # A presentation failure must neither leak evidence nor undo a
                # final evaluator decision and grant another attempt.
                raise ObjectiveInputError(
                    error_invalid_argument(
                        "Objective feedback could not be encoded",
                        details={"error": type(exc).__name__},
                    ),
                    terminal=evaluation.terminal,
                ) from None
        return ObjectiveResult(evaluation.ok, {"feedback": visible}, terminal=evaluation.terminal)


def submission_schema(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "done": {"type": "boolean"},
            "budget_left": {"type": "integer"},
            "queries_used": {"type": "integer"},
            "trap_hit": {"type": "boolean"},
            **fields,
        },
        "required": ["ok", "done", "budget_left", "queries_used", "trap_hit"],
    }
