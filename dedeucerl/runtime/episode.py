"""Generic episode runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Sequence, TypeVar

from pydantic import TypeAdapter

from dedeucerl.ir.actions import ActionContext, ActionValidationError, ToolActionContract
from dedeucerl.ir.hypotheses import HypothesisInputError
from dedeucerl.ir.objectives import (
    Execution,
    ExecutionStep,
    ObjectiveInputError,
    ObjectiveResult,
    ToolCall,
)
from dedeucerl.ir.types import TaskIR
from dedeucerl.kernel.types import (
    KernelInputError,
    KernelTransition,
    TaskInstance,
)
from dedeucerl.utils import (
    DedeuceError,
    error_budget_exhausted,
    error_episode_finished,
    error_invalid_argument,
    error_unknown_tool,
)

State = TypeVar("State")


@dataclass(frozen=True)
class EpisodeEvent:
    event: str
    tool_name: str
    args: Mapping[str, Any]
    action: Any | None
    output: Mapping[str, Any]
    error: Mapping[str, Any] | None
    kind: str | None
    cost: int
    budget_before: int
    budget_after: int
    queries_used: int
    tool_calls: int
    done: bool
    ok: bool
    trap_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "tool_name": self.tool_name,
            "args": dict(self.args),
            "action": self.action,
            "output": dict(self.output),
            "error": None if self.error is None else dict(self.error),
            "kind": self.kind,
            "cost": self.cost,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "queries_used": self.queries_used,
            "tool_calls": self.tool_calls,
            "done": self.done,
            "ok": self.ok,
            "trap_hit": self.trap_hit,
        }


@dataclass(frozen=True)
class ReplayResult:
    ok: bool
    events: Sequence[EpisodeEvent]
    mismatch: str | None = None


@dataclass
class EpisodeRuntime(Generic[State]):
    ir: TaskIR[State]
    instance: TaskInstance
    feedback: bool = False
    state: State = field(init=False)
    budget: int = 0
    budget_init: int = 0
    queries_used: int = 0
    tool_calls: int = 0
    evaluation_steps: int = 0
    done: bool = False
    ok: bool = False
    trap_hit: bool = False
    terminal_failure: bool = False
    events: list[EpisodeEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = self.ir.kernel.initial_state(self.instance)
        self.budget = max(0, int(self.instance.budget))
        self.budget_init = self.budget
        if self.budget <= 0:
            self.done = True

    def action_context(self) -> ActionContext:
        return self.ir.action_context(
            self.instance,
            self.state,
            budget=self.budget,
            queries_used=self.queries_used,
            tool_calls=self.tool_calls,
            done=self.done,
        )

    def contracts(self) -> list[ToolActionContract[Any]]:
        return list(self.ir.action_contracts(self.action_context()))

    def tool_schemas(self) -> list[dict[str, Any]]:
        return self.ir.tool_schemas(self.action_context())

    def call_tool(self, tool_name: str, raw_args: Mapping[str, Any] | None) -> EpisodeEvent:
        args = dict(raw_args or {})
        budget_before = self.budget

        contract = self._find_contract(tool_name)
        if self.done:
            return self._record_error(
                tool_name,
                args,
                error_episode_finished(),
                contract=contract,
                budget_before=budget_before,
            )

        if contract is None:
            cost = max(0, int(self.ir.resource_model.unknown_tool_cost))
            if not self._charge(cost):
                return self._record_error(
                    tool_name,
                    args,
                    error_budget_exhausted(),
                    contract=None,
                    budget_before=budget_before,
                )
            return self._record_error(
                tool_name,
                args,
                error_unknown_tool(tool_name, [c.name for c in self.contracts()]),
                contract=None,
                budget_before=budget_before,
            )

        cost = self.ir.resource_model.cost(contract)
        if not self._charge(cost):
            return self._record_error(
                tool_name,
                args,
                error_budget_exhausted(),
                contract=contract,
                budget_before=budget_before,
            )

        try:
            action = contract.canonicalize(args)
        except ActionValidationError as e:
            return self._record_error(
                tool_name,
                args,
                error_invalid_argument(
                    f"Invalid arguments for tool '{tool_name}'",
                    details={"reason": str(e), "tool": tool_name},
                ),
                contract=contract,
                budget_before=budget_before,
            )
        except Exception as e:
            return self._record_error(
                tool_name,
                args,
                error_invalid_argument(
                    f"Action canonicalization for tool '{tool_name}' raised exception",
                    details={"tool": tool_name, "error": str(e)},
                ),
                contract=contract,
                budget_before=budget_before,
            )

        result: ObjectiveResult | KernelTransition[State]
        try:
            if contract.kind == "submit":
                context = _EvaluationContext(self, deepcopy(self.instance))
                result = self.ir.objective.evaluate(
                    context,
                    tool_name,
                    action,
                    feedback=self.feedback,
                )
                if context.failure is not None:
                    raise ObjectiveInputError(context.failure, terminal=result.terminal)
            else:
                result = self.ir.call(self.instance, self.state, tool_name, action)
        except (KernelInputError, HypothesisInputError, ObjectiveInputError) as e:
            if isinstance(e, ObjectiveInputError) and e.terminal:
                self.done = True
                self.terminal_failure = True
            return self._record_error(
                tool_name,
                args,
                e.error,
                contract=contract,
                budget_before=budget_before,
                action=action,
            )
        except Exception as e:
            return self._record_error(
                tool_name,
                args,
                error_invalid_argument(
                    f"Tool '{tool_name}' raised exception",
                    details={
                        "tool": tool_name,
                        "error": (type(e).__name__ if contract.kind == "submit" else str(e)),
                    },
                ),
                contract=contract,
                budget_before=budget_before,
                action=action,
            )

        output: dict[str, Any] = {}
        if isinstance(result, KernelTransition):
            self.state = result.next_state
            if result.trap:
                self.trap_hit = True
                if self.ir.resource_model.trap_ends_episode:
                    self.done = True
                    self.ok = False
            output.update(result.observation)
            if contract.kind == "probe":
                output["t"] = self.tool_calls
        elif isinstance(result, ObjectiveResult):
            self.ok = bool(result.ok) and not self.trap_hit
            self.terminal_failure = result.terminal and not result.ok
            if result.ok or result.terminal:
                self.done = True
            output.update(result.observation)
            output["ok"] = self.ok
        else:
            return self._record_error(
                tool_name,
                args,
                error_invalid_argument(
                    f"Tool '{tool_name}' returned unsupported result type",
                    details={"tool": tool_name, "type": type(result).__name__},
                ),
                contract=contract,
                budget_before=budget_before,
                action=action,
            )

        if self.budget <= 0 and not self.ok:
            self.done = True
        output.update(self._runtime_fields())

        event = EpisodeEvent(
            event="tool_result",
            tool_name=tool_name,
            args=args,
            action=_json_action(action),
            output=output,
            error=None,
            kind=contract.kind,
            cost=budget_before - self.budget,
            budget_before=budget_before,
            budget_after=self.budget,
            queries_used=self.queries_used,
            tool_calls=self.tool_calls,
            done=self.done,
            ok=self.ok,
            trap_hit=self.trap_hit,
        )
        self.events.append(event)
        return event

    def replay(self, events: Sequence[EpisodeEvent | Mapping[str, Any]]) -> ReplayResult:
        runtime = EpisodeRuntime(self.ir, self.instance, feedback=self.feedback)
        replayed: list[EpisodeEvent] = []
        for idx, event in enumerate(events):
            event_dict = event.to_dict() if isinstance(event, EpisodeEvent) else dict(event)
            got = runtime.call_tool(
                str(event_dict.get("tool_name", "")),
                event_dict.get("args") if isinstance(event_dict.get("args"), Mapping) else {},
            )
            replayed.append(got)
            expected_action = event_dict.get("action")
            if got.action != expected_action:
                return ReplayResult(
                    ok=False,
                    events=replayed,
                    mismatch=f"event {idx}: action mismatch",
                )
            expected_output = event_dict.get("output")
            if got.output != expected_output:
                return ReplayResult(
                    ok=False,
                    events=replayed,
                    mismatch=f"event {idx}: output mismatch",
                )
        return ReplayResult(ok=True, events=replayed)

    def state_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "budget_init": self.budget_init,
            "queries_used": self.queries_used,
            "tool_calls": self.tool_calls,
            "evaluation_steps": self.evaluation_steps,
            "trap_hit": self.trap_hit,
            "ok": self.ok,
            "done": self.done,
            "terminal_failure": self.terminal_failure,
            "cs": self.state,
        }

    def _find_contract(self, tool_name: str) -> ToolActionContract[Any] | None:
        return next((c for c in self.contracts() if c.name == tool_name), None)

    def _charge(self, cost: int, *, tool_call: bool = True) -> bool:
        enough = self.budget >= cost
        charged = min(self.budget, cost)
        self.budget -= charged
        self.queries_used += charged
        if tool_call and (enough or charged > 0):
            self.tool_calls += 1
        if not enough:
            self.done = True
            self.ok = False
        return enough

    def _run_candidate(self, instance: TaskInstance, calls: Sequence[ToolCall]) -> Execution[State]:
        # An objective may test several candidates, but each starts fresh and
        # all work is charged to the same episode. Never copy exploration state.
        isolated = deepcopy(instance)
        state = self.ir.kernel.initial_state(isolated)
        steps: list[ExecutionStep] = []
        for call in calls:
            if self.done:
                raise ObjectiveInputError(error_episode_finished())
            context = self.ir.action_context(
                isolated,
                state,
                budget=self.budget,
                queries_used=self.queries_used,
                tool_calls=self.tool_calls,
                done=False,
            )
            contracts = self.ir.action_space.contracts_for_context(context)
            contract = next((c for c in contracts if c.name == call.name), None)
            if contract is None:
                # Objective tools are deliberately absent: evaluation cannot
                # recursively submit or obtain an uncharged grading oracle.
                raise ObjectiveInputError(
                    error_unknown_tool(call.name, [c.name for c in contracts])
                )
            cost = self.ir.resource_model.cost(contract)
            if not self._charge(cost, tool_call=False):
                raise ObjectiveInputError(error_budget_exhausted())
            try:
                action = contract.canonicalize(call.arguments)
            except ActionValidationError as error:
                raise ObjectiveInputError(error_invalid_argument(str(error))) from error
            transition = self.ir.call(isolated, state, call.name, action)
            self.evaluation_steps += 1
            self.trap_hit = self.trap_hit or transition.trap
            steps.append(
                ExecutionStep(
                    call.name,
                    _json_action(action),
                    deepcopy(transition.observation),
                    transition.trap,
                )
            )
            state = transition.next_state
            if transition.trap and self.ir.resource_model.trap_ends_episode:
                self.done = True
                self.ok = False
                raise ObjectiveInputError(error_episode_finished())
        return Execution(tuple(steps), state)

    def _runtime_fields(self) -> dict[str, Any]:
        return {
            "done": self.done,
            "budget_left": self.budget,
            "queries_used": self.queries_used,
            "trap_hit": self.trap_hit,
        }

    def _record_error(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        error: DedeuceError,
        *,
        contract: ToolActionContract[Any] | None,
        budget_before: int,
        action: Any | None = None,
    ) -> EpisodeEvent:
        if self.budget <= 0 and not self.ok:
            self.done = True
        output = {
            "error": error.to_dict(),
            **self._runtime_fields(),
        }
        if contract is not None and contract.kind == "submit":
            output["ok"] = False
        event = EpisodeEvent(
            event="tool_result",
            tool_name=tool_name,
            args=dict(args),
            action=_json_action(action),
            output=output,
            error=error.to_dict(),
            kind=None if contract is None else contract.kind,
            cost=budget_before - self.budget,
            budget_before=budget_before,
            budget_after=self.budget,
            queries_used=self.queries_used,
            tool_calls=self.tool_calls,
            done=self.done,
            ok=self.ok,
            trap_hit=self.trap_hit,
        )
        self.events.append(event)
        return event


@dataclass
class _EvaluationContext(Generic[State]):
    runtime: EpisodeRuntime[State]
    instance: TaskInstance
    failure: DedeuceError | None = None

    def run(self, calls: Sequence[ToolCall]) -> Execution[State]:
        try:
            return self.runtime._run_candidate(self.instance, calls)
        except (ObjectiveInputError, KernelInputError) as error:
            self.failure = error.error
            raise


def _json_action(action: Any) -> Any:
    return TypeAdapter(Any).dump_python(action, mode="json")
