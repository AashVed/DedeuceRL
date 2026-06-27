"""Generic episode runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from dedeucerl.ir.actions import ActionContext, ActionValidationError, ToolActionContract
from dedeucerl.ir.hypotheses import HypothesisInputError, HypothesisJudgment
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
class EpisodeRuntime:
    ir: TaskIR
    instance: TaskInstance
    feedback: bool = False
    state: Any = None
    budget: int = 0
    budget_init: int = 0
    queries_used: int = 0
    tool_calls: int = 0
    done: bool = False
    ok: bool = False
    trap_hit: bool = False
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
                cost=0,
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
                    cost=cost,
                    budget_before=budget_before,
                )
            return self._record_error(
                tool_name,
                args,
                error_unknown_tool(tool_name, [c.name for c in self.contracts()]),
                contract=None,
                cost=cost,
                budget_before=budget_before,
            )

        cost = self.ir.resource_model.cost(contract)
        if not self._charge(cost):
            return self._record_error(
                tool_name,
                args,
                error_budget_exhausted(),
                contract=contract,
                cost=cost,
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
                cost=cost,
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
                cost=cost,
                budget_before=budget_before,
            )

        try:
            if contract.kind == "submit":
                result = self.ir.submit(self.instance, tool_name, action)
            else:
                result = self.ir.call(self.instance, self.state, tool_name, action)
        except (KernelInputError, HypothesisInputError) as e:
            return self._record_error(
                tool_name,
                args,
                e.error,
                contract=contract,
                cost=cost,
                budget_before=budget_before,
                action=action,
            )
        except Exception as e:
            return self._record_error(
                tool_name,
                args,
                error_invalid_argument(
                    f"Tool '{tool_name}' raised exception",
                    details={"tool": tool_name, "error": str(e)},
                ),
                contract=contract,
                cost=cost,
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
            output.update(self._runtime_fields())
            if contract.kind == "probe":
                output["t"] = self.tool_calls
        elif isinstance(result, HypothesisJudgment):
            self.ok = bool(result.ok) and not self.trap_hit
            if result.ok:
                self.done = True
            output.update(result.observation)
            output["ok"] = self.ok
            output.update(self._runtime_fields())
            output["counterexample"] = self.ir.feedback_model.counterexample(
                feedback_enabled=self.feedback,
                judgment=result,
                runtime_ok=self.ok,
            )
            if result.distance is not None:
                output["distance"] = result.distance
        else:
            return self._record_error(
                tool_name,
                args,
                error_invalid_argument(
                    f"Tool '{tool_name}' returned unsupported result type",
                    details={"tool": tool_name, "type": type(result).__name__},
                ),
                contract=contract,
                cost=cost,
                budget_before=budget_before,
                action=action,
            )

        if self.budget <= 0 and not self.ok:
            self.done = True

        event = EpisodeEvent(
            event="tool_result",
            tool_name=tool_name,
            args=args,
            action=action,
            output=output,
            error=None,
            kind=contract.kind,
            cost=cost,
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
            "trap_hit": self.trap_hit,
            "ok": self.ok,
            "done": self.done,
            "cs": self.state,
        }

    def _find_contract(self, tool_name: str) -> ToolActionContract[Any] | None:
        return next((c for c in self.contracts() if c.name == tool_name), None)

    def _charge(self, cost: int) -> bool:
        if self.budget < cost:
            self.budget = 0
            self.done = True
            self.ok = False
            return False
        self.budget -= cost
        self.queries_used += cost
        self.tool_calls += 1
        return True

    def _runtime_fields(self) -> dict[str, Any]:
        return {
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
        cost: int,
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
            action=action,
            output=output,
            error=error.to_dict(),
            kind=None if contract is None else contract.kind,
            cost=cost,
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
