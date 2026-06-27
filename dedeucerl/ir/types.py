"""Executable task intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from dedeucerl.ir.actions import ActionContext, ToolActionContract, ToolActionSpace
from dedeucerl.kernel.types import (
    KernelJudgment,
    KernelParam,
    KernelTransition,
    SystemKernel,
    TaskInstance,
)


class ObservationModel(Protocol):
    """Public and tool-observation model for a task."""

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]: ...


class HypothesisContract(Protocol):
    """Submission contract and judgment behavior."""

    def handles(self, tool_name: str) -> bool: ...

    def judge(
        self,
        instance: TaskInstance,
        tool_name: str,
        action: Any,
    ) -> KernelJudgment: ...


@dataclass(frozen=True)
class ResourceModel:
    """Budget, cost, and trap policy."""

    unknown_tool_cost: int = 1
    trap_ends_episode: bool = False
    cost_overrides: Mapping[str, int] = field(default_factory=dict)

    def cost(self, contract: ToolActionContract[Any]) -> int:
        return max(0, int(self.cost_overrides.get(contract.name, contract.cost)))


@dataclass(frozen=True)
class FeedbackModel:
    """Incorrect-submission feedback policy."""

    reveal_counterexample: bool = True

    def counterexample(
        self,
        *,
        feedback_enabled: bool,
        judgment: KernelJudgment,
        runtime_ok: bool,
    ) -> Any | None:
        if not self.reveal_counterexample or not feedback_enabled or runtime_ok:
            return None
        return judgment.counterexample


class TaskGeneratorSpec(Protocol):
    """Deterministic task instance generator."""

    params: Mapping[str, KernelParam]

    def sample(self, *, seed: int, budget: int, **kwargs: Any) -> TaskInstance: ...


class Renderer(Protocol):
    """Surface renderer for prompts, traces, or human views."""

    def render(
        self,
        ir: TaskIR,
        instance: TaskInstance,
        contracts: Sequence[ToolActionContract[Any]],
        *,
        feedback: bool = False,
    ) -> Any: ...


@dataclass(frozen=True)
class TaskIR:
    """Executable task contract compiled by runtimes and surfaces."""

    name: str
    version: str
    kernel: SystemKernel
    action_space: ToolActionSpace
    observation_model: ObservationModel
    hypothesis_contract: HypothesisContract
    resource_model: ResourceModel
    feedback_model: FeedbackModel
    generator: TaskGeneratorSpec
    renderers: Mapping[str, Renderer] = field(default_factory=dict)

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        return self.observation_model.public_observation(instance)

    def action_context(
        self,
        instance: TaskInstance,
        state: Any,
        *,
        budget: int,
        queries_used: int,
        tool_calls: int,
        done: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActionContext:
        return ActionContext(
            instance=instance,
            state=state,
            budget=budget,
            queries_used=queries_used,
            tool_calls=tool_calls,
            done=done,
            metadata={} if metadata is None else dict(metadata),
        )

    def action_contracts(self, context: ActionContext) -> list[ToolActionContract[Any]]:
        return self.action_space.contracts_for_context(context)

    def tool_schemas(self, context: ActionContext) -> list[dict[str, Any]]:
        return self.action_space.to_tool_schemas(context)

    def handles_submission(self, tool_name: str) -> bool:
        return self.hypothesis_contract.handles(tool_name)

    def call(
        self,
        instance: TaskInstance,
        state: Any,
        tool_name: str,
        action: Any,
    ) -> KernelTransition | KernelJudgment:
        if self.handles_submission(tool_name):
            return self.hypothesis_contract.judge(instance, tool_name, action)
        return self.kernel.call(instance, state, tool_name, action)
