"""Executable task intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from dedeucerl.kernel.types import (
    KernelJudgment,
    KernelParam,
    KernelTransition,
    SystemKernel,
    TaskInstance,
    ToolContract,
)


class ActionSpace(Protocol):
    """Validated action vocabulary exposed by a task."""

    name: str

    def contains(self, value: Any) -> bool: ...

    def json_schema(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class EnumSpace:
    """Finite enum action/output space."""

    name: str
    values: Sequence[Any]

    def contains(self, value: Any) -> bool:
        return value in self.values

    def json_schema(self) -> Mapping[str, Any]:
        schema_type = "string"
        if self.values and all(isinstance(v, int) and not isinstance(v, bool) for v in self.values):
            schema_type = "integer"
        elif self.values and all(isinstance(v, bool) for v in self.values):
            schema_type = "boolean"
        return {"type": schema_type, "enum": list(self.values)}


class ObservationModel(Protocol):
    """Public and tool-observation model for a task."""

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]: ...


class HypothesisContract(Protocol):
    """Submission contract and judgment behavior."""

    def tool_contracts(self, instance: TaskInstance) -> Sequence[ToolContract]: ...

    def handles(self, tool_name: str) -> bool: ...

    def judge(
        self,
        instance: TaskInstance,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> KernelJudgment: ...


@dataclass(frozen=True)
class ResourceModel:
    """Budget, cost, and trap policy."""

    unknown_tool_cost: int = 1
    trap_ends_episode: bool = False
    cost_overrides: Mapping[str, int] = field(default_factory=dict)

    def cost(self, contract: ToolContract) -> int:
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
        contracts: Sequence[ToolContract],
        *,
        feedback: bool = False,
    ) -> Any: ...


@dataclass(frozen=True)
class TaskIR:
    """Executable task contract compiled by runtimes and surfaces."""

    name: str
    version: str
    kernel: SystemKernel
    action_space: ActionSpace
    observation_model: ObservationModel
    hypothesis_contract: HypothesisContract
    resource_model: ResourceModel
    feedback_model: FeedbackModel
    generator: TaskGeneratorSpec
    tools: Sequence[ToolContract] = field(default_factory=tuple)
    renderers: Mapping[str, Renderer] = field(default_factory=dict)

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        return self.observation_model.public_observation(instance)

    def tool_contracts(self, instance: TaskInstance, state: Any) -> list[ToolContract]:
        _ = state
        return [*self.tools, *self.hypothesis_contract.tool_contracts(instance)]

    def handles_submission(self, tool_name: str) -> bool:
        return self.hypothesis_contract.handles(tool_name)

    def call(
        self,
        instance: TaskInstance,
        state: Any,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> KernelTransition | KernelJudgment:
        if self.handles_submission(tool_name):
            return self.hypothesis_contract.judge(instance, tool_name, args)
        return self.kernel.call(instance, state, tool_name, args)
