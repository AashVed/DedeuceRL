"""Executable task intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar

from dedeucerl.ir.actions import ActionContext, ToolActionContract, ToolActionSpace
from dedeucerl.ir.objectives import ObjectiveContract
from dedeucerl.kernel.types import (
    KernelParam,
    KernelTransition,
    SystemKernel,
    TaskInstance,
)

State = TypeVar("State")


class ObservationModel(Protocol):
    """Public and tool-observation model for a task."""

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ResourceModel:
    """Budget, cost, and trap policy."""

    unknown_tool_cost: int = 1
    trap_ends_episode: bool = False
    cost_overrides: Mapping[str, int] = field(default_factory=dict)

    def cost(self, contract: ToolActionContract[Any]) -> int:
        return max(0, int(self.cost_overrides.get(contract.name, contract.cost)))


class TaskGeneratorSpec(Protocol):
    """Deterministic task instance generator."""

    @property
    def params(self) -> Mapping[str, KernelParam]: ...

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
class TaskIR(Generic[State]):
    """Executable task contract compiled by runtimes and surfaces."""

    name: str
    version: str
    kernel: SystemKernel[State]
    action_space: ToolActionSpace
    observation_model: ObservationModel
    objective: ObjectiveContract[State]
    resource_model: ResourceModel
    generator: TaskGeneratorSpec
    renderers: Mapping[str, Renderer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        non_submit_contracts = tuple(self.action_space.contracts)
        submit_contracts = tuple(self.objective.tool_contracts())

        bad_non_submit = [
            contract.name for contract in non_submit_contracts if contract.kind == "submit"
        ]
        if bad_non_submit:
            raise ValueError(
                "submit tools must be declared by the objective: "
                f"{bad_non_submit!r}"
            )

        bad_submit = [
            contract.name for contract in submit_contracts if contract.kind != "submit"
        ]
        if bad_submit:
            raise ValueError(
                "objectives may only declare submit tools: "
                f"{bad_submit!r}"
            )

        names = [contract.name for contract in (*non_submit_contracts, *submit_contracts)]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate TaskIR tool names: {names!r}")

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        return self.observation_model.public_observation(instance)

    def action_context(
        self,
        instance: TaskInstance,
        state: State,
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
        submit_contracts = [
            contract.mask(context) for contract in self.objective.tool_contracts()
        ]
        return [*self.action_space.contracts_for_context(context), *submit_contracts]

    def tool_schemas(self, context: ActionContext) -> list[dict[str, Any]]:
        return [contract.to_tool_schema() for contract in self.action_contracts(context)]

    def call(
        self,
        instance: TaskInstance,
        state: State,
        tool_name: str,
        action: Any,
    ) -> KernelTransition[State]:
        return self.kernel.call(instance, state, tool_name, action)
