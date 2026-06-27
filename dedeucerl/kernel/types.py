"""Pure semantic kernel contracts for DedeuceRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from dedeucerl.utils.errors import DedeuceError


ToolKind = Literal["probe", "submit", "diagnostic"]


@dataclass(frozen=True)
class TaskInstance:
    """A complete hidden-system task instance.

    `private` is intentionally opaque to surfaces. Kernels own its shape.
    """

    id: str
    seed: int | None
    kernel_name: str
    kernel_version: str
    private: Mapping[str, Any]
    params: Mapping[str, Any]
    budget: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelParam:
    """User-facing sampler parameter metadata."""

    type: Literal["int", "float", "bool", "str", "json"]
    description: str
    default: Any = None
    choices: Sequence[Any] | None = None


@dataclass(frozen=True)
class ToolContract:
    """A semantic operation exposed by a kernel."""

    name: str
    kind: ToolKind
    description: str
    args_schema: Mapping[str, Any]
    return_schema: Mapping[str, Any]
    cost: int = 1


@dataclass(frozen=True)
class KernelTransition:
    """Result of a state-changing probe or diagnostic operation."""

    next_state: Any
    observation: Mapping[str, Any]
    trap: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelJudgment:
    """Result of a hypothesis submission operation."""

    ok: bool
    observation: Mapping[str, Any]
    counterexample: Any | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


class KernelInputError(Exception):
    """A user-facing kernel input error returned through runtime envelopes."""

    def __init__(self, error: DedeuceError):
        super().__init__(error.message)
        self.error = error


class SystemKernel(Protocol):
    """Pure hidden-system semantics.

    Implementations must not depend on Verifiers, datasets, provider adapters,
    prompts, or CLIs.
    """

    name: str
    version: str

    def initial_state(self, instance: TaskInstance) -> Any: ...

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]: ...

    def tool_contracts(self, instance: TaskInstance, state: Any) -> Sequence[ToolContract]: ...

    def call(
        self,
        instance: TaskInstance,
        state: Any,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> KernelTransition | KernelJudgment: ...


class TaskSampler(Protocol):
    """Deterministic task sampler for a kernel."""

    kernel: SystemKernel
    params: Mapping[str, KernelParam]

    def sample(self, *, seed: int, budget: int, **kwargs: Any) -> TaskInstance: ...
