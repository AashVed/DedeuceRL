"""Pure semantic kernel contracts for DedeuceRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Mapping, Protocol, Sequence, TypeVar

from dedeucerl.utils.errors import DedeuceError

State = TypeVar("State")
State_co = TypeVar("State_co", covariant=True)


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
class KernelTransition(Generic[State_co]):
    """Result of a state-changing probe or diagnostic operation."""

    next_state: State_co
    observation: Mapping[str, Any]
    trap: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


class KernelInputError(Exception):
    """A user-facing kernel input error returned through runtime envelopes."""

    def __init__(self, error: DedeuceError):
        super().__init__(error.message)
        self.error = error


class SystemKernel(Protocol[State]):
    """Pure hidden-system semantics.

    Implementations must not depend on Verifiers, datasets, provider adapters,
    prompts, CLIs, or TaskIR surface compilers.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def initial_state(self, instance: TaskInstance) -> State: ...

    def call(
        self,
        instance: TaskInstance,
        state: State,
        tool_name: str,
        action: Any,
    ) -> KernelTransition[State]: ...
