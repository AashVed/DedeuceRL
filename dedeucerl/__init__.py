"""DedeuceRL: kernel/runtime/surface framework for active identification."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("dedeucerl")
except PackageNotFoundError:  # pragma: no cover
    # Source checkout without an installed distribution.
    __version__ = "0.0.0"

from dedeucerl.core.rubric import make_rubric, reward_identification
from dedeucerl.kernel import (
    KERNEL_REGISTRY,
    KernelJudgment,
    KernelTransition,
    MealyKernel,
    SystemKernel,
    TaskInstance,
    ToolContract,
)
from dedeucerl.runtime import EpisodeRuntime

__all__ = [
    "__version__",
    "EpisodeRuntime",
    "KERNEL_REGISTRY",
    "KernelJudgment",
    "KernelTransition",
    "MealyKernel",
    "SystemKernel",
    "TaskInstance",
    "ToolContract",
    "make_rubric",
    "reward_identification",
]
