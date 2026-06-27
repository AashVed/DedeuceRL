"""Kernel registry."""

from __future__ import annotations

from dataclasses import dataclass

from dedeucerl.kernel.mealy import MealyKernel, MealySampler
from dedeucerl.kernel.types import SystemKernel, TaskSampler


@dataclass(frozen=True)
class KernelEntry:
    name: str
    kernel: SystemKernel
    sampler: TaskSampler


MEALY_KERNEL = MealyKernel()
MEALY_SAMPLER = MealySampler()

KERNEL_REGISTRY: dict[str, KernelEntry] = {
    "mealy": KernelEntry(name="mealy", kernel=MEALY_KERNEL, sampler=MEALY_SAMPLER),
}


def get_kernel_entry(name: str) -> KernelEntry:
    try:
        return KERNEL_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(KERNEL_REGISTRY))
        raise KeyError(f"Unknown kernel '{name}'. Available kernels: {available}") from e
