from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dedeucerl.ir import EnumSpace, FeedbackModel, ResourceModel, TaskIR
from dedeucerl.kernel.types import (
    KernelJudgment,
    KernelParam,
    KernelTransition,
    TaskInstance,
    ToolContract,
)
from dedeucerl.runtime import EpisodeRuntime


@dataclass(frozen=True)
class FakeKernel:
    name = "fake"
    version = "0.1"

    def initial_state(self, instance: TaskInstance) -> int:
        _ = instance
        return 0

    def call(
        self,
        instance: TaskInstance,
        state: Any,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> KernelTransition:
        _ = instance
        if tool_name != "probe":
            raise KeyError(tool_name)
        return KernelTransition(next_state=int(state) + 1, observation={"seen": args["x"]})


@dataclass(frozen=True)
class FakeObservation:
    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        return {"task": instance.id}


@dataclass(frozen=True)
class FakeHypothesis:
    def tool_contracts(self, instance: TaskInstance) -> Sequence[ToolContract]:
        _ = instance
        return (
            ToolContract(
                name="submit",
                kind="submit",
                description="Submit a boolean answer.",
                args_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "boolean"}},
                    "required": ["answer"],
                },
                return_schema={"type": "object"},
                cost=2,
            ),
        )

    def handles(self, tool_name: str) -> bool:
        return tool_name == "submit"

    def judge(
        self,
        instance: TaskInstance,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> KernelJudgment:
        _ = (instance, tool_name)
        ok = bool(args["answer"])
        return KernelJudgment(ok=ok, observation={"ok": ok}, counterexample={"answer": True})


@dataclass(frozen=True)
class FakeGenerator:
    params: Mapping[str, KernelParam]

    def sample(self, *, seed: int, budget: int, **kwargs: Any) -> TaskInstance:
        _ = kwargs
        return TaskInstance(
            id=f"fake-{seed}",
            seed=seed,
            kernel_name="fake",
            kernel_version="0.1",
            private={},
            params={},
            budget=budget,
        )


def _fake_ir(*, reveal_counterexample: bool = True) -> TaskIR:
    return TaskIR(
        name="fake",
        version="0.1",
        kernel=FakeKernel(),
        action_space=EnumSpace("x", ["a", "b"]),
        observation_model=FakeObservation(),
        hypothesis_contract=FakeHypothesis(),
        resource_model=ResourceModel(unknown_tool_cost=3),
        feedback_model=FeedbackModel(reveal_counterexample=reveal_counterexample),
        generator=FakeGenerator(params={}),
        tools=(
            ToolContract(
                name="probe",
                kind="probe",
                description="Probe with x.",
                args_schema={
                    "type": "object",
                    "properties": {"x": {"type": "string", "enum": ["a", "b"]}},
                    "required": ["x"],
                },
                return_schema={"type": "object"},
                cost=1,
            ),
        ),
    )


def test_task_ir_contracts_and_public_observation() -> None:
    ir = _fake_ir()
    instance = ir.generator.sample(seed=1, budget=5)

    assert ir.action_space.contains("a")
    assert not ir.action_space.contains("z")
    assert ir.public_observation(instance) == {"task": "fake-1"}
    assert [contract.name for contract in ir.tool_contracts(instance, 0)] == ["probe", "submit"]


def test_runtime_uses_ir_resources_and_feedback() -> None:
    ir = _fake_ir(reveal_counterexample=False)
    instance = ir.generator.sample(seed=1, budget=5)
    runtime = EpisodeRuntime(ir, instance, feedback=True)

    unknown = runtime.call_tool("missing", {})
    assert unknown.cost == 3
    assert runtime.budget == 2

    failed = runtime.call_tool("submit", {"answer": False})
    assert failed.output["ok"] is False
    assert failed.output["counterexample"] is None
    assert runtime.budget == 0
