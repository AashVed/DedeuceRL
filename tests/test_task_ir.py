from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dedeucerl.ir import (
    EnumSpace,
    FeedbackModel,
    MaskedSpace,
    ProductSpace,
    ResourceModel,
    TaskIR,
    ToolActionContract,
    ToolActionSpace,
)
from dedeucerl.kernel.types import (
    KernelJudgment,
    KernelParam,
    KernelTransition,
    TaskInstance,
)
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.utils.errors import ErrorCode


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
        action: Any,
    ) -> KernelTransition:
        _ = instance
        if tool_name != "probe":
            raise KeyError(tool_name)
        return KernelTransition(next_state=int(state) + 1, observation={"seen": action["x"]})


@dataclass(frozen=True)
class FakeObservation:
    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        return {"task": instance.id}


@dataclass(frozen=True)
class FakeHypothesis:
    def handles(self, tool_name: str) -> bool:
        return tool_name == "submit"

    def judge(
        self,
        instance: TaskInstance,
        tool_name: str,
        action: Any,
    ) -> KernelJudgment:
        _ = (instance, tool_name)
        ok = bool(action["answer"])
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
        action_space=ToolActionSpace(
            contracts=(
                ToolActionContract(
                    name="probe",
                    kind="probe",
                    description="Probe with x.",
                    action_space=ProductSpace(
                        "probe_args",
                        {"x": EnumSpace("x", ["a", "b"])},
                    ),
                    return_schema={"type": "object"},
                    cost=1,
                ),
                ToolActionContract(
                    name="submit",
                    kind="submit",
                    description="Submit a boolean answer.",
                    action_space=ProductSpace(
                        "submit_args",
                        {"answer": EnumSpace("answer", [True, False])},
                    ),
                    return_schema={"type": "object"},
                    cost=2,
                ),
            )
        ),
        observation_model=FakeObservation(),
        hypothesis_contract=FakeHypothesis(),
        resource_model=ResourceModel(unknown_tool_cost=3),
        feedback_model=FeedbackModel(reveal_counterexample=reveal_counterexample),
        generator=FakeGenerator(params={}),
    )


def test_task_ir_contracts_and_public_observation() -> None:
    ir = _fake_ir()
    instance = ir.generator.sample(seed=1, budget=5)
    context = ir.action_context(
        instance,
        0,
        budget=instance.budget,
        queries_used=0,
        tool_calls=0,
        done=False,
    )

    probe = ir.action_space.get("probe", context)
    assert probe is not None
    assert probe.action_space.contains({"x": "a"})
    assert not probe.action_space.contains({"x": "z"})
    assert ir.public_observation(instance) == {"task": "fake-1"}
    assert [contract.name for contract in ir.action_contracts(context)] == ["probe", "submit"]


def test_runtime_uses_ir_resources_and_feedback() -> None:
    ir = _fake_ir(reveal_counterexample=False)
    instance = ir.generator.sample(seed=1, budget=5)
    runtime = EpisodeRuntime(ir, instance, feedback=True)

    unknown = runtime.call_tool("missing", {})
    assert unknown.cost == 3
    assert runtime.budget == 2

    failed = runtime.call_tool("submit", {"answer": False})
    assert failed.output["ok"] is False
    assert failed.action == {"answer": False}
    assert failed.output["counterexample"] is None
    assert runtime.budget == 0


def test_runtime_rejects_masked_action_before_kernel_dispatch() -> None:
    calls: list[Any] = []

    @dataclass(frozen=True)
    class RecordingKernel:
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
            action: Any,
        ) -> KernelTransition:
            _ = (instance, state, tool_name)
            calls.append(action)
            return KernelTransition(next_state=0, observation={"seen": action["x"]})

    ir = TaskIR(
        name="fake",
        version="0.1",
        kernel=RecordingKernel(),
        action_space=ToolActionSpace(
            contracts=(
                ToolActionContract(
                    name="probe",
                    kind="probe",
                    description="Probe with masked x.",
                    action_space=MaskedSpace(
                        "masked_probe_args",
                        ProductSpace("probe_args", {"x": EnumSpace("x", ["a", "b"])}),
                        lambda context, action: context.budget > 0 and action["x"] == "a",
                    ),
                    return_schema={"type": "object"},
                    cost=1,
                ),
            )
        ),
        observation_model=FakeObservation(),
        hypothesis_contract=FakeHypothesis(),
        resource_model=ResourceModel(),
        feedback_model=FeedbackModel(),
        generator=FakeGenerator(params={}),
    )
    instance = ir.generator.sample(seed=1, budget=2)
    runtime = EpisodeRuntime(ir, instance)

    event = runtime.call_tool("probe", {"x": "b"})

    assert event.error is not None
    assert event.action is None
    assert runtime.budget == 1
    assert calls == []


def test_runtime_wraps_action_canonicalization_exceptions() -> None:
    @dataclass(frozen=True)
    class RecordingKernel:
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
            action: Any,
        ) -> KernelTransition:
            _ = (instance, state, tool_name, action)
            raise AssertionError("kernel should not be called")

    def broken_mask(context: Any, action: Any) -> bool:
        _ = (context, action)
        raise RuntimeError("mask failed")

    ir = TaskIR(
        name="fake",
        version="0.1",
        kernel=RecordingKernel(),
        action_space=ToolActionSpace(
            contracts=(
                ToolActionContract(
                    name="probe",
                    kind="probe",
                    description="Probe with broken mask.",
                    action_space=MaskedSpace(
                        "broken_probe_args",
                        ProductSpace("probe_args", {"x": EnumSpace("x", ["a"])}),
                        broken_mask,
                    ),
                    return_schema={"type": "object"},
                    cost=1,
                ),
            )
        ),
        observation_model=FakeObservation(),
        hypothesis_contract=FakeHypothesis(),
        resource_model=ResourceModel(),
        feedback_model=FeedbackModel(),
        generator=FakeGenerator(params={}),
    )
    instance = ir.generator.sample(seed=1, budget=2)
    runtime = EpisodeRuntime(ir, instance)

    event = runtime.call_tool("probe", {"x": "a"})

    assert event.error is not None
    assert event.error["code"] == ErrorCode.INVALID_ARGUMENT.value
    assert "mask failed" in event.error["details"]["error"]
    assert event.action is None
    assert runtime.budget == 1
