"""A non-Mealy objective: configure a private project with viewer access.

Run with ``python examples/custom_objective.py``. No runtime or surface changes
are necessary to define a new submission, private evaluator, and feedback type.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from dedeucerl.ir import (
    EnumSpace,
    Evaluation,
    EvaluationContext,
    FeedbackSpec,
    Objective,
    ProductSpace,
    ResourceModel,
    TaskIR,
    ToolActionContract,
    ToolActionSpace,
    ToolCall,
    TypedSpace,
)
from dedeucerl.kernel.types import KernelParam, KernelTransition, TaskInstance
from dedeucerl.runtime import EpisodeRuntime

Operation = Literal["login", "verify", "create", "grant_viewer", "publish"]


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operations: list[Operation] = Field(min_length=1)


@dataclass(frozen=True)
class WorkflowState:
    authenticated: bool = False
    verified: bool = False
    project: bool = False
    viewer: bool = False
    public: bool = False


class WorkflowKernel:
    name = "workflow"
    version = "1.0"

    def initial_state(self, instance: TaskInstance) -> WorkflowState:
        return WorkflowState()

    def call(
        self,
        instance: TaskInstance,
        state: WorkflowState,
        tool_name: str,
        action: Mapping[str, Any],
    ) -> KernelTransition[WorkflowState]:
        operation = action["operation"]
        if operation == "login":
            state = replace(state, authenticated=True)
        elif not state.authenticated:
            return KernelTransition(state, {"status": "unauthorized"})
        elif operation == "verify":
            state = replace(state, verified=True)
        elif operation == "create":
            if instance.private["requires_verification"] and not state.verified:
                return KernelTransition(state, {"status": "verification_required"})
            state = replace(state, project=True)
        elif not state.project:
            return KernelTransition(state, {"status": "project_required"})
        elif operation == "grant_viewer":
            state = replace(state, viewer=True)
        elif operation == "publish":
            state = replace(state, public=True)
        return KernelTransition(state, {"status": "ok"})


class WorkflowObservation:
    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        return {
            "goal": "Create a private project and grant viewer access.",
            "evaluation_start": "initial",
            "budget": instance.budget,
        }


class WorkflowGenerator:
    params: Mapping[str, KernelParam] = {}

    def sample(self, *, seed: int, budget: int, **kwargs: Any) -> TaskInstance:
        if budget < 5:
            raise ValueError("budget must cover a complete workflow plus submission")
        return TaskInstance(
            id=f"workflow-{seed}",
            seed=seed,
            kernel_name="workflow",
            kernel_version="1.0",
            budget=budget,
            params={},
            private={"requires_verification": bool(seed % 2)},
        )


class WorkflowFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    missing: list[Literal["project", "viewer"]]
    unexpected_public_access: bool


def evaluate_workflow(
    context: EvaluationContext[WorkflowState],
    plan: WorkflowPlan,
) -> Evaluation[WorkflowState]:
    execution = context.run(
        [ToolCall("request", {"operation": operation}) for operation in plan.operations]
    )
    state = execution.final_state
    return Evaluation(state.project and state.viewer and not state.public, state)


def workflow_feedback(state: WorkflowState) -> WorkflowFeedback:
    missing: list[Literal["project", "viewer"]] = []
    if not state.project:
        missing.append("project")
    if not state.viewer:
        missing.append("viewer")
    return WorkflowFeedback(missing=missing, unexpected_public_access=state.public)


def build_workflow_ir() -> TaskIR[WorkflowState]:
    objective: Objective[WorkflowState, WorkflowPlan, WorkflowState, WorkflowFeedback] = Objective(
        name="private_project",
        version="1.0",
        submission=ToolActionContract[WorkflowPlan](
            name="submit_workflow",
            kind="submit",
            cost=1,
            description="Try a workflow from initial state; costs 1 plus each operation.",
            action_space=TypedSpace("workflow_plan", WorkflowPlan),
            return_schema={},
        ),
        evaluator=evaluate_workflow,
        feedback=FeedbackSpec(WorkflowFeedback, workflow_feedback),
    )
    return TaskIR(
        name="workflow",
        version="1.0",
        kernel=WorkflowKernel(),
        objective=objective,
        action_space=ToolActionSpace(
            (
                ToolActionContract(
                    name="request",
                    kind="probe",
                    description="Interact with the unknown service.",
                    action_space=ProductSpace(
                        "request",
                        {
                            "operation": EnumSpace(
                                "operation",
                                ["login", "verify", "create", "grant_viewer", "publish"],
                            )
                        },
                    ),
                    return_schema={"type": "object"},
                    cost=1,
                ),
            )
        ),
        observation_model=WorkflowObservation(),
        resource_model=ResourceModel(),
        generator=WorkflowGenerator(),
    )


if __name__ == "__main__":
    ir = build_workflow_ir()
    runtime = EpisodeRuntime(ir, ir.generator.sample(seed=1, budget=20), feedback=True)
    for operations in (["login", "create"], ["login", "verify", "create", "grant_viewer"]):
        print(runtime.call_tool("submit_workflow", {"operations": operations}).output)
