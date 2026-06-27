"""Mealy TaskIR reference implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from dedeucerl.ir.actions import (
    EnumSpace,
    ProductSpace,
    ToolActionContract,
    ToolActionSpace,
)
from dedeucerl.ir.hypotheses import FiniteTransducerIsomorphismContract
from dedeucerl.ir.types import FeedbackModel, ResourceModel, TaskIR
from dedeucerl.kernel.mealy import (
    ALPHABET,
    OUTPUTS,
    MealyKernel,
    generate_mealy_system,
)
from dedeucerl.kernel.types import (
    KernelParam,
    TaskInstance,
)


MEALY_TASK_NAME = "mealy"
MEALY_TASK_VERSION = "2.2"


@dataclass(frozen=True)
class MealyObservationModel:
    """Public observation model for Mealy tasks."""

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        table = instance.private["table"]
        return {
            "kernel": MEALY_TASK_NAME,
            "alphabet": list(ALPHABET),
            "outputs": list(OUTPUTS),
            "n_states": int(table["n"]),
            "budget": int(instance.budget),
            "trap": bool(instance.params.get("trap", True)),
        }


@dataclass(frozen=True)
class MealyGenerator:
    """Deterministic TaskIR generator for Mealy task instances."""

    params: Mapping[str, KernelParam] = field(
        default_factory=lambda: {
            "n_states": KernelParam(
                type="int",
                description="Number of states in the hidden Mealy machine.",
                default=3,
                choices=None,
            ),
            "trap": KernelParam(
                type="bool",
                description="Whether to include trap transitions.",
                default=True,
            ),
        }
    )

    def sample(self, *, seed: int, budget: int, **kwargs: Any) -> TaskInstance:
        n_states = int(kwargs.get("n_states", self.params["n_states"].default))
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if int(budget) < 0:
            raise ValueError("budget must be >= 0")
        trap = bool(kwargs.get("trap", self.params["trap"].default))
        private = generate_mealy_system(seed=seed, n_states=n_states, trap=trap)
        return TaskInstance(
            id=f"mealy-{seed}",
            seed=int(seed),
            kernel_name=MEALY_TASK_NAME,
            kernel_version=MEALY_TASK_VERSION,
            private=private,
            params={"n_states": n_states, "trap": trap},
            budget=int(budget),
        )


@dataclass(frozen=True)
class MealyPromptRenderer:
    """Prompt renderer for Mealy TaskIR instances."""

    def render(
        self,
        ir: TaskIR,
        instance: TaskInstance,
        contracts: Sequence[ToolActionContract[Any]],
        *,
        feedback: bool = False,
    ) -> list[dict[str, Any]]:
        tools_text = "\n".join(_format_tool(contract) for contract in contracts)
        system = (
            "You are an autonomous tool-using agent solving a hidden-system identification task.\n"
            "Return only function tool calls; do not answer in natural language.\n\n"
            "Episode semantics:\n"
            "- Probe/diagnostic tools may reveal observations and can change hidden state.\n"
            "- Submit tools judge a hypothesis. Correct submissions end the episode.\n"
            "- Every valid or invalid tool call can consume budget.\n"
            "- Budget exhaustion ends the episode.\n"
            "- Trap hits are reported in tool results and may affect reward.\n\n"
            "Mealy task: identify a finite-state transducer. "
            'Submit JSON shaped like {"n": <states>, "start": 0, '
            '"trans": {"0": {"A": [next_state, output], ...}, ...}}.\n\n'
            "Tools:\n"
            f"{tools_text}"
        )
        if feedback:
            system += "\n\nFeedback mode: incorrect submissions may include a counterexample."

        user = (
            "OBSERVATION:\n"
            + json.dumps(ir.public_observation(instance), sort_keys=True)
            + "\n\nUse the available tools to identify the hidden system, then submit a complete hypothesis."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_mealy_ir() -> TaskIR:
    action_space = ToolActionSpace(
        contracts=(
            ToolActionContract(
                name="act",
                kind="probe",
                description="Execute one input symbol on the hidden Mealy machine.",
                action_space=ProductSpace(
                    name="act_args",
                    fields={
                        "symbol": EnumSpace(
                            "symbol",
                            ALPHABET,
                            description="Input symbol to execute.",
                        )
                    },
                ),
                return_schema={
                    "type": "object",
                    "properties": {
                        "out": {"type": "integer"},
                        "budget_left": {"type": "integer"},
                        "t": {"type": "integer"},
                        "trap_hit": {"type": "boolean"},
                        "queries_used": {"type": "integer"},
                    },
                    "required": ["out", "budget_left", "t", "trap_hit", "queries_used"],
                },
                cost=1,
            ),
        )
    )
    return TaskIR(
        name=MEALY_TASK_NAME,
        version=MEALY_TASK_VERSION,
        kernel=MealyKernel(),
        action_space=action_space,
        observation_model=MealyObservationModel(),
        hypothesis_contract=FiniteTransducerIsomorphismContract(
            description="Submit a complete Mealy transition table as a JSON string.",
            alphabet=ALPHABET,
            outputs=OUTPUTS,
            start_const=0,
        ),
        resource_model=ResourceModel(unknown_tool_cost=1, trap_ends_episode=False),
        feedback_model=FeedbackModel(reveal_counterexample=True),
        generator=MealyGenerator(),
        renderers={"prompt": MealyPromptRenderer()},
    )


def _format_tool(contract: ToolActionContract[Any]) -> str:
    schema = contract.to_tool_schema()["parameters"]
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    args = []
    if isinstance(props, dict):
        for name, schema in props.items():
            if not isinstance(schema, dict):
                continue
            typ = schema.get("type", "any")
            enum = schema.get("enum")
            suffix = " required" if name in required else " optional"
            enum_text = f" one of {enum}" if enum else ""
            args.append(f"{name}: {typ}{enum_text}{suffix}")
    return f"- {contract.name}({', '.join(args)}) [{contract.kind}, cost={contract.cost}]"
