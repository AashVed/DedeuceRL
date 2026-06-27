"""Prompt compiler for kernel tasks."""

from __future__ import annotations

import json
from typing import Any

from dedeucerl.kernel.types import SystemKernel, TaskInstance, ToolContract


def compile_prompt(
    kernel: SystemKernel,
    instance: TaskInstance,
    contracts: list[ToolContract],
    *,
    feedback: bool = False,
) -> list[dict[str, Any]]:
    observation = kernel.public_observation(instance)
    guidance_fn = getattr(kernel, "prompt_guidance", None)
    guidance = guidance_fn(instance) if callable(guidance_fn) else _default_guidance(instance)
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
        f"{guidance}\n\n"
        "Tools:\n"
        f"{tools_text}"
    )
    if feedback:
        system += "\n\nFeedback mode: incorrect submissions may include a counterexample."

    user = (
        "OBSERVATION:\n"
        + json.dumps(observation, sort_keys=True)
        + "\n\nUse the available tools to identify the hidden system, then submit a complete hypothesis."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _format_tool(contract: ToolContract) -> str:
    props = contract.args_schema.get("properties", {})
    required = set(contract.args_schema.get("required", []))
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


def _default_guidance(instance: TaskInstance) -> str:
    if instance.kernel_name == "mealy":
        return (
            "Mealy task: identify a finite-state transducer. "
            'Submit JSON shaped like {"n": <states>, "start": 0, '
            '"trans": {"0": {"A": [next_state, output], ...}, ...}}.'
        )
    return "Identify the hidden system using the exposed tools."
