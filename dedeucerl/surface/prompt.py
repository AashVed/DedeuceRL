"""Prompt compiler for TaskIR tasks."""

from __future__ import annotations

import json
from typing import Any

from dedeucerl.ir.types import TaskIR
from dedeucerl.kernel.types import TaskInstance, ToolContract


def compile_prompt(
    ir: TaskIR,
    instance: TaskInstance,
    contracts: list[ToolContract] | None = None,
    *,
    feedback: bool = False,
) -> list[dict[str, Any]]:
    runtime_contracts = (
        list(contracts)
        if contracts is not None
        else ir.tool_contracts(instance, ir.kernel.initial_state(instance))
    )
    renderer = ir.renderers.get("prompt")
    if renderer is not None:
        return renderer.render(ir, instance, runtime_contracts, feedback=feedback)

    observation = ir.public_observation(instance)
    guidance = _default_guidance()
    tools_text = "\n".join(_format_tool(contract) for contract in runtime_contracts)

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


def _default_guidance() -> str:
    return "Identify the hidden system using the exposed tools."
