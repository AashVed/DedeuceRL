"""Verifiers surface for TaskIR runtimes."""

from __future__ import annotations

import json
import keyword
from typing import Callable, Iterable

import verifiers as vf
from verifiers.types import State, Tool

from dedeucerl.core.rubric import make_rubric
from dedeucerl.ir.registry import get_task_entry
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface.dataset import instance_from_dict


class KernelToolEnv(vf.StatefulToolEnv):
    """Compile kernel task instances into a Verifiers StatefulToolEnv."""

    def __init__(self, *, dataset, feedback: bool = False, max_turns: int | None = None, **kwargs):
        self.feedback_enabled = bool(feedback)
        self._runtime_ref: EpisodeRuntime | None = None
        self._state_ref: State | None = None
        tools, tool_defs = self._build_tools_from_dataset(dataset)
        resolved_max_turns = max_turns
        if resolved_max_turns is None:
            resolved_max_turns = self._derive_max_turns(
                (dataset, kwargs.get("eval_dataset")),
                feedback=self.feedback_enabled,
            )
        super().__init__(
            tools=tools,
            dataset=dataset,
            rubric=kwargs.pop("rubric", make_rubric()),
            max_turns=resolved_max_turns,
            **kwargs,
        )
        # Verifiers normally derives these definitions from Python annotations.
        # Runtime-compiled tools intentionally have dynamic signatures, so retain
        # the richer TaskIR schemas as the model-facing source of truth instead.
        self.tool_defs = tool_defs

    async def setup_state(self, state: State, **kwargs) -> State:
        _ = kwargs
        instance = instance_from_dict(json.loads(state["answer"]))
        entry = get_task_entry(instance.kernel_name)
        runtime = EpisodeRuntime(entry.ir, instance, feedback=self.feedback_enabled)
        state.update(runtime.state_dict())
        state["_runtime"] = runtime
        self._runtime_ref = runtime
        self._state_ref = state
        return state

    def update_tool_args(self, tool_name: str, tool_args: dict, messages, state, **kwargs) -> dict:
        _ = (tool_name, tool_args, messages, kwargs)
        runtime = state.get("_runtime")
        if isinstance(runtime, EpisodeRuntime):
            self._runtime_ref = runtime
            self._state_ref = state
        return tool_args

    def _build_tools_from_dataset(self, dataset) -> tuple[list[Callable[..., str]], list[Tool]]:
        if len(dataset) <= 0:
            return [], []
        instance = instance_from_dict(json.loads(dataset[0]["answer"]))
        entry = get_task_entry(instance.kernel_name)
        context = entry.ir.action_context(
            instance,
            entry.ir.kernel.initial_state(instance),
            budget=instance.budget,
            queries_used=0,
            tool_calls=0,
            done=instance.budget <= 0,
        )
        contracts = entry.ir.action_contracts(context)
        schemas = [contract.to_tool_schema() for contract in contracts]
        tools = [self._make_tool(schema["name"], schema["parameters"]) for schema in schemas]
        tool_defs = [
            Tool(
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"],
            )
            for schema in schemas
        ]
        return tools, tool_defs

    @staticmethod
    def _derive_max_turns(datasets: Iterable, *, feedback: bool) -> int:
        budgets: list[int] = []
        for dataset in datasets:
            if dataset is None:
                continue
            for row in dataset:
                instance = instance_from_dict(json.loads(row["answer"]))
                budgets.append(int(instance.budget))
        if not budgets:
            return 64
        return max(budgets) + (10 if feedback else 2)

    def _make_tool(self, name: str, parameters_schema: dict) -> Callable[..., str]:
        if not name.isidentifier():
            raise ValueError(f"Tool name must be a valid Python identifier: {name!r}")
        props = parameters_schema.get("properties", {})
        required = parameters_schema.get("required", [])
        if not isinstance(props, dict):
            props = {}
        if not isinstance(required, list):
            required = []

        required_names = [str(arg) for arg in required if arg in props]
        optional_names = [str(arg) for arg in props if arg not in required_names]
        arg_names = [*required_names, *optional_names]
        invalid = [arg for arg in arg_names if not arg.isidentifier() or keyword.iskeyword(arg)]
        if invalid:
            raise ValueError(f"Tool arguments must be valid Python identifiers: {invalid!r}")

        signature_parts = [*required_names, *(f"{arg}=None" for arg in optional_names)]
        lines = [f"def {name}({', '.join(signature_parts)}) -> str:", "    payload = {}"]
        lines.extend(f"    payload[{arg!r}] = {arg}" for arg in required_names)
        for arg in optional_names:
            lines.append(f"    if {arg} is not None:")
            lines.append(f"        payload[{arg!r}] = {arg}")
        lines.append(f"    return _dispatch({name!r}, payload)")
        source = "\n".join(lines) + "\n"
        namespace = {"_dispatch": self._dispatch_tool}
        exec(source, namespace)
        tool = namespace[name]
        tool.__doc__ = f"Runtime-compiled tool '{name}'."
        return tool

    def _dispatch_tool(self, name: str, kwargs: dict) -> str:
        if self._runtime_ref is None:
            raise RuntimeError("Runtime not initialized. Call setup_state first.")
        event = self._runtime_ref.call_tool(name, kwargs)
        if self._state_ref is not None:
            self._state_ref.update(self._runtime_ref.state_dict())
            self._state_ref["_runtime"] = self._runtime_ref
        return json.dumps(event.output)


def make_verifiers_env(*, dataset, feedback: bool = False, max_turns: int | None = None, **kwargs):
    return KernelToolEnv(dataset=dataset, feedback=feedback, max_turns=max_turns, **kwargs)
