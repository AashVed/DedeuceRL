"""Verifiers surface for TaskIR runtimes."""

from __future__ import annotations

import json
from typing import Callable

import verifiers as vf
from verifiers.types import State

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
        tools = self._build_tools_from_dataset(dataset)
        super().__init__(
            tools=tools,
            dataset=dataset,
            rubric=kwargs.pop("rubric", make_rubric()),
            max_turns=max_turns or 64,
            **kwargs,
        )

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

    def _build_tools_from_dataset(self, dataset) -> list[Callable[..., str]]:
        if len(dataset) <= 0:
            return []
        instance = instance_from_dict(json.loads(dataset[0]["answer"]))
        entry = get_task_entry(instance.kernel_name)
        contracts = entry.ir.tool_contracts(instance, entry.ir.kernel.initial_state(instance))
        return [self._make_tool(contract.name, contract.args_schema) for contract in contracts]

    def _make_tool(self, name: str, args_schema: dict) -> Callable[..., str]:
        if not name.isidentifier():
            raise ValueError(f"Tool name must be a valid Python identifier: {name!r}")
        props = args_schema.get("properties", {})
        required = args_schema.get("required", [])
        if not isinstance(props, dict):
            props = {}
        if not isinstance(required, list):
            required = []

        arg_names = [str(arg) for arg in required if arg in props]
        signature = ", ".join(arg_names)
        payload = ", ".join(f"{arg!r}: {arg}" for arg in arg_names)
        source = (
            f"def {name}({signature}) -> str:\n"
            f"    return _dispatch({name!r}, {{{payload}}})\n"
        )
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
