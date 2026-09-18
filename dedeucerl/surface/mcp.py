"""MCP surface for one self-contained DedeuceRL episode."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import mcp.server.stdio
from mcp.server import Server, ServerRequestContext
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)

from dedeucerl import __version__
from dedeucerl.core.rubric import score_identification
from dedeucerl.ir.actions import ToolActionContract
from dedeucerl.ir.types import TaskIR
from dedeucerl.kernel.types import TaskInstance
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface.prompt import compile_prompt
from dedeucerl.utils.errors import error_episode_finished


_ERROR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "error": {"type": "object"},
        "budget_left": {"type": "integer"},
        "queries_used": {"type": "integer"},
        "trap_hit": {"type": "boolean"},
        "ok": {"type": "boolean"},
    },
    "required": ["error", "budget_left", "queries_used", "trap_hit"],
}

_TERMINAL_OUTPUT_PROPERTIES: dict[str, Any] = {
    "score": {"type": "number", "description": "Final benchmark score."},
    "reward": {"type": "number", "description": "Alias of the final benchmark score."},
    "termination_reason": {
        "type": "string",
        "description": "Why the episode reached a terminal state.",
    },
}


@dataclass(frozen=True)
class EpisodeArtifactPaths:
    """Resolved output paths for one MCP episode."""

    run_id: str
    result: Path | None
    trace: Path | None


class MCPEpisodeServer:
    """Expose exactly one stateful benchmark episode as an MCP server."""

    def __init__(
        self,
        ir: TaskIR,
        instance: TaskInstance,
        *,
        feedback: bool = False,
        persist: bool = True,
        runs_dir: str | Path = ".dedeucerl/runs",
        run_id: str | None = None,
        result_path: str | Path | None = None,
        trace_path: str | Path | None = None,
    ) -> None:
        self.ir = ir
        self.instance = instance
        self.runtime = EpisodeRuntime(ir, instance, feedback=feedback)
        self.feedback = feedback
        self.started_at = _now()
        self.finished_at: str | None = None
        self.final_result: dict[str, Any] | None = None
        self._activated = False
        self.artifacts = _resolve_artifact_paths(
            ir=ir,
            instance=instance,
            persist=persist,
            runs_dir=Path(runs_dir),
            run_id=run_id,
            result_path=None if result_path is None else Path(result_path),
            trace_path=None if trace_path is None else Path(trace_path),
        )

        self.instructions = _compile_instructions(ir, instance, self.runtime, feedback=feedback)
        self.server: Server[Any] = Server(
            "DedeuceRL",
            version=__version__,
            title=f"DedeuceRL: {ir.name}",
            description="One stateful hidden-system identification benchmark episode.",
            instructions=self.instructions,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )

    async def _list_tools(
        self,
        ctx: ServerRequestContext[Any],
        params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        _ = (ctx, params)
        return ListToolsResult(
            tools=[_compile_mcp_tool(contract) for contract in self.runtime.contracts()]
        )

    async def _call_tool(
        self,
        ctx: ServerRequestContext[Any],
        params: CallToolRequestParams,
    ) -> CallToolResult:
        _ = ctx
        if self.final_result is not None:
            return self._post_terminal_result(params.name)
        self._activate()
        event = self.runtime.call_tool(params.name, params.arguments)
        self._append_trace(event.to_dict())
        output = dict(event.output)
        if self.runtime.done:
            final_result = self.finalize()
            output.update(
                score=final_result["score"],
                reward=final_result["reward"],
                termination_reason=final_result["termination_reason"],
            )

        return CallToolResult(
            content=[TextContent(text=json.dumps(output, sort_keys=True))],
            structuredContent=output,
            isError=event.error is not None,
        )

    def _post_terminal_result(self, tool_name: str) -> CallToolResult:
        output: dict[str, Any] = {
            "error": error_episode_finished().to_dict(),
            "budget_left": self.runtime.budget,
            "queries_used": self.runtime.queries_used,
            "trap_hit": self.runtime.trap_hit,
        }
        contract = next(
            (contract for contract in self.runtime.contracts() if contract.name == tool_name),
            None,
        )
        if contract is not None and contract.kind == "submit":
            output["ok"] = False
        return CallToolResult(
            content=[TextContent(text=json.dumps(output, sort_keys=True))],
            structuredContent=output,
            isError=True,
        )

    async def run_stdio(self) -> None:
        """Run the episode over STDIO until the MCP client disconnects."""
        reason = "disconnected"
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
        except BaseException:
            reason = "server_error"
            raise
        finally:
            self.finalize(reason)

    def finalize(self, reason: str | None = None) -> dict[str, Any]:
        """Finalize scoring and artifacts once; subsequent calls are idempotent."""
        if self.final_result is not None:
            return self.final_result

        termination_reason = _termination_reason(self.runtime, self.ir, fallback=reason)
        self.finished_at = _now()
        state = self.runtime.state_dict()
        score = score_identification(state)
        result = {
            "schema_version": 1,
            "run_id": self.artifacts.run_id,
            "task": self.ir.name,
            "task_version": self.ir.version,
            "episode_id": self.instance.id,
            "seed": self.instance.seed,
            "params": dict(self.instance.params),
            "feedback": self.feedback,
            "termination_reason": termination_reason,
            "done": self.runtime.done,
            "ok": self.runtime.ok,
            "trap_hit": self.runtime.trap_hit,
            "score": score,
            "reward": score,
            "budget_init": self.runtime.budget_init,
            "budget_remaining": self.runtime.budget,
            "queries_used": self.runtime.queries_used,
            "tool_calls": self.runtime.tool_calls,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result_path": (
                None
                if not self._activated or self.artifacts.result is None
                else str(self.artifacts.result)
            ),
            "trace_path": (
                None
                if not self._activated or self.artifacts.trace is None
                else str(self.artifacts.trace)
            ),
        }
        if self._activated and self.artifacts.result is not None:
            _write_json_atomic(self.artifacts.result, result)
        self._append_trace({"event": "episode_end", **result})
        self.final_result = result
        return result

    def _activate(self) -> None:
        if self._activated:
            return
        self._start_trace()
        self._activated = True

    def _start_trace(self) -> None:
        path = self.artifacts.trace
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "episode_start",
                        "schema_version": 1,
                        "run_id": self.artifacts.run_id,
                        "task": self.ir.name,
                        "task_version": self.ir.version,
                        "episode_id": self.instance.id,
                        "seed": self.instance.seed,
                        "params": dict(self.instance.params),
                        "feedback": self.feedback,
                        "budget_init": self.runtime.budget_init,
                        "started_at": self.started_at,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def _append_trace(self, event: Mapping[str, Any]) -> None:
        if not self._activated:
            return
        path = self.artifacts.trace
        if path is None:
            return
        payload = {"run_id": self.artifacts.run_id, "recorded_at": _now(), **dict(event)}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _compile_mcp_tool(contract: ToolActionContract[Any]) -> Tool:
    schema = contract.to_tool_schema()
    output_schema = {
        "type": "object",
        "properties": _TERMINAL_OUTPUT_PROPERTIES,
        "anyOf": [dict(contract.return_schema), _ERROR_OUTPUT_SCHEMA],
    }
    return Tool(
        name=contract.name,
        description=f"{contract.description} Cost: {contract.cost} budget unit(s).",
        inputSchema=schema["parameters"],
        outputSchema=output_schema,
        annotations=ToolAnnotations(
            title=contract.name,
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )


def _compile_instructions(
    ir: TaskIR,
    instance: TaskInstance,
    runtime: EpisodeRuntime,
    *,
    feedback: bool,
) -> str:
    messages = compile_prompt(ir, instance, runtime.contracts(), feedback=feedback)
    content = [str(message.get("content", "")).strip() for message in messages]
    return "\n\n".join(part for part in content if part)


def _resolve_artifact_paths(
    *,
    ir: TaskIR,
    instance: TaskInstance,
    persist: bool,
    runs_dir: Path,
    run_id: str | None,
    result_path: Path | None,
    trace_path: Path | None,
) -> EpisodeArtifactPaths:
    if not persist:
        if result_path is not None or trace_path is not None:
            raise ValueError("artifact paths cannot be used with persist=False")
        return EpisodeArtifactPaths(
            run_id=run_id or _new_run_id(ir, instance), result=None, trace=None
        )

    if run_id is not None and _slug(run_id) != run_id:
        raise ValueError("run_id may contain only letters, digits, '.', '_', and '-'")
    resolved_id = run_id or _new_run_id(ir, instance)
    run_dir = runs_dir / resolved_id
    result = result_path or run_dir / "result.json"
    trace = trace_path or run_dir / "trace.jsonl"
    if result.resolve(strict=False) == trace.resolve(strict=False):
        raise ValueError("result and trace paths must be different")
    if result.exists():
        raise FileExistsError(f"result path already exists: {result}")
    if trace.exists():
        raise FileExistsError(f"trace path already exists: {trace}")
    return EpisodeArtifactPaths(run_id=resolved_id, result=result, trace=trace)


def _new_run_id(ir: TaskIR, instance: TaskInstance) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    task = _slug(ir.name)
    episode = _slug(instance.id)
    return f"{timestamp}-{task}-{episode}-{uuid4().hex[:8]}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.") or "episode"


def _termination_reason(runtime: EpisodeRuntime, ir: TaskIR, *, fallback: str | None) -> str:
    if runtime.ok:
        return "solved"
    if runtime.done and runtime.trap_hit:
        if ir.resource_model.trap_ends_episode:
            return "trapped"
        if runtime.budget > 0:
            return "solved_with_trap"
    if runtime.done and runtime.budget <= 0:
        return "budget_exhausted"
    return fallback or ("finished" if runtime.done else "disconnected")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"result path already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
