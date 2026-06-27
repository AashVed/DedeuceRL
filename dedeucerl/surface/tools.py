"""Compile TaskIR action contracts to provider-neutral tool schemas."""

from __future__ import annotations

from typing import Any, Sequence

from dedeucerl.ir.actions import ToolActionContract


def compile_tool_schema(contract: ToolActionContract[Any]) -> dict[str, Any]:
    return contract.to_tool_schema()


def compile_tool_schemas(contracts: Sequence[ToolActionContract[Any]]) -> list[dict[str, Any]]:
    return [compile_tool_schema(contract) for contract in contracts]
