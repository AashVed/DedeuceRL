"""Compile kernel tool contracts to provider-neutral tool schemas."""

from __future__ import annotations

from typing import Any

from dedeucerl.kernel.types import ToolContract


def compile_tool_schema(contract: ToolContract) -> dict[str, Any]:
    return {
        "name": contract.name,
        "description": contract.description,
        "parameters": dict(contract.args_schema),
    }


def compile_tool_schemas(contracts: list[ToolContract]) -> list[dict[str, Any]]:
    return [compile_tool_schema(contract) for contract in contracts]
