"""Interactive CLI for playing TaskIR-backed DedeuceRL episodes."""

from __future__ import annotations

import json
from typing import Any

from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface import compile_prompt, compile_tool_schemas


def _prompt_int(label: str, *, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    return default if not raw else int(raw)


def _prompt_bool(label: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{label} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "true", "1")


def _parse_tool_input(raw: str, tool_names: set[str]) -> tuple[str | None, dict[str, Any]]:
    parts = raw.strip().split(maxsplit=1)
    if not parts:
        return None, {}
    name = parts[0]
    if name not in tool_names:
        return name, {}
    if len(parts) == 1:
        return name, {}
    arg_text = parts[1].strip()
    try:
        decoded = json.loads(arg_text)
        if isinstance(decoded, dict):
            return name, decoded
        return name, _wrap_single_arg(name, decoded)
    except Exception:
        return name, _wrap_single_arg(name, arg_text)


def _wrap_single_arg(tool_name: str, value: Any) -> dict[str, Any]:
    if tool_name == "act":
        return {"symbol": value}
    if tool_name == "submit_table":
        return {"table_json": value if isinstance(value, str) else json.dumps(value)}
    return {"value": value}


def _print_block(title: str, content: str) -> None:
    print(f"\n=== {title} ===")
    print(content)


def main() -> None:
    print("DedeuceRL Interactive Game")
    names = sorted(TASK_REGISTRY)
    for i, name in enumerate(names, start=1):
        print(f"{i}. {name}")
    choice = _prompt_int("Select kernel", default=1)
    kernel_name = names[max(0, min(choice - 1, len(names) - 1))]
    entry = TASK_REGISTRY[kernel_name]

    seed = _prompt_int("Seed", default=0)
    budget = _prompt_int("Budget", default=25)
    n_states = _prompt_int("n_states", default=3)
    trap = _prompt_bool("Use traps", default=True)

    instance = entry.ir.generator.sample(seed=seed, budget=budget, n_states=n_states, trap=trap)
    runtime = EpisodeRuntime(entry.ir, instance, feedback=True)
    contracts = runtime.contracts()
    tool_schemas = compile_tool_schemas(contracts)
    prompt = compile_prompt(entry.ir, instance, contracts, feedback=True)

    _print_block("SYSTEM PROMPT", str(prompt[0]["content"]))
    _print_block("USER PROMPT", str(prompt[1]["content"]))
    _print_block("TOOLS", "\n".join(f"- {schema['name']}" for schema in tool_schemas))
    print("\nCommands: :help :prompt :state :quit")

    tool_names = {contract.name for contract in contracts}
    while not runtime.done and runtime.budget > 0:
        raw = input("> ").rstrip("\n")
        if not raw.strip():
            continue
        if raw.startswith(":"):
            cmd = raw[1:].strip().lower()
            if cmd in ("q", "quit", "exit"):
                return
            if cmd == "help":
                _print_block(
                    "HELP",
                    "\n".join(
                        [
                            "Examples:",
                            "  act A",
                            '  act {"symbol":"A"}',
                            '  submit_table {"n":2,"start":0,"trans":{...}}',
                            "Commands: :prompt :state :quit",
                        ]
                    ),
                )
            elif cmd == "prompt":
                _print_block("SYSTEM PROMPT", str(prompt[0]["content"]))
                _print_block("USER PROMPT", str(prompt[1]["content"]))
            elif cmd == "state":
                print(json.dumps(runtime.state_dict(), indent=2))
            else:
                print(f"Unknown command :{cmd}")
            continue

        tool_name, args = _parse_tool_input(raw, tool_names)
        if tool_name is None:
            continue
        event = runtime.call_tool(tool_name, args)
        print(json.dumps(event.output))

    print("\nEpisode finished.")
    print(json.dumps(runtime.state_dict(), indent=2))
