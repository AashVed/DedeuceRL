"""dedeucerl-selfcheck: validate the TaskIR/runtime/surface installation."""

from __future__ import annotations

import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dedeucerl-selfcheck",
        description="Validate DedeuceRL installation and run basic sanity checks.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def check_imports() -> list[str]:
    errors: list[str] = []
    try:
        from dedeucerl.ir import TASK_REGISTRY, TaskIR
        from dedeucerl.kernel import MealyKernel, TaskInstance
        from dedeucerl.runtime import EpisodeRuntime
        from dedeucerl.surface import compile_prompt, compile_tool_schemas

        _ = (
            TASK_REGISTRY,
            MealyKernel,
            TaskIR,
            TaskInstance,
            EpisodeRuntime,
            compile_prompt,
            compile_tool_schemas,
        )
    except Exception as e:
        errors.append(f"Failed to import architecture modules: {e}")

    try:
        from dedeucerl.adapters import ADAPTER_REGISTRY, get_adapter

        _ = (ADAPTER_REGISTRY, get_adapter)
    except Exception as e:
        errors.append(f"Failed to import adapters: {e}")

    try:
        import verifiers as vf
        from datasets import Dataset

        _ = (vf.StatefulToolEnv, Dataset)
    except Exception as e:
        errors.append(f"Failed to import surface dependencies: {e}")
    return errors


def check_mealy_runtime(verbose: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        from dedeucerl.ir import TASK_REGISTRY
        from dedeucerl.runtime import EpisodeRuntime

        ir = TASK_REGISTRY["mealy"].ir
        instance = ir.generator.sample(seed=42, budget=10, n_states=3, trap=True)
        runtime = EpisodeRuntime(ir, instance, feedback=True)
        contracts = runtime.contracts()

        if verbose:
            print(f"  Instance: {instance.id}")
            print(f"  Tools: {[contract.name for contract in contracts]}")

        event = runtime.call_tool("act", {"symbol": "A"})
        if "out" not in event.output:
            errors.append("Mealy act did not produce an output")

        table = instance.private["table"]
        submit = runtime.call_tool("submit_table", {"table_json": json.dumps(table)})
        if submit.output.get("ok") is not True:
            errors.append("Correct Mealy submission did not succeed")
    except Exception as e:
        errors.append(f"Mealy runtime check failed: {e}")
    return errors


def check_surface_roundtrip(verbose: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        from dedeucerl.ir import TASK_REGISTRY
        from dedeucerl.surface import (
            build_dataset_from_split,
            compile_prompt,
            compile_tool_schemas,
            generate_split,
        )

        entry = TASK_REGISTRY["mealy"]
        split = generate_split(entry.ir, seeds=[0], budget=5, subset_name="dev", n_states=2)
        dataset = build_dataset_from_split(split, "dev", feedback=True)
        if len(dataset) != 1:
            errors.append("Dataset compiler did not produce one row")
        instance = entry.ir.generator.sample(seed=0, budget=5, n_states=2)
        context = entry.ir.action_context(
            instance,
            entry.ir.kernel.initial_state(instance),
            budget=instance.budget,
            queries_used=0,
            tool_calls=0,
            done=instance.budget <= 0,
        )
        contracts = entry.ir.action_contracts(context)
        prompt = compile_prompt(entry.ir, instance, contracts, feedback=True)
        schemas = compile_tool_schemas(contracts)
        if not prompt or not schemas:
            errors.append("Prompt/tool schema compilation returned empty output")
        if verbose:
            print(f"  Prompt messages: {len(prompt)}")
            print(f"  Tool schemas: {[schema['name'] for schema in schemas]}")
    except Exception as e:
        errors.append(f"Surface roundtrip failed: {e}")
    return errors


def check_adapter_tool_roundtrip() -> list[str]:
    errors: list[str] = []
    try:
        from dedeucerl.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter("claude-3-5-sonnet-latest")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do thing"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "act", "arguments": '{"symbol":"A"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"out":0}'},
        ]
        system_text, convo = adapter._to_anthropic_conversation(messages)  # type: ignore[attr-defined]
        if not system_text:
            errors.append("anthropic: missing system text")
        if not any(
            msg.get("role") == "user"
            and isinstance(msg.get("content"), list)
            and any(block.get("type") == "tool_result" for block in msg["content"])
            for msg in convo
        ):
            errors.append("anthropic: tool_result block not produced")
    except Exception as e:
        errors.append(f"anthropic: adapter transcript conversion failed: {e}")
    return errors


def main() -> None:
    args = parse_args()
    checks = [
        ("Checking imports", check_imports),
        ("Checking Mealy runtime", lambda: check_mealy_runtime(args.verbose)),
        ("Checking surface compilers", lambda: check_surface_roundtrip(args.verbose)),
        ("Checking adapter tool round-trip", check_adapter_tool_roundtrip),
    ]

    print("DedeuceRL Self-Check")
    print("=" * 40)
    all_errors: list[str] = []
    for idx, (label, fn) in enumerate(checks, start=1):
        print(f"\n{idx}. {label}...")
        errors = fn()
        if errors:
            for error in errors:
                print(f"   x {error}")
            all_errors.extend(errors)
        else:
            print("   OK")

    print("\n" + "=" * 40)
    if all_errors:
        print(f"{len(all_errors)} error(s) found")
        sys.exit(1)
    print("All checks passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
