"""Interactive CLI game runner.

This lets a human play as the "agent" against any DedeuceRL skin:
- shows the exact system/user prompt
- accepts tool calls
- prints the tool JSON outputs exactly as an agent would see them
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from dedeucerl.core import make_rubric
from dedeucerl.skins import SKIN_REGISTRY
from dedeucerl.utils import (
    error_invalid_argument,
    error_invalid_json,
    error_unknown_tool,
)

from cliGame.hf_defaults import get_hf_defaults


@dataclass
class Episode:
    skin_name: str
    seed: int
    budget: int
    trap: bool
    system: Dict[str, Any]
    answer: str
    prompt: List[Dict[str, Any]]
    env: Any
    state: Dict[str, Any]
    messages: List[Dict[str, Any]]
    tool_map: Dict[str, Any]
    tool_schemas: List[Dict[str, Any]]


def _print_block(title: str, text: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}\n{text}\n")


def _choose_skin_interactive() -> str:
    skins = sorted(SKIN_REGISTRY.keys())
    _print_block("DedeuceRL CLI Game", "Select a skin to play.")
    for i, s in enumerate(skins, start=1):
        print(f"{i}. {s}")

    while True:
        raw = input("\nSkin (number or name): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(skins):
                return skins[idx - 1]
            print(f"Invalid choice: {idx}")
            continue
        if raw in SKIN_REGISTRY:
            return raw
        print(f"Unknown skin '{raw}'. Available: {skins}")


def _prompt_int(label: str, *, default: Optional[int] = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if not raw and default is not None:
            return int(default)
        try:
            return int(raw)
        except Exception:
            print("Please enter an integer.")


def _derive_domain_spec(SkinClass, answer_data: Dict[str, Any], *, budget: int, trap: bool):
    extractor = getattr(SkinClass, "domain_params_from_answer", None)
    params: Dict[str, Any] = {}
    if callable(extractor):
        try:
            raw = extractor(answer_data)
        except Exception:
            raw = None
        if isinstance(raw, dict):
            params.update({k: v for k, v in raw.items() if v is not None})

    params["budget"] = int(budget)
    params["trap"] = bool(trap)
    return SkinClass.domain_spec(**params)


def _build_observation(
    spec, answer_data: Dict[str, Any], *, budget: int, trap: bool
) -> Dict[str, Any]:
    obs_values: Dict[str, Any] = {}
    # Fill from answer-derived params when possible.
    extractor = getattr(spec, "observation_fields", {})
    if isinstance(extractor, dict):
        for k in extractor.keys():
            if k in answer_data and answer_data[k] is not None:
                obs_values[k] = answer_data[k]
    obs_values["budget"] = int(budget)
    obs_values["trap"] = bool(trap)

    # Try common observation fields.
    # (Skins often expose these only via DomainSpec examples.)
    for k in ("n_states", "n_endpoints", "endpoints"):
        if k not in obs_values and k in getattr(spec, "__dict__", {}):
            obs_values[k] = getattr(spec, k)

    return spec.build_observation(**obs_values)


def _default_call_id(turn: int, idx: int) -> str:
    return f"call_{turn}_{idx}"


def _consume_invalid_budget(state: Dict[str, Any], amount: int = 1) -> None:
    # Mirror dedeucerl.cli.eval semantics: invalid tool calls still cost budget.
    amount = int(max(0, amount))
    if amount <= 0:
        return
    budget = int(state.get("budget", 0))
    used = min(budget, amount)
    state["budget"] = budget - used
    state["queries_used"] = int(state.get("queries_used", 0)) + used
    if int(state.get("budget", 0)) <= 0:
        state["budget"] = 0
        state["done"] = True
        state["ok"] = False


def _parse_tool_input(
    raw: str,
    *,
    tool_map: Dict[str, Any],
    tool_schemas: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Parse a single line into (tool_name, args, error).

    Accepted forms:
      act A
      act {"symbol":"A"}
      api_call {"method":"GET","endpoint":"/users"}
      submit_table { ...table object... }    (auto-wrapped into table_json)
      submit_spec { ...spec object... }      (auto-wrapped into spec_json)
      type_check <expr-with-spaces>
    """

    s = raw.strip()
    if not s:
        return None, None, "empty"
    if s.startswith(":"):
        return None, None, "command"

    parts = s.split(None, 1)
    name = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ""

    if name not in tool_map:
        return name, None, "unknown_tool"

    # Find schema for argument hints.
    schema = None
    for t in tool_schemas:
        if t.get("name") == name:
            schema = t
            break

    params = (schema or {}).get("parameters") or {}
    prop_map = (params.get("properties") or {}) if isinstance(params, dict) else {}
    arg_names = list(prop_map.keys()) if isinstance(prop_map, dict) else []

    if not rest:
        return name, {}, None

    # JSON attempt (object or scalar).
    parsed: Any
    # Only treat the remainder as JSON when it *looks* like JSON.
    # (Bare tokens like `true` are common DSL expressions; treat them as strings.)
    if rest[:1] in ("{", "[", '"'):
        try:
            parsed = json.loads(rest)
        except Exception:
            parsed = None
        else:
            if isinstance(parsed, dict):
                # If the parsed object already matches the tool-args shape, accept it.
                if arg_names and set(parsed.keys()).issubset(set(arg_names)):
                    return name, parsed, None

                # Convenience: allow submit_* and other single-arg tools to accept
                # a raw JSON object and wrap it into that one argument.
                if len(arg_names) == 1:
                    arg0 = arg_names[0]
                    if arg0.endswith("_json"):
                        return name, {arg0: json.dumps(parsed)}, None
                    return name, {arg0: parsed}, None

                return name, None, "bad_args"

            # Non-object JSON: only valid for single-arg tools.
            if len(arg_names) == 1:
                return name, {arg_names[0]: parsed}, None
            return name, None, "bad_args"

    # Non-JSON: if tool has exactly one arg, treat rest as that arg.
    if len(arg_names) == 1:
        return name, {arg_names[0]: rest}, None

    # Special convenience: allow submit_* <json> without wrapping.
    if name.startswith("submit"):
        try:
            obj = json.loads(rest)
        except Exception:
            return name, None, "bad_args"
        if len(arg_names) == 1 and arg_names[0].endswith("_json"):
            return name, {arg_names[0]: json.dumps(obj)}, None

    return name, None, "bad_args"


def _start_episode(skin_name: str, seed: int) -> Episode:
    SkinClass = SKIN_REGISTRY[skin_name]

    hf = get_hf_defaults(skin_name, subset="test")
    budget = int(hf.budget)
    trap = bool(hf.gen_kwargs.get("trap", False))

    gen_kwargs = dict(hf.gen_kwargs)
    # Budget is episode-level; not used for generation.
    gen_kwargs.pop("budget", None)
    system = SkinClass.generate_system_static(seed=seed, **gen_kwargs)

    answer_data: Dict[str, Any] = {"seed": int(seed), "budget": int(budget), **system}
    answer = json.dumps(answer_data)

    spec = _derive_domain_spec(SkinClass, answer_data, budget=budget, trap=trap)
    obs = _build_observation(spec, answer_data, budget=budget, trap=trap)
    prompt = SkinClass.get_prompt_template(obs, feedback=True)

    ds = Dataset.from_dict({"prompt": [prompt], "answer": [answer]})
    env = SkinClass(dataset=ds, rubric=make_rubric(), feedback=True, max_turns=10_000)

    state = {"prompt": prompt, "answer": answer}
    state = asyncio.run(env.setup_state(state))
    # Ensure internal state pointer is available even before first tool call.
    env._state_ref = state

    messages: List[Dict[str, Any]] = list(prompt)
    tool_map = {t.__name__: t for t in env._get_tools()}
    tool_schemas = spec.get_tools()

    return Episode(
        skin_name=skin_name,
        seed=seed,
        budget=budget,
        trap=trap,
        system=system,
        answer=answer,
        prompt=prompt,
        env=env,
        state=state,
        messages=messages,
        tool_map=tool_map,
        tool_schemas=tool_schemas,
    )


def _print_prompt(prompt: List[Dict[str, Any]]) -> None:
    sys_msg = next((m for m in prompt if m.get("role") == "system"), None)
    user_msg = next((m for m in prompt if m.get("role") == "user"), None)

    if sys_msg is not None:
        _print_block("SYSTEM PROMPT", str(sys_msg.get("content") or ""))
    if user_msg is not None:
        _print_block("USER PROMPT", str(user_msg.get("content") or ""))


def _print_tools(tool_schemas: List[Dict[str, Any]]) -> None:
    lines = []
    for t in tool_schemas:
        name = t.get("name", "")
        params = t.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        arglist = ", ".join(sorted(props.keys()))
        lines.append(f"- {name}({arglist})")
    _print_block("TOOLS", "\n".join(lines) if lines else "(none)")


def _print_state_brief(state: Dict[str, Any]) -> None:
    keys = ["budget", "queries_used", "trap_hit", "ok", "done"]
    view = {k: state.get(k) for k in keys}
    print(json.dumps(view, indent=2))


def _help() -> None:
    _print_block(
        "HELP",
        "\n".join(
            [
                "Enter tool calls as: <tool_name> <args>",
                "Examples:",
                "  act A",
                '  act {"symbol":"A"}',
                '  api_call {"method":"GET","endpoint":"/users"}',
                '  api_call {"method":"POST","endpoint":"/login","variant":"valid"}',
                "  type_check <expr>",
                '  run_tests {"expr":"true","suite":"public"}',
                '  submit_table {"n":4,"start":0,"trans":{...}}   (auto-wrapped)',
                '  submit_spec {"n_states":4,"start":0,"transitions":{...}} (auto-wrapped)',
                "Commands:",
                "  :help   show help",
                "  :tools  show tools",
                "  :prompt show prompts again",
                "  :state  show budget/flags",
                "  :quit   exit",
            ]
        ),
    )


def play_episode(ep: Episode) -> None:
    print(f"\nPlaying skin='{ep.skin_name}' seed={ep.seed} budget={ep.budget} trap={ep.trap}")
    _print_prompt(ep.prompt)
    _print_tools(ep.tool_schemas)
    print("Type :help for commands.\n")

    turn = 0
    while not bool(ep.state.get("done", False)) and int(ep.state.get("budget", 0)) > 0:
        raw = input("> ").rstrip("\n")
        if not raw.strip():
            continue

        if raw.strip().startswith(":"):
            cmd = raw.strip()[1:].strip().lower()
            if cmd in ("q", "quit", "exit"):
                print("Exiting.")
                return
            if cmd == "help":
                _help()
                continue
            if cmd == "tools":
                _print_tools(ep.tool_schemas)
                continue
            if cmd == "prompt":
                _print_prompt(ep.prompt)
                continue
            if cmd == "state":
                _print_state_brief(ep.state)
                continue
            print(f"Unknown command :{cmd}")
            continue

        tool_name, args, err = _parse_tool_input(
            raw,
            tool_map=ep.tool_map,
            tool_schemas=ep.tool_schemas,
        )

        turn += 1
        call_id = _default_call_id(turn, 0)

        # Attach state for any helper errors.
        ep.env._state_ref = ep.state

        if err == "unknown_tool":
            _consume_invalid_budget(ep.state, 1)
            payload = ep.env._tool_error(
                error_unknown_tool(tool_name or "", list(ep.tool_map.keys())), extra={"ok": False}
            )
            print(payload)
            ep.messages.append({"role": "tool", "tool_call_id": call_id, "content": payload})
            continue

        if err in ("bad_args",):
            _consume_invalid_budget(ep.state, 1)
            payload = ep.env._tool_error(
                error_invalid_argument(
                    "Malformed tool arguments. Provide a JSON object or a single raw arg for single-arg tools.",
                    details={"input": raw},
                ),
                extra={"ok": False},
            )
            print(payload)
            ep.messages.append({"role": "tool", "tool_call_id": call_id, "content": payload})
            continue

        if err in ("empty",):
            continue

        if tool_name is None or args is None:
            _consume_invalid_budget(ep.state, 1)
            payload = ep.env._tool_error(error_invalid_json("tool call"), extra={"ok": False})
            print(payload)
            ep.messages.append({"role": "tool", "tool_call_id": call_id, "content": payload})
            continue

        tool_fn = ep.tool_map.get(tool_name)
        if tool_fn is None:
            _consume_invalid_budget(ep.state, 1)
            payload = ep.env._tool_error(
                error_unknown_tool(tool_name, list(ep.tool_map.keys())), extra={"ok": False}
            )
            print(payload)
            ep.messages.append({"role": "tool", "tool_call_id": call_id, "content": payload})
            continue

        # Convenience: if the user passed a JSON object but the tool expects a single "*_json" string,
        # wrap it. (This is common for submit_* tools.)
        try:
            schema = next((t for t in ep.tool_schemas if t.get("name") == tool_name), None)
            prop_map = ((schema or {}).get("parameters") or {}).get("properties") or {}
            if isinstance(args, dict) and len(args) == 1:
                ((k, v),) = list(args.items())
                if (
                    k.endswith("_json")
                    and isinstance(prop_map.get(k), dict)
                    and prop_map[k].get("type") == "string"
                    and not isinstance(v, str)
                ):
                    args[k] = json.dumps(v)
        except Exception:
            pass

        # Execute tool.
        ep.env.update_tool_args(tool_name, args, ep.messages, ep.state)
        try:
            result = tool_fn(**args)
        except TypeError as e:
            _consume_invalid_budget(ep.state, 1)
            result = ep.env._tool_error(
                error_invalid_argument(
                    f"Tool invocation failed: {e}", details={"tool": tool_name, "args": args}
                ),
                extra={"ok": False},
            )
        except Exception as e:
            _consume_invalid_budget(ep.state, 1)
            result = ep.env._tool_error(
                error_invalid_argument(
                    "Tool raised exception",
                    details={"tool": tool_name, "error": str(e), "args": args},
                ),
                extra={"ok": False},
            )

        print(result)
        ep.messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    print("\nEpisode finished.")
    _print_state_brief(ep.state)


def main() -> None:
    skin = _choose_skin_interactive()
    seed = _prompt_int("Seed", default=0)
    ep = _start_episode(skin, seed)
    play_episode(ep)
