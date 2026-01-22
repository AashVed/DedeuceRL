"""Universal contract tests for all registered DedeuceRL skins.

Goal: a centralized, futureproof suite that validates the *core framework
contract* for any skin added to `dedeucerl.skins.SKIN_REGISTRY`.

This intentionally avoids skin-specific assertions except where the framework
exposes a standard hook (e.g., `domain_params_from_answer`, tool schemas).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Dict, Optional, Tuple, Type

import pytest
from datasets import Dataset
from verifiers.types import State

from dedeucerl.core import TaskGenerator, make_rubric
from dedeucerl.core.config import SkinConfig
from dedeucerl.core.domain_spec import DomainSpec, ToolSchema
from dedeucerl.skins import SKIN_REGISTRY
from dedeucerl.utils.errors import ErrorCode


def _get_default_gen_kwargs(skin_cls: Type) -> Dict[str, Any]:
    sig = inspect.signature(skin_cls.generate_system_static)
    out: Dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name == "seed":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is not inspect.Parameter.empty:
            out[name] = p.default
    return out


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return dict(kwargs)
    allowed = set(sig.parameters.keys())
    allowed.discard("cls")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _build_domain_spec(skin_cls: Type, answer_data: Dict[str, Any]) -> DomainSpec:
    sig = inspect.signature(skin_cls.domain_spec)
    assert "budget" in sig.parameters, f"{skin_cls.__name__}.domain_spec must accept budget"
    assert "trap" in sig.parameters, f"{skin_cls.__name__}.domain_spec must accept trap"

    params: Dict[str, Any] = {}
    extractor = getattr(skin_cls, "domain_params_from_answer", None)
    if callable(extractor):
        raw = extractor(answer_data)
        if isinstance(raw, dict):
            params.update({k: v for k, v in raw.items() if v is not None})

    kwargs: Dict[str, Any] = {
        "budget": int(answer_data.get("budget", 0)),
        "trap": bool(answer_data.get("trap", True)),
        **params,
    }
    spec_kwargs = _filter_kwargs_for_callable(skin_cls.domain_spec, kwargs)
    return skin_cls.domain_spec(**spec_kwargs)


def _make_env(
    skin_cls: Type, system: Dict[str, Any], *, budget: int = 25, feedback: bool = False
) -> Tuple[Any, State]:
    answer_data = {"budget": int(budget), **system}
    answer = json.dumps(answer_data)
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "test"}]],
            "answer": [answer],
        }
    )
    env = skin_cls(dataset=dataset, rubric=make_rubric(), feedback=feedback, max_turns=256)

    state = State({"answer": answer})
    asyncio.run(env.setup_state(state))
    env._state_ref = state
    return env, state


def _tool_map(env) -> Dict[str, Any]:
    return {t.__name__: t for t in env._get_tools()}


def _extract_observation_from_prompt(prompt) -> Dict[str, Any]:
    for msg in prompt:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if "OBSERVATION:\n" not in content:
            continue
        tail = content.split("OBSERVATION:\n", 1)[1]
        raw = tail.split("\n\n", 1)[0]
        return json.loads(raw)
    raise AssertionError("Prompt did not contain OBSERVATION block")


def _assert_error_envelope(payload: Dict[str, Any]) -> None:
    assert "error" in payload
    err = payload["error"]
    assert isinstance(err, dict)
    assert "code" in err and "message" in err
    assert "budget_left" in payload
    assert "queries_used" in payload
    assert "trap_hit" in payload


def _assert_matches_tool_schema(payload: Dict[str, Any], schema: ToolSchema) -> None:
    if "error" in payload:
        _assert_error_envelope(payload)
        return

    expected = set(schema.returns.keys())
    got = set(payload.keys())
    assert got == expected, (
        f"{schema.name}: return keys drift (expected={sorted(expected)} got={sorted(got)})"
    )


def _build_args_for_tool(
    schema: ToolSchema,
    *,
    system: Dict[str, Any],
    default_expr: Optional[str] = None,
    use_correct_submission: bool = False,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {}

    correct_payload: Any = None
    if use_correct_submission:
        if "table" in system:
            correct_payload = system["table"]
        elif "spec" in system:
            correct_payload = system["spec"]
        elif "target_expr" in system:
            correct_payload = system["target_expr"]

    for arg_name, arg_schema in schema.args.items():
        if arg_schema.enum:
            args[arg_name] = arg_schema.enum[0]
            continue

        if arg_name == "expr":
            if use_correct_submission and isinstance(correct_payload, str):
                args[arg_name] = correct_payload
            else:
                args[arg_name] = default_expr or system.get("starter_expr") or "true"
            continue

        if arg_schema.type == "string":
            if "json" in arg_name.lower() or arg_name.endswith("_json"):
                if use_correct_submission:
                    if correct_payload is None:
                        raise AssertionError(
                            f"Can't construct correct submission for {schema.name}"
                        )
                    args[arg_name] = (
                        correct_payload
                        if isinstance(correct_payload, str)
                        else json.dumps(correct_payload)
                    )
                else:
                    args[arg_name] = json.dumps({"wrong": True})
            else:
                args[arg_name] = "test"
            continue

        if arg_schema.type == "integer":
            args[arg_name] = 0
            continue
        if arg_schema.type == "boolean":
            args[arg_name] = True
            continue
        if arg_schema.type == "object":
            args[arg_name] = {}
            continue
        if arg_schema.type == "array":
            args[arg_name] = []
            continue

        raise AssertionError(f"Unsupported arg type: {arg_schema.type}")

    return args


class TestUniversalRegistry:
    def test_registry_non_empty(self):
        assert SKIN_REGISTRY, "SKIN_REGISTRY should not be empty"

    def test_all_skins_have_config(self):
        for skin_name, skin_cls in SKIN_REGISTRY.items():
            assert hasattr(skin_cls, "config"), f"{skin_name}: missing config"
            assert isinstance(skin_cls.config, SkinConfig), f"{skin_name}: config not SkinConfig"

    def test_unique_skin_names(self):
        names = [skin_cls.config.skin_name for skin_cls in SKIN_REGISTRY.values()]
        assert len(names) == len(set(names)), "Duplicate config.skin_name"


@pytest.mark.parametrize("skin_name,skin_cls", list(SKIN_REGISTRY.items()))
class TestUniversalSkinContract:
    def test_generation_deterministic(self, skin_name: str, skin_cls: Type):
        kwargs = _get_default_gen_kwargs(skin_cls)
        sys1 = skin_cls.generate_system_static(seed=123, **kwargs)
        sys2 = skin_cls.generate_system_static(seed=123, **kwargs)
        assert sys1 == sys2, f"{skin_name}: generation not deterministic"

    def test_generation_trap_toggle(self, skin_name: str, skin_cls: Type):
        sig = inspect.signature(skin_cls.generate_system_static)
        assert "trap" in sig.parameters, f"{skin_name}: generate_system_static must accept trap"

        kwargs = _get_default_gen_kwargs(skin_cls)
        kwargs_no_trap = {k: v for k, v in kwargs.items() if k != "trap"}
        sys_trap = skin_cls.generate_system_static(seed=7, trap=True, **kwargs_no_trap)
        sys_no_trap = skin_cls.generate_system_static(seed=7, trap=False, **kwargs_no_trap)
        assert isinstance(sys_trap, dict)
        assert isinstance(sys_no_trap, dict)

    def test_domain_spec_accepts_budget_trap(self, skin_name: str, skin_cls: Type):
        sig = inspect.signature(skin_cls.domain_spec)
        assert "budget" in sig.parameters, f"{skin_name}: domain_spec must accept budget"
        assert "trap" in sig.parameters, f"{skin_name}: domain_spec must accept trap"

    def test_task_generator_prompt_observation_valid(self, skin_name: str, skin_cls: Type):
        gen = TaskGenerator(skin_cls)
        split = gen.generate_split(seeds=[0], budget=15, subset_name="dev")

        # Save/load not required; build dataset directly from dict via temp file semantics.
        # TaskGenerator expects a path, so roundtrip through a real file.
        # (pytest tmp_path is available via fixture in other tests; keep this local.)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"{skin_name}.json"
            gen.save_split(split, str(p))
            ds = gen.build_dataset(str(p), "dev", feedback=True)

        assert len(ds) == 1
        prompt = ds[0]["prompt"]
        answer = json.loads(ds[0]["answer"])

        assert isinstance(prompt, list)
        roles = [m.get("role") for m in prompt]
        assert "system" in roles
        assert "user" in roles

        obs = _extract_observation_from_prompt(prompt)
        assert isinstance(obs, dict)

        # Schema-first invariant: observation defaults/examples should not be overwritten by null.
        spec = _build_domain_spec(skin_cls, answer)
        for field_name, field_spec in spec.observation_fields.items():
            if field_spec.example is None:
                continue
            if field_name in obs:
                assert obs[field_name] is not None, (
                    f"{skin_name}: observation field '{field_name}' is null; "
                    "likely None leaked into build_observation"
                )

    def test_tools_match_domain_spec_and_return_schema(self, skin_name: str, skin_cls: Type):
        kwargs = _get_default_gen_kwargs(skin_cls)
        system = skin_cls.generate_system_static(seed=11, **kwargs)
        env, state = _make_env(skin_cls, system, budget=50, feedback=True)

        spec = _build_domain_spec(skin_cls, {"budget": 50, **system})
        tools = _tool_map(env)
        assert len(spec.tool_schemas) >= 2, f"{skin_name}: expected at least probe + submit"

        # Tools must be discoverable by name.
        for ts in spec.tool_schemas:
            assert ts.name in tools, f"{skin_name}: tool '{ts.name}' not found on env"

        # Call each tool in an isolated env so submit doesn't end the others.
        for ts in spec.tool_schemas:
            env_i, state_i = _make_env(skin_cls, system, budget=80, feedback=True)
            tool = _tool_map(env_i)[ts.name]
            args = _build_args_for_tool(ts, system=system)
            raw = tool(**args)
            payload = json.loads(raw)
            _assert_matches_tool_schema(payload, ts)
            assert state_i["queries_used"] >= 0

    def test_budget_consumption_and_episode_end(self, skin_name: str, skin_cls: Type):
        kwargs = _get_default_gen_kwargs(skin_cls)
        system = skin_cls.generate_system_static(seed=21, **kwargs)
        env, state = _make_env(skin_cls, system, budget=1, feedback=False)

        spec = _build_domain_spec(skin_cls, {"budget": 1, **system})
        tools = _tool_map(env)

        # Call the first non-submit tool (or fallback to first tool).
        probe_schema = next(
            (ts for ts in spec.tool_schemas if "submit" not in ts.name.lower()),
            spec.tool_schemas[0],
        )
        tool = tools[probe_schema.name]
        args = _build_args_for_tool(probe_schema, system=system)
        raw = tool(**args)
        payload = json.loads(raw)
        _assert_matches_tool_schema(payload, probe_schema)

        assert state["budget"] == 0
        assert state["done"] is True

    def test_episode_finished_error(self, skin_name: str, skin_cls: Type):
        kwargs = _get_default_gen_kwargs(skin_cls)
        system = skin_cls.generate_system_static(seed=31, **kwargs)
        env, state = _make_env(skin_cls, system, budget=5, feedback=False)

        spec = _build_domain_spec(skin_cls, {"budget": 5, **system})
        tools = _tool_map(env)
        ts = spec.tool_schemas[0]

        state["done"] = True
        raw = tools[ts.name](**_build_args_for_tool(ts, system=system))
        payload = json.loads(raw)
        _assert_error_envelope(payload)
        assert payload["error"]["code"] == ErrorCode.EPISODE_FINISHED.value

    def test_budget_exhausted_error_when_insufficient_for_tool(
        self, skin_name: str, skin_cls: Type
    ):
        # Only meaningful for skins with variable or >1 tool costs.
        kwargs = _get_default_gen_kwargs(skin_cls)
        system = skin_cls.generate_system_static(seed=41, **kwargs)

        tool_costs = None
        if isinstance(system.get("dsl"), dict):
            tool_costs = system["dsl"].get("tool_costs")

        if not isinstance(tool_costs, dict):
            pytest.skip(f"{skin_name}: no tool_costs available to test insufficient budget")

        expensive = None
        for k, v in tool_costs.items():
            if isinstance(v, int) and v > 1:
                expensive = (k, v)
                break

        if expensive is None:
            pytest.skip(f"{skin_name}: no tool with cost > 1")

        _, cost = expensive
        budget = cost - 1
        env, state = _make_env(skin_cls, system, budget=budget, feedback=False)

        spec = _build_domain_spec(skin_cls, {"budget": budget, **system})
        tools = _tool_map(env)

        # Prefer submit (usually most expensive) but accept any tool name match.
        candidate = next(
            (ts for ts in spec.tool_schemas if "submit" in ts.name.lower()),
            spec.tool_schemas[0],
        )
        raw = tools[candidate.name](**_build_args_for_tool(candidate, system=system))
        payload = json.loads(raw)

        # The framework-level contract for insufficient budget is E002.
        _assert_error_envelope(payload)
        assert payload["error"]["code"] == ErrorCode.BUDGET_EXHAUSTED.value
        assert state["done"] is True
        assert state["budget"] == 0

    def test_trap_mechanics(self, skin_name: str, skin_cls: Type):
        kwargs = _get_default_gen_kwargs(skin_cls)
        system = skin_cls.generate_system_static(seed=45, **kwargs)

        env, state = _make_env(skin_cls, system, budget=50, feedback=False)
        tools = _tool_map(env)

        # Try to trigger a trap using standardized metadata fields.
        if system.get("trap_pairs"):
            s, a = system["trap_pairs"][0]
            state["cs"] = int(s)
            payload = json.loads(tools["act"](str(a)))
            assert payload.get("trap_hit") is True

        elif system.get("trap_calls"):
            first = system["trap_calls"][0]
            state["cs"] = int(first[0])
            if len(first) == 3:
                _, ep, method = first
                payload = json.loads(tools["api_call"](str(method), str(ep)))
            elif len(first) == 4:
                _, ep, method, variant = first
                payload = json.loads(tools["api_call"](str(method), str(ep), str(variant)))
            else:
                pytest.skip(f"{skin_name}: unknown trap_calls shape")
            assert payload.get("trap_hit") is True

        elif isinstance(system.get("dsl"), dict) and system["dsl"].get("banned_tokens"):
            tok = system["dsl"]["banned_tokens"][0]
            if "type_check" not in tools:
                pytest.skip(f"{skin_name}: no type_check tool")
            payload = json.loads(tools["type_check"](str(tok)))
            assert payload.get("trap_hit") is True

        else:
            pytest.skip(f"{skin_name}: no trap metadata present")

        assert bool(state.get("trap_hit", False)) is True
        if skin_cls.config.trap_ends_episode:
            assert state.get("done") is True
            assert state.get("ok") is False

    def test_submission_success_and_failure(self, skin_name: str, skin_cls: Type):
        kwargs = _get_default_gen_kwargs(skin_cls)

        # Avoid traps interfering with success path.
        if "trap" in kwargs:
            kwargs = {**kwargs, "trap": False}
        system = skin_cls.generate_system_static(seed=51, **kwargs)

        env_ok, state_ok = _make_env(skin_cls, system, budget=200, feedback=True)
        spec = _build_domain_spec(skin_cls, {"budget": 200, **system})
        tools = _tool_map(env_ok)

        submit_schema = next(
            (ts for ts in spec.tool_schemas if "submit" in ts.name.lower()),
            spec.tool_schemas[-1],
        )

        raw_ok = tools[submit_schema.name](
            **_build_args_for_tool(submit_schema, system=system, use_correct_submission=True)
        )
        payload_ok = json.loads(raw_ok)
        _assert_matches_tool_schema(payload_ok, submit_schema)
        assert payload_ok.get("ok") is True, f"{skin_name}: correct submission did not succeed"
        assert state_ok["done"] is True
        assert state_ok["ok"] is True

        env_bad, state_bad = _make_env(skin_cls, system, budget=200, feedback=True)
        tools_bad = _tool_map(env_bad)
        raw_bad = tools_bad[submit_schema.name](
            **_build_args_for_tool(submit_schema, system=system, use_correct_submission=False)
        )
        payload_bad = json.loads(raw_bad)
        _assert_matches_tool_schema(payload_bad, submit_schema)
        assert payload_bad.get("ok") is not True, f"{skin_name}: wrong submission succeeded"

    def test_exprpolicy_starter_vs_target(self, skin_name: str, skin_cls: Type):
        # Centralized, but only runs for ExprPolicy-like skins that expose these fields.
        kwargs = _get_default_gen_kwargs(skin_cls)
        system = skin_cls.generate_system_static(seed=61, **kwargs)

        if not {"starter_expr", "target_expr", "public_cases"}.issubset(system.keys()):
            pytest.skip(f"{skin_name}: no starter/target expression contract")

        env, _ = _make_env(skin_cls, system, budget=200, feedback=False)
        tools = _tool_map(env)
        if "run_tests" not in tools:
            pytest.skip(f"{skin_name}: missing run_tests tool")

        res_starter = json.loads(tools["run_tests"](system["starter_expr"], "public"))
        assert res_starter.get("ok") is True
        assert res_starter.get("passed") is False

        res_target = json.loads(tools["run_tests"](system["target_expr"], "public"))
        assert res_target.get("ok") is True
        assert res_target.get("passed") is True

    def test_max_expr_len_is_enforced(self, skin_name: str, skin_cls: Type):
        # Centralized safety/perf contract for expression-based skins.
        kwargs = _get_default_gen_kwargs(skin_cls)

        sig = inspect.signature(skin_cls.generate_system_static)
        if "max_expr_len" not in sig.parameters:
            pytest.skip(f"{skin_name}: no max_expr_len in generation")

        # Force a very small max length.
        filtered = {k: v for k, v in kwargs.items() if k not in ("max_expr_len", "trap")}
        system = skin_cls.generate_system_static(seed=71, max_expr_len=10, trap=False, **filtered)
        if not isinstance(system.get("dsl"), dict) or "max_expr_len" not in system["dsl"]:
            pytest.skip(f"{skin_name}: no dsl.max_expr_len in system")

        env, _ = _make_env(skin_cls, system, budget=200, feedback=False)
        tools = _tool_map(env)

        long_expr = "true && true && true"  # definitely > 10 chars

        if "type_check" in tools:
            res = json.loads(tools["type_check"](long_expr))
            assert res.get("ok") is False
            assert res.get("errors"), f"{skin_name}: expected errors for long expr"

        if "run_tests" in tools:
            res = json.loads(tools["run_tests"](long_expr, "public"))
            assert res.get("ok") is False
            assert res.get("errors"), f"{skin_name}: expected errors for long expr"

        if "submit" in tools:
            res = json.loads(tools["submit"](long_expr))
            assert res.get("ok") is False
            assert res.get("errors"), f"{skin_name}: expected errors for long expr"
