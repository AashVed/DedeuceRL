"""dedeucerl-selfcheck: Validate installation and run basic tests."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="dedeucerl-selfcheck",
        description="Validate DedeuceRL installation and run basic sanity checks.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output.",
    )

    return parser.parse_args()


def check_imports() -> List[str]:
    """Check that all required imports work."""
    errors = []

    try:
        from dedeucerl.core import HiddenSystemEnv, SkinConfig, TaskGenerator

        _ = (HiddenSystemEnv, SkinConfig, TaskGenerator)
    except ImportError as e:
        errors.append(f"Failed to import core: {e}")

    try:
        from dedeucerl.skins import SKIN_REGISTRY, MealyEnv

        _ = (SKIN_REGISTRY, MealyEnv)
    except ImportError as e:
        errors.append(f"Failed to import skins: {e}")

    try:
        from dedeucerl.adapters import ADAPTER_REGISTRY, get_adapter

        _ = (ADAPTER_REGISTRY, get_adapter)
    except ImportError as e:
        errors.append(f"Failed to import adapters: {e}")

    try:
        from verifiers.envs.stateful_tool_env import StatefulToolEnv

        _ = StatefulToolEnv
    except ImportError as e:
        errors.append(f"Failed to import verifiers: {e}")

    try:
        from datasets import Dataset

        _ = Dataset
    except ImportError as e:
        errors.append(f"Failed to import datasets: {e}")

    return errors


def check_mealy_generation(verbose: bool = False) -> List[str]:
    """Check that Mealy machine generation works."""
    errors = []

    try:
        from dedeucerl.skins import MealyEnv

        # Generate a system
        system = MealyEnv.generate_system_static(seed=42, n_states=3, trap=True)

        if verbose:
            print(
                f"  Generated system: n={system['table']['n']}, traps={len(system['trap_pairs'])}"
            )

        # Validate structure
        table = system["table"]
        if table["n"] != 3:
            errors.append(f"Expected 3 states, got {table['n']}")
        if table["start"] != 0:
            errors.append(f"Expected start=0, got {table['start']}")

        trans = table["trans"]
        for s in range(3):
            if str(s) not in trans:
                errors.append(f"Missing state {s} in transitions")
                continue
            for a in ["A", "B", "C"]:
                if a not in trans[str(s)]:
                    errors.append(f"Missing transition ({s}, {a})")

    except Exception as e:
        errors.append(f"Mealy generation failed: {e}")

    return errors


def check_isomorphism(verbose: bool = False) -> List[str]:
    """Check that isomorphism checking works."""
    errors = []

    try:
        from dedeucerl.skins import MealyEnv
        from dedeucerl.core import make_rubric
        from datasets import Dataset

        # Create a simple 2-state machine
        table = {
            "n": 2,
            "start": 0,
            "trans": {
                "0": {"A": [1, 0], "B": [0, 1], "C": [0, 2]},
                "1": {"A": [0, 1], "B": [1, 0], "C": [1, 2]},
            },
        }

        # Create minimal dataset
        answer = json.dumps({"table": table, "trap_pairs": [], "budget": 10})
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "test"}]],
                "answer": [answer],
            }
        )

        # Create environment
        rubric = make_rubric()
        env = MealyEnv(dataset=dataset, rubric=rubric, feedback=False, max_turns=20)

        # Manually configure
        env._configure_from_metadata(json.loads(answer))

        # Test exact match
        hypothesis = {"n": 2, "start": 0, "trans": env._trans}
        if not env.is_isomorphic(hypothesis, env._ground_truth):
            errors.append("Isomorphism check failed for exact match")
        elif verbose:
            print("  Exact match: PASS")

        # Test wrong output
        wrong = {
            "n": 2,
            "start": 0,
            "trans": {
                0: {"A": (1, 1), "B": (0, 1), "C": (0, 2)},  # Changed output
                1: {"A": (0, 1), "B": (1, 0), "C": (1, 2)},
            },
        }
        if env.is_isomorphic(wrong, env._ground_truth):
            errors.append("Isomorphism check accepted wrong hypothesis")
        elif verbose:
            print("  Wrong output: PASS (correctly rejected)")

    except Exception as e:
        errors.append(f"Isomorphism check failed: {e}")

    return errors


def _infer_trap_from_answer(answer_data: Dict[str, Any]) -> bool:
    if "trap_pairs" in answer_data:
        return bool(answer_data.get("trap_pairs"))
    if "trap_calls" in answer_data:
        return bool(answer_data.get("trap_calls"))
    return False


def check_all_skins(verbose: bool = False) -> List[str]:
    """Check determinism + reachability + traps + DomainSpec consistency for all skins."""
    errors: List[str] = []

    from dedeucerl.skins import SKIN_REGISTRY
    from dedeucerl.core.automata import is_fully_reachable, verify_trap_free_path_exists

    seed = 123
    budget = 10

    for name, Skin in SKIN_REGISTRY.items():
        if verbose:
            print(f"  Skin: {name}")

        # 1) Determinism
        try:
            sys1 = Skin.generate_system_static(seed=seed)
            sys2 = Skin.generate_system_static(seed=seed)
            if sys1 != sys2:
                errors.append(f"{name}: non-deterministic generation for seed={seed}")
        except Exception as e:
            errors.append(f"{name}: generation failed: {e}")
            continue

        # 2) DomainSpec consistency (via new hook if present)
        answer_data: Dict[str, Any] = {"seed": seed, "budget": budget, **sys1}
        trap = _infer_trap_from_answer(answer_data)

        extractor = getattr(Skin, "domain_params_from_answer", None)
        params: Dict[str, Any] = {}
        if callable(extractor):
            try:
                raw_params = extractor(answer_data)
            except Exception as e:
                errors.append(f"{name}: domain_params_from_answer() failed: {e}")
                raw_params = None
            if isinstance(raw_params, dict):
                params = raw_params

        params["budget"] = budget
        params["trap"] = trap

        try:
            spec = Skin.domain_spec(**params)
            # Ensure tool schemas are JSON-serializable
            json.dumps(spec.get_tools())
            # Ensure prompt template builds without error
            obs_values = {k: v for k, v in params.items() if k in spec.observation_fields}
            obs = spec.build_observation(**obs_values)
            prompt = Skin.get_prompt_template(obs, feedback=False)
            if not isinstance(prompt, list) or not prompt:
                errors.append(f"{name}: get_prompt_template returned invalid prompt")
        except Exception as e:
            errors.append(f"{name}: domain_spec/prompt build failed: {e}")

        # 3) Reachability + trap encoding consistency (best-effort for built-in formats)
        try:
            if "table" in sys1:
                table = sys1["table"]
                n = int(table.get("n", 0))
                start = int(table.get("start", 0))
                trans = table.get("trans", {})

                mealy_actions = ["A", "B", "C"]

                def get_next_mealy(state: int, action: str) -> int:
                    return int(trans[str(state)][action][0])

                if not is_fully_reachable(n, start, mealy_actions, get_next_mealy):
                    errors.append(f"{name}: not fully reachable")

                trap_pairs = sys1.get("trap_pairs", [])
                trap_set: Set[Tuple[int, Any]] = set()
                for s, a in trap_pairs:
                    s_i = int(s)
                    a_s = str(a)
                    if not (0 <= s_i < n):
                        errors.append(f"{name}: invalid trap state {s}")
                    if a_s not in mealy_actions:
                        errors.append(f"{name}: invalid trap action {a}")
                    trap_set.add((s_i, a_s))

                if trap_set and not verify_trap_free_path_exists(
                    n, start, mealy_actions, get_next_mealy, trap_set
                ):
                    errors.append(f"{name}: traps make system unsolvable")

            elif "spec" in sys1:
                spec_data = sys1["spec"]
                n = int(spec_data.get("n_states", 0))
                start = int(spec_data.get("start", 0))
                transitions = spec_data.get("transitions", {})

                if not isinstance(transitions, dict) or "0" not in transitions:
                    raise ValueError("missing transitions")

                # Detect protocol-style vs APIEnv-style transitions
                first_state = transitions.get("0", {})
                # Pick any endpoint/method to inspect depth
                any_ep = next(iter(first_state.keys()))
                any_methods = first_state[any_ep]
                any_m = next(iter(any_methods.keys()))
                leaf = any_methods[any_m]

                if isinstance(leaf, dict):
                    # APIEnv: transitions[state][ep][method][variant] = [ns, status, schema]
                    ap_actions: List[Tuple[str, str, str]] = []
                    for ep, methods in first_state.items():
                        for m, variants in methods.items():
                            for v in variants.keys():
                                ap_actions.append((m, ep, v))

                    def get_next_apienv(state: int, action: Tuple[str, str, str]) -> int:
                        m, ep, v = action
                        return int(transitions[str(state)][ep][m][v][0])

                    if not is_fully_reachable(n, start, ap_actions, get_next_apienv):
                        errors.append(f"{name}: not fully reachable")

                    trap_calls = sys1.get("trap_calls", [])
                    trap_set2: Set[Tuple[int, Any]] = set()
                    for s, ep, m, v in trap_calls:
                        a = (str(m), str(ep), str(v))
                        trap_set2.add((int(s), a))
                        if a not in ap_actions:
                            errors.append(f"{name}: trap action not in action set: {a}")

                    if trap_set2 and not verify_trap_free_path_exists(
                        n, start, ap_actions, get_next_apienv, trap_set2
                    ):
                        errors.append(f"{name}: traps make system unsolvable")

                else:
                    # Protocol: transitions[state][ep][method] = [ns, status]
                    actions2: List[Tuple[str, str]] = []
                    for ep, methods in first_state.items():
                        for m in methods.keys():
                            actions2.append((ep, m))

                    def get_next2(state: int, action: Tuple[str, str]) -> int:
                        ep, m = action
                        return int(transitions[str(state)][ep][m][0])

                    if not is_fully_reachable(n, start, actions2, get_next2):
                        errors.append(f"{name}: not fully reachable")

                    trap_calls = sys1.get("trap_calls", [])
                    trap_set3: Set[Tuple[int, Any]] = set()
                    for s, ep, m in trap_calls:
                        a = (str(ep), str(m))
                        trap_set3.add((int(s), a))
                        if a not in actions2:
                            errors.append(f"{name}: trap action not in action set: {a}")

                    if trap_set3 and not verify_trap_free_path_exists(
                        n, start, actions2, get_next2, trap_set3
                    ):
                        errors.append(f"{name}: traps make system unsolvable")

        except Exception as e:
            errors.append(f"{name}: reachability/trap checks failed: {e}")

    return errors


def check_counterexample(verbose: bool = False) -> List[str]:
    """Check that counterexample generation works."""
    errors = []

    try:
        from dedeucerl.skins import MealyEnv
        from dedeucerl.core import make_rubric
        from datasets import Dataset

        # Create a simple 2-state machine
        table = {
            "n": 2,
            "start": 0,
            "trans": {
                "0": {"A": [1, 0], "B": [0, 1], "C": [0, 2]},
                "1": {"A": [0, 1], "B": [1, 0], "C": [1, 2]},
            },
        }

        answer = json.dumps({"table": table, "trap_pairs": [], "budget": 10})
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "test"}]],
                "answer": [answer],
            }
        )

        rubric = make_rubric()
        env = MealyEnv(dataset=dataset, rubric=rubric, feedback=True, max_turns=20)
        env._configure_from_metadata(json.loads(answer))

        # Create wrong hypothesis
        wrong = {
            "n": 2,
            "start": 0,
            "trans": {
                0: {"A": (1, 1), "B": (0, 1), "C": (0, 2)},  # Changed output
                1: {"A": (0, 1), "B": (1, 0), "C": (1, 2)},
            },
        }

        cex = env.get_counterexample(wrong, env._ground_truth)

        if cex is None:
            errors.append("Counterexample generation returned None")
        elif verbose:
            print(f"  Counterexample: {cex}")
            print("  Counterexample generation: PASS")

    except Exception as e:
        errors.append(f"Counterexample check failed: {e}")

    return errors


def check_adapter_tool_roundtrip(verbose: bool = False) -> List[str]:
    """Check that adapters preserve tool round-trip transcript.

    This is a *local* check that doesn't call any external APIs.
    It verifies that adapters can translate:
    - assistant tool_calls
    - tool results (role="tool")

    into provider-native message formats.
    """
    errors: List[str] = []

    # Anthropic: ensure tool result messages are converted to tool_result blocks.
    try:
        from dedeucerl.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter("claude-3-5-sonnet-latest")
        msgs = [
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
        system_text, convo = adapter._to_anthropic_conversation(msgs)  # type: ignore[attr-defined]
        if not system_text:
            errors.append("anthropic: missing system text")

        # Expect a user message containing tool_result
        def _has_tool_result(msg: Dict[str, Any]) -> bool:
            if msg.get("role") != "user":
                return False
            content = msg.get("content")
            if not isinstance(content, list):
                return False
            return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)

        has_tool_result = any(_has_tool_result(m) for m in convo)
        if not has_tool_result:
            errors.append("anthropic: tool_result block not produced")
    except Exception as e:
        errors.append(f"anthropic: adapter transcript conversion failed: {e}")

    # Gemini: ensure adapter import path errors are informative if SDK missing.
    try:
        from dedeucerl.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter("gemini-1.5-pro")
        try:
            adapter._import_backend()  # type: ignore[attr-defined]
        except ImportError:
            # Acceptable when SDK not installed in the current environment.
            pass
    except Exception as e:
        errors.append(f"gemini: adapter initialization failed: {e}")

    return errors


def main():
    """Entry point for dedeucerl-selfcheck CLI."""
    args = parse_args()

    all_errors = []

    print("DedeuceRL Self-Check")
    print("=" * 40)

    # Run checks
    print("\n1. Checking imports...")
    errors = check_imports()
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        all_errors.extend(errors)
    else:
        print("   ✓ All imports successful")

    print("\n2. Checking Mealy generation...")
    errors = check_mealy_generation(args.verbose)
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        all_errors.extend(errors)
    else:
        print("   ✓ Mealy generation working")

    print("\n3. Checking isomorphism...")
    errors = check_isomorphism(args.verbose)
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        all_errors.extend(errors)
    else:
        print("   ✓ Isomorphism checking working")

    print("\n4. Checking counterexamples...")
    errors = check_counterexample(args.verbose)
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        all_errors.extend(errors)
    else:
        print("   ✓ Counterexample generation working")

    print("\n5. Checking adapter tool round-trip...")
    errors = check_adapter_tool_roundtrip(args.verbose)
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        all_errors.extend(errors)
    else:
        print("   ✓ Adapter transcript translation looks good")

    print("\n6. Checking all skins (determinism, reachability, traps, DomainSpec)...")
    errors = check_all_skins(args.verbose)
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        all_errors.extend(errors)
    else:
        print("   ✓ All skins passed invariants")

    # Summary
    print("\n" + "=" * 40)
    if all_errors:
        print(f"❌ {len(all_errors)} error(s) found")
        sys.exit(1)
    else:
        print("✓ All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
