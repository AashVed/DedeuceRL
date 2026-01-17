"""Test suite for ProtocolEnv skin."""

import json
import pytest
from verifiers.types import State

from dedeucerl.skins import ProtocolEnv
from dedeucerl.core import make_rubric, TaskGenerator
from datasets import Dataset


class TestProtocolEnvGeneration:
    """Tests for Protocol environment generation."""

    def test_basic_generation(self):
        """Test basic API spec generation with state-dependent transitions."""
        system = ProtocolEnv.generate_system_static(seed=42, n_endpoints=3, n_states=3, trap=True)

        assert "spec" in system
        assert "trap_calls" in system

        spec = system["spec"]
        assert spec["n_states"] == 3
        assert spec["start"] == 0
        assert "transitions" in spec
        # All states should have transitions
        assert len(spec["transitions"]) == 3

    def test_deterministic(self):
        """Test that same seed produces same spec."""
        sys1 = ProtocolEnv.generate_system_static(seed=123, n_endpoints=3, n_states=4)
        sys2 = ProtocolEnv.generate_system_static(seed=123, n_endpoints=3, n_states=4)

        assert sys1["spec"] == sys2["spec"]

    def test_no_trap(self):
        """Test generation without traps."""
        system = ProtocolEnv.generate_system_static(seed=42, n_endpoints=3, n_states=3, trap=False)
        assert system["trap_calls"] == []

    def test_state_dependent_transitions(self):
        """Test that transitions are state-dependent."""
        system = ProtocolEnv.generate_system_static(seed=42, n_endpoints=3, n_states=3)
        transitions = system["spec"]["transitions"]

        # Each state should have transitions for all endpoints
        for state_str in transitions:
            state_trans = transitions[state_str]
            assert len(state_trans) > 0
            for ep, methods in state_trans.items():
                for method, (ns, status) in methods.items():
                    assert method in ["GET", "POST", "PUT", "DELETE"]
                    assert 0 <= ns < 3
                    assert status in [200, 201, 400, 404]


class TestProtocolEnvBehavior:
    """Tests for Protocol environment behavior."""

    @pytest.fixture
    def env_with_spec(self):
        """Create an environment with a known state-dependent spec."""
        # State-dependent transitions: different behavior from state 0 vs state 1
        spec = {
            "n_states": 2,
            "start": 0,
            "transitions": {
                "0": {
                    "/users": {"GET": [0, 200], "POST": [1, 201]},
                    "/items": {"GET": [1, 200]},
                },
                "1": {
                    "/users": {"GET": [1, 404], "POST": [0, 201]},
                    "/items": {"GET": [0, 200]},
                },
            },
        }
        answer = json.dumps({"spec": spec, "trap_calls": [], "budget": 10})
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "test"}]],
                "answer": [answer],
            }
        )
        rubric = make_rubric()
        env = ProtocolEnv(dataset=dataset, rubric=rubric, feedback=False, max_turns=20)
        env._configure_from_metadata(json.loads(answer))
        env._state_ref = State(
            {
                "cs": 0,
                "budget": 10,
                "budget_init": 10,
                "queries_used": 0,
                "steps": 0,
                "trap_hit": False,
                "ok": False,
                "done": False,
            }
        )

        return env

    def test_api_call_valid(self, env_with_spec):
        """Test valid API call from state 0."""
        result = json.loads(env_with_spec.api_call("GET", "/users"))
        assert result["status"] == 200
        assert result["budget_left"] == 9

    def test_api_call_invalid_endpoint(self, env_with_spec):
        """Test API call to invalid endpoint."""
        result = json.loads(env_with_spec.api_call("GET", "/invalid"))
        assert result["status"] == 404

    def test_state_dependent_behavior(self, env_with_spec):
        """Test that same call from different states produces different results."""
        # From state 0: GET /users -> 200
        result0 = json.loads(env_with_spec.api_call("GET", "/users"))
        assert result0["status"] == 200

        # Transition to state 1 via POST /users
        env_with_spec._state_ref["cs"] = 1

        # From state 1: GET /users -> 404 (different!)
        result1 = json.loads(env_with_spec.api_call("GET", "/users"))
        assert result1["status"] == 404


class TestTaskGeneratorNewSkins:
    """Tests for TaskGenerator with new skins."""

    def test_protocol_split_generation(self):
        """Test split generation for ProtocolEnv."""
        generator = TaskGenerator(ProtocolEnv)
        split = generator.generate_split(
            seeds=[0, 1, 2],
            budget=25,
            subset_name="test",
            n_endpoints=3,
            n_states=3,
        )

        assert "test" in split
        assert split["test"]["budget"] == 25
        assert len(split["test"]["items"]) == 3
