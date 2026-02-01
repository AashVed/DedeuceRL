"""Test suite for MealySkin."""

import json
import pytest

from dedeucerl.skins import MealyEnv
from dedeucerl.core import make_rubric, TaskGenerator
from datasets import Dataset


class TestMealyGeneration:
    """Tests for Mealy machine generation."""

    def test_basic_generation(self):
        """Test basic machine generation."""
        system = MealyEnv.generate_system_static(seed=42, n_states=3, trap=True)

        assert "table" in system
        assert "trap_pairs" in system

        table = system["table"]
        assert table["n"] == 3
        assert table["start"] == 0
        assert len(table["trans"]) == 3

    def test_deterministic(self):
        """Test that same seed produces same machine."""
        sys1 = MealyEnv.generate_system_static(seed=123, n_states=4)
        sys2 = MealyEnv.generate_system_static(seed=123, n_states=4)

        assert sys1["table"] == sys2["table"]

    def test_different_seeds(self):
        """Test that different seeds produce different machines."""
        sys1 = MealyEnv.generate_system_static(seed=1, n_states=4)
        sys2 = MealyEnv.generate_system_static(seed=2, n_states=4)

        # Very unlikely to be identical
        assert sys1["table"]["trans"] != sys2["table"]["trans"]

    def test_no_trap(self):
        """Test generation without traps."""
        system = MealyEnv.generate_system_static(seed=42, n_states=3, trap=False)
        assert system["trap_pairs"] == []

    def test_all_states_have_transitions(self):
        """Test that all states have all transitions."""
        system = MealyEnv.generate_system_static(seed=42, n_states=5)
        trans = system["table"]["trans"]

        for s in range(5):
            assert str(s) in trans
            for a in ["A", "B", "C"]:
                assert a in trans[str(s)]
                ns, out = trans[str(s)][a]
                assert 0 <= ns < 5
                assert out in [0, 1, 2]


class TestMealyIsomorphism:
    """Tests for isomorphism checking."""

    @pytest.fixture
    def env_with_table(self):
        """Create an environment with a known table."""
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
        env = MealyEnv(dataset=dataset, rubric=rubric, feedback=False, max_turns=20)
        env._configure_from_metadata(json.loads(answer))
        return env

    def test_exact_match(self, env_with_table):
        """Test that exact match is recognized."""
        hypothesis = {"n": 2, "start": 0, "trans": env_with_table._trans}
        assert env_with_table.is_isomorphic(hypothesis, env_with_table._ground_truth)

    def test_wrong_output(self, env_with_table):
        """Test that wrong output is rejected."""
        wrong = {
            "n": 2,
            "start": 0,
            "trans": {
                0: {"A": (1, 1), "B": (0, 1), "C": (0, 2)},  # Changed output
                1: {"A": (0, 1), "B": (1, 0), "C": (1, 2)},
            },
        }
        assert not env_with_table.is_isomorphic(wrong, env_with_table._ground_truth)

    def test_wrong_transition(self, env_with_table):
        """Test that wrong transition is rejected."""
        wrong = {
            "n": 2,
            "start": 0,
            "trans": {
                0: {"A": (0, 0), "B": (0, 1), "C": (0, 2)},  # Changed next state
                1: {"A": (0, 1), "B": (1, 0), "C": (1, 2)},
            },
        }
        assert not env_with_table.is_isomorphic(wrong, env_with_table._ground_truth)

    def test_wrong_size(self, env_with_table):
        """Test that wrong size is rejected."""
        wrong = {
            "n": 3,
            "start": 0,
            "trans": {
                0: {"A": (1, 0), "B": (0, 1), "C": (0, 2)},
                1: {"A": (0, 1), "B": (1, 0), "C": (1, 2)},
                2: {"A": (0, 0), "B": (1, 1), "C": (2, 2)},
            },
        }
        assert not env_with_table.is_isomorphic(wrong, env_with_table._ground_truth)


class TestMealyCounterexample:
    """Tests for counterexample generation."""

    @pytest.fixture
    def env_with_table(self):
        """Create an environment for counterexample testing."""
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
        return env

    def test_counterexample_for_wrong_output(self, env_with_table):
        """Test counterexample generation for wrong output."""
        wrong = {
            "n": 2,
            "start": 0,
            "trans": {
                0: {"A": (1, 1), "B": (0, 1), "C": (0, 2)},  # Changed output
                1: {"A": (0, 1), "B": (1, 0), "C": (1, 2)},
            },
        }
        cex = env_with_table.get_counterexample(wrong, env_with_table._ground_truth)
        assert cex is not None
        assert len(cex) > 0
        # First step should show the difference
        assert cex[0]["in"] == "A"
        assert cex[0]["out"] == 0  # True output


class TestTaskGenerator:
    """Tests for TaskGenerator."""

    def test_generate_split(self):
        """Test split generation."""
        generator = TaskGenerator(MealyEnv)
        split = generator.generate_split(
            seeds=[0, 1, 2],
            budget=20,
            subset_name="test",
            n_states=3,
        )

        assert "test" in split
        assert split["test"]["budget"] == 20
        assert len(split["test"]["items"]) == 3

    def test_derive_max_turns(self):
        """Test max_turns derivation."""
        generator = TaskGenerator(MealyEnv)

        # Without feedback
        mt = generator.derive_max_turns(budget=25, n_states=5, feedback=False)
        assert mt == 27  # 25 + 2

        # With feedback
        mt = generator.derive_max_turns(budget=25, n_states=5, feedback=True)
        assert mt == 35  # 25 + max(3, min(10, 10))
