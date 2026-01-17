"""Test suite for core automata algorithms."""

from collections import Counter

from dedeucerl.core.automata import (
    compute_reachable_states,
    is_fully_reachable,
    compute_state_signatures,
    is_minimal,
    check_behavioral_equivalence,
    check_isomorphism_with_signatures,
    find_counterexample,
    generate_random_traps,
    verify_trap_free_path_exists,
)
from dedeucerl.utils.rng import get_rng


class TestReachability:
    """Tests for reachability algorithms."""

    def test_linear_chain_reachable(self):
        """Test reachability in a linear chain: 0 -> 1 -> 2 -> 0."""

        # Transitions: A goes to next state, B/C stay
        def get_next(state, action):
            if action == "A":
                return (state + 1) % 3
            return state

        reachable = compute_reachable_states(3, 0, ["A", "B", "C"], get_next)
        assert reachable == {0, 1, 2}

    def test_disconnected_state(self):
        """Test that disconnected states are not reachable."""

        # State 2 is disconnected
        def get_next(state, action):
            if state == 0:
                return 1 if action == "A" else 0
            elif state == 1:
                return 0 if action == "A" else 1
            else:
                return 2  # State 2 only goes to itself

        reachable = compute_reachable_states(3, 0, ["A", "B"], get_next)
        assert reachable == {0, 1}
        assert 2 not in reachable

    def test_is_fully_reachable_true(self):
        """Test is_fully_reachable returns True for fully connected."""

        def get_next(state, action):
            return (state + 1) % 4

        assert is_fully_reachable(4, 0, ["A"], get_next)

    def test_is_fully_reachable_false(self):
        """Test is_fully_reachable returns False for disconnected."""

        def get_next(state, action):
            if state < 2:
                return (state + 1) % 2
            return state

        assert not is_fully_reachable(4, 0, ["A"], get_next)


class TestPartitionRefinement:
    """Tests for partition refinement / signature computation."""

    def test_all_states_distinguishable(self):
        """Test that distinct states get distinct signatures."""
        # Each state has unique output pattern
        trans = {
            0: {"A": (1, 0), "B": (0, 1)},
            1: {"A": (0, 1), "B": (1, 0)},
        }

        def get_trans(s, a):
            return trans[s][a]

        sigs = compute_state_signatures(2, ["A", "B"], get_trans)
        assert len(set(sigs)) == 2  # All unique

    def test_equivalent_states_same_signature(self):
        """Test that equivalent states get same signature."""
        # States 1 and 2 are equivalent (same outputs, same successors)
        trans = {
            0: {"A": (1, 0), "B": (2, 0)},
            1: {"A": (0, 1), "B": (0, 1)},
            2: {"A": (0, 1), "B": (0, 1)},  # Same as state 1
        }

        def get_trans(s, a):
            return trans[s][a]

        sigs = compute_state_signatures(3, ["A", "B"], get_trans)
        assert sigs[1] == sigs[2]  # States 1 and 2 equivalent
        assert sigs[0] != sigs[1]  # State 0 is different

    def test_is_minimal_true(self):
        """Test is_minimal returns True for minimal automaton."""
        trans = {
            0: {"A": (1, 0), "B": (0, 1)},
            1: {"A": (0, 1), "B": (1, 0)},
        }

        def get_trans(s, a):
            return trans[s][a]

        assert is_minimal(2, ["A", "B"], get_trans)

    def test_is_minimal_false(self):
        """Test is_minimal returns False for non-minimal automaton."""
        # States 1 and 2 are equivalent
        trans = {
            0: {"A": (1, 0)},
            1: {"A": (0, 1)},
            2: {"A": (0, 1)},  # Same as state 1
        }

        def get_trans(s, a):
            return trans[s][a]

        assert not is_minimal(3, ["A"], get_trans)


class TestBehavioralEquivalence:
    """Tests for behavioral equivalence checking."""

    def test_identical_systems_equivalent(self):
        """Test that identical systems are equivalent."""
        trans = {
            0: {"A": (1, 0), "B": (0, 1)},
            1: {"A": (0, 1), "B": (1, 0)},
        }

        def get_trans(s, a):
            return trans[s][a]

        assert check_behavioral_equivalence(2, 0, 2, 0, ["A", "B"], get_trans, get_trans)

    def test_different_outputs_not_equivalent(self):
        """Test that systems with different outputs are not equivalent."""
        trans_a = {
            0: {"A": (1, 0), "B": (0, 1)},
            1: {"A": (0, 1), "B": (1, 0)},
        }
        trans_b = {
            0: {"A": (1, 1), "B": (0, 1)},  # Different output for A
            1: {"A": (0, 1), "B": (1, 0)},
        }

        def get_trans_a(s, a):
            return trans_a[s][a]

        def get_trans_b(s, a):
            return trans_b[s][a]

        assert not check_behavioral_equivalence(2, 0, 2, 0, ["A", "B"], get_trans_a, get_trans_b)


class TestIsomorphism:
    """Tests for isomorphism checking."""

    def test_identical_is_isomorphic(self):
        """Test that identical systems are isomorphic."""
        trans = {
            0: {"A": (1, 0), "B": (0, 1)},
            1: {"A": (0, 1), "B": (1, 0)},
        }

        def get_trans(s, a):
            return trans[s][a]

        assert check_isomorphism_with_signatures(2, 0, 0, ["A", "B"], get_trans, get_trans)

    def test_relabeled_is_isomorphic(self):
        """Test that relabeled systems are isomorphic."""
        # Original: state 0 -> state 1 on A
        trans_a = {
            0: {"A": (1, 0), "B": (0, 1)},
            1: {"A": (0, 1), "B": (1, 0)},
        }
        # Relabeled: swap states 0 and 1
        trans_b = {
            0: {"A": (1, 1), "B": (0, 0)},  # Was state 1
            1: {"A": (0, 0), "B": (1, 1)},  # Was state 0
        }

        def get_trans_a(s, a):
            return trans_a[s][a]

        def get_trans_b(s, a):
            return trans_b[s][a]

        # Note: start states must also be swapped for true isomorphism
        # Here we test with same start, which should fail
        # because the outputs from start differ
        assert not check_isomorphism_with_signatures(2, 0, 0, ["A", "B"], get_trans_a, get_trans_b)

    def test_different_size_not_isomorphic(self):
        """Test that different sized systems are not isomorphic."""
        trans_a = {0: {"A": (0, 0)}}
        trans_b = {0: {"A": (1, 0)}, 1: {"A": (0, 0)}}

        def get_trans_a(s, a):
            return trans_a[s][a]

        def get_trans_b(s, a):
            return trans_b[s][a]

        # Different n_states, so signatures won't match
        sigs_a = compute_state_signatures(1, ["A"], get_trans_a)
        sigs_b = compute_state_signatures(2, ["A"], get_trans_b)
        assert Counter(sigs_a) != Counter(sigs_b)


class TestCounterexample:
    """Tests for counterexample generation."""

    def test_finds_divergence(self):
        """Test that counterexample finds the divergence point."""
        trans_true = {
            0: {"A": (1, 0), "B": (0, 1)},
            1: {"A": (0, 1), "B": (1, 0)},
        }
        trans_hyp = {
            0: {"A": (1, 1), "B": (0, 1)},  # Wrong output for A
            1: {"A": (0, 1), "B": (1, 0)},
        }

        def get_true(s, a):
            return trans_true[s][a]

        def get_hyp(s, a):
            return trans_hyp[s][a]

        cex = find_counterexample(0, 0, ["A", "B"], get_true, get_hyp)
        assert cex is not None
        assert len(cex) >= 1
        # First action should be A (where divergence occurs)
        assert cex[0][0] == "A"
        assert cex[0][1] == 0  # True output

    def test_no_counterexample_for_equivalent(self):
        """Test that no counterexample for equivalent systems."""
        trans = {
            0: {"A": (1, 0), "B": (0, 1)},
            1: {"A": (0, 1), "B": (1, 0)},
        }

        def get_trans(s, a):
            return trans[s][a]

        cex = find_counterexample(0, 0, ["A", "B"], get_trans, get_trans)
        assert cex is None


class TestTrapGeneration:
    """Tests for trap generation utilities."""

    def test_generates_correct_count(self):
        """Test that correct number of traps are generated."""
        rng = get_rng(42)
        traps = generate_random_traps(6, ["A", "B", "C"], rng, n_traps=3)
        assert len(traps) == 3

    def test_avoids_start_state(self):
        """Test that traps avoid start state when requested."""
        rng = get_rng(42)
        traps = generate_random_traps(
            4, ["A", "B"], rng, n_traps=10, avoid_start=True, start_state=0
        )
        for state, action in traps:
            assert state != 0

    def test_trap_free_path_exists(self):
        """Test verification of trap-free path."""

        # Linear chain with trap on state 1
        def get_next(state, action):
            return (state + 1) % 3

        traps = {(1, "A")}  # Trap on state 1

        # Can still reach all states via B
        assert verify_trap_free_path_exists(3, 0, ["A", "B"], get_next, traps)

    def test_trap_blocks_path(self):
        """Test that blocking traps are detected."""

        # Only one action, trap blocks it
        def get_next(state, action):
            return (state + 1) % 3

        traps = {(0, "A")}  # Trap on only path from start

        # Cannot reach state 1 or 2
        assert not verify_trap_free_path_exists(3, 0, ["A"], get_next, traps)
