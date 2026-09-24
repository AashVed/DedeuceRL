"""Core automata algorithms for active system identification.

This module provides domain-agnostic algorithms that kernels can leverage:
- Reachability analysis (BFS from start state)
- Partition refinement for minimization/signature computation
- Behavioral equivalence checking via product automaton
- Counterexample generation via BFS
- Trap generation utilities

These algorithms work with a generic transition system abstraction:
    TransitionFn: (state, action) -> (next_state, output)

Tasks provide adapters to convert their domain-specific structures
to this generic interface.

References:
- Angluin, D. (1987). Learning regular sets from queries and counterexamples.
- Hopcroft, J. (1971). An n log n algorithm for minimizing states in a finite automaton.
- Lee, D. & Yannakakis, M. (1996). Principles and methods of testing finite state machines.
"""

from __future__ import annotations

from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field
from random import Random
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

# Type variables for generic algorithms
State = TypeVar("State")
Action = TypeVar("Action")
Output = TypeVar("Output")


# ═══════════════════════════════════════════════════════════════════════════════
# Core Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TransitionSystem(Generic[State, Action, Output]):
    """Generic labeled transition system abstraction.

    This is the universal interface for all automata-like structures:
    - Mealy machines: State × Symbol → State × Output
    - Moore machines: State × Symbol → State (output is state-based)
    - DFA/NFA: State × Symbol → State (output is accept/reject)
    - Protocols: State × (Method, Endpoint) → State × StatusCode

    Tasks convert their domain-specific structures to this interface
    to leverage core algorithms.

    Attributes:
        n_states: Number of states (states are 0..n_states-1).
        start: Initial state.
        actions: List of valid actions.
        transition_fn: Function (state, action) -> (next_state, output).
        outputs: Optional list of valid outputs (for validation).
    """

    n_states: int
    start: State
    actions: List[Action]
    transition_fn: Callable[[State, Action], Tuple[State, Output]]
    outputs: List[Output] = field(default_factory=list)

    def transition(self, state: State, action: Action) -> Tuple[State, Output]:
        """Execute a transition. Returns (next_state, output)."""
        return self.transition_fn(state, action)

    def get_output(self, state: State, action: Action) -> Output:
        """Get output for a transition without caring about next state."""
        _, output = self.transition(state, action)
        return output

    def get_next_state(self, state: State, action: Action) -> State:
        """Get next state for a transition without caring about output."""
        next_state, _ = self.transition(state, action)
        return next_state


@dataclass
class CounterexampleTrace(Generic[Action, Output]):
    """A counterexample showing where two systems differ.

    Attributes:
        trace: Sequence of (action, expected_output) pairs.
        divergence_index: Index where outputs first differ.
        expected_output: What the ground truth produces.
        actual_output: What the hypothesis produces (if available).
    """

    trace: List[Tuple[Action, Output]]
    divergence_index: int = -1
    expected_output: Optional[Output] = None
    actual_output: Optional[Output] = None

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list format for JSON serialization."""
        return [{"action": a, "output": o} for a, o in self.trace]


# ═══════════════════════════════════════════════════════════════════════════════
# Reachability Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def compute_reachable_states(
    n_states: int,
    start: int,
    actions: List[Any],
    get_next_state: Callable[[int, Any], int],
) -> Set[int]:
    """Compute all states reachable from start via BFS.

    This is the fundamental reachability algorithm used to verify
    that generated systems have no unreachable states.

    Args:
        n_states: Total number of states.
        start: Initial state.
        actions: List of valid actions.
        get_next_state: Function (state, action) -> next_state.

    Returns:
        Set of reachable state indices.

    Example:
        >>> def next_fn(s, a): return transitions[s][a][0]
        >>> reachable = compute_reachable_states(5, 0, ["A", "B", "C"], next_fn)
        >>> assert len(reachable) == 5  # All states reachable
    """
    visited: Set[int] = {start}
    frontier: List[int] = [start]

    while frontier:
        next_frontier: List[int] = []
        for state in frontier:
            for action in actions:
                try:
                    next_state = get_next_state(state, action)
                    if next_state not in visited:
                        visited.add(next_state)
                        next_frontier.append(next_state)
                except (KeyError, IndexError):
                    continue
        frontier = next_frontier

    return visited


def is_fully_reachable(
    n_states: int,
    start: int,
    actions: List[Any],
    get_next_state: Callable[[int, Any], int],
) -> bool:
    """Check if all states are reachable from start.

    Args:
        n_states: Total number of states.
        start: Initial state.
        actions: List of valid actions.
        get_next_state: Function (state, action) -> next_state.

    Returns:
        True if all states 0..n_states-1 are reachable.
    """
    reachable = compute_reachable_states(n_states, start, actions, get_next_state)
    return len(reachable) == n_states


# ═══════════════════════════════════════════════════════════════════════════════
# Partition Refinement (Minimization / Signature Computation)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_state_signatures(
    n_states: int,
    actions: List[Any],
    get_transition: Callable[[int, Any], Tuple[int, Any]],
    max_iterations: Optional[int] = None,
) -> List[int]:
    """Compute stable state signatures via partition refinement.

    This is the core algorithm for:
    1. Checking minimality (all signatures unique)
    2. Isomorphism checking (signature multisets must match)
    3. State equivalence classes (states with same signature are equivalent)

    The algorithm iteratively refines partitions based on:
    - Immediate outputs for each action
    - Signature of successor states for each action

    Convergence is guaranteed in O(n) iterations for n states.

    Args:
        n_states: Number of states.
        actions: List of valid actions (must be consistent order).
        get_transition: Function (state, action) -> (next_state, output).
        max_iterations: Maximum refinement iterations (default: 4*n_states).

    Returns:
        List of signature IDs, one per state. States with the same ID
        are behaviorally equivalent.

    Example:
        >>> def trans(s, a): return transitions[s][a]
        >>> sigs = compute_state_signatures(5, ["A", "B", "C"], trans)
        >>> assert len(set(sigs)) == 5  # All states distinguishable (minimal)
    """
    if max_iterations is None:
        max_iterations = max(1, 4 * n_states)

    def reindex(tuples: List[Tuple]) -> List[int]:
        """Map tuples to dense integer IDs."""
        mapping: Dict[Tuple, int] = {}
        result: List[int] = []
        for t in tuples:
            if t not in mapping:
                mapping[t] = len(mapping)
            result.append(mapping[t])
        return result

    # Initial signatures based on immediate outputs only
    initial_sigs: List[Tuple] = []
    for state in range(n_states):
        outputs = tuple(get_transition(state, a)[1] for a in actions)
        initial_sigs.append(outputs)

    signatures = reindex(initial_sigs)

    # Iteratively refine until stable
    for _ in range(max_iterations):
        refined_sigs: List[Tuple] = []
        for state in range(n_states):
            # Build signature: (output_a, sig[next_a], output_b, sig[next_b], ...)
            sig_parts: List[Any] = []
            for action in actions:
                next_state, output = get_transition(state, action)
                sig_parts.append(output)
                sig_parts.append(signatures[next_state])
            refined_sigs.append(tuple(sig_parts))

        new_signatures = reindex(refined_sigs)
        if new_signatures == signatures:
            break
        signatures = new_signatures

    return signatures


def is_minimal(
    n_states: int,
    actions: List[Any],
    get_transition: Callable[[int, Any], Tuple[int, Any]],
) -> bool:
    """Check if an automaton is minimal (no equivalent states).

    An automaton is minimal if all states have unique signatures,
    meaning no two states are behaviorally equivalent.

    Args:
        n_states: Number of states.
        actions: List of valid actions.
        get_transition: Function (state, action) -> (next_state, output).

    Returns:
        True if all states are distinguishable.
    """
    signatures = compute_state_signatures(n_states, actions, get_transition)
    return len(set(signatures)) == n_states


# ═══════════════════════════════════════════════════════════════════════════════
# Behavioral Equivalence / Isomorphism Checking
# ═══════════════════════════════════════════════════════════════════════════════


def check_behavioral_equivalence(
    n_states_a: int,
    start_a: int,
    n_states_b: int,
    start_b: int,
    actions: List[Any],
    get_transition_a: Callable[[int, Any], Tuple[int, Any]],
    get_transition_b: Callable[[int, Any], Tuple[int, Any]],
) -> bool:
    """Check if two transition systems are behaviorally equivalent.

    Two systems are equivalent if they produce the same output sequence
    for every possible input sequence starting from their initial states.

    This is checked via BFS on the product automaton (state_a, state_b).
    If we ever find a transition where outputs differ, they're not equivalent.

    Args:
        n_states_a: Number of states in system A.
        start_a: Initial state of system A.
        n_states_b: Number of states in system B.
        start_b: Initial state of system B.
        actions: List of valid actions (same for both systems).
        get_transition_a: Transition function for system A.
        get_transition_b: Transition function for system B.

    Returns:
        True if systems are behaviorally equivalent.

    Note:
        This checks behavioral equivalence, not structural isomorphism.
        Two systems can be behaviorally equivalent with different structures
        if one is a minimization of the other.
    """
    if n_states_a <= 0 or n_states_b <= 0:
        return False
    if start_a < 0 or start_b < 0:
        return False

    visited: Set[Tuple[int, int]] = set()
    queue: deque = deque([(start_a, start_b)])
    visited.add((start_a, start_b))

    while queue:
        state_a, state_b = queue.popleft()

        for action in actions:
            try:
                next_a, output_a = get_transition_a(state_a, action)
                next_b, output_b = get_transition_b(state_b, action)
            except (KeyError, IndexError, TypeError):
                return False

            # Outputs must match
            if output_a != output_b:
                return False

            # Validate next states
            if not (0 <= next_a < n_states_a) or not (0 <= next_b < n_states_b):
                return False

            pair = (next_a, next_b)
            if pair not in visited:
                visited.add(pair)
                queue.append(pair)

    return True


def check_isomorphism_with_signatures(
    n_states: int,
    start_true: int,
    start_hyp: int,
    actions: List[Any],
    get_transition_true: Callable[[int, Any], Tuple[int, Any]],
    get_transition_hyp: Callable[[int, Any], Tuple[int, Any]],
) -> bool:
    """Check if hypothesis is isomorphic to ground truth (up to state relabeling).

    This is a more rigorous check than behavioral equivalence:
    it verifies that there exists a bijective mapping between states
    that preserves all transitions and outputs.

    Algorithm:
    1. Compute signatures for both systems
    2. Check signature multisets match
    3. Use backtracking to find a valid state mapping

    Args:
        n_states: Number of states (must be same for both).
        start_true: Start state of ground truth.
        start_hyp: Start state of hypothesis.
        actions: List of valid actions.
        get_transition_true: Transition function for ground truth.
        get_transition_hyp: Transition function for hypothesis.

    Returns:
        True if systems are isomorphic.
    """
    # Compute signatures for both
    sig_true = compute_state_signatures(n_states, actions, get_transition_true)
    sig_hyp = compute_state_signatures(n_states, actions, get_transition_hyp)

    # Signature multisets must match
    if Counter(sig_true) != Counter(sig_hyp):
        return False

    # Start states must have same signature
    if sig_true[start_true] != sig_hyp[start_hyp]:
        return False

    # Build signature classes
    classes_true: Dict[int, List[int]] = defaultdict(list)
    classes_hyp: Dict[int, List[int]] = defaultdict(list)
    for i, sig in enumerate(sig_true):
        classes_true[sig].append(i)
    for i, sig in enumerate(sig_hyp):
        classes_hyp[sig].append(i)

    # Initialize mapping with start -> start
    mapping: Dict[int, int] = {start_hyp: start_true}
    used_true: Set[int] = {start_true}

    def consistent(s_hyp: int, s_true: int) -> bool:
        """Check if mapping s_hyp -> s_true is locally consistent."""
        for action in actions:
            next_hyp, out_hyp = get_transition_hyp(s_hyp, action)
            next_true, out_true = get_transition_true(s_true, action)

            if out_hyp != out_true:
                return False

            if next_hyp in mapping:
                if mapping[next_hyp] != next_true:
                    return False
            else:
                if sig_hyp[next_hyp] != sig_true[next_true]:
                    return False

        return True

    # Order classes by size for efficient backtracking
    class_order = sorted(classes_true.keys(), key=lambda c: len(classes_true[c]))

    # Build candidate pairs per class (excluding already-mapped start)
    per_class: List[Tuple[int, List[int], List[int]]] = []
    for sig_id in class_order:
        hyps = [s for s in classes_hyp[sig_id] if s != start_hyp]
        trues = [s for s in classes_true[sig_id] if s != start_true]
        if len(hyps) != len(trues):
            return False
        if hyps:
            per_class.append((sig_id, hyps, trues))

    def backtrack(class_idx: int) -> bool:
        if class_idx >= len(per_class):
            return len(mapping) == n_states

        _, hyps, trues = per_class[class_idx]
        unmapped_hyps = [h for h in hyps if h not in mapping]

        if not unmapped_hyps:
            return backtrack(class_idx + 1)

        # Build candidates for each unmapped hypothesis state
        candidates: Dict[int, List[int]] = {}
        for h in unmapped_hyps:
            candidates[h] = [t for t in trues if t not in used_true and consistent(h, t)]
            if not candidates[h]:
                return False

        # Try to assign in order of fewest candidates (MRV heuristic)
        sorted_hyps = sorted(unmapped_hyps, key=lambda h: len(candidates[h]))

        def assign(idx: int) -> bool:
            if idx >= len(sorted_hyps):
                return backtrack(class_idx + 1)

            h = sorted_hyps[idx]
            for t in candidates[h]:
                if t in used_true:
                    continue

                mapping[h] = t
                used_true.add(t)

                if all(consistent(h2, mapping[h2]) for h2 in mapping):
                    if assign(idx + 1):
                        return True

                del mapping[h]
                used_true.remove(t)

            return False

        return assign(0)

    return backtrack(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Counterexample Generation
# ═══════════════════════════════════════════════════════════════════════════════


def find_counterexample(
    start_true: int,
    start_hyp: int,
    actions: List[Any],
    get_transition_true: Callable[[int, Any], Tuple[int, Any]],
    get_transition_hyp: Callable[[int, Any], Tuple[int, Any]],
    max_depth: int = 100,
) -> Optional[List[Tuple[Any, Any]]]:
    """Find shortest counterexample showing behavioral divergence.

    Uses BFS on the product automaton to find the shortest input sequence
    where the two systems produce different outputs.

    Args:
        start_true: Initial state of ground truth.
        start_hyp: Initial state of hypothesis.
        actions: List of valid actions.
        get_transition_true: Transition function for ground truth.
        get_transition_hyp: Transition function for hypothesis.
        max_depth: Maximum search depth.

    Returns:
        List of (action, expected_output) pairs showing the divergence,
        or None if systems are equivalent up to max_depth.

    Example:
        >>> cex = find_counterexample(0, 0, ["A", "B"], trans_true, trans_hyp)
        >>> if cex:
        ...     print(f"Divergence at: {cex}")
    """
    visited: Set[Tuple[int, int]] = set()
    queue: deque = deque([((start_true, start_hyp), [])])
    visited.add((start_true, start_hyp))

    while queue:
        (s_true, s_hyp), path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for action in actions:
            try:
                next_true, out_true = get_transition_true(s_true, action)
            except (KeyError, IndexError, TypeError):
                # Ground truth should always be valid
                continue

            try:
                next_hyp, out_hyp = get_transition_hyp(s_hyp, action)
            except (KeyError, IndexError, TypeError):
                # Hypothesis is malformed - return trace to this point
                return _build_trace(path + [action], start_true, get_transition_true)

            if out_true != out_hyp:
                # Found divergence!
                return _build_trace(path + [action], start_true, get_transition_true)

            pair = (next_true, next_hyp)
            if pair not in visited:
                visited.add(pair)
                queue.append((pair, path + [action]))

    return None


def _build_trace(
    actions: List[Any],
    start: int,
    get_transition: Callable[[int, Any], Tuple[int, Any]],
) -> List[Tuple[Any, Any]]:
    """Build a trace of (action, output) pairs by simulating."""
    trace: List[Tuple[Any, Any]] = []
    state = start
    for action in actions:
        try:
            next_state, output = get_transition(state, action)
            trace.append((action, output))
            state = next_state
        except (KeyError, IndexError, TypeError):
            trace.append((action, None))
            break
    return trace


# ═══════════════════════════════════════════════════════════════════════════════
# Trap Generation Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def generate_random_traps(
    n_states: int,
    actions: List[Any],
    rng: Any,
    n_traps: Optional[int] = None,
    avoid_start: bool = True,
    start_state: int = 0,
) -> List[Tuple[int, Any]]:
    """Generate random trap (state, action) pairs.

    Traps are transitions that, once taken, make success impossible.
    This utility generates traps uniformly at random while optionally
    avoiding traps from the start state (to ensure solvability).

    Args:
        n_states: Number of states.
        actions: List of valid actions.
        rng: Random number generator (e.g., from get_rng(seed)).
        n_traps: Number of traps to generate (default: n_states // 3).
        avoid_start: If True, don't place traps on transitions from start.
        start_state: The start state to avoid if avoid_start is True.

    Returns:
        List of (state, action) trap pairs.
    """
    if n_traps is None:
        n_traps = max(1, n_states // 3)

    # Build candidate pool
    candidates: List[Tuple[int, Any]] = []
    for state in range(n_states):
        if avoid_start and state == start_state:
            continue
        for action in actions:
            candidates.append((state, action))

    if not candidates:
        return []

    # Sample without replacement
    n_traps = min(n_traps, len(candidates))
    selected: Set[Tuple[int, Any]] = set()

    while len(selected) < n_traps:
        idx = rng.randrange(len(candidates))
        selected.add(candidates[idx])

    return list(selected)


def verify_trap_free_path_exists(
    n_states: int,
    start: int,
    actions: List[Any],
    get_next_state: Callable[[int, Any], int],
    traps: Set[Tuple[int, Any]],
    target_states: Optional[Set[int]] = None,
) -> bool:
    """Verify that a trap-free path exists from start to target states.

    This ensures the problem is solvable: there must be at least one
    way to explore the system without hitting traps.

    Args:
        n_states: Number of states.
        start: Initial state.
        actions: List of valid actions.
        get_next_state: Function (state, action) -> next_state.
        traps: Set of (state, action) trap pairs.
        target_states: States that must be reachable trap-free.
                      If None, checks if all states are reachable.

    Returns:
        True if all target states are reachable without hitting traps.
    """
    if target_states is None:
        target_states = set(range(n_states))

    visited: Set[int] = {start}
    frontier: List[int] = [start]

    while frontier:
        next_frontier: List[int] = []
        for state in frontier:
            for action in actions:
                if (state, action) in traps:
                    continue  # Skip trap transitions

                try:
                    next_state = get_next_state(state, action)
                    if next_state not in visited:
                        visited.add(next_state)
                        next_frontier.append(next_state)
                except (KeyError, IndexError):
                    continue

        frontier = next_frontier

    return target_states.issubset(visited)


def is_strongly_connected(
    n_states: int,
    actions: List[Any],
    get_next_state: Callable[[int, Any], int],
    blocked: Optional[Set[Tuple[int, Any]]] = None,
) -> bool:
    """Whether every state can reach every other, excluding blocked edges.

    Two traversals from state zero, forward and backward, suffice. This checks
    recoverability without requiring any particular action or route to provide it.
    """
    if n_states < 1:
        return False
    forward: List[List[int]] = [[] for _ in range(n_states)]
    reverse: List[List[int]] = [[] for _ in range(n_states)]
    for state in range(n_states):
        for action in actions:
            if blocked and (state, action) in blocked:
                continue
            target = get_next_state(state, action)
            forward[state].append(target)
            reverse[target].append(state)
    for graph in (forward, reverse):
        seen = {0}
        pending = [0]
        for state in pending:
            for target in graph[state]:
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        if len(seen) != n_states:
            return False
    return True


def _partition_counts(slots: int, groups: int) -> List[List[int]]:
    """Stirling numbers S(i, j): partitions of i slots into j nonempty blocks."""
    counts = [[0] * (groups + 1) for _ in range(slots + 1)]
    counts[0][0] = 1
    for i in range(1, slots + 1):
        for j in range(1, min(i, groups) + 1):
            counts[i][j] = j * counts[i - 1][j] + counts[i - 1][j - 1]
    return counts


def _sample_surjection(counts: List[List[int]], groups: int, rng: Random) -> List[int]:
    """Uniformly map slots onto labelled destinations, each occurring at least once."""
    labels = list(range(groups))
    rng.shuffle(labels)
    targets = [0] * (len(counts) - 1)
    for i in range(len(targets), 0, -1):
        # Either the last slot starts its own block, or joins one of j blocks.
        if rng.randrange(counts[i][groups]) < counts[i - 1][groups - 1]:
            groups -= 1
            targets[i - 1] = labels[groups]
        else:
            targets[i - 1] = labels[rng.randrange(groups)]
    return targets


def sample_strongly_connected_transitions(
    n_states: int, actions: List[Any], rng: Random
) -> Dict[int, Dict[Any, int]]:
    """Sample uniformly over complete strongly connected labelled transition maps.

    Positive indegree is necessary for strong connectivity. Sampling onto maps
    first avoids the vanishing acceptance of unrestricted independent targets.
    SCC rejection imposes no planted cycle, permutation action, or spanning tree.
    All valid labelled maps have equal probability, conditional on returning.

    Exact partition counts use O(n_states**2 * len(actions)) integer entries;
    integer sizes also grow. This is intended for small/medium finite-state
    benchmarks (measured through 256 states), not very large graphs. Counts are
    local to this call rather than retained in a global cache.
    """
    if n_states < 1 or not actions or len(set(actions)) != len(actions):
        raise ValueError("Require n_states >= 1 and distinct, nonempty actions")
    counts = _partition_counts(n_states * len(actions), n_states)
    for _ in range(10_000):
        targets = iter(_sample_surjection(counts, n_states, rng))
        transitions = {s: {a: next(targets) for a in actions} for s in range(n_states)}
        if is_strongly_connected(n_states, actions, lambda s, a: transitions[s][a]):
            return transitions
    raise RuntimeError(f"Failed to sample strongly connected transitions for n_states={n_states}")
