"""
DedeuceRL Skin Template
========================

This is a template for creating new skins. Copy this file to create your own.

Quick Start:
1. Copy: cp _template.py myskin.py
2. Replace 'MySkin' with your skin name throughout
3. Implement all TODO sections (search for "TODO")
4. Register in skins/__init__.py:

   from .myskin import MySkinEnv
   SKIN_REGISTRY["myskin"] = MySkinEnv

Testing:
    python -c "
    from dedeucerl.skins import MySkinEnv
    print(MySkinEnv.generate_system_static(seed=42))
    "

Skin Design Checklist:
□ What is the hidden system? (state machine, protocol, etc.)
□ What actions can the agent take? (how does it probe?)
□ What observations does the agent receive?
□ What does a correct hypothesis look like? (JSON schema)
□ When are two hypotheses equivalent? (isomorphism)
□ What counterexamples help debug wrong hypotheses?
□ Are there safety traps? (forbidden actions)

Core Utilities Available (from dedeucerl.core.automata):
─────────────────────────────────────────────────────────
Instead of reimplementing common algorithms, leverage core utilities:

- is_fully_reachable(n_states, start, actions, get_next_state)
  → Verify all states are reachable from start

- is_minimal(n_states, actions, get_transition)
  → Check no two states are behaviorally equivalent

- compute_state_signatures(n_states, actions, get_transition)
  → Partition refinement for equivalence classes

- check_isomorphism_with_signatures(n, start_a, start_b, actions, trans_a, trans_b)
  → Check if two systems are isomorphic (up to state relabeling)

- find_counterexample(start_a, start_b, actions, trans_a, trans_b)
  → Find shortest distinguishing input sequence

- generate_random_traps(n_states, actions, rng, n_traps, avoid_start)
  → Generate random trap (state, action) pairs

See MealyEnv for a reference implementation using these utilities.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dedeucerl.core.env import HiddenSystemEnv
from dedeucerl.core.config import SkinConfig
from dedeucerl.core.automata import (
    is_fully_reachable,
    is_minimal,
    check_isomorphism_with_signatures,
    find_counterexample,
    generate_random_traps,
    verify_trap_free_path_exists,
)
from dedeucerl.utils.rng import get_rng
from dedeucerl.utils import error_invalid_argument, error_invalid_json


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Define Your Domain Constants
# ═══════════════════════════════════════════════════════════════════════════════
#
# These define what actions the agent can take and what outputs the system
# can produce. Customize these for your domain.
#
# Examples:
#   - Mealy machine: ACTIONS = ["A", "B", "C"], OUTPUTS = [0, 1, 2]
#   - API skin: ACTIONS = ["GET", "POST", "DELETE"], endpoints list
#   - Graph skin: ACTIONS = ["N", "S", "E", "W"], room IDs
#   - Circuit skin: ACTIONS = ["HIGH", "LOW"], OUTPUTS = ["HIGH", "LOW"]

ACTIONS = ["A", "B", "C"]  # TODO: Define your action alphabet
OUTPUTS = [0, 1, 2]  # TODO: Define possible outputs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Configure Your Skin
# ═══════════════════════════════════════════════════════════════════════════════

MYSKIN_CONFIG = SkinConfig(
    # Correctness: Accept solutions that are equivalent up to relabeling?
    # Set True if multiple state labelings can represent the same system.
    isomorphism_check=True,
    # Feedback: Provide counterexamples when hypothesis is wrong?
    # This helps agents learn but makes the task easier.
    feedback_enabled=False,
    # Traps: Are there forbidden actions that cause failure?
    trap_enabled=True,
    trap_ends_episode=False,  # True = immediate fail, False = can still submit
    # Budget: How many probe actions before timeout?
    default_budget=25,
    submission_cost=1,  # Cost per submission attempt
    # Episode limits
    max_turns=64,
    # Metadata
    skin_name="myskin",  # TODO: Change this
    skin_version="1.0",
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Implement Your Skin
# ═══════════════════════════════════════════════════════════════════════════════


class MySkinEnv(HiddenSystemEnv):
    """
    TODO: Describe your skin in detail.

    Hidden system: [What the agent is trying to identify]
    Probe action: [How the agent queries the system]
    Hypothesis: [What format the agent submits]

    Example:
        Hidden system: A finite-state transducer with 3-5 states
        Probe action: Execute input symbol, observe output
        Hypothesis: Complete transition table as JSON
    """

    config = MYSKIN_CONFIG

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Add skin-specific instance variables
        self._ground_truth: Dict[str, Any] = {}
        self._n_states: int = 0
        self._start: int = 0

    # ───────────────────────────────────────────────────────────────────────────
    # REQUIRED: Abstract Method Implementations
    # ───────────────────────────────────────────────────────────────────────────

    def _configure_from_metadata(self, meta: Dict[str, Any]) -> None:
        """
        Parse ground truth from episode metadata.

        Called at episode start with JSON-decoded answer data.
        Must set: self._ground_truth, self._trap_pairs

        Args:
            meta: Parsed JSON containing system spec, traps, budget, etc.
        """
        # TODO: Parse your ground truth structure
        system = meta.get("system", {})
        self._ground_truth = system
        self._n_states = int(system.get("n_states", 0))
        self._start = int(system.get("start", 0))

        # Parse trap pairs: set of (state, action) that trigger trap
        self._trap_pairs = set((int(s), str(a)) for s, a in meta.get("traps", []))

    def _get_start_state(self) -> Any:
        """Return the initial state for the hidden system."""
        return self._start

    def _get_tools(self) -> List:
        """
        Return list of tool methods exposed to the agent.

        Tools must:
        - Have explicit type hints (used to build JSON schema)
        - Return JSON strings
        - Update state via self._state()
        """
        return [self.probe, self.submit]

    # ───────────────────────────────────────────────────────────────────────────
    # REQUIRED: Tool Methods (Exposed to LLM Agent)
    # ───────────────────────────────────────────────────────────────────────────

    def probe(self, action: str) -> str:
        """
        Execute a probe action on the hidden system.

        TODO: Document your probe action semantics.

        Args:
            action: One of the valid actions (e.g., 'A', 'B', 'C').

        Returns:
            JSON string with observation and state info.
        """
        state = self._state()

        # Standard checks
        if state["done"]:
            return self._episode_finished()

        # Each probe consumes budget (episode ends at 0).
        if not self._consume_budget(1):
            return self._budget_exhausted()

        state["steps"] = state.get("steps", 0) + 1

        # Validate action
        if action not in ACTIONS:
            return self._tool_error(
                error_invalid_argument(
                    f"Invalid action '{action}'. Must be one of: {ACTIONS}",
                    details={"received": action, "valid": ACTIONS},
                )
            )

        cs = state["cs"]

        # TODO: Execute action on hidden system, compute next state and output
        ns = cs  # Next state
        output = 0  # Observation

        # Update state
        state["cs"] = ns

        # Check for trap - use base helper for config-aware behavior
        if (cs, action) in self._trap_pairs:
            self._mark_trap_hit()

        return json.dumps(
            {
                "output": output,
                "budget_left": state["budget"],
                "step": state["steps"],
                "trap_hit": state["trap_hit"],
                "queries_used": state["queries_used"],
            }
        )

    def submit(self, hypothesis_json: str) -> str:
        """
        Submit a hypothesis about the hidden system.

        TODO: Document your hypothesis JSON schema.

        Args:
            hypothesis_json: JSON string matching your schema.

        Returns:
            JSON string with ok, budget_left, counterexample (if feedback enabled).
        """
        state = self._state()

        if state["done"]:
            return self._episode_finished()

        # Every submission attempt consumes budget.
        if not self._consume_budget(int(self.config.submission_cost)):
            return self._budget_exhausted()

        # Parse hypothesis
        try:
            hypothesis = json.loads(hypothesis_json)
        except Exception:
            return self._tool_error(error_invalid_json("hypothesis_json"), extra={"ok": False})

        # TODO: Validate hypothesis structure
        # Check required fields, valid ranges, etc.

        # Check correctness (with isomorphism if configured)
        ok = self._check_equivalence(hypothesis)

        if not ok:
            # Generate counterexample if feedback enabled
            cex = None
            if self.feedback_enabled:
                cex = self._generate_counterexample(hypothesis)

            payload = {
                "ok": False,
                "budget_left": state["budget"],
                "queries_used": state["queries_used"],
                "trap_hit": state["trap_hit"],
            }
            if cex is not None:
                payload["counterexample"] = cex
            return json.dumps(payload)

        # Correct submission
        state["ok"] = not state["trap_hit"]
        state["done"] = True

        return json.dumps(
            {
                "ok": state["ok"],
                "budget_left": state["budget"],
                "queries_used": state["queries_used"],
                "trap_hit": state["trap_hit"],
            }
        )

    # ───────────────────────────────────────────────────────────────────────────
    # REQUIRED: Equivalence Checking (using core.automata utilities)
    # ───────────────────────────────────────────────────────────────────────────

    def _check_equivalence(self, hypothesis: Dict) -> bool:
        """
        Check if hypothesis is equivalent to ground truth.

        Uses core.automata.check_isomorphism_with_signatures for the heavy
        lifting. You just need to provide adapter functions.

        Args:
            hypothesis: Parsed hypothesis dict.

        Returns:
            True if equivalent, False otherwise.
        """
        # Parse hypothesis structure
        try:
            n = int(hypothesis.get("n_states", -1))
            start = int(hypothesis.get("start", -1))
            trans = hypothesis.get("transitions", {})
        except Exception:
            return False

        if n != self._n_states or start != self._start:
            return False

        # Create adapters for core.automata functions
        def get_trans_true(s: int, a: str) -> Tuple[int, Any]:
            # TODO: Adapt to your ground truth structure
            return self._ground_truth["transitions"][str(s)][a]

        def get_trans_hyp(s: int, a: str) -> Tuple[int, Any]:
            # TODO: Adapt to your hypothesis structure
            return tuple(trans[str(s)][a])

        # Use core isomorphism checking
        return check_isomorphism_with_signatures(
            n, self._start, start, ACTIONS, get_trans_true, get_trans_hyp
        )

    # ───────────────────────────────────────────────────────────────────────────
    # OPTIONAL: Counterexample Generation (using core.automata utilities)
    # ───────────────────────────────────────────────────────────────────────────
    #
    # Counterexamples help agents debug their hypotheses. The format is
    # skin-specific and should show WHERE the hypothesis differs from ground truth.
    #
    # Examples by skin:
    #   - Mealy: Trace showing [{"in": "A", "out": 0}, {"in": "B", "out": 1}]
    #   - Protocol: API call sequence with expected status codes
    #   - Graph: Path showing [{"direction": "N", "room": 1}, ...]
    # ───────────────────────────────────────────────────────────────────────────

    def _generate_counterexample(self, hypothesis: Dict) -> Optional[List[Dict[str, Any]]]:
        """
        Generate a counterexample showing where hypothesis differs.

        Uses core.automata.find_counterexample for the search,
        then formats the result for your skin-specific output.

        Args:
            hypothesis: The incorrect hypothesis.

        Returns:
            List of steps showing divergence, or None if cannot find one.
        """
        try:
            start_hyp = int(hypothesis.get("start", 0))
            trans = hypothesis.get("transitions", {})
        except Exception:
            return [{"action": ACTIONS[0], "output": OUTPUTS[0]}]

        # Create adapters for core.automata functions
        def get_trans_true(s: int, a: str) -> Tuple[int, Any]:
            # TODO: Adapt to your ground truth structure
            return self._ground_truth["transitions"][str(s)][a]

        def get_trans_hyp(s: int, a: str) -> Tuple[int, Any]:
            # TODO: Adapt to your hypothesis structure
            return tuple(trans[str(s)][a])

        # Use core counterexample generation
        cex = find_counterexample(self._start, start_hyp, ACTIONS, get_trans_true, get_trans_hyp)

        if cex is None:
            return None

        # Format for your skin-specific output
        # TODO: Customize the output format
        return [{"action": action, "output": output} for action, output in cex]

    # ───────────────────────────────────────────────────────────────────────────
    # REQUIRED: Static Generation Methods
    # ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def generate_system_static(
        seed: int,
        n_states: int = 3,  # TODO: Add your skin-specific params
        trap: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a random hidden system.

        MUST be deterministic: same seed → same system.

        Uses core.automata utilities for reachability and minimality checking.
        See MealyEnv.generate_system_static for a complete reference.

        Args:
            seed: Random seed for reproducibility.
            n_states: Number of states (skin-specific).
            trap: Whether to include trap transitions.

        Returns:
            Dict with 'system' and 'traps' keys.
        """
        rng = get_rng(seed)
        S = n_states

        # TODO: Define your transition structure
        # transitions[state][action] = (next_state, output)
        transitions: Dict[int, Dict[str, Tuple[int, Any]]] = {}

        def gen_once() -> Dict[int, Dict[str, Tuple[int, Any]]]:
            """Generate a random system with reachability backbone."""
            t: Dict[int, Dict[str, Tuple[int, Any]]] = {s: {} for s in range(S)}
            # Backbone on first action for reachability
            for s in range(S):
                ns = (s + 1) % S
                t[s][ACTIONS[0]] = (ns, rng.choice(OUTPUTS))
            # Fill remaining actions randomly
            for s in range(S):
                for a in ACTIONS[1:]:
                    t[s][a] = (rng.randrange(S), rng.choice(OUTPUTS))
            return t

        # Create adapters for core.automata functions
        def make_get_next(t: Dict[int, Dict[str, Tuple[int, Any]]]) -> Callable[[int, str], int]:
            return lambda s, a: int(t[s][a][0])

        def make_get_trans(
            t: Dict[int, Dict[str, Tuple[int, Any]]],
        ) -> Callable[[int, str], Tuple[int, Any]]:
            return lambda s, a: t[s][a]

        # Regenerate until reachable and minimal (using core utilities)
        while True:
            transitions = gen_once()
            # Use core.automata.is_fully_reachable
            if not is_fully_reachable(S, 0, ACTIONS, make_get_next(transitions)):
                continue
            # Use core.automata.is_minimal (if isomorphism checking enabled)
            if not is_minimal(S, ACTIONS, make_get_trans(transitions)):
                continue
            break

        system = {
            "n_states": n_states,
            "start": 0,
            "transitions": {
                str(s): {a: list(v) for a, v in transitions[s].items()} for s in range(S)
            },
        }

        # Generate traps using core utility with solvability verification
        # IMPORTANT: Use avoid_start=True and verify_trap_free_path_exists to ensure solvability
        traps: List[Tuple[int, str]] = []
        if trap:
            k = max(1, n_states // 3)
            seen_traps: Set[Tuple[int, str]] = set()
            attempts = 0

            while len(traps) < k and attempts < 100:
                attempts += 1
                # Generate single candidate trap with avoid_start=True
                candidate_traps = generate_random_traps(
                    S, ACTIONS, rng, n_traps=1, avoid_start=True, start_state=0
                )
                if not candidate_traps:
                    continue

                candidate = candidate_traps[0]
                if candidate in seen_traps:
                    continue

                # Only add trap if trap-free exploration path still exists
                test_traps = seen_traps | {candidate}
                if verify_trap_free_path_exists(
                    S, 0, ACTIONS, make_get_next(transitions), test_traps
                ):
                    seen_traps.add(candidate)
                    traps.append(candidate)

        return {
            "system": system,
            "traps": traps,
        }

    @classmethod
    def get_prompt_template(
        cls,
        obs: Dict[str, Any],
        *,
        feedback: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Build system and user prompts for an episode.

        The prompt should:
        - Explain what the agent is trying to identify
        - Document available tools and their schemas
        - Show the hypothesis JSON format with example
        - NOT leak any ground truth

        Args:
            obs: Observation dict (budget, n_states, etc.)
            feedback: Whether counterexample feedback is enabled.

        Returns:
            List of message dicts [{"role": "system", ...}, {"role": "user", ...}]
        """
        n_states = obs.get("n_states", 3)
        budget = obs.get("budget", 25)
        trap = obs.get("trap", True)

        # TODO: Customize this prompt for your domain
        sys_msg = {
            "role": "system",
            "content": (
                "You are an agent identifying a hidden system.\n"
                "Objective: Discover the system and submit a complete specification.\n"
                "Return ONLY function tool calls; never output natural language.\n\n"
                "Episode semantics:\n"
                f"- The system has {n_states} states.\n"
                "- Each probe() consumes 1 query from the budget.\n"
                + ("- Traps exist: some actions cause trap_hit=true.\n" if trap else "")
                + "- submit(hypothesis_json): consumes 1 query. If correct, ends the episode; otherwise it continues"
                + (" (counterexample returned)." if feedback else ".")
                + " The episode also ends when budget is exhausted.\n\n"
                "Tools:\n"
                "- probe(action: str) -> {output, budget_left, trap_hit, ...}\n"
                "- submit(hypothesis_json: str) -> {ok, budget_left, counterexample?}\n\n"
                "Hypothesis JSON schema:\n"
                '{"n_states": <int>, "start": 0, ...}\n'
            ),
        }

        obs_json = {
            "n_states": n_states,
            "budget": budget,
            "trap": trap,
            "actions": ACTIONS,
        }

        usr_msg = {
            "role": "user",
            "content": (
                "OBSERVATION:\n" + json.dumps(obs_json) + "\n\n"
                "Task: Use probe() to gather evidence, then call submit() with complete specification.\n"
                + ("If incorrect, a counterexample will be returned.\n" if feedback else "")
                + "Respond only with tool calls."
            ),
        }

        return [sys_msg, usr_msg]
