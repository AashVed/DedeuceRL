"""MealySkin: Mealy machine identification environment.

Purpose (for ML Practitioners)
-------------------------------
This skin tests an agent's ability to:
1. **Systematic exploration**: Cover all states and transitions efficiently
2. **Hypothesis formation**: Build a complete mental model from partial observations
3. **Memory and reasoning**: Track state across interactions without direct visibility
4. **Equivalence understanding**: Recognize that structurally different tables can be
   behaviorally identical (isomorphism checking allows equivalent relabelings)

The Mealy machine paradigm is foundational in automata learning (Angluin's L* algorithm)
and represents the simplest non-trivial active identification task. Success requires
more than pattern matching—agents must reason about hidden state transitions.

Key features:
- Uses core.automata for reachability, minimality, isomorphism checking
- Demonstrates the adapter pattern for domain-specific structures
- Provides counterexample generation via product BFS
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dedeucerl.core.env import HiddenSystemEnv
from dedeucerl.core.config import SkinConfig
from dedeucerl.core.domain_spec import (
    DomainSpec,
    ToolSchema,
    ArgSchema,
    ReturnField,
    ObservationField,
    ParamSpec,
)
from dedeucerl.core.automata import (
    is_fully_reachable,
    is_minimal,
    check_isomorphism_with_signatures,
    find_counterexample,
    generate_random_traps,
    verify_trap_free_path_exists,
)
from dedeucerl.utils.rng import get_rng
from dedeucerl.utils import error_invalid_symbol


ALPHABET = ["A", "B", "C"]
OUTPUTS = [0, 1, 2]

# Mealy-specific configuration
MEALY_CONFIG = SkinConfig(
    isomorphism_check=True,
    trap_enabled=True,
    default_budget=25,
    max_turns=64,
    skin_name="mealy",
    skin_version="1.1",  # Bumped for core.automata refactor
)


class MealyEnv(HiddenSystemEnv):
    """
    Mealy machine identification environment.

    Hidden system: Transition table (state × symbol → next_state × output)
    Probe action: Execute a symbol (A, B, C)
    Hypothesis: Full transition table as JSON

    This is the reference implementation for active system identification,
    ported from the original DedeuceBench.
    """

    config = MEALY_CONFIG

    # ─────────────────────────────────────────────────────────────
    # Domain Specification (single source of truth)
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def domain_spec(
        cls,
        n_states: int = 5,
        budget: int = 25,
        trap: bool = True,
    ) -> DomainSpec:
        """
        Return the complete domain specification for Mealy skin.

        This is the single source of truth for:
        - Action vocabulary (symbols A, B, C)
        - Output vocabulary (0, 1, 2)
        - Tool schemas with enums
        - Hypothesis schema for validation
        - Observation fields
        """
        return DomainSpec(
            actions=ALPHABET,
            outputs=OUTPUTS,
            tool_schemas=[
                ToolSchema(
                    name="act",
                    description="Execute one input symbol on the hidden Mealy machine",
                    args={
                        "symbol": ArgSchema(
                            type="string",
                            enum=ALPHABET,
                            description="Input symbol to execute",
                        )
                    },
                    returns={
                        "out": ReturnField("int", "Output value 0, 1, or 2"),
                        "budget_left": ReturnField("int", "Remaining query budget"),
                        "t": ReturnField("int", "Step count since start"),
                        "trap_hit": ReturnField("bool", "Whether a trap was triggered"),
                        "queries_used": ReturnField("int", "Total queries consumed"),
                    },
                ),
                ToolSchema(
                    name="submit_table",
                    description=(
                        "Submit hypothesis transition table as JSON string. "
                        'Schema: {"n": <states>, "start": 0, "trans": {"0": {"A": [next, out], "B": [...], "C": [...]}, ...}}'
                    ),
                    args={
                        "table_json": ArgSchema(
                            type="string",
                            description=(
                                "JSON with keys: n (int), start (0), trans (dict of state -> symbol -> [next_state, output]). "
                                'Example: {"n":2,"start":0,"trans":{"0":{"A":[1,2],"B":[0,1],"C":[0,0]},"1":{"A":[0,0],"B":[1,1],"C":[1,2]}}}'
                            ),
                        )
                    },
                    returns={
                        "ok": ReturnField("bool", "Whether hypothesis is correct"),
                        "budget_left": ReturnField("int", "Remaining budget"),
                        "queries_used": ReturnField("int", "Total queries consumed"),
                        "trap_hit": ReturnField("bool", "Whether trap was triggered"),
                        "counterexample": ReturnField(
                            "list", "Distinguishing trace (if feedback enabled)"
                        ),
                    },
                ),
            ],
            hypothesis_schema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of states"},
                    "start": {
                        "type": "integer",
                        "const": 0,
                        "description": "Start state (always 0)",
                    },
                    "trans": {
                        "type": "object",
                        "description": "Transition table: state -> symbol -> [next_state, output]",
                        "patternProperties": {
                            "^[0-9]+$": {
                                "type": "object",
                                "properties": {
                                    sym: {
                                        "type": "array",
                                        "items": [
                                            {"type": "integer"},
                                            {"type": "integer"},
                                        ],
                                        "minItems": 2,
                                        "maxItems": 2,
                                    }
                                    for sym in ALPHABET
                                },
                                "required": ALPHABET,
                            }
                        },
                    },
                },
                "required": ["n", "start", "trans"],
            },
            observation_fields={
                "alphabet": ObservationField("list", "Allowed input symbols", ALPHABET),
                "n_states": ObservationField("int", "Number of states to identify", n_states),
                "budget": ObservationField("int", "Query budget", budget),
                "trap": ObservationField("bool", "Whether traps exist", trap),
            },
            params={
                "n_states": ParamSpec(
                    type="int",
                    description="Number of states in the hidden Mealy machine.",
                    default=n_states,
                    bounds=(1, None),
                ),
            },
            skin_name="mealy",
            n_states=n_states,
            has_traps=trap,
        )

    @classmethod
    def domain_params_from_answer(cls, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DomainSpec parameters from episode answer payload."""
        table = answer_data.get("table", {})
        return {
            "n_states": int(table.get("n", 5)),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trans: Dict[int, Dict[str, Tuple[int, int]]] = {}
        self._n_states: int = 0
        self._start: int = 0

    # ─────────────────────────────────────────────────────────────
    # Abstract method implementations
    # ─────────────────────────────────────────────────────────────

    def _configure_from_metadata(self, meta: Dict[str, Any]) -> None:
        """Parse ground truth Mealy machine from metadata."""
        table = meta.get("table", {})
        self._trans = {
            int(s): {a: (int(ns), int(o)) for a, (ns, o) in v.items()}
            for s, v in table.get("trans", {}).items()
        }
        self._n_states = int(table.get("n", 0))
        self._start = int(table.get("start", 0))
        self._trap_pairs = set((int(s), str(a)) for s, a in meta.get("trap_pairs", []))
        self._ground_truth = table

    def _get_start_state(self) -> int:
        """Return the starting state (always 0)."""
        return self._start

    def _get_tools(self) -> List:
        """Return Mealy-specific tools."""
        return [self.act, self.submit_table]

    # ─────────────────────────────────────────────────────────────
    # Tools (exposed to LLM agent)
    # ─────────────────────────────────────────────────────────────

    def act(self, symbol: str) -> str:
        """
        Execute one input symbol on the hidden Mealy machine.

        Args:
            symbol: One of 'A', 'B', 'C'.

        Returns:
            JSON string: {out, budget_left, t, trap_hit, queries_used}
        """
        state = self._state()

        if state["done"]:
            return self._episode_finished()

        if state["budget"] <= 0:
            return self._budget_exhausted()

        # Consume budget (ends episode when it reaches 0)
        self._consume_budget(1)
        state["steps"] = state.get("steps", 0) + 1

        if symbol not in ALPHABET:
            return self._tool_error(error_invalid_symbol(symbol, ALPHABET))

        cs = int(state["cs"])
        ns, out = self._trans[cs][symbol]
        state["cs"] = ns

        # Check for trap - use base helper for config-aware behavior
        if (cs, symbol) in self._trap_pairs:
            self._mark_trap_hit()

        return json.dumps(
            {
                "out": int(out),
                "budget_left": state["budget"],
                "t": state["steps"],
                "trap_hit": state["trap_hit"],
                "queries_used": state["queries_used"],
            }
        )

    def submit_table(self, table_json: str) -> str:
        """
        Submit exact transition table as JSON string.

        Args:
            table_json: JSON string matching schema {n, start, trans}

        Returns:
            JSON string: {ok, budget_left, queries_used, trap_hit, counterexample?}
        """
        state = self._state()

        if state["done"]:
            return self._episode_finished()

        if state["budget"] <= 0:
            return self._budget_exhausted()

        # Every submission attempt consumes budget (even if correct).
        cost = int(self.config.submission_cost)
        if not self._consume_budget(cost):
            return self._budget_exhausted()

        table, parse_err = self._parse_json_arg(table_json, context="table_json")
        if parse_err is not None:
            return self._tool_error(parse_err, extra={"ok": False})

        schema = self.__class__.domain_spec(
            n_states=self._n_states,
            budget=int(state.get("budget_init", self.config.default_budget)),
            trap=bool(self._trap_pairs),
        ).hypothesis_schema
        validation_err = self._prevalidate_hypothesis(table, schema)
        if validation_err is not None:
            return self._tool_error(validation_err, extra={"ok": False})

        assert isinstance(table, dict)

        try:
            n = int(table.get("n", -1))
            start = int(table.get("start", -1))
            trans = {
                int(s): {a: (int(v[0]), int(v[1])) for a, v in m.items()}
                for s, m in table.get("trans", {}).items()
            }
        except Exception:
            n, start, trans = -1, -1, {}

        # Check isomorphism
        ok = self._check_isomorphism(n, start, trans)

        if not ok:
            # Generate counterexample if feedback enabled
            cex = None
            if self.feedback_enabled:
                cex = self._generate_counterexample(n, start, trans)

            payload = {
                "ok": False,
                "budget_left": state["budget"],
                "queries_used": state["queries_used"],
                "trap_hit": state["trap_hit"],
                "counterexample": cex,
            }
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
                "counterexample": None,
            }
        )

    # ─────────────────────────────────────────────────────────────
    # Isomorphism checking (using core.automata utilities)
    # ─────────────────────────────────────────────────────────────

    def _check_isomorphism(self, n: int, start: int, trans: Dict) -> bool:
        """Check if submitted table is isomorphic to ground truth.

        Uses core.automata.check_isomorphism_with_signatures for the
        heavy lifting, demonstrating how skins leverage core infrastructure.
        """
        if n != self._n_states or start != self._start:
            return False

        # Structural sanity check
        try:
            for s in range(n):
                m = trans[s]
                if not all(k in m for k in ALPHABET):
                    return False
                for a in ALPHABET:
                    ns, o = m[a]
                    if not (0 <= ns < n):
                        return False
        except Exception:
            return False

        # Create adapters for core.automata functions
        def get_trans_true(s: int, a: str) -> Tuple[int, int]:
            return self._trans[s][a]

        def get_trans_hyp(s: int, a: str) -> Tuple[int, int]:
            return trans[s][a]

        # Use core isomorphism checking
        return check_isomorphism_with_signatures(
            n, self._start, start, ALPHABET, get_trans_true, get_trans_hyp
        )

    def _generate_counterexample(
        self, n: int, start: int, trans: Dict
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate counterexample via product BFS.

        Uses core.automata.find_counterexample for the search,
        then formats the result for Mealy-specific output.
        """
        if n <= 0 or start < 0:
            return [{"in": "A", "out": 0}]

        # Create adapters for core.automata functions
        def get_trans_true(s: int, a: str) -> Tuple[int, int]:
            return self._trans[s][a]

        def get_trans_hyp(s: int, a: str) -> Tuple[int, int]:
            return trans[s][a]

        # Use core counterexample generation
        cex = find_counterexample(self._start, start, ALPHABET, get_trans_true, get_trans_hyp)

        if cex is None:
            return None

        # Format for Mealy-specific output: [{"in": action, "out": output}, ...]
        return [{"in": action, "out": output} for action, output in cex]

    # ─────────────────────────────────────────────────────────────
    # Override base class methods
    # ─────────────────────────────────────────────────────────────

    def is_isomorphic(self, hypothesis: Any, ground_truth: Any) -> bool:
        """Check if hypothesis matches ground truth up to state relabeling."""
        try:
            n = int(hypothesis.get("n", -1))
            start = int(hypothesis.get("start", -1))
            trans = {
                int(s): {a: (int(v[0]), int(v[1])) for a, v in m.items()}
                for s, m in hypothesis.get("trans", {}).items()
            }
            return self._check_isomorphism(n, start, trans)
        except Exception:
            return False

    def get_counterexample(self, hypothesis: Any, ground_truth: Any) -> Optional[Any]:
        """Generate counterexample for incorrect hypothesis."""
        try:
            n = int(hypothesis.get("n", -1))
            start = int(hypothesis.get("start", -1))
            trans = {
                int(s): {a: (int(v[0]), int(v[1])) for a, v in m.items()}
                for s, m in hypothesis.get("trans", {}).items()
            }
            return self._generate_counterexample(n, start, trans)
        except Exception:
            return [{"in": "A", "out": 0}]

    # ─────────────────────────────────────────────────────────────
    # Static methods for task generation
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def generate_system_static(
        seed: int,
        n_states: int = 5,
        trap: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a random Mealy machine with guaranteed reachability and minimality.

        Uses core.automata utilities for reachability and minimality checking,
        demonstrating how skins leverage the core infrastructure.

        Args:
            seed: Random seed for reproducibility.
            n_states: Number of states.
            trap: Whether to include trap transitions.

        Returns:
            Dict with 'table' and 'trap_pairs'.
        """
        rng = get_rng(seed)
        S = n_states

        def gen_once() -> Dict[int, Dict[str, Tuple[int, int]]]:
            """Generate a random machine with reachability backbone."""
            t: Dict[int, Dict[str, Tuple[int, int]]] = {s: {} for s in range(S)}
            # Backbone on 'A' for reachability
            for s in range(S):
                ns = (s + 1) % S
                t[s]["A"] = (ns, rng.choice(OUTPUTS))
            # Fill remaining symbols
            for s in range(S):
                for a in [x for x in ALPHABET if x != "A"]:
                    t[s][a] = (rng.randrange(S), rng.choice(OUTPUTS))
            return t

        def make_get_next(t: Dict[int, Dict[str, Tuple[int, int]]]) -> "Callable[[int, str], int]":
            """Create adapter for core.automata functions."""
            return lambda s, a: t[s][a][0]

        def make_get_trans(
            t: Dict[int, Dict[str, Tuple[int, int]]],
        ) -> "Callable[[int, str], Tuple[int, int]]":
            """Create adapter for core.automata functions."""
            return lambda s, a: t[s][a]

        # Regenerate until reachable and minimal (using core utilities)
        while True:
            trans = gen_once()
            # Use core.automata.is_fully_reachable
            if not is_fully_reachable(S, 0, ALPHABET, make_get_next(trans)):
                continue
            # Use core.automata.is_minimal
            if not is_minimal(S, ALPHABET, make_get_trans(trans)):
                continue
            break

        # Generate trap pairs using core utility with solvability verification
        trap_pairs: List[Tuple[int, str]] = []
        if trap:
            k = max(1, S // 3)
            attempts = 0
            seen_traps: Set[Tuple[int, str]] = set()

            while len(trap_pairs) < k and attempts < 100:
                attempts += 1
                # Generate single candidate trap with avoid_start=True
                candidate_traps = generate_random_traps(
                    S, ALPHABET, rng, n_traps=1, avoid_start=True, start_state=0
                )
                if not candidate_traps:
                    continue

                candidate = candidate_traps[0]
                if candidate in seen_traps:
                    continue

                # Test if adding this trap still allows trap-free exploration
                test_traps = seen_traps | {candidate}
                if verify_trap_free_path_exists(S, 0, ALPHABET, make_get_next(trans), test_traps):
                    seen_traps.add(candidate)
                    trap_pairs.append(candidate)

        # Build JSON-friendly table
        table = {
            "n": S,
            "start": 0,
            "trans": {
                str(s): {a: [ns, out] for a, (ns, out) in trans[s].items()} for s in range(S)
            },
        }

        return {
            "table": table,
            "trap_pairs": [[s, a] for s, a in trap_pairs],
        }

    @classmethod
    def get_prompt_template(
        cls,
        obs: Dict[str, Any],
        *,
        feedback: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Build the prompt messages for an episode.

        Uses domain_spec() to generate tools and observation schema.

        Args:
            obs: Observation dict with budget, n_states, etc.
            feedback: Whether feedback mode is enabled.

        Returns:
            List of message dicts (system + user).
        """
        n_states = obs.get("n_states", 5)
        budget = obs.get("budget", 25)
        trap = obs.get("trap", True)

        # Get domain spec for this configuration
        spec = cls.domain_spec(n_states=n_states, budget=budget, trap=trap)

        # Build tools section from spec
        tools_text = spec.format_tools_for_prompt()

        sys_msg = {
            "role": "system",
            "content": (
                "You are an autonomous tool-using agent interacting with a hidden Mealy machine (finite-state transducer).\n"
                "Objective: exactly identify the machine and submit the full transition table via submit_table(table_json).\n"
                "Return ONLY function tool calls; never output natural language.\n\n"
                "Episode semantics:\n"
                "- Stateful episode: the machine's state persists across all act() calls; there are no resets.\n"
                "- Start state is 0. Each act(symbol) consumes 1 query, produces an output, and advances the hidden state.\n"
                "- Invalid symbols still consume 1 query and return an error.\n"
                "- submit_table(table_json): consumes 1 query. Correct submission ends the episode; incorrect continues"
                + (" (with counterexample)." if feedback else ".")
                + " Budget=0 also ends the episode.\n\n"
                + "Tool return fields (meaning):\n"
                + "- out: the output value produced by the hidden machine for the last act().\n"
                + "- budget_left: remaining query budget AFTER this tool call.\n"
                + "- t: 1-based count of act() probes taken so far (submit_table does not increment t).\n"
                + "- queries_used: total queries consumed so far (includes both act() and submit_table()).\n\n"
                + (
                    "Counterexample semantics (feedback mode):\n"
                    "- If submit_table is incorrect, counterexample is a short distinguishing trace from the START state (0).\n"
                    "- The counterexample does NOT change your live episode state (your current hidden state remains unchanged).\n"
                    "- Format: [{\"in\": \"A\", \"out\": 0}, {\"in\": \"B\", \"out\": 2}, ...] where out is the TRUE output.\n\n"
                    if feedback
                    else ""
                )
                + "Submission schema footguns (avoid these mistakes):\n"
                + "- Use string state keys: \"0\", \"1\", ... \"n-1\".\n"
                + "- trans must include EVERY state 0..n-1 and EACH symbol A,B,C exactly once.\n"
                + "- Each transition value is [next_state, out] (do NOT swap order).\n\n"
                + "Tools:\n"
                + tools_text
                + "\n\n"
                + "Submit-table JSON schema:\n"
                + '{"n": <int>, "start": 0, "trans": {"0": {"A": [<ns>, <out>], "B": [...], "C": [...]}, ...}}\n\n'
                + "Skeleton example (n=2):\n"
                + '{"n":2,"start":0,"trans":{"0":{"A":[1,2],"B":[0,1],"C":[0,0]},"1":{"A":[0,0],"B":[1,1],"C":[1,2]}}}'
            ),
        }

        # Build observation from spec
        obs_json = spec.build_observation(
            alphabet=ALPHABET,
            budget=budget,
            n_states=n_states,
            trap=trap,
        )

        usr_msg = {
            "role": "user",
            "content": (
                "OBSERVATION:\n" + json.dumps(obs_json) + "\n\n"
                "Task: Use act() to gather evidence, then call submit_table(table_json) with a complete table.\n"
                + ("If incorrect, a counterexample will be returned.\n" if feedback else "")
                + "Respond only with tool calls."
            ),
        }

        return [sys_msg, usr_msg]
