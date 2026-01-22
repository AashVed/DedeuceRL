"""ProtocolSkin: API reverse engineering environment with state-dependent transitions.

Purpose (for ML Practitioners)
-------------------------------
This skin tests an agent's ability to:
1. **Systematic state-space exploration**: Track hidden API states across calls
2. **Stateful reasoning**: Recognize that the same call produces different results
   depending on the current state (real-world APIs work this way: auth flows, sessions)
3. **Hypothesis formation under uncertainty**: Build a mental model of API behavior
4. **Safety awareness**: Avoid trap transitions that represent forbidden operations

Unlike simple API enumeration, this requires the agent to maintain a state graph
and systematically explore state × endpoint × method combinations.
"""

from __future__ import annotations

import json

from typing import Any, Dict, List, Optional, Set, Tuple

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
    check_behavioral_equivalence,
    find_counterexample,
    verify_trap_free_path_exists,
)
from dedeucerl.utils.rng import get_rng
from dedeucerl.utils import error_invalid_method


# HTTP methods and status codes
METHODS = ["GET", "POST", "PUT", "DELETE"]
ENDPOINTS = ["/users", "/items", "/orders", "/auth", "/config"]
STATUS_CODES = [200, 201, 400, 401, 404, 500]

# Protocol-specific configuration
PROTOCOL_CONFIG = SkinConfig(
    isomorphism_check=True,
    trap_enabled=True,
    default_budget=30,
    max_turns=64,
    skin_name="protocol",
    skin_version="1.2",  # Bumped for trap-free verification
)


class ProtocolEnv(HiddenSystemEnv):
    """
    API reverse engineering environment with state-dependent behavior.

    Hidden system: State machine where API responses depend on current state
    Probe action: Make an API call {method, endpoint}
    Hypothesis: Full state-dependent transition specification

    The agent must discover:
    1. Which endpoints exist and accept which methods
    2. How responses CHANGE based on current API state
    3. How each call transitions to a new state
    4. Which (state, endpoint, method) combinations trigger traps

    Key difference from v1.0: Transitions are now state-dependent.
    The same API call from different states can produce different responses.

    Acceptance Criteria (Correctness Contract)
    ------------------------------------------
    Unlike MealyEnv which checks structural isomorphism, ProtocolEnv uses
    **behavioral equivalence**: a submission is correct if it produces the
    same outputs for all possible input sequences from the start state.

    This means:
    - Minimized specs with fewer states are accepted if behaviorally equivalent
    - State relabeling is implicitly allowed (behavioral, not structural)
    - Agent need not discover the exact internal structure, only the I/O behavior
    """

    config = PROTOCOL_CONFIG

    # ─────────────────────────────────────────────────────────────
    # Domain Specification (single source of truth)
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def domain_spec(
        cls,
        n_states: int = 3,
        n_endpoints: int = 3,
        budget: int = 30,
        trap: bool = True,
        endpoints: Optional[List[str]] = None,
    ) -> DomainSpec:
        """
        Return the complete domain specification for Protocol skin.

        This is the single source of truth for:
        - Action vocabulary (HTTP methods)
        - Output vocabulary (status codes)
        - Tool schemas with enums
        - Hypothesis schema for validation
        - Observation fields
        """
        if endpoints is None:
            endpoints = ENDPOINTS[:n_endpoints]

        return DomainSpec(
            actions=METHODS,
            outputs=STATUS_CODES,
            tool_schemas=[
                ToolSchema(
                    name="api_call",
                    description="Make an API call to the hidden system",
                    args={
                        "method": ArgSchema(
                            type="string",
                            enum=METHODS,
                            description="HTTP method",
                        ),
                        "endpoint": ArgSchema(
                            type="string",
                            enum=endpoints,
                            description="Endpoint path",
                        ),
                    },
                    returns={
                        "status": ReturnField("int", "HTTP status code"),
                        "body": ReturnField("object", "Response body"),
                        "budget_left": ReturnField("int", "Remaining budget"),
                        "trap_hit": ReturnField("bool", "Whether trap was triggered"),
                        "queries_used": ReturnField("int", "Total queries consumed"),
                    },
                ),
                ToolSchema(
                    name="submit_spec",
                    description=(
                        "Submit API specification as JSON string. "
                        'Schema: {"n_states": <int>, "start": 0, "transitions": {"0": {"/endpoint": {"METHOD": [next_state, status]}, ...}, ...}}'
                    ),
                    args={
                        "spec_json": ArgSchema(
                            type="string",
                            description=(
                                "JSON with keys: n_states (int), start (0), transitions (state -> endpoint -> method -> [next_state, status]). "
                                'Example: {"n_states":2,"start":0,"transitions":{"0":{"/users":{"GET":[0,200],"POST":[1,201]}},"1":{"/users":{"GET":[1,404]}}}}'
                            ),
                        )
                    },
                    returns={
                        "ok": ReturnField("bool", "Whether specification is correct"),
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
                    "n_states": {"type": "integer"},
                    "start": {"type": "integer", "const": 0},
                    "transitions": {
                        "type": "object",
                        "description": "state -> endpoint -> method -> [next_state, status]",
                    },
                },
                "required": ["n_states", "start", "transitions"],
            },
            observation_fields={
                "endpoints": ObservationField("list", "Available API endpoints", endpoints),
                "n_states": ObservationField("int", "Number of API states", n_states),
                "budget": ObservationField("int", "Query budget", budget),
                "trap": ObservationField("bool", "Whether traps exist", trap),
            },
            params={
                "n_states": ParamSpec(
                    type="int",
                    description="Number of hidden API states.",
                    default=n_states,
                    bounds=(1, None),
                ),
                "n_endpoints": ParamSpec(
                    type="int",
                    description="Number of API endpoints exposed to the agent.",
                    default=n_endpoints,
                    bounds=(1, len(ENDPOINTS)),
                ),
            },
            skin_name="protocol",
            n_states=n_states,
            has_traps=trap,
        )

    @classmethod
    def domain_params_from_answer(cls, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DomainSpec parameters from episode answer payload."""
        spec = answer_data.get("spec", {})
        n_states = int(spec.get("n_states", 3))
        transitions = spec.get("transitions", {})

        endpoints: List[str] = []
        if (
            isinstance(transitions, dict)
            and "0" in transitions
            and isinstance(transitions["0"], dict)
        ):
            endpoints = list(transitions["0"].keys())

        n_endpoints = len(endpoints) if endpoints else int(answer_data.get("n_endpoints", 3))
        return {
            "n_states": n_states,
            "n_endpoints": n_endpoints,
            "endpoints": endpoints or None,
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # transitions[state][endpoint][method] = (next_state, status)
        self._transitions: Dict[int, Dict[str, Dict[str, Tuple[int, int]]]] = {}
        self._n_states: int = 0
        self._start: int = 0
        self._endpoint_list: List[str] = []

    # ─────────────────────────────────────────────────────────────
    # Abstract method implementations
    # ─────────────────────────────────────────────────────────────

    def _configure_from_metadata(self, meta: Dict[str, Any]) -> None:
        """Parse ground truth API specification from metadata."""
        spec = meta.get("spec", {})
        self._n_states = int(spec.get("n_states", 0))
        self._start = int(spec.get("start", 0))

        # Parse state-dependent transitions
        raw_transitions = spec.get("transitions", {})
        self._transitions = {}
        for state_str, endpoints in raw_transitions.items():
            state = int(state_str)
            self._transitions[state] = {}
            for ep, methods in endpoints.items():
                self._transitions[state][ep] = {}
                for method, (ns, status) in methods.items():
                    self._transitions[state][ep][method] = (int(ns), int(status))

        # Extract endpoint list for validation
        if self._transitions:
            first_state = next(iter(self._transitions.values()))
            self._endpoint_list = list(first_state.keys())
        else:
            self._endpoint_list = []

        # Parse trap pairs: (state, endpoint, method)
        self._trap_pairs = set((int(s), str(ep), str(m)) for s, ep, m in meta.get("trap_calls", []))
        self._ground_truth = spec

    def _get_start_state(self) -> int:
        """Return the starting state (always 0)."""
        return self._start

    def _get_tools(self) -> List:
        """Return Protocol-specific tools."""
        return [self.api_call, self.submit_spec]

    # ─────────────────────────────────────────────────────────────
    # Tools (exposed to LLM agent)
    # ─────────────────────────────────────────────────────────────

    def api_call(self, method: str, endpoint: str) -> str:
        """Make an API call to the hidden system.

        The response depends on the current hidden state. The same call
        from different states may produce different status codes and
        transition to different next states.

        Returns:
            JSON string: {status, body, budget_left, trap_hit, queries_used}
        """
        state = self._state()

        if state["done"]:
            return self._episode_finished()

        if state["budget"] <= 0:
            return self._budget_exhausted()

        # Consume budget (ends episode when it reaches 0)
        self._consume_budget(1)
        state["steps"] = state.get("steps", 0) + 1

        if method not in METHODS:
            return self._tool_error(
                error_invalid_method(method, METHODS),
                extra={"status": 400, "body": {"error": "Invalid method"}},
            )

        cs = int(state["cs"])

        if cs not in self._transitions:
            return json.dumps(
                {
                    "status": 500,
                    "body": {"error": "Invalid API state"},
                    "budget_left": state["budget"],
                    "trap_hit": state["trap_hit"],
                    "queries_used": state["queries_used"],
                }
            )

        if endpoint not in self._transitions[cs]:
            return json.dumps(
                {
                    "status": 404,
                    "body": {"error": "Endpoint not found"},
                    "budget_left": state["budget"],
                    "trap_hit": state["trap_hit"],
                    "queries_used": state["queries_used"],
                }
            )

        if method not in self._transitions[cs][endpoint]:
            return json.dumps(
                {
                    "status": 405,
                    "body": {"error": "Method not allowed"},
                    "budget_left": state["budget"],
                    "trap_hit": state["trap_hit"],
                    "queries_used": state["queries_used"],
                }
            )

        ns, status = self._transitions[cs][endpoint][method]
        state["cs"] = ns

        if (cs, endpoint, method) in self._trap_pairs:
            self._mark_trap_hit()

        body = self._generate_response_body(endpoint, method, status)

        return json.dumps(
            {
                "status": status,
                "body": body,
                "budget_left": state["budget"],
                "trap_hit": state["trap_hit"],
                "queries_used": state["queries_used"],
            }
        )

    def submit_spec(self, spec_json: str) -> str:
        """Submit API specification as JSON string."""
        state = self._state()

        if state["done"]:
            return self._episode_finished()

        if state["budget"] <= 0:
            return self._budget_exhausted()

        # Every submission attempt consumes budget (even if correct).
        cost = int(self.config.submission_cost)
        if not self._consume_budget(cost):
            return self._budget_exhausted()

        spec, parse_err = self._parse_json_arg(spec_json, context="spec_json")
        if parse_err is not None:
            return self._tool_error(parse_err, extra={"ok": False})

        schema = self.__class__.domain_spec(
            n_states=self._n_states,
            n_endpoints=len(self._endpoint_list) or 1,
            endpoints=self._endpoint_list or None,
            budget=int(state.get("budget_init", self.config.default_budget)),
            trap=bool(self._trap_pairs),
        ).hypothesis_schema
        validation_err = self._prevalidate_hypothesis(spec, schema)
        if validation_err is not None:
            return self._tool_error(validation_err, extra={"ok": False})

        assert isinstance(spec, dict)

        try:
            n_states = int(spec.get("n_states", -1))
            start = int(spec.get("start", -1))

            raw_transitions = spec.get("transitions", {})
            transitions: Dict[int, Dict[str, Dict[str, Tuple[int, int]]]] = {}
            for state_str, endpoints in raw_transitions.items():
                s = int(state_str)
                transitions[s] = {}
                for ep, methods in endpoints.items():
                    transitions[s][ep] = {}
                    for m, trans in methods.items():
                        transitions[s][ep][m] = (int(trans[0]), int(trans[1]))
        except Exception:
            n_states, start, transitions = -1, -1, {}

        ok = self._check_behavioral_equivalence(n_states, start, transitions)

        if not ok:
            cex = None
            if self.feedback_enabled:
                cex = self._generate_counterexample(n_states, start, transitions)

            payload: Dict[str, Any] = {
                "ok": False,
                "budget_left": state["budget"],
                "queries_used": state["queries_used"],
                "trap_hit": state["trap_hit"],
                "counterexample": cex,
            }
            return json.dumps(payload)

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
    # Behavioral equivalence checking
    # ─────────────────────────────────────────────────────────────

    def _check_behavioral_equivalence(
        self,
        n_states: int,
        start: int,
        transitions: Dict[int, Dict[str, Dict[str, Tuple[int, int]]]],
    ) -> bool:
        """Check if submitted spec is behaviorally equivalent to ground truth."""
        if n_states <= 0 or start < 0:
            return False

        # Use a fixed action alphabet derived from the ground truth.
        actions: List[Tuple[str, str]] = []
        for st_transitions in self._transitions.values():
            for ep, methods in st_transitions.items():
                for m in methods.keys():
                    actions.append((m, ep))
        actions = sorted(set(actions))

        def get_trans_true(s: int, a: Tuple[str, str]) -> Tuple[int, int]:
            m, ep = a
            return self._transitions[s][ep][m]

        def get_trans_hyp(s: int, a: Tuple[str, str]) -> Tuple[int, int]:
            m, ep = a
            return transitions[s][ep][m]

        return check_behavioral_equivalence(
            self._n_states,
            self._start,
            n_states,
            start,
            actions,
            get_trans_true,
            get_trans_hyp,
        )

    def _generate_counterexample(
        self,
        n_states: int,
        start: int,
        transitions: Dict[int, Dict[str, Dict[str, Tuple[int, int]]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate a shortest counterexample trace (if any)."""
        if n_states <= 0 or start < 0:
            return [{"call": "GET /users", "expected_status": 200, "reason": "invalid spec"}]

        actions: List[Tuple[str, str]] = []
        for st_transitions in self._transitions.values():
            for ep, methods in st_transitions.items():
                for m in methods.keys():
                    actions.append((m, ep))
        actions = sorted(set(actions))

        def get_trans_true(s: int, a: Tuple[str, str]) -> Tuple[int, int]:
            m, ep = a
            return self._transitions[s][ep][m]

        def get_trans_hyp(s: int, a: Tuple[str, str]) -> Tuple[int, int]:
            m, ep = a
            return transitions[s][ep][m]

        cex = find_counterexample(self._start, start, actions, get_trans_true, get_trans_hyp)
        if cex is None:
            return None

        return [
            {"call": f"{m} {ep}", "expected_status": expected_status}
            for (m, ep), expected_status in cex
        ]

    def _generate_response_body(self, endpoint: str, method: str, status: int) -> Dict[str, Any]:
        """Generate a response body based on endpoint and status."""
        if status == 200:
            return {"success": True, "data": f"{method} {endpoint} completed"}
        elif status == 201:
            return {"success": True, "id": 1, "message": "Created"}
        elif status == 400:
            return {"error": "Bad request"}
        elif status == 401:
            return {"error": "Unauthorized"}
        elif status == 404:
            return {"error": "Not found"}
        else:
            return {"error": "Internal server error"}

    # ─────────────────────────────────────────────────────────────
    # Static methods for task generation
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def generate_system_static(
        seed: int,
        n_endpoints: int = 3,
        n_states: int = 3,
        trap: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a random API specification with TRUE state-dependent behavior.

        Each (state, endpoint, method) triple has its own transition and response,
        making the API a genuine state machine where the same call from different
        states can produce different results.

        Args:
            seed: Random seed for reproducibility.
            n_endpoints: Number of endpoints.
            n_states: Number of API states.
            trap: Whether to include trap transitions.

        Returns:
            Dict with 'spec' and 'trap_calls'.
        """
        rng = get_rng(seed)

        # Select endpoints
        selected_endpoints = ENDPOINTS[:n_endpoints]

        # For each endpoint, decide which methods it supports (globally)
        # This ensures the same methods are available from all states
        endpoint_methods: Dict[str, List[str]] = {}
        for ep in selected_endpoints:
            n_methods = rng.randint(1, min(3, len(METHODS)))
            endpoint_methods[ep] = rng.sample(METHODS, n_methods)

        # Generate state-dependent transitions
        # transitions[state][endpoint][method] = [next_state, status]
        transitions: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

        for state in range(n_states):
            transitions[str(state)] = {}
            for ep in selected_endpoints:
                transitions[str(state)][ep] = {}
                for method in endpoint_methods[ep]:
                    # State-dependent: different (ns, status) for each state
                    ns = rng.randrange(n_states)
                    status = rng.choice([200, 201, 400, 404])
                    transitions[str(state)][ep][method] = [ns, status]

        # Ensure reachability: all states should be reachable from start
        # We do this by creating a backbone path through states
        if n_states > 1:
            for s in range(n_states - 1):
                # Pick a random endpoint/method and make it transition to s+1
                ep = rng.choice(selected_endpoints)
                method = rng.choice(endpoint_methods[ep])
                transitions[str(s)][ep][method][0] = s + 1

        # Generate trap calls with solvability verification
        trap_calls: List[Tuple[int, str, str]] = []
        if trap:
            k = max(1, n_endpoints // 2)
            seen_traps: Set[Tuple[int, str, str]] = set()
            attempts = 0

            # Build action list for trap verification
            all_actions: List[Tuple[str, str]] = [
                (ep, method) for ep in selected_endpoints for method in endpoint_methods[ep]
            ]

            while len(trap_calls) < k and attempts < 100:
                attempts += 1
                state = rng.randrange(n_states)
                ep = rng.choice(selected_endpoints)
                method = rng.choice(endpoint_methods[ep])
                key = (state, ep, method)
                if key not in seen_traps:
                    # Tentatively add trap and check solvability
                    test_traps = seen_traps | {key}

                    # Build trap set in format expected by verify_trap_free_path_exists
                    trap_set = {(s, (e, m)) for s, e, m in test_traps}

                    def get_next_state_fn(s: int, action: Tuple[str, str]) -> int:
                        ep_action, method_action = action
                        return transitions[str(s)][ep_action][method_action][0]

                    # Only add trap if trap-free exploration path still exists
                    if verify_trap_free_path_exists(
                        n_states, 0, all_actions, get_next_state_fn, trap_set
                    ):
                        seen_traps.add(key)
                        trap_calls.append(key)

        spec = {
            "n_states": n_states,
            "start": 0,
            "transitions": transitions,
        }

        return {
            "spec": spec,
            "trap_calls": [[s, ep, m] for s, ep, m in trap_calls],
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

        Uses domain_spec() for tool descriptions.

        Args:
            obs: Observation dict with budget, n_states, etc.
            feedback: Whether feedback mode is enabled.

        Returns:
            List of message dicts (system + user).
        """
        n_states = obs.get("n_states", 3)
        n_endpoints = obs.get("n_endpoints", 3)
        budget = obs.get("budget", 30)
        trap = obs.get("trap", True)
        endpoints = obs.get("endpoints", ENDPOINTS[:n_endpoints])

        # Get domain spec for tool descriptions
        spec = cls.domain_spec(
            n_states=n_states,
            n_endpoints=n_endpoints,
            budget=budget,
            trap=trap,
            endpoints=endpoints,
        )
        tools_text = spec.format_tools_for_prompt()

        sys_msg = {
            "role": "system",
            "content": (
                "You are an autonomous agent reverse-engineering a hidden REST API.\n"
                "Objective: discover the API's STATE-DEPENDENT behavior and submit a specification.\n"
                "Return ONLY function tool calls; never output natural language.\n\n"
                "IMPORTANT: The API has hidden states. The SAME call from DIFFERENT states\n"
                "may produce DIFFERENT responses and transitions. You must track the state.\n\n"
                "Episode semantics:\n"
                f"- The API has {n_states} hidden states (you start in state 0).\n"
                "- Each api_call() consumes 1 query from the budget.\n"
                + (
                    "- Traps exist: some (state, endpoint, method) combinations cause trap_hit=true.\n"
                    if trap
                    else ""
                )
                + "- submit_spec(spec_json): consumes 1 query. If correct, ends the episode; otherwise it continues"
                + (" (counterexample returned)." if feedback else ".")
                + " The episode also ends when budget is exhausted.\n\n"
                "CORRECTNESS CRITERIA (BEHAVIORAL EQUIVALENCE):\n"
                "Your submission is correct if it produces the SAME output sequence as the hidden API\n"
                "for ALL possible input sequences. You do NOT need to match the exact number of states;\n"
                "a minimized/equivalent specification with fewer states is also accepted.\n\n"
                "Tools:\n" + tools_text + "\n\n"
                "Specification JSON schema (STATE-DEPENDENT):\n"
                '{"n_states": <int>, "start": 0, "transitions": {\n'
                '  "0": {"/endpoint": {"METHOD": [next_state, status], ...}, ...},\n'
                '  "1": {"/endpoint": {"METHOD": [next_state, status], ...}, ...},\n'
                "  ...\n"
                "}}\n\n"
                "Example (n_states=2):\n"
                '{"n_states":2,"start":0,"transitions":{\n'
                '  "0":{"/users":{"GET":[0,200],"POST":[1,201]}},\n'
                '  "1":{"/users":{"GET":[1,404],"POST":[0,201]}}\n'
                "}}"
            ),
        }

        # Build observation from spec
        obs_json = spec.build_observation(
            endpoints=endpoints,
            budget=budget,
            n_states=n_states,
            trap=trap,
        )

        usr_msg = {
            "role": "user",
            "content": (
                "OBSERVATION:\n" + json.dumps(obs_json) + "\n\n"
                "Task: Use api_call() to discover STATE-DEPENDENT endpoint behavior,\n"
                "then call submit_spec() with a complete specification.\n"
                + ("If incorrect, a counterexample will be returned.\n" if feedback else "")
                + "Respond only with tool calls."
            ),
        }

        return [sys_msg, usr_msg]
