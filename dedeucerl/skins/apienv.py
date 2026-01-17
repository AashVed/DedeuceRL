"""APIEnv: Practitioner-oriented stateful SaaS API reverse engineering skin.

Purpose (for ML Practitioners)
-------------------------------
This skin models a realistic, stateful REST API workflow such as onboarding,
authentication, org selection, billing/plan upgrade, and feature-gated actions.

Why this matters:
- Real APIs are *stateful*: the same request can behave differently depending on
  auth/session/onboarding/plan state.
- Reverse-engineering that hidden state machine is a practical skill for
  debugging, black-box QA, contract testing, and agentic API integration.

Task:
- You may call `api_call(method, endpoint, variant)` to probe the hidden API.
- You must submit a full specification via `submit_spec(spec_json)`.

Correctness Contract (Behavioral Equivalence)
--------------------------------------------
A submission is correct if it is behaviorally equivalent to the hidden system:
for ALL possible input sequences from the start state, the output sequence
(status code + response schema tag) must match.

Important:
- You do NOT need to match the exact number of internal states.
- Minimized equivalent specifications with fewer states are accepted.

Traps:
Some (state, method, endpoint, variant) operations are marked as traps. If hit,
`trap_hit=true` is returned (and may end the episode depending on config).

Design notes:
- The API is deterministic and has a finite operation alphabet.
- Generation ensures reachability, minimality (by construction), and trap-free
  solvability when traps are enabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from datasets import Dataset

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
    verify_trap_free_path_exists,
    check_behavioral_equivalence,
    find_counterexample,
)
from dedeucerl.utils.rng import get_rng


METHODS = ["GET", "POST", "DELETE"]
STATUS_CODES = [200, 201, 400, 401, 403, 404, 405, 409, 422]

# Response schema tags are a deliberately small vocabulary that provides
# realistic information without leaking internal state IDs.
SCHEMA_TAGS = [
    "AuthOk",
    "AuthFail",
    "AlreadyAuthed",
    "LoggedOut",
    "NeedAuth",
    "NeedVerify",
    "AlreadyVerified",
    "OrgSelectedA",
    "OrgSelectedB",
    "BadOrg",
    "NeedOrg",
    "UpgradeOk",
    "AlreadyPro",
    "ProjectsListA",
    "ProjectsListB",
    "ProjectCreatedFree",
    "ProjectCreatedPro",
    "OrgDeleteDryRun",
    "OrgDeleted",
    "NotFound",
    "MethodNotAllowed",
    "InvalidVariant",
]


# Canonical endpoint/method/variant catalog.
# Variants are finite request "modes" (valid/invalid, org choice, etc.).
ENDPOINT_CATALOG: Dict[str, Dict[str, List[str]]] = {
    "/login": {"POST": ["valid", "invalid"]},
    "/logout": {"POST": ["logout"]},
    "/verify_email": {"POST": ["code_ok", "code_bad"]},
    "/select_org": {"POST": ["orgA", "orgB", "invalid"]},
    "/upgrade_plan": {"POST": ["to_pro", "noop"]},
    "/projects": {"GET": ["list"], "POST": ["create"]},
    "/org": {"DELETE": ["dry_run", "confirm"]},
}


APIENV_CONFIG = SkinConfig(
    isomorphism_check=True,
    trap_enabled=True,
    default_budget=35,
    submission_cost=1,
    max_turns=80,
    skin_name="apienv",
    skin_version="1.0",
)


@dataclass(frozen=True)
class Profile:
    """Interpretable latent workflow profile for a hidden state."""

    authed: bool
    verified: bool
    org: str  # "none" | "A" | "B"
    plan: str  # "free" | "pro"


def _canonical_call(method: str, endpoint: str, variant: str) -> str:
    return f"{method} {endpoint}#{variant}"


class APIEnv(HiddenSystemEnv):
    """Stateful SaaS API reverse engineering environment."""

    config = APIENV_CONFIG

    # ─────────────────────────────────────────────────────────────
    # Domain Specification (single source of truth)
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def domain_spec(
        cls,
        n_states: int = 7,
        n_endpoints: int = 6,
        budget: int = 35,
        trap: bool = True,
        endpoints: Optional[List[str]] = None,
    ) -> DomainSpec:
        if endpoints is None:
            endpoints = list(ENDPOINT_CATALOG.keys())[:n_endpoints]

        variants_map: Dict[str, Dict[str, List[str]]] = {
            ep: ENDPOINT_CATALOG[ep] for ep in endpoints if ep in ENDPOINT_CATALOG
        }
        all_variants: List[str] = sorted(
            {v for ep in variants_map.values() for vs in ep.values() for v in vs}
        )

        return DomainSpec(
            actions=[
                _canonical_call(m, ep, v)
                for ep, mm in variants_map.items()
                for m, vs in mm.items()
                for v in vs
            ],
            outputs=[f"{code}:{tag}" for code in STATUS_CODES for tag in SCHEMA_TAGS],
            tool_schemas=[
                ToolSchema(
                    name="api_call",
                    description="Make an API call (method + endpoint + request variant)",
                    args={
                        "method": ArgSchema(
                            type="string",
                            enum=[
                                m
                                for m in METHODS
                                if any(m in variants_map.get(ep, {}) for ep in variants_map)
                            ],
                            description="HTTP method",
                        ),
                        "endpoint": ArgSchema(
                            type="string",
                            enum=endpoints,
                            description="Endpoint path",
                        ),
                        "variant": ArgSchema(
                            type="string",
                            enum=all_variants,
                            description="Finite request variant for this endpoint/method",
                        ),
                    },
                    returns={
                        "status": ReturnField("int", "HTTP status code"),
                        "schema": ReturnField("string", "Coarse response schema tag"),
                        "budget_left": ReturnField("int", "Remaining budget"),
                        "trap_hit": ReturnField("bool", "Whether a trap was triggered"),
                        "queries_used": ReturnField("int", "Total queries consumed"),
                    },
                ),
                ToolSchema(
                    name="submit_spec",
                    description=(
                        "Submit a behaviorally equivalent API state machine specification. "
                        'Schema: {"n_states": <int>, "start": 0, "transitions": {"0": {"/endpoint": {"METHOD": {"variant": [next_state, status, schema]}}}}}'
                    ),
                    args={
                        "spec_json": ArgSchema(
                            type="string",
                            description=(
                                "JSON with keys: n_states (int), start (0), transitions (state -> endpoint -> method -> variant -> [next_state, status, schema_tag]). "
                                'Example: {"n_states":2,"start":0,"transitions":{"0":{"/login":{"POST":{"valid":[1,200,"AuthOk"]}}}}}'
                            ),
                        )
                    },
                    returns={
                        "ok": ReturnField("bool", "Whether submission is correct"),
                        "budget_left": ReturnField("int", "Remaining budget"),
                        "queries_used": ReturnField("int", "Total queries consumed"),
                        "trap_hit": ReturnField("bool", "Whether a trap was triggered"),
                        "counterexample": ReturnField(
                            "list", "Shortest distinguishing trace (if feedback enabled)"
                        ),
                    },
                ),
            ],
            hypothesis_schema={
                "type": "object",
                "required": ["n_states", "start", "transitions"],
                "properties": {
                    "n_states": {"type": "integer", "minimum": 1},
                    "start": {"type": "integer", "minimum": 0},
                    "transitions": {"type": "object"},
                },
            },
            observation_fields={
                "budget": ObservationField("int", "Query budget", budget),
                "trap": ObservationField("bool", "Whether traps exist", trap),
                "n_states": ObservationField("int", "Hidden system state count", n_states),
                "endpoints": ObservationField("list", "Available endpoints", endpoints),
                "methods": ObservationField("list", "Available HTTP methods", METHODS),
                "variants": ObservationField(
                    "object",
                    "Per-endpoint supported methods and variants",
                    variants_map,
                ),
                "status_codes": ObservationField(
                    "list", "Possible response status codes", STATUS_CODES
                ),
                "response_schemas": ObservationField(
                    "list", "Possible response schema tags", SCHEMA_TAGS
                ),
            },
            params={
                "n_states": ParamSpec(
                    type="int",
                    description="Number of hidden states (upper bound; minimized solutions accepted)",
                    default=n_states,
                    bounds=(4, 8),
                ),
                "n_endpoints": ParamSpec(
                    type="int",
                    description="Number of endpoints included from the catalog",
                    default=n_endpoints,
                    bounds=(3, len(ENDPOINT_CATALOG)),
                ),
                "trap": ParamSpec(
                    type="bool",
                    description="Whether trap operations exist",
                    default=trap,
                ),
            },
            skin_name="apienv",
            n_states=n_states,
            has_traps=trap,
        )

    @classmethod
    def domain_params_from_answer(cls, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DomainSpec parameters from episode answer payload."""
        spec = answer_data.get("spec", {})
        n_states = int(spec.get("n_states", 7))

        endpoints = spec.get("endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            endpoints = None

        n_endpoints = len(endpoints) if endpoints else int(answer_data.get("n_endpoints", 6))
        return {
            "n_states": n_states,
            "n_endpoints": n_endpoints,
            "endpoints": endpoints,
        }

    # ─────────────────────────────────────────────────────────────
    # Skin initialization and metadata
    # ─────────────────────────────────────────────────────────────

    def __init__(self, dataset: Dataset, rubric, **kwargs):
        super().__init__(dataset=dataset, rubric=rubric, **kwargs)
        self._n_states: int = 0
        self._start: int = 0
        self._endpoints: List[str] = []
        self._variants_map: Dict[str, Dict[str, List[str]]] = {}
        self._ops: List[Tuple[str, str, str]] = []
        # transitions[state][endpoint][method][variant] = (next_state, status, schema)
        self._transitions: Dict[int, Dict[str, Dict[str, Dict[str, Tuple[int, int, str]]]]] = {}

    def _configure_from_metadata(self, meta: Dict[str, Any]) -> None:
        spec = meta.get("spec", {})
        self._n_states = int(spec.get("n_states", 0))
        self._start = int(spec.get("start", 0))
        self._endpoints = list(spec.get("endpoints", []))
        self._variants_map = dict(spec.get("variants", {}))

        raw_transitions = spec.get("transitions", {})
        transitions: Dict[int, Dict[str, Dict[str, Dict[str, Tuple[int, int, str]]]]] = {}
        for s_str, ep_map in raw_transitions.items():
            s = int(s_str)
            transitions[s] = {}
            for ep, m_map in ep_map.items():
                transitions[s][ep] = {}
                for method, v_map in m_map.items():
                    transitions[s][ep][method] = {}
                    for variant, triple in v_map.items():
                        transitions[s][ep][method][variant] = (
                            int(triple[0]),
                            int(triple[1]),
                            str(triple[2]),
                        )
        self._transitions = transitions

        # Trap calls: [state, endpoint, method, variant]
        self._trap_pairs = set(tuple(x) for x in meta.get("trap_calls", []))

        # Build op list (finite operation alphabet) from variants_map
        ops: List[Tuple[str, str, str]] = []
        for ep in self._endpoints:
            for method, variants in self._variants_map.get(ep, {}).items():
                for variant in variants:
                    ops.append((method, ep, variant))
        self._ops = ops

    def _get_start_state(self) -> int:
        return self._start

    def _get_tools(self) -> List:
        return [self.api_call, self.submit_spec]

    # ─────────────────────────────────────────────────────────────
    # Tools
    # ─────────────────────────────────────────────────────────────

    def api_call(self, method: str, endpoint: str, variant: str) -> str:
        """Make an API call (method + endpoint + variant)."""
        state = self._state()

        if state["done"]:
            return self._episode_finished()

        if state["budget"] <= 0:
            return self._budget_exhausted()

        # Consume budget (ends episode when it reaches 0)
        self._consume_budget(1)
        state["steps"] = state.get("steps", 0) + 1

        if endpoint not in self._variants_map:
            return json.dumps(
                {
                    "status": 404,
                    "schema": "NotFound",
                    "budget_left": state["budget"],
                    "trap_hit": state["trap_hit"],
                    "queries_used": state["queries_used"],
                }
            )

        if method not in self._variants_map[endpoint]:
            return json.dumps(
                {
                    "status": 405,
                    "schema": "MethodNotAllowed",
                    "budget_left": state["budget"],
                    "trap_hit": state["trap_hit"],
                    "queries_used": state["queries_used"],
                }
            )

        allowed_variants = set(self._variants_map[endpoint][method])
        if variant not in allowed_variants:
            return json.dumps(
                {
                    "status": 422,
                    "schema": "InvalidVariant",
                    "budget_left": state["budget"],
                    "trap_hit": state["trap_hit"],
                    "queries_used": state["queries_used"],
                }
            )

        cs = int(state["cs"])
        ns, status_code, schema = self._transitions[cs][endpoint][method][variant]
        state["cs"] = ns

        if (cs, endpoint, method, variant) in self._trap_pairs:
            self._mark_trap_hit()

        return json.dumps(
            {
                "status": int(status_code),
                "schema": str(schema),
                "budget_left": state["budget"],
                "trap_hit": state["trap_hit"],
                "queries_used": state["queries_used"],
            }
        )

    def submit_spec(self, spec_json: str) -> str:
        """Submit a candidate API specification."""
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
            n_endpoints=len(self._endpoints) or 1,
            endpoints=self._endpoints or None,
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

            transitions: Dict[int, Dict[str, Dict[str, Dict[str, Tuple[int, int, str]]]]] = {}
            for s_str, ep_map in raw_transitions.items():
                s = int(s_str)
                transitions[s] = {}
                for ep, m_map in ep_map.items():
                    transitions[s][ep] = {}
                    for m, v_map in m_map.items():
                        transitions[s][ep][m] = {}
                        for v, triple in v_map.items():
                            transitions[s][ep][m][v] = (
                                int(triple[0]),
                                int(triple[1]),
                                str(triple[2]),
                            )
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
            }
            if cex is not None:
                payload["counterexample"] = cex
            return json.dumps(payload)

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

    # ─────────────────────────────────────────────────────────────
    # Behavioral equivalence checking (using core.automata utilities)
    # ─────────────────────────────────────────────────────────────

    def _make_transition_adapter(
        self,
        transitions: Dict[int, Dict[str, Dict[str, Dict[str, Tuple[int, int, str]]]]],
    ) -> "Callable[[int, Tuple[str, str, str]], Tuple[int, Tuple[int, str]]]":
        """Create adapter from nested dict to flat (state, action) -> (next, output).

        This allows us to leverage core.automata utilities with compound outputs.
        Output is (status, schema) as a tuple for equality comparison.
        """

        def get_transition(state: int, action: Tuple[str, str, str]) -> Tuple[int, Tuple[int, str]]:
            method, ep, variant = action
            ns, status, schema = transitions[state][ep][method][variant]
            return (int(ns), (int(status), str(schema)))

        return get_transition

    def _make_ground_truth_adapter(
        self,
    ) -> "Callable[[int, Tuple[str, str, str]], Tuple[int, Tuple[int, str]]]":
        """Create adapter for ground truth transitions."""

        def get_transition(state: int, action: Tuple[str, str, str]) -> Tuple[int, Tuple[int, str]]:
            method, ep, variant = action
            ns, status, schema = self._transitions[state][ep][method][variant]
            return (int(ns), (int(status), str(schema)))

        return get_transition

    def _check_behavioral_equivalence(
        self,
        n_states: int,
        start: int,
        transitions: Dict[int, Dict[str, Dict[str, Dict[str, Tuple[int, int, str]]]]],
    ) -> bool:
        """Check behavioral equivalence using core.automata utility.

        Uses check_behavioral_equivalence with compound output (status, schema).
        """
        if n_states <= 0 or start < 0:
            return False

        # Build action list: tuples of (method, ep, variant)
        actions = [(m, ep, v) for m, ep, v in self._ops]

        try:
            return check_behavioral_equivalence(
                n_states_a=self._n_states,
                start_a=self._start,
                n_states_b=n_states,
                start_b=start,
                actions=actions,
                get_transition_a=self._make_ground_truth_adapter(),
                get_transition_b=self._make_transition_adapter(transitions),
            )
        except (KeyError, IndexError, TypeError):
            return False

    def _generate_counterexample(
        self,
        n_states: int,
        start: int,
        transitions: Dict[int, Dict[str, Dict[str, Dict[str, Tuple[int, int, str]]]]],
    ) -> List[Dict[str, Any]]:
        """Generate counterexample using core.automata utility."""
        if n_states <= 0 or start < 0:
            return [
                {
                    "call": _canonical_call("POST", "/login", "valid"),
                    "expected_status": 200,
                    "expected_schema": "AuthOk",
                    "reason": "invalid spec",
                }
            ]

        actions = [(m, ep, v) for m, ep, v in self._ops]

        try:
            cex = find_counterexample(
                start_true=self._start,
                start_hyp=start,
                actions=actions,
                get_transition_true=self._make_ground_truth_adapter(),
                get_transition_hyp=self._make_transition_adapter(transitions),
            )
        except (KeyError, IndexError, TypeError):
            # Malformed spec - return first action as counterexample
            if self._ops:
                m, ep, v = self._ops[0]
                ns, status, schema = self._transitions[self._start][ep][m][v]
                return [
                    {
                        "call": _canonical_call(m, ep, v),
                        "expected_status": int(status),
                        "expected_schema": str(schema),
                    }
                ]
            return []

        if cex is None:
            return []

        # Convert from [(action, output), ...] to skin-specific format
        # Output is (status, schema) tuple from our adapter
        result: List[Dict[str, Any]] = []
        for action, output in cex:
            method, ep, variant = action
            status, schema = output
            result.append(
                {
                    "call": _canonical_call(method, ep, variant),
                    "expected_status": int(status),
                    "expected_schema": str(schema),
                }
            )
        return result

    # ─────────────────────────────────────────────────────────────
    # Prompting
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def get_prompt_template(
        cls, obs: Dict[str, Any], *, feedback: bool = False
    ) -> List[Dict[str, str]]:
        n_states = int(obs.get("n_states", 7))
        n_endpoints = int(obs.get("n_endpoints", 6))
        budget = int(obs.get("budget", 35))
        trap = bool(obs.get("trap", True))

        endpoints = obs.get("endpoints")
        if endpoints is None:
            endpoints = list(ENDPOINT_CATALOG.keys())[:n_endpoints]

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
                "You are an autonomous agent reverse-engineering a hidden stateful SaaS REST API.\n"
                "Your goal is to infer a behaviorally equivalent specification of the API state machine.\n"
                "Return ONLY function tool calls; never output natural language.\n\n"
                "Episode semantics:\n"
                "- You start in hidden state 0.\n"
                "- Each api_call consumes 1 query from the budget.\n"
                "- submit_spec consumes 1 query. If correct, it ends the episode; otherwise it continues"
                + (" (a counterexample is returned)." if feedback else ".")
                + " The episode also ends when budget is exhausted.\n"
                + ("- Traps exist: some calls set trap_hit=true.\n" if trap else "")
                + "\n"
                "CORRECTNESS (BEHAVIORAL EQUIVALENCE):\n"
                "Your submission is correct if it matches the hidden API for ALL possible input sequences\n"
                "(matching both status code and response schema tag).\n"
                "You do NOT need to match the exact number of internal states; minimized equivalent specs are accepted.\n\n"
                "Tools:\n" + tools_text + "\n\n"
                "Specification JSON schema (STATE-DEPENDENT):\n"
                '{"n_states": <int>, "start": 0, "transitions": {"0": {"/endpoint": {"METHOD": {"variant": [next_state, status, schema]}}}}}\n\n'
                "Respond only with tool calls."
            ),
        }

        # Provide explicit domain alphabets in the observation.
        obs_json = spec.build_observation(
            endpoints=endpoints,
            methods=METHODS,
            variants={ep: ENDPOINT_CATALOG[ep] for ep in endpoints},
            status_codes=STATUS_CODES,
            response_schemas=SCHEMA_TAGS,
            budget=budget,
            n_states=n_states,
            trap=trap,
        )

        usr_msg = {
            "role": "user",
            "content": (
                "OBSERVATION:\n" + json.dumps(obs_json) + "\n\n"
                "Task: Use api_call() to explore the hidden API behavior, then submit_spec() with a complete specification.\n"
                + ("If incorrect, a counterexample will be returned.\n" if feedback else "")
                + "Respond only with tool calls."
            ),
        }

        return [sys_msg, usr_msg]

    # ─────────────────────────────────────────────────────────────
    # Static generation
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def generate_system_static(
        seed: int,
        n_states: int = 7,
        n_endpoints: int = 6,
        trap: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a deterministic, realistic workflow API system.

        The hidden system is created by selecting a subset of interpretable
        workflow profiles and compiling them into a finite automaton.

        The resulting system is:
        - Fully reachable from start (by construction)
        - Minimal for the chosen profiles (profiles have intentionally distinct behavior)
        - Trap-solvable when traps are enabled
        """

        rng = get_rng(seed)

        endpoints = list(ENDPOINT_CATALOG.keys())[:n_endpoints]
        variants_map = {ep: ENDPOINT_CATALOG[ep] for ep in endpoints}

        # Build a set of profiles. Keep n_states within supported range.
        base_profiles: List[Profile] = [
            Profile(False, False, "none", "free"),  # 0: anon
            Profile(True, False, "none", "free"),  # 1: authed, not verified
            Profile(True, True, "none", "free"),  # 2: verified, no org
            Profile(True, True, "A", "free"),  # 3: org A, free
            Profile(True, True, "B", "free"),  # 4: org B, free
            Profile(True, True, "A", "pro"),  # 5: org A, pro
            Profile(True, True, "B", "pro"),  # 6: org B, pro
            Profile(True, True, "none", "pro"),  # 7: verified, no org, pro
        ]

        n_states = int(max(4, min(n_states, len(base_profiles))))

        # Always include first 4 for reachability chain; sample remaining.
        selected: List[Profile] = base_profiles[:4]
        remaining = base_profiles[4:]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, n_states - len(selected))])

        # Ensure start profile is first.
        profiles = selected[:n_states]

        # Map profile -> state id; duplicates are not expected.
        profile_to_state = {p: i for i, p in enumerate(profiles)}

        def pick_state(wanted: Profile) -> int:
            if wanted in profile_to_state:
                return profile_to_state[wanted]

            # Fallbacks: relax org, then plan.
            if wanted.org != "none":
                alt = Profile(wanted.authed, wanted.verified, "none", wanted.plan)
                if alt in profile_to_state:
                    return profile_to_state[alt]
            if wanted.plan != "free":
                alt2 = Profile(wanted.authed, wanted.verified, wanted.org, "free")
                if alt2 in profile_to_state:
                    return profile_to_state[alt2]
            # As last resort, go to start.
            return 0

        def transition(
            p: Profile, method: str, endpoint: str, variant: str
        ) -> Tuple[Profile, int, str]:
            """Pure transition function over profiles."""

            # Default: forbidden unless handled.
            if endpoint == "/login" and method == "POST":
                if variant == "valid":
                    if p.authed:
                        return p, 200, "AlreadyAuthed"
                    return Profile(True, False, "none", "free"), 200, "AuthOk"
                return p, 401, "AuthFail"

            if endpoint == "/logout" and method == "POST":
                if not p.authed:
                    return p, 401, "NeedAuth"
                return Profile(False, False, "none", "free"), 200, "LoggedOut"

            if endpoint == "/verify_email" and method == "POST":
                if not p.authed:
                    return p, 401, "NeedAuth"
                if p.verified:
                    return p, 200, "AlreadyVerified"
                if variant == "code_ok":
                    return Profile(True, True, "none", p.plan), 200, "AlreadyVerified"
                return p, 403, "NeedVerify"

            if endpoint == "/select_org" and method == "POST":
                if not p.authed:
                    return p, 401, "NeedAuth"
                if not p.verified:
                    return p, 403, "NeedVerify"
                if variant == "orgA":
                    return Profile(True, True, "A", p.plan), 200, "OrgSelectedA"
                if variant == "orgB":
                    return Profile(True, True, "B", p.plan), 200, "OrgSelectedB"
                return p, 422, "BadOrg"

            if endpoint == "/upgrade_plan" and method == "POST":
                if not p.authed:
                    return p, 401, "NeedAuth"
                if not p.verified:
                    return p, 403, "NeedVerify"
                if variant == "noop":
                    return p, 200, "UpgradeOk" if p.plan == "free" else "AlreadyPro"
                if p.plan == "pro":
                    return p, 409, "AlreadyPro"
                return Profile(True, True, p.org, "pro"), 201, "UpgradeOk"

            if endpoint == "/projects" and method == "GET":
                if not p.authed:
                    return p, 401, "NeedAuth"
                if not p.verified:
                    return p, 403, "NeedVerify"
                if p.org == "none":
                    return p, 403, "NeedOrg"
                if p.org == "A":
                    return p, 200, "ProjectsListA"
                return p, 200, "ProjectsListB"

            if endpoint == "/projects" and method == "POST":
                if not p.authed:
                    return p, 401, "NeedAuth"
                if not p.verified:
                    return p, 403, "NeedVerify"
                if p.org == "none":
                    return p, 403, "NeedOrg"
                if p.plan == "free":
                    return p, 201, "ProjectCreatedFree"
                return p, 201, "ProjectCreatedPro"

            if endpoint == "/org" and method == "DELETE":
                if not p.authed:
                    return p, 401, "NeedAuth"
                if not p.verified:
                    return p, 403, "NeedVerify"
                if p.org == "none":
                    return p, 403, "NeedOrg"
                if variant == "dry_run":
                    return p, 200, "OrgDeleteDryRun"
                # confirm
                return Profile(False, False, "none", "free"), 200, "OrgDeleted"

            # If the call isn't in the domain, treat as not found.
            return p, 404, "NeedAuth"

        # Build transitions table
        transitions: Dict[str, Dict[str, Dict[str, Dict[str, List[Any]]]]] = {}
        for s, p in enumerate(profiles):
            transitions[str(s)] = {}
            for ep in endpoints:
                transitions[str(s)][ep] = {}
                for method, variants in variants_map[ep].items():
                    transitions[str(s)][ep][method] = {}
                    for variant in variants:
                        p2, status, schema = transition(p, method, ep, variant)
                        ns = pick_state(p2)
                        transitions[str(s)][ep][method][variant] = [ns, int(status), str(schema)]

        # Build operation list for trap feasibility checks
        all_ops: List[Tuple[str, str, str]] = []
        for ep in endpoints:
            for method, variants in variants_map[ep].items():
                for variant in variants:
                    all_ops.append((method, ep, variant))

        # Generate trap calls with solvability verification
        trap_calls: List[Tuple[int, str, str, str]] = []
        if trap:
            k = max(1, min(3, n_states // 2))
            seen: Set[Tuple[int, str, str, str]] = set()
            attempts = 0

            def get_next_state_fn(s: int, op: Tuple[str, str, str]) -> int:
                method, ep, variant = op
                return int(transitions[str(s)][ep][method][variant][0])

            while len(trap_calls) < k and attempts < 200:
                attempts += 1
                s = rng.randrange(n_states)
                if s == 0:
                    continue
                method, ep, variant = rng.choice(all_ops)
                cand = (s, ep, method, variant)
                if cand in seen:
                    continue

                test = seen | {cand}
                # Trap set must use the same action representation as `all_ops`.
                trap_set = {(st, (m, e, v)) for st, e, m, v in test}
                if verify_trap_free_path_exists(n_states, 0, all_ops, get_next_state_fn, trap_set):
                    seen.add(cand)
                    trap_calls.append(cand)

        spec = {
            "n_states": n_states,
            "start": 0,
            "endpoints": endpoints,
            "methods": METHODS,
            "variants": variants_map,
            "transitions": transitions,
        }

        return {
            "spec": spec,
            "trap_calls": [[s, ep, method, variant] for s, ep, method, variant in trap_calls],
        }
