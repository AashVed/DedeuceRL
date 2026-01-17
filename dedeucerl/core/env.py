"""Base environment class for active system identification."""

from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from verifiers.envs.stateful_tool_env import StatefulToolEnv
from verifiers.types import State
from verifiers import Rubric
from datasets import Dataset

from .config import SkinConfig

from dedeucerl.utils import (
    DedeuceError,
    error_budget_exhausted,
    error_episode_finished,
    error_invalid_json,
    error_malformed_hypothesis,
)
from dedeucerl.utils.schema import validate_jsonschema


class HiddenSystemEnv(StatefulToolEnv):
    """
    Base class for active system identification environments.

    Inherits from verifiers.StatefulToolEnv for full ecosystem compatibility.
    Skins implement domain-specific logic by overriding abstract methods.

    The active identification paradigm:
    1. Agent probes the hidden system via domain-specific tools
    2. Each probe consumes budget
    3. Agent submits a hypothesis about the hidden system
    4. Hypothesis is checked for correctness (possibly up to isomorphism)
    """

    # Class-level configuration (override in subclass)
    config: SkinConfig = SkinConfig()

    def __init__(
        self,
        dataset: Dataset,
        rubric: Rubric,
        *,
        feedback: bool = False,
        max_turns: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the environment.

        Args:
            dataset: HuggingFace Dataset with (prompt, answer) pairs.
            rubric: Scoring rubric for evaluating episodes.
            feedback: Whether to provide counterexamples on incorrect submissions.
            max_turns: Maximum tool calls per episode.
        """
        self.feedback_enabled = feedback or self.config.feedback_enabled

        tools = self._get_tools()

        super().__init__(
            tools=tools,
            max_turns=max_turns or self.config.max_turns,
            dataset=dataset,
            rubric=rubric,
            **kwargs,
        )

        # Internal state (set per-episode in setup_state)
        self._ground_truth: Any = None
        self._trap_pairs: set = set()
        self._state_ref: Optional[State] = None

    # ─────────────────────────────────────────────────────────────
    # verifiers lifecycle methods
    # ─────────────────────────────────────────────────────────────

    async def setup_state(self, state: State, **kwargs) -> State:
        """Initialize episode from dataset item.

        The state["answer"] contains JSON-encoded episode metadata:
        - Ground truth system specification
        - Budget, trap settings, etc.
        """
        meta = json.loads(state.get("answer", "{}"))

        # Parse ground truth (skin-specific)
        self._configure_from_metadata(meta)

        budget = int(meta.get("budget", self.config.default_budget))
        if budget < 0:
            budget = 0

        done = budget <= 0

        # Initialize mutable episode state
        state.update(
            {
                "cs": self._get_start_state(),
                "budget": budget,
                "budget_init": budget,
                "queries_used": 0,
                "steps": 0,
                "trap_hit": False,
                "ok": False,
                "done": done,
            }
        )

        return state

    def update_tool_args(self, tool_name: str, tool_args: dict, messages, state, **kwargs) -> dict:
        """Attach state reference for tool methods."""
        self._state_ref = state
        return tool_args

    # ─────────────────────────────────────────────────────────────
    # Abstract methods (MUST be implemented by skins)
    # ─────────────────────────────────────────────────────────────

    @abstractmethod
    def _configure_from_metadata(self, meta: Dict[str, Any]) -> None:
        """
        Parse ground truth from answer metadata.

        Should set:
        - self._ground_truth: The hidden system specification
        - self._trap_pairs: Set of trap transitions (skin-specific format)
        - Any other skin-specific state
        """
        pass

    @abstractmethod
    def _get_start_state(self) -> Any:
        """Return the starting state for the hidden system."""
        pass

    @abstractmethod
    def _get_tools(self) -> List:
        """
        Return the list of tool methods for this skin.

        Each skin defines its own probe and submit methods with explicit signatures.
        Tool signatures vary by domain:

        - MealySkin: act(symbol: str), submit_table(table_json: str)
        - ProtocolSkin: api_call(method, endpoint), submit_spec(spec_json)
        """
        pass

    # ─────────────────────────────────────────────────────────────
    # Optional overrides (sensible defaults)
    # ─────────────────────────────────────────────────────────────

    def is_isomorphic(self, hypothesis: Any, ground_truth: Any) -> bool:
        """
        Check if hypothesis is equivalent to ground truth.

        Default: exact equality. Override for domain-specific equivalence.
        If config.isomorphism_fn is set, uses that function.
        """
        if not self.config.isomorphism_check:
            return hypothesis == ground_truth

        if self.config.isomorphism_fn is not None:
            return self.config.isomorphism_fn(hypothesis, ground_truth)

        return hypothesis == ground_truth

    def get_counterexample(self, hypothesis: Any, ground_truth: Any) -> Optional[Any]:
        """
        Generate a counterexample showing where hypothesis differs.

        Default: None. Override for domain-specific counterexamples.
        """
        if self.config.counterexample_fn is not None:
            return self.config.counterexample_fn(hypothesis, ground_truth)
        return None

    # ─────────────────────────────────────────────────────────────
    # Helper methods (available to all skins)
    # ─────────────────────────────────────────────────────────────

    def _state(self) -> Dict[str, Any]:
        """Get the current episode state dict."""
        if self._state_ref is None:
            raise RuntimeError("State not initialized. Call setup_state first.")
        return self._state_ref

    def _consume_budget(self, amount: int = 1) -> bool:
        """Consume query budget.

        Returns False if insufficient budget; in that case the episode is ended.
        Also ends the episode once budget reaches 0.
        """
        state = self._state()
        budget = int(state.get("budget", 0))

        if budget < amount:
            state["budget"] = 0
            state["done"] = True
            state["ok"] = False
            return False

        state["budget"] = budget - amount
        state["queries_used"] = int(state.get("queries_used", 0)) + amount

        if int(state.get("budget", 0)) <= 0:
            state["budget"] = 0
            state["done"] = True
            # If they haven't already succeeded, budget exhaustion is failure.
            if not bool(state.get("ok", False)):
                state["ok"] = False

        return True

    def _parse_json_arg(
        self, raw: str, *, context: str
    ) -> tuple[Optional[Any], Optional[DedeuceError]]:
        """Parse a JSON string argument.

        Returns (value, error). This is a small helper for tool methods.
        """
        try:
            return json.loads(raw), None
        except Exception:
            return None, error_invalid_json(context)

    def _prevalidate_hypothesis(
        self,
        hypothesis: Any,
        schema: Optional[Dict[str, Any]],
    ) -> Optional[DedeuceError]:
        """Optionally validate a hypothesis against a JSONSchema-like dict.

        This is opt-in: skins call it from their submit tools if they want
        consistent "malformed hypothesis" taxonomy.

        Returns:
            A DedeuceError if invalid, else None.
        """
        if not schema:
            return None

        reason = validate_jsonschema(hypothesis, schema)
        if reason:
            return error_malformed_hypothesis(reason)
        return None

    def _tool_error(self, err: DedeuceError, *, extra: Optional[Dict[str, Any]] = None) -> str:
        """Return a standard tool error envelope.

        Skins may include additional domain-specific fields via `extra`.
        """
        state = self._state()
        payload: Dict[str, Any] = {
            "error": err.to_dict(),
            "budget_left": int(state.get("budget", 0)),
            "queries_used": int(state.get("queries_used", 0)),
            "trap_hit": bool(state.get("trap_hit", False)),
        }
        if extra:
            payload.update(extra)
        return json.dumps(payload)

    def _episode_finished(self) -> str:
        """Return a standard "episode finished" tool error."""
        return self._tool_error(error_episode_finished())

    def _budget_exhausted(self) -> str:
        """Mark episode finished due to budget exhaustion and return tool error."""
        state = self._state()
        state["budget"] = 0
        state["done"] = True
        state["ok"] = False
        return self._tool_error(error_budget_exhausted())

    def _mark_trap_hit(self) -> None:
        """Mark the episode as having hit a trap."""
        state = self._state()
        state["trap_hit"] = True
        if self.config.trap_ends_episode:
            state["done"] = True
            state["ok"] = False

    def _end_episode(self, success: bool) -> None:
        """End the episode with given success status."""
        state = self._state()
        state["done"] = True
        state["ok"] = success and not state["trap_hit"]
