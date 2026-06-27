"""Mealy TaskIR reference implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from dedeucerl.core.automata import check_isomorphism_with_signatures, find_counterexample
from dedeucerl.ir.types import EnumSpace, FeedbackModel, ResourceModel, TaskIR
from dedeucerl.kernel.mealy import (
    ALPHABET,
    OUTPUTS,
    MealyKernel,
    generate_mealy_system,
    parse_transitions,
)
from dedeucerl.kernel.types import (
    KernelInputError,
    KernelJudgment,
    KernelParam,
    TaskInstance,
    ToolContract,
)
from dedeucerl.utils import error_invalid_json, error_malformed_hypothesis
from dedeucerl.utils.schema import validate_jsonschema


MEALY_TASK_NAME = "mealy"
MEALY_TASK_VERSION = "2.1"


@dataclass(frozen=True)
class MealyObservationModel:
    """Public observation model for Mealy tasks."""

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        table = instance.private["table"]
        return {
            "kernel": MEALY_TASK_NAME,
            "alphabet": list(ALPHABET),
            "outputs": list(OUTPUTS),
            "n_states": int(table["n"]),
            "budget": int(instance.budget),
            "trap": bool(instance.params.get("trap", True)),
        }


@dataclass(frozen=True)
class MealyHypothesisContract:
    """Complete-table submission contract for Mealy tasks."""

    def tool_contracts(self, instance: TaskInstance) -> Sequence[ToolContract]:
        _ = instance
        return (
            ToolContract(
                name="submit_table",
                kind="submit",
                description="Submit a complete Mealy transition table as a JSON string.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "table_json": {
                            "type": "string",
                            "description": (
                                'JSON: {"n": <int>, "start": 0, "trans": '
                                '{"0": {"A": [next_state, output], ...}, ...}}'
                            ),
                        }
                    },
                    "required": ["table_json"],
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "budget_left": {"type": "integer"},
                        "queries_used": {"type": "integer"},
                        "trap_hit": {"type": "boolean"},
                        "counterexample": {"type": ["array", "null"]},
                    },
                    "required": ["ok", "budget_left", "queries_used", "trap_hit"],
                },
                cost=1,
            ),
        )

    def handles(self, tool_name: str) -> bool:
        return tool_name == "submit_table"

    def judge(
        self,
        instance: TaskInstance,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> KernelJudgment:
        if not self.handles(tool_name):
            raise KeyError(tool_name)

        raw = args.get("table_json")
        if not isinstance(raw, str):
            raise KernelInputError(error_malformed_hypothesis("table_json must be a string"))

        try:
            table = json.loads(raw)
        except Exception as e:
            raise KernelInputError(error_invalid_json("table_json")) from e

        reason = validate_jsonschema(table, _hypothesis_schema())
        if reason:
            raise KernelInputError(error_malformed_hypothesis(reason))

        if not isinstance(table, dict):
            raise KernelInputError(error_malformed_hypothesis("table_json must encode an object"))

        ok = _isomorphic(instance.private["table"], table)
        cex = None if ok else _counterexample(instance.private["table"], table)
        return KernelJudgment(ok=ok, observation={"ok": ok}, counterexample=cex)


@dataclass(frozen=True)
class MealyGenerator:
    """Deterministic TaskIR generator for Mealy task instances."""

    params: Mapping[str, KernelParam] = field(
        default_factory=lambda: {
            "n_states": KernelParam(
                type="int",
                description="Number of states in the hidden Mealy machine.",
                default=3,
                choices=None,
            ),
            "trap": KernelParam(
                type="bool",
                description="Whether to include trap transitions.",
                default=True,
            ),
        }
    )

    def sample(self, *, seed: int, budget: int, **kwargs: Any) -> TaskInstance:
        n_states = int(kwargs.get("n_states", self.params["n_states"].default))
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if int(budget) < 0:
            raise ValueError("budget must be >= 0")
        trap = bool(kwargs.get("trap", self.params["trap"].default))
        private = generate_mealy_system(seed=seed, n_states=n_states, trap=trap)
        return TaskInstance(
            id=f"mealy-{seed}",
            seed=int(seed),
            kernel_name=MEALY_TASK_NAME,
            kernel_version=MEALY_TASK_VERSION,
            private=private,
            params={"n_states": n_states, "trap": trap},
            budget=int(budget),
        )


@dataclass(frozen=True)
class MealyPromptRenderer:
    """Prompt renderer for Mealy TaskIR instances."""

    def render(
        self,
        ir: TaskIR,
        instance: TaskInstance,
        contracts: Sequence[ToolContract],
        *,
        feedback: bool = False,
    ) -> list[dict[str, Any]]:
        tools_text = "\n".join(_format_tool(contract) for contract in contracts)
        system = (
            "You are an autonomous tool-using agent solving a hidden-system identification task.\n"
            "Return only function tool calls; do not answer in natural language.\n\n"
            "Episode semantics:\n"
            "- Probe/diagnostic tools may reveal observations and can change hidden state.\n"
            "- Submit tools judge a hypothesis. Correct submissions end the episode.\n"
            "- Every valid or invalid tool call can consume budget.\n"
            "- Budget exhaustion ends the episode.\n"
            "- Trap hits are reported in tool results and may affect reward.\n\n"
            "Mealy task: identify a finite-state transducer. "
            'Submit JSON shaped like {"n": <states>, "start": 0, '
            '"trans": {"0": {"A": [next_state, output], ...}, ...}}.\n\n'
            "Tools:\n"
            f"{tools_text}"
        )
        if feedback:
            system += "\n\nFeedback mode: incorrect submissions may include a counterexample."

        user = (
            "OBSERVATION:\n"
            + json.dumps(ir.public_observation(instance), sort_keys=True)
            + "\n\nUse the available tools to identify the hidden system, then submit a complete hypothesis."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_mealy_ir() -> TaskIR:
    action_space = EnumSpace("symbol", ALPHABET)
    return TaskIR(
        name=MEALY_TASK_NAME,
        version=MEALY_TASK_VERSION,
        kernel=MealyKernel(),
        action_space=action_space,
        observation_model=MealyObservationModel(),
        hypothesis_contract=MealyHypothesisContract(),
        resource_model=ResourceModel(unknown_tool_cost=1, trap_ends_episode=False),
        feedback_model=FeedbackModel(reveal_counterexample=True),
        generator=MealyGenerator(),
        tools=(
            ToolContract(
                name="act",
                kind="probe",
                description="Execute one input symbol on the hidden Mealy machine.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            **dict(action_space.json_schema()),
                            "description": "Input symbol to execute.",
                        }
                    },
                    "required": ["symbol"],
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "out": {"type": "integer"},
                        "budget_left": {"type": "integer"},
                        "t": {"type": "integer"},
                        "trap_hit": {"type": "boolean"},
                        "queries_used": {"type": "integer"},
                    },
                    "required": ["out", "budget_left", "t", "trap_hit", "queries_used"],
                },
                cost=1,
            ),
        ),
        renderers={"prompt": MealyPromptRenderer()},
    )


def _hypothesis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1},
            "start": {"type": "integer", "const": 0},
            "trans": {
                "type": "object",
                "patternProperties": {
                    "^[0-9]+$": {
                        "type": "object",
                        "properties": {
                            sym: {
                                "type": "array",
                                "items": [{"type": "integer"}, {"type": "integer"}],
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
    }


def _isomorphic(truth: Mapping[str, Any], hypothesis: Mapping[str, Any]) -> bool:
    try:
        n = int(hypothesis.get("n", -1))
        start = int(hypothesis.get("start", -1))
        true_n = int(truth.get("n", -2))
        true_start = int(truth.get("start", -2))
        hyp_trans = parse_transitions(hypothesis)
        true_trans = parse_transitions(truth)
    except Exception:
        return False

    if n != true_n or start != true_start:
        return False

    try:
        for s in range(n):
            for a in ALPHABET:
                ns, _ = hyp_trans[s][a]
                if not (0 <= ns < n):
                    return False
    except Exception:
        return False

    return check_isomorphism_with_signatures(
        n,
        true_start,
        start,
        ALPHABET,
        lambda s, a: true_trans[s][a],
        lambda s, a: hyp_trans[s][a],
    )


def _counterexample(
    truth: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    try:
        true_start = int(truth.get("start", 0))
        hyp_start = int(hypothesis.get("start", 0))
        true_trans = parse_transitions(truth)
        hyp_trans = parse_transitions(hypothesis)
    except Exception:
        return [{"in": "A", "out": 0}]

    cex = find_counterexample(
        true_start,
        hyp_start,
        ALPHABET,
        lambda s, a: true_trans[s][a],
        lambda s, a: hyp_trans[s][a],
    )
    if cex is None:
        return None
    return [{"in": action, "out": output} for action, output in cex]


def _format_tool(contract: ToolContract) -> str:
    props = contract.args_schema.get("properties", {})
    required = set(contract.args_schema.get("required", []))
    args = []
    if isinstance(props, dict):
        for name, schema in props.items():
            if not isinstance(schema, dict):
                continue
            typ = schema.get("type", "any")
            enum = schema.get("enum")
            suffix = " required" if name in required else " optional"
            enum_text = f" one of {enum}" if enum else ""
            args.append(f"{name}: {typ}{enum_text}{suffix}")
    return f"- {contract.name}({', '.join(args)}) [{contract.kind}, cost={contract.cost}]"
