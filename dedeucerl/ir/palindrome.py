"""A Mealy plan objective built entirely on the public custom-objective API."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from dedeucerl.ir.actions import ToolActionContract
from dedeucerl.ir.mealy import MealyGenerator, MealyObservationModel, build_mealy_ir
from dedeucerl.ir.objectives import Evaluation, EvaluationContext, FeedbackSpec, Objective, ToolCall
from dedeucerl.ir.typed_space import TypedSpace
from dedeucerl.ir.types import TaskIR
from dedeucerl.kernel.types import KernelParam, TaskInstance

PALINDROME_TASK_VERSION = "1.2"


class PalindromePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["sequence"]
    actions: list[Literal["A", "B", "C"]] = Field(min_length=1)


class ImpossibleClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["impossible"]


class PalindromeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    answer: Annotated[PalindromePlan | ImpossibleClaim, Field(discriminator="kind")]


class PalindromeGoal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_length: int = Field(default=5, ge=2)
    min_distinct: int = Field(default=2, ge=2, le=3)


@dataclass(frozen=True)
class PalindromeEvidence:
    outputs: tuple[int, ...]
    goal: PalindromeGoal


class PalindromeFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reason: Literal["too_short", "insufficient_variety", "not_palindrome", "solution_exists"]
    mismatches: list[tuple[int, int]] = Field(default_factory=list)


def evaluate_palindrome(
    context: EvaluationContext[int],
    submission: PalindromeAnswer,
) -> Evaluation[PalindromeEvidence | None]:
    plan = submission.answer
    if isinstance(plan, ImpossibleClaim):
        # Truth concerns the fixed goal, never the remaining exploration budget.
        # A wrong claim is final so this cannot become a retryable existence oracle.
        return Evaluation(palindrome_witness(context.instance) is None, None, terminal=True)
    goal = _goal(context.instance)
    execution = context.run([ToolCall("act", {"symbol": symbol}) for symbol in plan.actions])
    outputs = tuple(int(step.observation["out"]) for step in execution.steps)
    ok = (
        len(outputs) >= goal.min_length
        and len(set(outputs)) >= goal.min_distinct
        and outputs == outputs[::-1]
    )
    return Evaluation(ok, PalindromeEvidence(outputs, goal))


def palindrome_feedback(evidence: PalindromeEvidence | None) -> PalindromeFeedback:
    if evidence is None:
        return PalindromeFeedback(reason="solution_exists")
    outputs, goal = evidence.outputs, evidence.goal
    if len(outputs) < goal.min_length:
        return PalindromeFeedback(reason="too_short")
    if len(set(outputs)) < goal.min_distinct:
        return PalindromeFeedback(reason="insufficient_variety")
    return PalindromeFeedback(
        reason="not_palindrome",
        mismatches=[
            (i, len(outputs) - 1 - i)
            for i in range(len(outputs) // 2)
            if outputs[i] != outputs[-1 - i]
        ],
    )


def _goal(instance: TaskInstance) -> PalindromeGoal:
    return PalindromeGoal.model_validate(
        {name: instance.params[name] for name in ("min_length", "min_distinct")}
    )


@dataclass(frozen=True)
class PalindromeObservation:
    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        return {
            **MealyObservationModel().public_observation(instance),
            "goal": {"kind": "palindrome", **_goal(instance).model_dump()},
            "evaluation_start": "initial",
            "submission_cost": "1 plus the cost of each executed action",
            "impossible_answer_rule": (
                "You may answer kind='impossible' if you conclude that no trap-free action "
                "sequence of any length from the initial state satisfies the goal. "
                "Whether this machine has a solution is hidden; this rule does not tell you "
                "that it is impossible. The claim costs 1 and ends the episode whether "
                "correct or incorrect. Spending your budget does not make the goal impossible."
            ),
        }


@dataclass(frozen=True)
class PalindromeGenerator:
    params: Mapping[str, KernelParam] = field(
        default_factory=lambda: {
            **MealyGenerator().params,
            "min_length": KernelParam(
                type="int", description="Minimum palindrome length.", default=5
            ),
            "min_distinct": KernelParam(
                type="int",
                description="Minimum distinct output symbols.",
                default=2,
                choices=[2, 3],
            ),
        }
    )

    def sample(self, *, seed: int, budget: int, **kwargs: Any) -> TaskInstance:
        unknown = kwargs.keys() - self.params.keys()
        if unknown:
            raise ValueError(f"Unknown palindrome parameters: {sorted(unknown)}")
        goal = PalindromeGoal(
            min_length=kwargs.get("min_length", 5), min_distinct=kwargs.get("min_distinct", 2)
        )
        if budget < 1:
            raise ValueError("budget must cover the submission fee")
        for attempt in range(1000):
            machine = MealyGenerator().sample(
                seed=seed + attempt * 1_000_003,
                budget=budget,
                **{name: kwargs[name] for name in MealyGenerator().params if name in kwargs},
            )
            instance = replace(
                machine,
                id=f"mealy-palindrome-{seed}",
                seed=seed,
                kernel_name="mealy_palindrome",
                kernel_version=PALINDROME_TASK_VERSION,
                params={**machine.params, **goal.model_dump()},
            )
            witness = palindrome_witness(instance)
            # Retain naturally impossible instances. Only reject a possible goal
            # whose cheapest execution exceeds the declared initial budget.
            if witness is None or len(witness) + 1 <= budget:
                return instance
        raise ValueError("Could not sample a palindrome task satisfying the budget constraint")


def palindrome_witness(instance: TaskInstance) -> list[str] | None:
    """Return a shortest qualifying safe plan, or prove none exists at any length.

    Pair-state palindrome reachability (Anderson et al., arXiv:0711.3183,
    section 3), extended with an output mask and length saturated at the minimum.
    Saturation preserves all future acceptance conditions, so exhausting the
    finite graph proves impossibility without a budget or arbitrary length cap.
    At most n**2 * 2**q * (minimum + 1) nodes are explored; q is the output count.
    Parent pointers retain one shortest witness per node without copying plans.
    This private analysis never changes exploration state or consumes its budget.
    """
    goal = _goal(instance)
    table = instance.private["table"]
    traps = {(int(s), str(a)) for s, a in instance.private.get("trap_pairs", [])}
    incoming: dict[int, list[tuple[int, str, int]]] = {s: [] for s in range(table["n"])}
    outgoing: dict[int, list[tuple[int, str, int]]] = {s: [] for s in range(table["n"])}
    for raw_state, row in table["trans"].items():
        state = int(raw_state)
        for symbol, (target, output) in row.items():
            if (state, symbol) not in traps:
                outgoing[state].append((target, symbol, output))
                incoming[target].append((state, symbol, output))
    Node = tuple[int, int, int, int]
    # (inner node, first action, last action); base nodes contain only a center.
    parents: dict[Node, tuple[Node | None, str, str]] = {
        (s, s, 0, 0): (None, "", "") for s in range(table["n"])
    }
    for s, edges in outgoing.items():
        for target, symbol, out in edges:
            parents.setdefault((s, target, 1 << out, 1), (None, symbol, ""))
    pending = deque(parents)
    while pending:
        node = pending.popleft()
        left, right, mask, length = node
        if (
            left == table["start"]
            and length == goal.min_length
            and mask.bit_count() >= goal.min_distinct
        ):
            prefix, suffix = [], []
            current: Node | None = node
            while current is not None:
                current, first, last = parents[current]
                if first:
                    prefix.append(first)
                if last:
                    suffix.append(last)
            return prefix + suffix[::-1]
        for source, first, out in incoming[left]:
            for target, last, other in outgoing[right]:
                if out != other:
                    continue
                key = (source, target, mask | (1 << out), min(goal.min_length, length + 2))
                if key not in parents:
                    parents[key] = (node, first, last)
                    pending.append(key)
    return None


def build_palindrome_ir() -> TaskIR[int]:
    base = build_mealy_ir()
    objective: Objective[int, PalindromeAnswer, PalindromeEvidence | None, PalindromeFeedback] = (
        Objective(
            name="palindrome",
            version="1.1",
            submission=ToolActionContract[PalindromeAnswer](
                name="submit_answer",
                kind="submit",
                description=(
                    "Submit answer={kind:'sequence', actions:[...]} to execute from the initial "
                    "state in isolation. Outputs must satisfy the palindrome goal. Failed plans "
                    "may be retried; costs 1 plus each executed action. Or submit "
                    "answer={kind:'impossible'} to claim no safe sequence of any length satisfies "
                    "the goal. That claim costs 1 and ends the episode whether correct or wrong."
                ),
                action_space=TypedSpace("palindrome_answer", PalindromeAnswer),
                return_schema={},
                cost=1,
            ),
            evaluator=evaluate_palindrome,
            feedback=FeedbackSpec(PalindromeFeedback, palindrome_feedback),
        )
    )
    return replace(
        base,
        name="mealy_palindrome",
        version=PALINDROME_TASK_VERSION,
        objective=objective,
        generator=PalindromeGenerator(),
        observation_model=PalindromeObservation(),
        renderers={},
    )
