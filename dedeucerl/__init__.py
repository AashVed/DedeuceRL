"""DedeuceRL: TaskIR/runtime/surface framework for hidden-system objectives."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("dedeucerl")
except PackageNotFoundError:  # pragma: no cover
    # Source checkout without an installed distribution.
    __version__ = "0.0.0"

from dedeucerl.core.rubric import make_rubric, reward_identification, score_identification
from dedeucerl.ir import (
    TASK_REGISTRY,
    ActionContext,
    ActionSpace,
    ActionValidationError,
    EnumSpace,
    ExactJSONContract,
    FiniteTransducerIsomorphismContract,
    HypothesisContract,
    HypothesisInputError,
    HypothesisJudgment,
    HypothesisParseResult,
    HypothesisValidationResult,
    JsonSchemaSpace,
    MaskedSpace,
    NonEnumerableActionSpace,
    ProductSpace,
    SequenceSpace,
    TaskEntry,
    TaskIR,
    ToolActionContract,
    ToolActionSpace,
    ToolKind,
    UnionSpace,
    get_task_entry,
)
from dedeucerl.ir.hypotheses import HypothesisObjective
from dedeucerl.ir.objectives import (
    Evaluation,
    EvaluationContext,
    Execution,
    ExecutionStep,
    FeedbackSpec,
    Objective,
    ObjectiveContract,
    ObjectiveInputError,
    ObjectiveResult,
    ToolCall,
)
from dedeucerl.ir.typed_space import TypedSpace
from dedeucerl.kernel import (
    KernelTransition,
    MealyKernel,
    SystemKernel,
    TaskInstance,
)
from dedeucerl.runtime import EpisodeRuntime

__all__ = [
    "__version__",
    "ActionContext",
    "ActionSpace",
    "ActionValidationError",
    "EpisodeRuntime",
    "EnumSpace",
    "ExactJSONContract",
    "FiniteTransducerIsomorphismContract",
    "HypothesisContract",
    "HypothesisInputError",
    "HypothesisJudgment",
    "HypothesisParseResult",
    "HypothesisValidationResult",
    "JsonSchemaSpace",
    "KernelTransition",
    "MaskedSpace",
    "MealyKernel",
    "NonEnumerableActionSpace",
    "ProductSpace",
    "SequenceSpace",
    "SystemKernel",
    "TASK_REGISTRY",
    "TaskEntry",
    "TaskIR",
    "TaskInstance",
    "ToolActionContract",
    "ToolActionSpace",
    "ToolKind",
    "UnionSpace",
    "get_task_entry",
    "make_rubric",
    "reward_identification",
    "score_identification",
    "Evaluation",
    "EvaluationContext",
    "Execution",
    "ExecutionStep",
    "FeedbackSpec",
    "Objective",
    "ObjectiveContract",
    "ObjectiveInputError",
    "ObjectiveResult",
    "ToolCall",
    "HypothesisObjective",
    "TypedSpace",
]
