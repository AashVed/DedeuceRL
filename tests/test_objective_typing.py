"""Negative authoring examples prevent accidental Any holes in the public API."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def test_type_checker_rejects_incompatible_objective_components(tmp_path):
    source = dedent("""\
        from dedeucerl.ir import FeedbackSpec, Objective, ObjectiveContract, TaskIR, ToolActionContract, TypedSpace
        from dedeucerl.ir.palindrome import PalindromeAnswer, PalindromeEvidence, PalindromeFeedback, evaluate_palindrome, palindrome_feedback
        from dedeucerl.kernel.mealy import MealyKernel
        from examples.custom_objective import WorkflowPlan, WorkflowState, build_workflow_ir, workflow_feedback

        workflow = build_workflow_ir()
        compatible: ObjectiveContract[WorkflowState] = workflow.objective
        wrong_state: ObjectiveContract[int] = workflow.objective  # reject
        wrong_task = TaskIR(name="bad", version="1", kernel=MealyKernel(), objective=workflow.objective, action_space=workflow.action_space, observation_model=workflow.observation_model, resource_model=workflow.resource_model, generator=workflow.generator)  # reject
        wrong_feedback: FeedbackSpec[PalindromeEvidence, PalindromeFeedback] = FeedbackSpec(PalindromeFeedback, workflow_feedback)  # reject
        wrong_space: TypedSpace[PalindromeAnswer] = TypedSpace("bad", WorkflowPlan)  # reject
        submission = ToolActionContract[PalindromeAnswer](name="submit", kind="submit", description="test", action_space=TypedSpace("plan", PalindromeAnswer), return_schema={})
        wrong_evaluator: Objective[WorkflowState, PalindromeAnswer, PalindromeEvidence | None, PalindromeFeedback] = Objective(name="bad", version="1", submission=submission, evaluator=evaluate_palindrome, feedback=FeedbackSpec(PalindromeFeedback, palindrome_feedback))  # reject
    """)
    path = tmp_path / "invalid_objectives.py"
    path.write_text(source)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", str(path), "--no-error-summary"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    errors = {int(line.split(":")[1]) for line in result.stdout.splitlines() if ": error:" in line}
    expected = {number for number, line in enumerate(source.splitlines(), 1) if "# reject" in line}
    assert errors == expected, result.stdout + result.stderr
