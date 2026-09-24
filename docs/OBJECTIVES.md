# Custom objectives

A task defines what the agent should accomplish with a hidden system. Reconstructing
its transition table is one option. Producing an output, configuring a service,
reaching a target state, or satisfying a trace property are others.

`TaskIR[State]` combines a `SystemKernel[State]` with an `ObjectiveContract[State]`.
The built-in `Objective[State, Candidate, Evidence, Feedback]` covers custom tasks:

- **Candidate:** the agent's typed submission, decoded from JSON by `TypedSpace`.
- **Evaluator:** an author function receiving a private `EvaluationContext[State]`
  and the candidate. It returns `Evaluation(ok, evidence, terminal=False)`.
- **Evidence:** any author-defined Python value used internally to judge the attempt.
- **Feedback:** an optional typed, public explanation derived from that evidence.

There is no required transition-table format and no required counterexample format.

## Exploration and submission

Exploration calls operate on the live episode state. A submission's evaluator can
call `context.run([ToolCall(...)])` to execute a plan on an isolated copy of the
hidden instance. **Every run starts from the kernel's initial state.** It never
reads or changes the live exploration state. There is no current-state mode.

Exploration, submission fees, and actions executed during evaluation all spend
one episode budget. There is no separate attempt limit or framework sequence-length
cap. Authors can put semantic constraints on candidates through their model.
By default, an unsuccessful submission permits more exploration or another submission
while budget remains. An evaluator can return `Evaluation(False, evidence, terminal=True)`
to close the episode with failure. Every successful submission closes the episode.
Tool responses include `done` so hosts can stop on either outcome. A plan that runs out of budget fails; a successful complete plan
may spend the last budget unit. Invalid submissions still pay their submission fee.

Existing trap policy applies to executed candidate actions too: a trap taints the
episode, and a terminal trap stops execution. Isolation does not remove trap costs.
`tool_calls` counts external calls; `evaluation_steps` counts successfully executed
kernel calls inside submissions. `queries_used` counts consumed budget units.
Each submission trace event records the total fee plus execution cost. Intermediate
execution evidence is private, and replay repeats the isolated evaluation.

## Palindrome example

The registered `mealy_palindrome` task lets an agent explore using `act` and submit:

```json
{"answer": {"kind": "sequence", "actions": ["A", "A", "B"]}}
```

`submit_answer` executes those actions from the initial state. If their outputs
are `0, 1, 0`, the sequence satisfies a palindrome goal with minimum length three
and at least two distinct symbols. The agent chooses the palindrome: no particular
output string or exact submitted length is prescribed. The default minimum length
is five and minimum variety is two; both appear in the public goal.

```bash
dedeucerl-mcp serve --task mealy_palindrome --seed 7 --budget 25 --feedback \
  --param min_length=5 --param min_distinct=2

dedeucerl-generate --task mealy_palindrome --seeds 0-9 --budget 25 \
  --param min_length=5 -o palindrome.json
```

The same tool accepts an impossibility claim:

```json
{"answer": {"kind": "impossible"}}
```

This claims that **no trap-free sequence of any length from the initial state**
satisfies the public goal. It costs one budget unit and ends the episode whether
correct or wrong. Spending the exploration budget does not change that truth.
A wrong claim cannot be used as a retryable existence query.

The generator samples natural seeded Mealy machines and applies one constraint:
a machine must either have no qualifying safe solution, or have a shortest solution
whose execution plus submission fee fits the initial budget. Possible but
unaffordable candidates are skipped using deterministic seed progression. There is
no requested possible/impossible ratio and no guarantee a small cohort contains
both. Setting `min_distinct=3` requires all three output symbols; the default is two.

A private exact checker belongs to this objective, not the generic runtime. It
searches a finite graph of paired machine states, output-symbol masks, and lengths
saturated at the goal minimum. Equal-output edges extend a palindrome at both ends.
The first accepting path gives a shortest witness; exhausting the graph proves
impossibility at every length, without a search cutoff or dependence on budget.
This extends the pair-state construction in
[Anderson et al., section 3](https://arxiv.org/abs/0711.3183). There are at most
`n² × 2^q × (min_length + 1)` nodes; this is practical for the six-state,
three-output benchmark, not a claim of cheap verification for arbitrary objectives.

The checker is reused during generation and when grading an impossibility claim.
Its private computation consumes no episode budget and exposes no witness. With
feedback enabled, failed plans receive a reason and, for a non-palindrome, positions
of unequal output pairs. A wrong impossibility claim returns `solution_exists`
after closing the episode; correct claims and disabled feedback return null.
See [the implementation](../dedeucerl/ir/palindrome.py).

Default reward is binary correctness: **1 for either correct answer, 0 for failure
or a trap**. Calls, executed actions, and budget usage remain separate metrics.
The optional dense training rubric remains an explicit alternative.

This example demonstrates the API; it does not establish difficulty against a
particular model. It shares the [Mealy generator](GENERATION.md) and scoring policy.
Benchmark difficulty and distribution diversity still need measured model runs.

## Defining an objective

The runnable [workflow example](../examples/custom_objective.py) uses a non-Mealy
kernel. Its hidden service may require verification before project creation. The
agent must submit a workflow that creates a private project with viewer access.
Its evaluator uses the typed final state directly:

```python
def evaluate_workflow(
    context: EvaluationContext[WorkflowState], plan: WorkflowPlan,
) -> Evaluation[WorkflowState]:
    execution = context.run([
        ToolCall("request", {"operation": op}) for op in plan.operations
    ])
    state = execution.final_state
    return Evaluation(state.project and state.viewer and not state.public, state)
```

`WorkflowPlan` is a Pydantic model with a list of allowed operations. Wiring it up
requires a submission contract, evaluator, and optional feedback formatter:

```python
objective: Objective[WorkflowState, WorkflowPlan, WorkflowState, WorkflowFeedback] = Objective(
    name="private_project",
    version="1.0",
    submission=ToolActionContract[WorkflowPlan](
        name="submit_workflow", kind="submit", cost=1,
        description="Execute a workflow from initial state; costs 1 plus each operation.",
        action_space=TypedSpace("workflow_plan", WorkflowPlan), return_schema={},
    ),
    evaluator=evaluate_workflow,
    feedback=FeedbackSpec(WorkflowFeedback, workflow_feedback),
)
```

Attach it as `TaskIR[WorkflowState](kernel=WorkflowKernel(), objective=objective, ...)`.
The objective derives the submission's return schema from the feedback type.
`TypedSpace` accepts Pydantic object models and dataclasses, including tuples and
recursive structures, using strict JSON validation. Scalar coercions are rejected.
Agent-facing submissions always enter through this JSON boundary. Direct Python
instances passed by task-author code instead follow Pydantic's validation policy
(including `revalidate_instances`); existing instances are normally trusted. Thus
`canonicalize` and `contains` are not strict validators for untrusted in-memory
objects. This preserves model-specific codecs and validators without imposing a
second serialization policy. JSON submissions retain the model's configured
alias and extra-field rules.

The state type links kernel, runtime, evaluation context, and objective. Candidate,
evidence, and feedback types link the submission and author functions. Run `mypy`
to check these relationships; the package includes a `py.typed` marker. Python
itself does not enforce generic annotations. Tool names, tool arguments, and
heterogeneous observation mappings remain dynamically validated at dispatch.

An evaluator need not execute a plan. It can judge an artifact against private
instance data, as table reconstruction does. If it executes the hidden system,
use `context.run` so work is isolated and charged. Author callbacks are trusted
Python code, not a sandbox; agents only supply validated JSON candidates.

## Programmable feedback

Omit `feedback` for verdict-only grading. With a `FeedbackSpec`, the formatter runs
only for failed evaluations when the episode enables feedback. It can return a
failed constraint, output mismatch, counterexample, or richer diagnostic. Authors
choose the disclosure level in their formatter; no extra policy hierarchy is
required. Successful submissions and disabled feedback return `feedback: null`.

The framework validates the serialized feedback against its generated schema,
including aliases, nested types, and computed fields. Private evidence is never
automatically serialized. A broken formatter produces a structured error without
including its private validation input; consumed budget and terminal decisions are
retained. Choose public
error messages carefully in any author-defined kernel or objective.

For datasets and standard surfaces, add a `TaskEntry` to `TASK_REGISTRY` and have
the generator use that entry's name/version as the instance's `kernel_name` and
`kernel_version`. These existing serialized fields identify the complete task,
so two objectives over the same kernel can coexist under different task entries.

## Migration from 1.1.0

Replace `TaskIR(hypothesis_contract=contract, feedback_model=...)` with
`TaskIR(objective=HypothesisObjective(contract), ...)` for existing identification
tasks. The adapter retains parse, validate, normalize, judge, counterexample, and
distance hooks. The `mealy` task and its `submit_table` interface remain available.
Counterexample and distance hooks now run only after a failed submission with
feedback enabled. Previously, distance ran for every judgment. Return intentionally
unconditional public metadata through `HypothesisJudgment.observation` instead.
If an old task used `FeedbackModel(reveal_counterexample=False)`, keep that
disclosure policy by returning `None` from its contract's `counterexample` hook,
or define an objective without feedback.

`FeedbackModel` and `TaskIR.submit` have been removed; feedback now belongs to the
objective. `HypothesisJudgment` contains `ok` and `observation`; its unused
counterexample, distance, and info fields are removed. Custom objectives should
use private `Evaluation.evidence` and explicit `FeedbackSpec` instead of mixing
private diagnostics into public observations.

The default benchmark reward is now binary correctness for all tasks; historical
efficiency-weighted rewards are not directly comparable. Compare success rates
and report budget usage separately. Palindrome task 1.2 introduces the two-answer
`submit_answer` schema and natural impossible cases; regenerate task splits for
this objective version. Frozen old machine tables retain their transition behavior,
but replaying older palindrome tool traces requires their original runtime.
