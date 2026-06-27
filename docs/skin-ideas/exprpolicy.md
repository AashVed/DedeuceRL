# ExprPolicy Kernel Idea

Goal: rebuild the former ExprPolicy skin as a DSL debugging kernel rather than a
transition-system kernel.

## Task Shape

- Hidden task: repair a typed policy expression.
- Diagnostic operations: `type_check(expr)` and `run_tests(expr, suite)`.
- Submission operation: `submit(expr)`.
- Observation: compiler diagnostics, public test results, hidden-test judgment.
- Correctness: submitted expression passes hidden cases and respects DSL limits.

## Kernel Requirements

- Keep parser, type checker, evaluator, and test generation pure.
- Model tool costs explicitly; public tests can cost more than type checks.
- Support traps via banned tokens or unsafe expression forms.
- Enforce expression length and DSL grammar inside kernel validation.
- Return useful diagnostics without leaking hidden tests.

## Surface Expectations

- Prompt guidance should emphasize compiler/test feedback loops.
- Runtime must support non-transition diagnostic tools, which is why kernels use
  generic `call()` rather than only `step()`.
- Counterexamples should expose public or allowed diagnostic cases only.
