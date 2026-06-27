# Protocol Kernel Idea

Goal: rebuild the former Protocol skin as a pure kernel for stateful REST-style
API reverse engineering.

## Task Shape

- Hidden system: finite API state machine.
- Probe operation: `api_call(method, endpoint)`.
- Observation: HTTP-like status code and lightweight response body/schema tag.
- Submission: state-dependent transition spec.
- Correctness: behavioral equivalence from the start state, not exact state labels.

## Kernel Requirements

- Keep API semantics pure; no prompt, Verifiers, provider, or dataset imports.
- Represent actions as finite `(method, endpoint)` operations.
- Support traps for forbidden state/action pairs.
- Generate reachable systems with trap-free solvability when traps are enabled.
- Return counterexamples as distinguishing API call traces.

## Surface Expectations

- Prompt guidance should emphasize stateful behavior: the same call can produce
  different responses depending on hidden state.
- Tool contracts should expose method and endpoint enums.
- Submission schema should be compiled from the hypothesis contract.
