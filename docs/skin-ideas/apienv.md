# APIEnv Kernel Idea

Goal: rebuild the former APIEnv skin as a practitioner-oriented SaaS workflow
kernel.

## Task Shape

- Hidden system: stateful SaaS workflow such as login, verification, org
  selection, plan upgrade, projects, and organization deletion.
- Probe operation: `api_call(method, endpoint, variant)`.
- Observation: status code plus coarse response schema tag.
- Submission: behaviorally equivalent API state machine.
- Correctness: output-equivalence over all operation sequences from the start
  state.

## Kernel Requirements

- Use finite endpoint/method/variant catalogs as action spaces.
- Keep response schema tags small and inspectable.
- Generate interpretable latent workflow profiles while exposing only public
  API behavior.
- Support trap operations for unsafe workflows.
- Keep minimized equivalent submissions acceptable.

## Surface Expectations

- Prompt guidance should read like black-box API QA, not automata theory.
- Tool contracts should expose endpoint, method, and variant enums.
- Counterexamples should be API call traces with true status/schema outputs.
