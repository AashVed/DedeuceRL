# Generating diverse hidden systems

The Mealy distribution changes in task version **2.3**, and the shared palindrome
distribution is version **1.2**. Both use the same transition sampler. Palindrome
adds its own feasibility constraint and two-answer objective; see
[custom objectives](OBJECTIVES.md).

## Why the A cycle existed

The initial repository commit, `337248a`, explicitly described the Mealy generator
as using a “Backbone on 'A' for reachability” (`dedeucerl/skins/mealy.py`). The kernel
and TaskIR refactors preserved it. It was a convenient construction guaranteeing
reachability, rather than a documented requirement that A provide navigation.

It also forced one third of successor entries, made every state reachable by
repeating a known action, and guaranteed returning to the current state after
exactly `n` As. This is not a reset to state zero. A solver could combine that route
with observed output patterns to locate states and test other transitions.

Renaming A, permuting state labels, or planting a cycle with varying action labels
would retain a structural restriction. Requiring every action to be a permutation
would introduce another restriction: transitions could never merge states.

## The new distribution

For `n` states and `k` actions, generation has four steps:

1. Sample a uniform assignment of the `n*k` transition destinations in which every
   state has at least one incoming edge.
2. Reject unless the graph is strongly connected: every state can reach every
   other state. This preserves recoverability without prescribing a route.
3. Assign each transition an independent uniform output in `0, 1, 2`. Reject the
   **whole machine** if two states are behaviorally indistinguishable.
4. If traps are enabled, shuffle non-start transitions and mark a transition only
   when all states still have safe paths to each other. Stop at the existing target
   `max(1, n//3)`, or when no further safe placement is available. A one-state machine
   has no traps because every transition starts at zero.

The transition/output table before trap decoration is uniform over complete,
strongly connected, minimal **labelled** Mealy machines with those dimensions.
Cycles, self-loops, and merging transitions remain legitimate outcomes. No action
is privileged, and no specific easy topology is banned merely to lower scores.
Trap layouts are not uniformly sampled. The palindrome generator additionally
rejects possible-but-unaffordable goals while retaining naturally impossible goals,
so its final distribution is not the unconditional Mealy distribution. It does not
balance answer classes or resample a machine merely because its goal is impossible.

Initial reachability alone would admit one-way regions that cannot be left during
stateful exploration. Strong connectivity rules them out for this task. Preserving
it after excluding traps is an upgrade: the former trap selector only guaranteed
safe reachability from state zero. Neither property guarantees an agent can discover
safe routes without risk, infer trap-edge behavior safely, or solve within its budget.
The palindrome objective therefore retains its separate feasibility check.

## Exact sampling without a planted backbone

Naively choosing all destinations independently and rejecting disconnected graphs
becomes inefficient as the state count grows. Every strongly connected graph must
already have positive indegree at every state, so excluding zero-indegree graphs
before the connectivity check removes no eligible topology.

The implementation counts partitions of transition slots into nonempty destination
groups using `S(m,j) = j*S(m-1,j) + S(m-1,j-1)`. It samples a partition by those
integer counts, then uniformly assigns state labels to the groups. Every mapping
onto the state set has the same probability. Rejecting non-connected graphs and
then nonminimal graph/output pairs preserves conditional uniformity. Tests enumerate
all random branches on small cases and check each mapping's exact probability.

Set partitions and explicit sampling distributions are established tools in random
automata generation; see [Bassino and Nicaud (2007)](https://doi.org/10.1016/j.tcs.2007.04.001)
and [Nicaud's survey (2014)](https://www-igm.univ-mlv.fr/~nicaud/articles/mfcs14.pdf).
This implementation uses a small exact-count sampler followed by connectivity
rejection; it does not implement their accessible-automaton Boltzmann sampler or
claim its asymptotic complexity. The contribution here is the choice of explicit
benchmark invariants and distribution, integrated with objectives and validation.

The count table has `O(k*n²)` integer entries, with growing integer sizes. It is
local to a generation call and is not globally cached. Validation covers up to
256 states; the exact-count table uses approximately 51 MiB at that size for three
actions, versus about 6 MiB at 128. Very large graph generation would need a different
sampling implementation, not a silent switch to a differently biased distribution.

## Difficulty has several dimensions

Removing A's navigation shortcut broadens topology support. It does not establish
that an objective is hard or that uniform random machines resemble real services.
The research audit records the following separately:

| Axis | Measurement or guarantee |
|---|---|
| Topology | Per-action full cycles, merging transitions, self-loops, and repeated-action coverage |
| Access and recovery | Shortest access depth, directed diameter, full and safe strong connectivity |
| Observational ambiguity | Shortest distinguishing-word lengths for state pairs; one-step output collisions |
| Diversity | Duplicate detection after canonicalizing state labels, action symmetry, and seed cohorts |
| Objective difficulty | Required palindrome length/variety, actual costs, safe witness feasibility, and model strategies |
| Evaluation conditions | Fixed feedback, budgets, traps, model effort, host tools, and explicit time limits |

Independent random output labels still make many states distinguishable with short
experiments. The new distribution does not fix that axis merely by changing topology.
State identification and reset costs are distinct concerns in automata learning;
[Frohme's ADT work](https://arxiv.org/abs/1902.01139) studies that distinction.
Any later delayed-observation or bottleneck family should have its own measured,
versioned distribution, rather than being hidden behind an ambiguous difficulty knob.

## Other objectives and skins

`TaskGeneratorSpec` already lets each task own its distribution. No runtime plugin
system or universal domain generator is needed. `sample_strongly_connected_transitions`
and `is_strongly_connected` are optional pure graph utilities; tests also exercise
tuple-valued API actions. Mealy identification and palindrome share their actual
machine sampler, and palindrome forwards the shared generator's parameters.

Historical Protocol generation planted a chain with varying API actions, so it had
a different structural prior. APIEnv generated meaningful workflow profiles;
ExprPolicy generated typed expressions and test cases. Those domains should validate
their own observable behavior, feasibility, and diversity. They should not inherit
Mealy's connectivity requirements when irreversible workflows are intentional.

The unused `create_reachability_backbone` and `apply_backbone` exports have been
removed. They embedded the first-action cycle and had no repository callers. For
domain-specific construction, define the intended graph directly and validate its
properties; for this uniform finite-state family, use the new sampler.

## Reproducibility and compatibility

Seeds alone do not identify a machine across generator changes. Existing dataset
files contain full instances and keep their original task versions and transition
behavior. Objective or scoring changes can still require the original runtime for
old trace replay; palindrome 1.2 changes both its answer schema and sampling rule.
MCP artifacts report the served instance's version, including when an older dataset
is loaded by the newer runtime. Kernel transition semantics retain their version.

The original model-study instances are now frozen alongside their traces. Its auditor
replays those snapshots instead of regenerating them with a different sampler.
Generating the same seed with Mealy 2.3 intentionally creates a different machine.
Record the task version and frozen instance or split hash for comparisons.

The scripts under `research/generator_v2_3/` provide sampling measurements and
model-study reproduction commands through `--help`. Write their generated evidence
under `.dedeucerl/` or outside the checkout; keep benchmark results out of the
library source tree.
