from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from fractions import Fraction

import pytest

from dedeucerl.core.automata import (
    _partition_counts,
    _sample_surjection,
    is_minimal,
    is_strongly_connected,
    sample_strongly_connected_transitions,
)
from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.ir.palindrome import palindrome_witness
from dedeucerl.kernel.mealy import ALPHABET, generate_mealy_system
from dedeucerl.runtime import EpisodeRuntime


class NeedChoice(Exception):
    def __init__(self, stop):
        self.stop = stop


class EnumeratedRandom:
    """Enumerate RNG branches with their exact probabilities, without seed statistics."""

    def __init__(self, choices):
        self.choices = iter(choices)

    def randrange(self, stop):
        try:
            choice = next(self.choices)
        except StopIteration:
            raise NeedChoice(stop) from None
        assert 0 <= choice < stop
        return choice

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            j = self.randrange(i + 1)
            items[i], items[j] = items[j], items[i]


@pytest.mark.parametrize("slots,groups", [(3, 1), (4, 2), (4, 3)])
def test_surjection_sampler_is_exactly_uniform(slots, groups):
    counts = _partition_counts(slots, groups)
    probabilities = defaultdict(Fraction)
    pending = [((), Fraction(1))]
    while pending:
        choices, probability = pending.pop()
        try:
            targets = _sample_surjection(counts, groups, EnumeratedRandom(choices))
        except NeedChoice as branch:
            pending.extend(
                (choices + (choice,), probability / branch.stop) for choice in range(branch.stop)
            )
        else:
            probabilities[tuple(targets)] += probability
    expected = {
        targets
        for targets in itertools.product(range(groups), repeat=slots)
        if len(set(targets)) == groups
    }
    assert probabilities.keys() == expected
    assert set(probabilities.values()) == {Fraction(1, len(expected))}


def test_strong_connectivity_requires_a_return_path_and_respects_traps():
    graph = {0: {"go": 1, "wait": 0}, 1: {"go": 1, "wait": 0}}

    def next_state(s, a):
        return graph[s][a]

    assert is_strongly_connected(2, ["go", "wait"], next_state)
    # Still reachable from zero, but no safe return from one.
    assert not is_strongly_connected(2, ["go", "wait"], next_state, {(1, "wait")})
    assert is_strongly_connected(2, ["go", "wait"], next_state, {(1, "go")})


def _reachable(table, start, blocked):
    seen, pending = {start}, [start]
    for state in pending:
        for action, (target, _) in table["trans"][str(state)].items():
            if (state, action) not in blocked and target not in seen:
                pending.append(target)
                seen.add(target)
    return seen


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64])
def test_generated_machines_are_minimal_and_safely_recoverable(n):
    for seed in range(20):
        machine = generate_mealy_system(seed, n, trap=True)
        assert machine == generate_mealy_system(seed, n, trap=True)
        table = machine["table"]
        assert table == generate_mealy_system(seed, n, trap=False)["table"]
        traps = {tuple(pair) for pair in machine["trap_pairs"]}
        assert len(traps) == (0 if n == 1 else max(1, n // 3))
        assert all(state != 0 for state, _ in traps)
        if n == 1:
            assert traps == set()
        assert is_minimal(n, ALPHABET, lambda s, a: tuple(table["trans"][str(s)][a]))
        for start in range(n):
            assert _reachable(table, start, traps) == set(range(n))


def test_sampling_is_equivariant_under_action_renaming_and_supports_merging():
    actions = [("GET", "/status"), ("POST", "/verify"), ("PUT", "/profile")]
    for seed in range(20):
        graph = sample_strongly_connected_transitions(8, ALPHABET, random.Random(seed))
        renamed = sample_strongly_connected_transitions(8, actions, random.Random(seed))
        for state in range(8):
            assert list(graph[state].values()) == list(renamed[state].values())
    # Cohort checks prevent reintroducing a designated permutation/cycle or
    # removing legitimate self-loops and merging actions in the name of hardness.
    graphs = [
        sample_strongly_connected_transitions(8, ALPHABET, random.Random(s)) for s in range(30)
    ]
    for action in ALPHABET:
        assert any(len({g[s][action] for s in g}) < 8 for g in graphs)
        assert any(g[s][action] == s for g in graphs for s in g)


@pytest.mark.parametrize("actions", [[], ["A", "A"]])
def test_sampler_rejects_invalid_alphabets(actions):
    with pytest.raises(ValueError):
        sample_strongly_connected_transitions(3, actions, random.Random(0))


def test_new_palindrome_instances_keep_affordable_correct_answers():
    ir = TASK_REGISTRY["mealy_palindrome"].ir
    for seed in range(12):
        instance = ir.generator.sample(
            seed=seed, budget=80, n_states=8, trap=True, min_length=12, min_distinct=3
        )
        assert instance.kernel_version == ir.version == "1.2"
        witness = palindrome_witness(instance)
        assert witness is None or len(witness) + 1 <= instance.budget
        answer = (
            {"kind": "impossible"} if witness is None else {"kind": "sequence", "actions": witness}
        )
        runtime = EpisodeRuntime(ir, instance)
        result = runtime.call_tool("submit_answer", {"answer": answer})
        assert result.output["ok"] and not runtime.trap_hit
        assert (
            runtime.queries_used
            == (len(witness) if witness is not None else 0) + 1
            <= instance.budget
        )


def test_new_mealy_instances_still_accept_the_true_table():
    ir = TASK_REGISTRY["mealy"].ir
    instance = ir.generator.sample(seed=91, budget=2, n_states=12, trap=True)
    assert instance.kernel_version == ir.version == "2.3"
    result = EpisodeRuntime(ir, instance).call_tool(
        "submit_table", {"table_json": json.dumps(instance.private["table"])}
    )
    assert result.output["ok"]
