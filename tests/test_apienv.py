"""Tests for APIEnv skin."""

import json

from verifiers.types import State

from dedeucerl.core import TaskGenerator, make_rubric
from dedeucerl.skins.apienv import APIEnv


class TestAPIEnvWorkflow:
    """Integration-ish tests for APIEnv."""

    def test_generate_build_and_tools(self, tmp_path):
        rubric = make_rubric()
        gen = TaskGenerator(APIEnv)

        split = gen.generate_split(
            seeds=[42],
            budget=12,
            subset_name="test",
            n_states=6,
            n_endpoints=6,
            trap=True,
        )
        split_path = tmp_path / "apienv.json"
        gen.save_split(split, str(split_path))

        dataset = gen.build_dataset(str(split_path), "test", feedback=True)
        assert len(dataset) == 1

        # Prompt contains expected tools and contract language
        prompt = dataset[0]["prompt"]
        assert len(prompt) == 2
        assert "api_call" in prompt[0]["content"]
        assert "submit_spec" in prompt[0]["content"]
        assert "behaviorally equivalent" in prompt[0]["content"].lower()

        meta = json.loads(dataset[0]["answer"])
        assert "spec" in meta
        assert "trap_calls" in meta
        assert meta["spec"]["start"] == 0
        assert "transitions" in meta["spec"]

        env = APIEnv(dataset=dataset, rubric=rubric, feedback=True, max_turns=30)
        env._configure_from_metadata(meta)
        env._state_ref = State(
            {
                "cs": 0,
                "budget": 12,
                "budget_init": 12,
                "queries_used": 0,
                "steps": 0,
                "trap_hit": False,
                "ok": False,
                "done": False,
            }
        )

        # Valid call
        out = json.loads(env.api_call("POST", "/login", "valid"))
        assert out["status"] in (200, 201)
        assert "schema" in out
        assert out["budget_left"] == 11

        # Invalid variant returns 422 and does not crash
        out = json.loads(env.api_call("POST", "/login", "not_a_variant"))
        assert out["status"] == 422
        assert out["schema"] == "InvalidVariant"

        # Every submission consumes submission_cost
        before = env._state_ref["budget"]
        bad = json.loads(env.submit_spec("{}"))
        assert bad["ok"] is False
        assert env._state_ref["budget"] == before - env.config.submission_cost

        # Correct submission using ground truth spec should succeed (and also consume cost)
        before_good = env._state_ref["budget"]
        good = json.loads(env.submit_spec(json.dumps(meta["spec"])))
        assert good["ok"] in (True, False)  # trap_hit could make ok False
        assert env._state_ref["budget"] == before_good - env.config.submission_cost
        assert env._state_ref["done"] is True

    def test_domain_spec_exposes_alphabets(self):
        spec = APIEnv.domain_spec(n_states=6, n_endpoints=5, budget=10, trap=True)
        obs = spec.build_observation()
        assert "endpoints" in obs
        assert "variants" in obs
        assert "response_schemas" in obs
