"""Integration tests for full skin workflows."""

import json
import pytest
from datasets import Dataset
from verifiers.types import State

from dedeucerl.skins import MealyEnv, ProtocolEnv, SKIN_REGISTRY
from dedeucerl.core import TaskGenerator, make_rubric


class TestSkinIntegration:
    """Integration tests verifying full workflow for each skin."""

    @pytest.fixture
    def rubric(self):
        """Create a standard rubric."""
        return make_rubric()

    def test_mealy_full_workflow(self, rubric, tmp_path):
        """Test MealyEnv: generate -> build dataset -> tool calls."""
        # Generate
        gen = TaskGenerator(MealyEnv)
        split = gen.generate_split(seeds=[42], budget=10, subset_name="test", n_states=2, trap=True)
        split_path = tmp_path / "mealy.json"
        gen.save_split(split, str(split_path))

        # Build dataset
        dataset = gen.build_dataset(str(split_path), "test", feedback=True)
        assert len(dataset) == 1

        # Verify prompt structure
        prompt = dataset[0]["prompt"]
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert "Mealy" in prompt[0]["content"]
        assert "act" in prompt[0]["content"]
        assert "submit_table" in prompt[0]["content"]

        # Verify answer structure
        answer = json.loads(dataset[0]["answer"])
        assert "table" in answer
        assert "trap_pairs" in answer
        assert answer["table"]["n"] == 2

        # Test tool calls
        env = MealyEnv(dataset=dataset, rubric=rubric, feedback=True, max_turns=20)
        env._configure_from_metadata(answer)
        env._state_ref = State(
            {
                "cs": 0,
                "budget": 10,
                "budget_init": 10,
                "queries_used": 0,
                "steps": 0,
                "trap_hit": False,
                "ok": False,
                "done": False,
            }
        )

        # act() should return valid JSON with expected fields
        result = json.loads(env.act("A"))
        assert "out" in result
        assert "budget_left" in result
        assert result["budget_left"] == 9

        # Invalid symbol should error
        result = json.loads(env.act("X"))
        assert "error" in result

    def test_protocol_full_workflow(self, rubric, tmp_path):
        """Test ProtocolEnv: generate -> build dataset -> tool calls."""
        # Generate
        gen = TaskGenerator(ProtocolEnv)
        split = gen.generate_split(
            seeds=[42], budget=10, subset_name="test", n_endpoints=2, n_states=2
        )
        split_path = tmp_path / "protocol.json"
        gen.save_split(split, str(split_path))

        # Build dataset
        dataset = gen.build_dataset(str(split_path), "test", feedback=True)
        assert len(dataset) == 1

        # Verify prompt structure
        prompt = dataset[0]["prompt"]
        assert "REST API" in prompt[0]["content"] or "API" in prompt[0]["content"]
        assert "api_call" in prompt[0]["content"]
        assert "submit_spec" in prompt[0]["content"]

        # Verify answer structure (now uses transitions, not endpoints)
        answer = json.loads(dataset[0]["answer"])
        assert "spec" in answer
        assert "trap_calls" in answer
        assert answer["spec"]["n_states"] == 2
        assert "transitions" in answer["spec"]

        # Test tool calls
        env = ProtocolEnv(dataset=dataset, rubric=rubric, feedback=True, max_turns=20)
        env._configure_from_metadata(answer)
        env._state_ref = State(
            {
                "cs": 0,
                "budget": 10,
                "budget_init": 10,
                "queries_used": 0,
                "steps": 0,
                "trap_hit": False,
                "ok": False,
                "done": False,
            }
        )

        # api_call() to valid endpoint (extract from state 0 transitions)
        state_0_trans = answer["spec"]["transitions"]["0"]
        endpoints = list(state_0_trans.keys())
        ep = endpoints[0]
        method = list(state_0_trans[ep].keys())[0]
        result = json.loads(env.api_call(method, ep))
        assert "status" in result
        assert "budget_left" in result

        # api_call() to invalid endpoint
        result = json.loads(env.api_call("GET", "/nonexistent"))
        assert result["status"] == 404

    def test_budget_exhaustion_ends_episode(self, rubric):
        """Budget exhaustion should hard-stop an episode."""
        # Mealy: budget=0 should return budget exhausted and end episode
        table = MealyEnv.generate_system_static(seed=0, n_states=2, trap=False)["table"]
        answer = json.dumps({"table": table, "trap_pairs": [], "budget": 0})
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "test"}]],
                "answer": [answer],
            }
        )
        env = MealyEnv(dataset=dataset, rubric=rubric, feedback=False, max_turns=10)
        env._configure_from_metadata(json.loads(answer))
        env._state_ref = State(
            {
                "cs": 0,
                "budget": 0,
                "budget_init": 0,
                "queries_used": 0,
                "steps": 0,
                "trap_hit": False,
                "ok": False,
                "done": False,
            }
        )
        out = json.loads(env.act("A"))
        assert out.get("error", {}).get("code") == "E002"
        assert env._state_ref["done"] is True

        # Mealy: consuming last query sets done=True
        env._state_ref = State(
            {
                "cs": 0,
                "budget": 1,
                "budget_init": 1,
                "queries_used": 0,
                "steps": 0,
                "trap_hit": False,
                "ok": False,
                "done": False,
            }
        )
        out = json.loads(env.act("A"))
        assert out.get("budget_left") == 0
        assert env._state_ref["done"] is True

    def test_all_skins_have_unique_prompts(self, tmp_path):
        """Verify each skin generates distinct prompts."""
        prompts = {}

        for skin_name, skin_cls in SKIN_REGISTRY.items():
            gen = TaskGenerator(skin_cls)

            # Use skin-specific kwargs
            kwargs = {"seeds": [0], "budget": 10, "subset_name": "test"}
            if skin_name == "mealy":
                kwargs["n_states"] = 2
            elif skin_name == "protocol":
                kwargs["n_endpoints"] = 2
                kwargs["n_states"] = 2

            split = gen.generate_split(**kwargs)
            split_path = tmp_path / f"{skin_name}.json"
            gen.save_split(split, str(split_path))
            dataset = gen.build_dataset(str(split_path), "test", feedback=True)

            prompts[skin_name] = dataset[0]["prompt"][0]["content"]

        # Verify prompts are distinct
        prompt_set = set(prompts.values())
        assert len(prompt_set) == len(SKIN_REGISTRY), "Each skin should have a unique system prompt"

    def test_all_skins_have_tools(self):
        """Verify each skin defines tools via _get_tools()."""
        rubric = make_rubric()

        for skin_name, skin_cls in SKIN_REGISTRY.items():
            # Create minimal dataset
            dataset = Dataset.from_dict(
                {
                    "prompt": [[{"role": "user", "content": "test"}]],
                    "answer": ["{}"],
                }
            )
            env = skin_cls(dataset=dataset, rubric=rubric, max_turns=10)

            tools = env._get_tools()
            assert len(tools) >= 2, f"{skin_name} should have at least 2 tools (probe + submit)"

            # Verify tools are callable
            for tool in tools:
                assert callable(tool), f"{skin_name} tool {tool} should be callable"
