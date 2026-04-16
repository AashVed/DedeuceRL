"""CLI integration tests."""

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestSelfcheck:
    """Tests for dedeucerl-selfcheck CLI."""

    def test_selfcheck_runs(self):
        """Test that selfcheck completes successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.selfcheck"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "All checks passed" in result.stdout


class TestAggregate:
    """Tests for dedeucerl-aggregate CLI."""

    @staticmethod
    def _write_results(path, results):
        with open(path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    def test_aggregate_csv_output(self, tmp_path):
        """Test CSV output format."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 1,
                "rollout": 0,
                "ok": False,
                "trap_hit": True,
                "queries_used": 15,
                "reward": 0.0,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.aggregate", str(results_file), "--format", "csv"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert (
            "model,skin,split_hash,eval_config_hash,n_runs,n_episodes,success_rate,trap_rate,"
            "avg_queries,avg_reward,max_complete_k,n_episode_success_at_1,pass_at_1,"
            "n_episode_success_at_3,pass_at_3"
            in result.stdout
        )
        assert "test:model,mealy,hash-a,cfg-a,2,2,0.5000,0.5000,12.50,0.4500,1,1,0.5000,," in result.stdout

    def test_aggregate_json_output(self, tmp_path):
        """Test JSON output format."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dedeucerl.cli.aggregate",
                str(results_file),
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["model"] == "test:model"
        assert data[0]["skin"] == "mealy"
        assert data[0]["split_hash"] == "hash-a"
        assert data[0]["eval_config_hash"] == "cfg-a"
        assert data[0]["n_runs"] == 1
        assert data[0]["n_episodes"] == 1
        assert data[0]["max_complete_k"] == 1
        assert data[0]["pass_at_1"] == 1.0
        assert data[0]["pass_at_3"] is None

    def test_aggregate_markdown_output(self, tmp_path):
        """Test Markdown output format."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dedeucerl.cli.aggregate",
                str(results_file),
                "--format",
                "markdown",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "| Model | Skin | Split Hash | Eval Config | Runs | Episodes |" in result.stdout
        assert "test:model" in result.stdout

    def test_aggregate_splits_mixed_groups(self, tmp_path):
        """Same model across skins or splits should produce separate groups."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "skin": "apienv",
                "split_hash": "hash-b",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": False,
                "trap_hit": True,
                "queries_used": 5,
                "reward": 0.0,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.aggregate", str(results_file), "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0

        data = json.loads(result.stdout)
        assert len(data) == 2
        assert {(row["model"], row["skin"], row["split_hash"], row["eval_config_hash"]) for row in data} == {
            ("test:model", "mealy", "hash-a", "cfg-a"),
            ("test:model", "apienv", "hash-b", "cfg-a"),
        }

    def test_aggregate_multi_rollout_counts_runs_and_unique_episodes(self, tmp_path):
        """Multiple rollouts of one episode should count as multiple runs, one episode."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 3,
                "rollout": 0,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 3,
                "rollout": 1,
                "ok": False,
                "trap_hit": False,
                "queries_used": 20,
                "reward": 0.0,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.aggregate", str(results_file), "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0

        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["n_runs"] == 2
        assert data[0]["n_episodes"] == 1
        assert data[0]["success_rate"] == 0.5
        assert data[0]["max_complete_k"] == 2
        assert data[0]["pass_at_1"] == 1.0
        assert data[0]["pass_at_3"] is None

    def test_aggregate_groups_by_eval_config_hash(self, tmp_path):
        """Distinct eval configs should not collapse into one aggregate row."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-b",
                "episode_idx": 0,
                "rollout": 0,
                "ok": False,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.0,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.aggregate", str(results_file), "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 2
        assert {row["eval_config_hash"] for row in data} == {"cfg-a", "cfg-b"}

    def test_aggregate_computes_pass_at_k(self, tmp_path):
        """Pass@k should be derived from rollout prefixes per episode."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": False,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.0,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 1,
                "ok": True,
                "trap_hit": False,
                "queries_used": 11,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 2,
                "ok": True,
                "trap_hit": False,
                "queries_used": 12,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 1,
                "rollout": 0,
                "ok": False,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.0,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 1,
                "rollout": 1,
                "ok": False,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.0,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 1,
                "rollout": 2,
                "ok": False,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.0,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.aggregate", str(results_file), "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["max_complete_k"] == 3
        assert data[0]["n_episode_success_at_1"] == 0
        assert data[0]["pass_at_1"] == 0.0
        assert data[0]["n_episode_success_at_3"] == 1
        assert data[0]["pass_at_3"] == 0.5

    def test_aggregate_rejects_duplicate_rollout_rows(self, tmp_path):
        """Duplicate (episode_idx, rollout) rows should fail clearly."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": False,
                "trap_hit": False,
                "queries_used": 11,
                "reward": 0.0,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.aggregate", str(results_file), "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode != 0
        assert "Duplicate result row" in result.stderr

    def test_aggregate_incomplete_rollouts_report_max_complete_k(self, tmp_path):
        """Pass@k should only be reported up to the shared rollout depth."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 0,
                "ok": False,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.0,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 1,
                "ok": False,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.0,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 0,
                "rollout": 2,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 1,
                "rollout": 0,
                "ok": False,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.0,
            },
            {
                "model": "test:model",
                "skin": "mealy",
                "split_hash": "hash-a",
                "eval_config_hash": "cfg-a",
                "episode_idx": 1,
                "rollout": 1,
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
        ]
        self._write_results(results_file, results)

        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.aggregate", str(results_file), "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["max_complete_k"] == 2
        assert data[0]["pass_at_3"] is None


class TestSkinRegistry:
    """Tests for skin registry."""

    def test_all_skins_registered(self):
        """Test that all skins are in the registry."""
        from dedeucerl.skins import SKIN_REGISTRY

        assert "mealy" in SKIN_REGISTRY
        assert "protocol" in SKIN_REGISTRY
        assert "apienv" in SKIN_REGISTRY
        assert "exprpolicy" in SKIN_REGISTRY

    def test_skins_have_required_methods(self):
        """Test that all skins implement required methods."""
        from dedeucerl.skins import SKIN_REGISTRY

        for name, skin_cls in SKIN_REGISTRY.items():
            # Check static method
            assert hasattr(skin_cls, "generate_system_static")
            assert callable(getattr(skin_cls, "generate_system_static"))

            # Check class method
            assert hasattr(skin_cls, "get_prompt_template")
            assert callable(getattr(skin_cls, "get_prompt_template"))

            # New v0 contract: schema-first domain specification
            assert hasattr(skin_cls, "domain_spec")
            assert callable(getattr(skin_cls, "domain_spec"))
