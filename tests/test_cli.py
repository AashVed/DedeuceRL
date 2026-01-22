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

    def test_aggregate_csv_output(self, tmp_path):
        """Test CSV output format."""
        # Create test results
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
            {
                "model": "test:model",
                "ok": False,
                "trap_hit": True,
                "queries_used": 15,
                "reward": 0.0,
            },
        ]
        with open(results_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        result = subprocess.run(
            [sys.executable, "-m", "dedeucerl.cli.aggregate", str(results_file), "--format", "csv"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "model,n_episodes,success_rate" in result.stdout
        assert "test:model" in result.stdout

    def test_aggregate_json_output(self, tmp_path):
        """Test JSON output format."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
        ]
        with open(results_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

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

    def test_aggregate_markdown_output(self, tmp_path):
        """Test Markdown output format."""
        results_file = tmp_path / "results.jsonl"
        results = [
            {
                "model": "test:model",
                "ok": True,
                "trap_hit": False,
                "queries_used": 10,
                "reward": 0.9,
            },
        ]
        with open(results_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

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
        assert "| Model |" in result.stdout
        assert "test:model" in result.stdout


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
