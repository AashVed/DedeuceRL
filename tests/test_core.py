"""Test suite for DedeuceRL core."""

from dedeucerl.core import (
    SkinConfig,
    EpisodeState,
    ProbeResult,
    SubmitResult,
    TaskGenerator,
    make_rubric,
)


class TestSkinConfig:
    """Tests for SkinConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SkinConfig()
        assert config.isomorphism_check is True
        assert config.trap_enabled is True
        assert config.default_budget == 25
        assert config.max_turns == 64
        assert config.skin_name == "unknown"

    def test_custom_values(self):
        """Test custom configuration."""
        config = SkinConfig(
            isomorphism_check=False,
            default_budget=50,
            skin_name="test",
        )
        assert config.isomorphism_check is False
        assert config.default_budget == 50
        assert config.skin_name == "test"


class TestEpisodeState:
    """Tests for EpisodeState dataclass."""

    def test_initialization(self):
        """Test default initialization."""
        state = EpisodeState(current_state=0, budget=25, budget_init=25)
        assert state.queries_used == 0
        assert state.steps == 0
        assert state.trap_hit is False
        assert state.ok is False
        assert state.done is False

    def test_mutable_fields(self):
        """Test mutable field updates."""
        state = EpisodeState(current_state=0, budget=25, budget_init=25)
        state.queries_used = 5
        state.trap_hit = True
        assert state.queries_used == 5
        assert state.trap_hit is True


class TestProbeResult:
    """Tests for ProbeResult dataclass."""

    def test_basic_result(self):
        """Test basic probe result."""
        result = ProbeResult(
            observation=1,
            budget_remaining=24,
            step=1,
            trap_hit=False,
        )
        assert result.observation == 1
        assert result.budget_remaining == 24
        assert result.metadata == {}


class TestSubmitResult:
    """Tests for SubmitResult dataclass."""

    def test_correct_submission(self):
        """Test correct submission result."""
        result = SubmitResult(
            correct=True,
            budget_remaining=20,
            queries_used=5,
            trap_hit=False,
        )
        assert result.correct is True
        assert result.counterexample is None

    def test_incorrect_with_counterexample(self):
        """Test incorrect submission with counterexample."""
        cex = [{"in": "A", "out": 0}]
        result = SubmitResult(
            correct=False,
            budget_remaining=19,
            queries_used=6,
            trap_hit=False,
            counterexample=cex,
        )
        assert result.correct is False
        assert result.counterexample == cex


class TestMakeRubric:
    """Tests for rubric creation."""

    def test_rubric_creation(self):
        """Test that make_rubric returns a valid rubric."""
        rubric = make_rubric()
        assert rubric is not None
        assert len(rubric.funcs) == 5
        assert len(rubric.weights) == 5


class TestTaskGeneratorKwargsForwarding:
    """Tests that TaskGenerator forwards domain_spec kwargs generically."""

    def test_domain_spec_receives_custom_param(self):
        from dedeucerl.core.domain_spec import DomainSpec, ObservationField

        class DummySkin:
            # Minimal skin-like contract for TaskGenerator
            class config:
                default_budget = 25

            @staticmethod
            def generate_system_static(seed: int, **kwargs):
                return {}

            @classmethod
            def get_prompt_template(cls, obs, *, feedback: bool = False):
                return []

            @classmethod
            def domain_spec(cls, *, budget: int = 25, trap: bool = True, foo: int = 0):
                return DomainSpec(
                    actions=["A"],
                    outputs=[0],
                    tool_schemas=[],
                    hypothesis_schema={},
                    observation_fields={
                        "budget": ObservationField("int", "Query budget", budget),
                        "trap": ObservationField("bool", "Whether traps exist", trap),
                        "foo": ObservationField("int", "Custom param", foo),
                    },
                    skin_name="dummy",
                    n_states=1,
                    has_traps=trap,
                )

        gen = TaskGenerator(DummySkin)
        cfg = {"budget": 10, "trap": False, "foo": 7, "items": []}
        answer_data = {}  # Minimal answer payload
        obs = gen._build_observation_from_answer(answer_data, cfg)
        assert obs["foo"] == 7
        assert obs["budget"] == 10
        assert obs["trap"] is False
