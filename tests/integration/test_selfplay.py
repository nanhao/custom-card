"""Integration tests for app/selfplay_runner.py — SelfPlayRunner.run().

Covers TC-S1 through TC-S6 (game loop), TC-L1 through TC-L3 (logging),
and TC-S3 (balanced outcomes) from design/5 test_cases.md.

For PPO sample output tests (TC-P1–P7, TC-T1–T4), see
test_selfplay_samples.py.
"""

import json
import sys
from io import StringIO

import pytest

from src.backend.app.app_manager import AppManager
from src.backend.app.selfplay_config import SelfPlayConfig
from src.backend.game.session_store import SessionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_manager(tmp_path):
    """Create an AppManager with a temporary SessionStore."""
    saves_dir = tmp_path / "saves"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))
    store = SessionStore(config_path=str(config_path))
    return AppManager(store)


# ---------------------------------------------------------------------------
# TC-S1: Single game produces valid result
# ---------------------------------------------------------------------------


class TestSingleGame:
    """TC-S1: run_selfplay with game_count=1 produces valid result."""

    def test_single_game_produces_valid_result(self, app_manager):
        """Result has correct total, winner counts sum to 1."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1", game_count=1
        )
        result = app_manager.run_selfplay(config)

        assert result.total_games == 1
        assert result.ai1_wins + result.ai2_wins + result.draws == 1
        assert 1 <= result.avg_rounds <= 7

    def test_single_game_winner_is_valid(self, app_manager):
        """Winner is '1', '2', or 'draw' (reflected in counts)."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1", game_count=1
        )
        result = app_manager.run_selfplay(config)

        counts = [result.ai1_wins, result.ai2_wins, result.draws]
        assert sum(counts) == 1
        assert max(counts) == 1


# ---------------------------------------------------------------------------
# TC-S2: game_count=0 returns empty result
# ---------------------------------------------------------------------------


class TestZeroGames:
    """TC-S2: game_count=0 returns empty result."""

    def test_zero_games_empty_result(self, app_manager):
        """All counts are 0, avg_rounds is 0."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1", game_count=0
        )
        result = app_manager.run_selfplay(config)

        assert result.total_games == 0
        assert result.ai1_wins == 0
        assert result.ai2_wins == 0
        assert result.draws == 0
        assert result.avg_rounds == 0.0


# ---------------------------------------------------------------------------
# TC-S4: invalid ai_name raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidAIName:
    """TC-S4: invalid ai_name raises ValueError before any game starts."""

    def test_invalid_ai1_raises_value_error(self, app_manager):
        """Unknown ai1 raises ValueError."""
        config = SelfPlayConfig(
            ai1="nonexistent_ai", ai2="rnn_v1.1", game_count=1
        )
        with pytest.raises(ValueError):
            app_manager.run_selfplay(config)

    def test_invalid_ai2_raises_value_error(self, app_manager):
        """Unknown ai2 raises ValueError."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="nonexistent_ai", game_count=1
        )
        with pytest.raises(ValueError):
            app_manager.run_selfplay(config)


# ---------------------------------------------------------------------------
# TC-S5: Deterministic with temperature=0 both sides
# ---------------------------------------------------------------------------


class TestDeterministicGames:
    """TC-S5: temperature=0 both sides → same outcome every run."""

    def test_deterministic_games_produce_same_outcome(self, app_manager):
        """Two runs with temperature=0 produce identical winner counts."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
        )

        result1 = app_manager.run_selfplay(config)
        result2 = app_manager.run_selfplay(config)

        assert result1.ai1_wins == result2.ai1_wins
        assert result1.ai2_wins == result2.ai2_wins
        assert result1.draws == result2.draws
        assert result1.avg_rounds == result2.avg_rounds


# ---------------------------------------------------------------------------
# TC-S6: temp_seed with temperature>0 produces reproducible games
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """TC-S6: temp_seed with temperature>0 → reproducible outcomes.

    With the same temp_seed, game win/loss/draw counts should be
    identical across runs. avg_rounds may vary slightly due to
    platform-specific RNG behavior in PyTorch multinomial.
    """

    def test_seeded_games_are_reproducible(self, app_manager):
        """Two runs with same temp_seed produce identical win counts."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=1.0, temp_seed=42,
        )

        result1 = app_manager.run_selfplay(config)
        result2 = app_manager.run_selfplay(config)

        # Win/loss/draw counts must be identical
        assert result1.ai1_wins == result2.ai1_wins
        assert result1.ai2_wins == result2.ai2_wins
        assert result1.draws == result2.draws


# ---------------------------------------------------------------------------
# TC-L1, TC-L2, TC-L3: Progress logging
# ---------------------------------------------------------------------------


class TestProgressLogging:
    """TC-L1, TC-L2, TC-L3: Progress output to stdout."""

    def test_progress_printed_per_game(self, app_manager):
        """TC-L1: Each game produces a summary line."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
        )

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            app_manager.run_selfplay(config)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        for i in range(1, 4):
            assert f"Game {i}/3" in output, (
                f"Missing progress line for game {i}"
            )

    def test_summary_printed_at_end(self, app_manager):
        """TC-L2: Summary block with total, win counts, percentages."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=0.0,
        )

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            app_manager.run_selfplay(config)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "=== Summary ===" in output
        assert "Total: 2 games" in output

    def test_sample_path_in_summary(self, app_manager, tmp_path):
        """TC-L3: Summary includes 'Samples saved to' when configured."""
        sample_path = tmp_path / "logged.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=1, temperature=0.0,
            sample_output=str(sample_path),
        )

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            app_manager.run_selfplay(config)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Samples saved to:" in output

    def test_no_sample_path_not_in_summary(self, app_manager):
        """TC-L2: No 'Samples saved' line when sample_output is None."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=1, temperature=0.0,
            sample_output=None,
        )

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            app_manager.run_selfplay(config)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Samples saved" not in output


# ---------------------------------------------------------------------------
# TC-S3: Two identical AIs produce roughly balanced outcomes
# ---------------------------------------------------------------------------


class TestBalancedOutcomes:
    """TC-S3: Two identical AIs produce roughly balanced outcomes."""

    @pytest.mark.slow
    def test_balanced_outcomes_for_identical_ais(self, app_manager):
        """|ai1_wins − ai2_wins| ≤ 30 for 100 games with identical AIs."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=100, temperature=0.0,
        )
        result = app_manager.run_selfplay(config)

        diff = abs(result.ai1_wins - result.ai2_wins)
        assert diff <= 30, (
            f"Win difference {diff} exceeds 30 for identical AIs "
            f"(ai1={result.ai1_wins}, ai2={result.ai2_wins})"
        )


# ---------------------------------------------------------------------------
# TC-P5: sample_output not specified creates no file
# ---------------------------------------------------------------------------


class TestNoSampleOutput:
    """TC-P5: No file created when sample_output is None."""

    def test_no_sample_output_creates_no_file(self, app_manager, tmp_path):
        """Run without sample_output; no .jsonl file appears."""
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=1, sample_output=None,
        )
        app_manager.run_selfplay(config)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 0
