"""Integration tests for PPO sample output in self-play runs.

Covers TC-P1 through TC-P7, and TC-T1 through TC-T4 from
design/5 test_cases.md.

Tests exercise sample JSONL file format, reward assignment, temp_step
and temp_side configuration. Uses tmp_path for file output.
"""

import json

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
# TC-P1: sample_output produces valid JSONL file
# ---------------------------------------------------------------------------


class TestSampleOutputFile:
    """TC-P1, TC-P2: sample_output produces valid JSONL."""

    def test_sample_output_creates_valid_jsonl(self, app_manager, tmp_path):
        """File exists; each line is valid JSON with required keys."""
        sample_path = tmp_path / "samples.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        assert sample_path.exists()
        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) > 0

        required_keys = {"state", "action", "reward", "value", "log_prob"}
        for line in lines:
            obj = json.loads(line)
            assert required_keys.issubset(obj.keys())

    def test_sample_line_count_matches_rounds(self, app_manager, tmp_path):
        """TC-P2: Each game step produces one sample per player."""
        sample_path = tmp_path / "samples2.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=1, temperature=0.0,
            sample_output=str(sample_path),
        )
        result = app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        # Each round produces 2 sample lines (one per player)
        expected_lines = int(result.avg_rounds) * 2
        assert len(lines) == expected_lines


# ---------------------------------------------------------------------------
# TC-P3, TC-P4: Rewards in PPO samples
# ---------------------------------------------------------------------------


class TestSampleRewards:
    """TC-P3, TC-P4: Rewards in PPO samples match game outcome."""

    def test_winning_side_gets_positive_reward(self, app_manager, tmp_path):
        """Winner's steps have reward=1.0, loser's have reward=-1.0."""
        sample_path = tmp_path / "reward_samples.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=5, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        rewards = set()
        for line in lines:
            obj = json.loads(line)
            rewards.add(obj["reward"])

        assert rewards, "Sample file should not be empty"

    def test_reward_values_are_valid(self, app_manager, tmp_path):
        """All reward values are 1.0, -1.0, or 0.0."""
        sample_path = tmp_path / "reward_check.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            assert obj["reward"] in (1.0, -1.0, 0.0)


# ---------------------------------------------------------------------------
# TC-P6: log_prob is None for greedy steps
# ---------------------------------------------------------------------------


class TestGreedyLogProb:
    """TC-P6: log_prob is null (None→null in JSON) for temperature=0."""

    def test_greedy_steps_have_null_log_prob(self, app_manager, tmp_path):
        """All sample lines have log_prob: null when temperature=0."""
        sample_path = tmp_path / "greedy_samples.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=1, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            assert obj["log_prob"] is None, (
                f"Expected null log_prob for greedy, got {obj['log_prob']}"
            )


# ---------------------------------------------------------------------------
# TC-P7: state vector length is 14
# ---------------------------------------------------------------------------


class TestStateVectorLength:
    """TC-P7: Every sample's state array has exactly 14 elements."""

    def test_state_vector_length_is_14(self, app_manager, tmp_path):
        """All sample lines have state arrays of length 14."""
        sample_path = tmp_path / "state_length.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=1.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            assert len(obj["state"]) == 14, (
                f"Expected 14 elements, got {len(obj['state'])}"
            )


# ---------------------------------------------------------------------------
# TC-T1: temp_step=2 switches to greedy after round 2
# ---------------------------------------------------------------------------


class TestTempStep:
    """TC-T1, TC-T2: temp_step controls when sampling stops."""

    def test_temp_step_switches_to_greedy(self, app_manager, tmp_path):
        """TC-T1: Some samples have non-null and some have null log_prob."""
        sample_path = tmp_path / "temp_step.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=10, temperature=1.0,
            temp_step=2, temp_side="both",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        non_null = [line for line in lines
                     if json.loads(line)["log_prob"] is not None]
        null = [line for line in lines
                if json.loads(line)["log_prob"] is None]

        assert len(non_null) > 0, "Expected some non-null log_probs"
        assert len(null) > 0, "Expected some null log_probs (greedy steps)"

    def test_temp_step_none_samples_all(self, app_manager, tmp_path):
        """TC-T2: temp_step=None → all rounds have non-null log_prob."""
        sample_path = tmp_path / "no_step.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=5, temperature=1.0,
            temp_step=None, temp_side="both",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            assert obj["log_prob"] is not None, (
                "All rounds should have non-null log_prob"
            )


# ---------------------------------------------------------------------------
# TC-T3, TC-T4: temp_side controls which player is sampled
# ---------------------------------------------------------------------------


class TestTempSide:
    """TC-T3, TC-T4: temp_side controls which player gets sampling."""

    def test_temp_side_player1_only_samples_p1(self, app_manager, tmp_path):
        """TC-T3: Only P1 sample lines have non-null log_prob."""
        sample_path = tmp_path / "side_p1.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=5, temperature=1.0,
            temp_side="player1",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        for i, line in enumerate(lines):
            obj = json.loads(line)
            if i % 2 == 0:  # P1 samples (even indices)
                assert obj["log_prob"] is not None, (
                    f"P1 sample at index {i} should have non-null log_prob"
                )
            else:
                assert obj["log_prob"] is None, (
                    f"P2 sample at index {i} should have null log_prob"
                )

    def test_temp_side_player2_only_samples_p2(self, app_manager, tmp_path):
        """TC-T4: Only P2 sample lines have non-null log_prob."""
        sample_path = tmp_path / "side_p2.jsonl"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=5, temperature=1.0,
            temp_side="player2",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
        for i, line in enumerate(lines):
            obj = json.loads(line)
            if i % 2 == 0:  # P1 samples
                assert obj["log_prob"] is None, (
                    f"P1 sample at index {i} should have null log_prob"
                )
            else:  # P2 samples
                assert obj["log_prob"] is not None, (
                    f"P2 sample at index {i} should have non-null log_prob"
                )
