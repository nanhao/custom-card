"""Unit tests for SelfPlayConfig perspective validation and
SelfPlayRunner perspective filtering (design/10 D2).

Covers:
- TC-PC-01/02/03: SelfPlayConfig perspective field validation
- TC-PF-01/02/03: _build_game_sample perspective filtering
"""

import pytest

from src.backend.app.selfplay_config import SelfPlayConfig
from src.backend.app.selfplay_runner import SelfPlayRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(player, input_vector, action, value, log_prob):
    """Build a single per-step sample dict."""
    return {
        "player": player,
        "input": input_vector,
        "action": action,
        "value": value,
        "log_prob": log_prob,
    }


# ---------------------------------------------------------------------------
# TC-PC-01/02/03: SelfPlayConfig perspective validation
# ---------------------------------------------------------------------------


class TestSelfPlayConfigPerspective:
    """TC-PC-01/02/03: perspective field validation in SelfPlayConfig."""

    def test_default_perspective_is_both(self):
        """TC-PC-01: Default perspective is 'both'."""
        config = SelfPlayConfig(ai1="rnn_v1.1", ai2="rnn_v1.1")
        assert config.perspective == "both"

    def test_valid_perspective_values_accepted(self):
        """TC-PC-02: 'ai1', 'ai2', 'both' are accepted."""
        for val in ["ai1", "ai2", "both"]:
            config = SelfPlayConfig(
                ai1="rnn_v1.1", ai2="rnn_v1.1", perspective=val,
            )
            assert config.perspective == val

    def test_invalid_perspective_rejected(self):
        """TC-PC-03: Invalid perspective raises ValueError."""
        for bad_val in ["player1", "", "none"]:
            with pytest.raises(ValueError, match="Invalid perspective"):
                SelfPlayConfig(
                    ai1="rnn_v1.1", ai2="rnn_v1.1",
                    perspective=bad_val,
                )


# ---------------------------------------------------------------------------
# TC-PF-01/02/03: _build_game_sample perspective filtering
# ---------------------------------------------------------------------------


class TestPerspectiveFiltering:
    """TC-PF-01/02/03: perspective filtering in _build_game_sample."""

    def test_perspective_both_includes_both(self):
        """TC-PF-01: perspective='both' includes both perspectives."""
        game_samples = [
            _make_step("1", [float(i)] + [0.0] * 13, "attack", 0.0, None)
            for i in range(3)
        ] + [
            _make_step("2", [0.0, float(i)] + [0.0] * 12, "defense", 0.0, None)
            for i in range(3)
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1, winner="1", rounds_played=3,
            p1_reward=1.0, p2_reward=-1.0,
            ai1="a", ai2="b", perspective="both",
        )
        assert len(result["perspectives"]) == 2
        assert result["perspectives"][0]["player"] == "1"
        assert result["perspectives"][1]["player"] == "2"

    def test_perspective_ai1_includes_only_p1(self):
        """TC-PF-02: perspective='ai1' includes only player 1."""
        game_samples = [
            _make_step("1", [float(i)] + [0.0] * 13, "attack", 0.0, None)
            for i in range(3)
        ] + [
            _make_step("2", [0.0, float(i)] + [0.0] * 12, "defense", 0.0, None)
            for i in range(3)
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1, winner="1", rounds_played=3,
            p1_reward=1.0, p2_reward=-1.0,
            ai1="a", ai2="b", perspective="ai1",
        )
        assert len(result["perspectives"]) == 1
        assert result["perspectives"][0]["player"] == "1"
        assert len(result["perspectives"][0]["steps"]) == 3

    def test_perspective_ai2_includes_only_p2(self):
        """TC-PF-03: perspective='ai2' includes only player 2."""
        game_samples = [
            _make_step("1", [float(i)] + [0.0] * 13, "attack", 0.0, None)
            for i in range(3)
        ] + [
            _make_step("2", [0.0, float(i)] + [0.0] * 12, "defense", 0.0, None)
            for i in range(3)
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1, winner="1", rounds_played=3,
            p1_reward=1.0, p2_reward=-1.0,
            ai1="a", ai2="b", perspective="ai2",
        )
        assert len(result["perspectives"]) == 1
        assert result["perspectives"][0]["player"] == "2"
        assert len(result["perspectives"][0]["steps"]) == 3

    def test_perspective_filtering_with_empty_game_samples(self):
        """Perspective filter works with empty game samples."""
        result = SelfPlayRunner._build_game_sample(
            game_samples=[],
            game_id=1, winner=None, rounds_played=0,
            p1_reward=0.0, p2_reward=0.0,
            ai1="a", ai2="b", perspective="ai1",
        )
        assert len(result["perspectives"]) == 1
        assert result["perspectives"][0]["steps"] == []
