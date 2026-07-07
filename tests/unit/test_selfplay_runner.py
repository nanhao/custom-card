"""Unit tests for SelfPlayRunner._build_game_sample().

Covers the static method that builds a structured game sample dict
from per-step samples. Tests validate the game → perspective → step
hierarchy, reward assignment, and edge cases.

Per design/7 test_cases.md: validates structure (TC-F2/F4/F5/F6/F7),
reward assignment (TC-R1/R2/R3/R4), and content integrity.
"""

import pytest

from src.backend.app.selfplay_runner import SelfPlayRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(player, state, action, value, log_prob):
    """Build a single per-step sample dict matching _play_one_game output.

    Args:
        player: "1" or "2".
        state: List of 14 floats (state vector).
        action: Card action string.
        value: Value estimate float.
        log_prob: Log probability float or None.

    Returns:
        dict with keys player, state, action, value, log_prob.
    """
    return {
        "player": player,
        "state": state,
        "action": action,
        "value": value,
        "log_prob": log_prob,
    }


def _make_game_samples(num_rounds=3):
    """Build a realistic game_samples list for num_rounds.

    Returns alternating P1/P2 steps: [P1_r1, P2_r1, P1_r2, P2_r2, ...].
    Each player gets a distinct state vector prefix for validation.
    """
    samples = []
    for r in range(num_rounds):
        # P1 state: prefix [r+1, 0, ...], P2 state: prefix [0, r+1, ...]
        p1_state = [float(r + 1)] + [0.0] * 13
        p2_state = [0.0, float(r + 1)] + [0.0] * 12
        samples.append(_make_step(
            "1", p1_state, "attack", -0.1, -1.2,
        ))
        samples.append(_make_step(
            "2", p2_state, "defense", -0.2, -0.8,
        ))
    return samples


# ---------------------------------------------------------------------------
# TC-F2/F4/F5/F7: Structure validation
# ---------------------------------------------------------------------------


class TestBuildGameSampleStructure:
    """Validates the game → perspective → step hierarchy structure."""

    def test_returns_dict_with_required_top_level_keys(self):
        """TC-F2: result has exactly game_id, winner, rounds, perspectives."""
        game_samples = _make_game_samples(2)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=2,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="rnn_v2.1",
            ai2="rnn_v2.2",
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {"game_id", "winner", "rounds", "perspectives"}

    def test_game_id_is_preserved(self):
        """game_id is passed through unchanged."""
        game_samples = _make_game_samples(1)
        for gid in [1, 5, 42]:
            result = SelfPlayRunner._build_game_sample(
                game_samples=game_samples,
                game_id=gid,
                winner=None,
                rounds_played=1,
                p1_reward=0.0,
                p2_reward=0.0,
                ai1="a", ai2="b",
            )
            assert result["game_id"] == gid

    def test_rounds_is_preserved(self):
        """rounds field matches rounds_played argument."""
        game_samples = _make_game_samples(3)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=3,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        assert result["rounds"] == 3

    def test_winner_is_preserved(self):
        """winner field matches argument exactly."""
        game_samples = _make_game_samples(1)
        for winner in ["1", "2", None]:
            result = SelfPlayRunner._build_game_sample(
                game_samples=game_samples,
                game_id=1,
                winner=winner,
                rounds_played=1,
                p1_reward=0.0,
                p2_reward=0.0,
                ai1="a", ai2="b",
            )
            assert result["winner"] == winner

    def test_perspectives_has_exactly_two_elements(self):
        """TC-F4: perspectives array has length 2."""
        game_samples = _make_game_samples(2)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=2,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        assert len(result["perspectives"]) == 2

    def test_perspectives_player_order_is_p1_then_p2(self):
        """TC-F5: perspectives[0].player == "1", perspectives[1].player == "2"."""
        game_samples = _make_game_samples(2)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=2,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        assert result["perspectives"][0]["player"] == "1"
        assert result["perspectives"][1]["player"] == "2"

    def test_perspectives_contain_ai_names(self):
        """TC-F6: each perspective has correct ai field."""
        game_samples = _make_game_samples(1)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner=None,
            rounds_played=1,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="model_alpha",
            ai2="model_beta",
        )
        assert result["perspectives"][0]["ai"] == "model_alpha"
        assert result["perspectives"][1]["ai"] == "model_beta"

    def test_each_perspective_keys_are_player_ai_steps(self):
        """Each perspective has exactly keys: player, ai, steps."""
        game_samples = _make_game_samples(1)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner=None,
            rounds_played=1,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="a", ai2="b",
        )
        for p in result["perspectives"]:
            assert set(p.keys()) == {"player", "ai", "steps"}

    def test_step_keys_are_state_action_reward_value_log_prob(self):
        """TC-F7: each step has exactly the 5 required keys."""
        game_samples = _make_game_samples(2)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=2,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        for p in result["perspectives"]:
            for step in p["steps"]:
                assert set(step.keys()) == {"state", "action", "reward", "value", "log_prob"}


# ---------------------------------------------------------------------------
# TC-R1/R2/R3/R4: Reward assignment
# ---------------------------------------------------------------------------


class TestBuildGameSampleRewards:
    """Validates reward assignment by outcome."""

    def test_p1_wins_p1_steps_get_reward_1(self):
        """TC-R1: When winner="1", only last P1 step has reward=1.0, others 0.0."""
        game_samples = _make_game_samples(3)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=3,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        p1_steps = result["perspectives"][0]["steps"]
        assert len(p1_steps) == 3
        # Only the last step gets the terminal reward
        assert p1_steps[0]["reward"] == 0.0
        assert p1_steps[1]["reward"] == 0.0
        assert p1_steps[2]["reward"] == 1.0

    def test_p1_wins_p2_steps_get_reward_minus_1(self):
        """TC-R2: When winner="1", only last P2 step has reward=-1.0, others 0.0."""
        game_samples = _make_game_samples(3)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=3,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        p2_steps = result["perspectives"][1]["steps"]
        assert len(p2_steps) == 3
        # Only the last step gets the terminal reward
        assert p2_steps[0]["reward"] == 0.0
        assert p2_steps[1]["reward"] == 0.0
        assert p2_steps[2]["reward"] == -1.0

    def test_p2_wins_p2_steps_get_reward_1(self):
        """When winner="2", only last P2 step has reward=1.0, others 0.0."""
        game_samples = _make_game_samples(2)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="2",
            rounds_played=2,
            p1_reward=-1.0,
            p2_reward=1.0,
            ai1="a", ai2="b",
        )
        p2_steps = result["perspectives"][1]["steps"]
        assert len(p2_steps) == 2
        # Only the last step gets the terminal reward
        assert p2_steps[0]["reward"] == 0.0
        assert p2_steps[1]["reward"] == 1.0

    def test_draw_both_sides_get_reward_0(self):
        """TC-R3: Draw assigns reward=0.0 to both sides."""
        game_samples = _make_game_samples(3)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner=None,
            rounds_played=3,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="a", ai2="b",
        )
        for p in result["perspectives"]:
            for step in p["steps"]:
                assert step["reward"] == 0.0

    def test_reward_sparse_only_last_step(self):
        """TC-R4: Only the last step in each perspective gets the terminal reward."""
        game_samples = _make_game_samples(5)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=5,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        for p in result["perspectives"]:
            steps = p["steps"]
            assert len(steps) > 0
            # All but the last step should have reward=0.0
            for step in steps[:-1]:
                assert step["reward"] == 0.0, (
                    f"Intermediate step should have reward=0.0"
                )
            # Only last step has the terminal reward
            expected_terminal = 1.0 if p["player"] == "1" else -1.0
            assert steps[-1]["reward"] == expected_terminal, (
                f"Last step should have reward={expected_terminal}"
            )


# ---------------------------------------------------------------------------
# Content integrity
# ---------------------------------------------------------------------------


class TestBuildGameSampleContent:
    """Validates step-level content is preserved correctly."""

    def test_state_vector_preserved(self):
        """State vector from game_samples is preserved in output."""
        game_samples = _make_game_samples(2)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=2,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        # P1 r1 state should be [1.0] + 13 zeros
        p1_step0_state = result["perspectives"][0]["steps"][0]["state"]
        assert p1_step0_state[0] == 1.0
        assert len(p1_step0_state) == 14
        # P2 r1 state should be [0.0, 1.0] + 12 zeros
        p2_step0_state = result["perspectives"][1]["steps"][0]["state"]
        assert p2_step0_state[0] == 0.0
        assert p2_step0_state[1] == 1.0

    def test_action_preserved(self):
        """Action strings are preserved in output."""
        game_samples = [
            _make_step("1", [0.0] * 14, "heavy", 0.5, -0.3),
            _make_step("2", [0.0] * 14, "counter", 0.6, -0.4),
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner=None,
            rounds_played=1,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="a", ai2="b",
        )
        assert result["perspectives"][0]["steps"][0]["action"] == "heavy"
        assert result["perspectives"][1]["steps"][0]["action"] == "counter"

    def test_log_prob_null_preserved(self):
        """log_prob=None is preserved (for greedy steps)."""
        game_samples = [
            _make_step("1", [0.0] * 14, "attack", 0.0, None),
            _make_step("2", [0.0] * 14, "defense", 0.0, None),
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner=None,
            rounds_played=1,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="a", ai2="b",
        )
        assert result["perspectives"][0]["steps"][0]["log_prob"] is None
        assert result["perspectives"][1]["steps"][0]["log_prob"] is None

    def test_log_prob_float_preserved(self):
        """log_prob as float is preserved (for sampled steps)."""
        game_samples = [
            _make_step("1", [0.0] * 14, "attack", 0.1, -1.341),
            _make_step("2", [0.0] * 14, "defense", 0.2, -0.579),
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner=None,
            rounds_played=1,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="a", ai2="b",
        )
        assert result["perspectives"][0]["steps"][0]["log_prob"] == pytest.approx(-1.341)
        assert result["perspectives"][1]["steps"][0]["log_prob"] == pytest.approx(-0.579)

    def test_value_preserved(self):
        """Value estimates are preserved."""
        game_samples = [
            _make_step("1", [0.0] * 14, "attack", 0.85, None),
            _make_step("2", [0.0] * 14, "defense", -0.42, None),
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner=None,
            rounds_played=1,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="a", ai2="b",
        )
        assert result["perspectives"][0]["steps"][0]["value"] == pytest.approx(0.85)
        assert result["perspectives"][1]["steps"][0]["value"] == pytest.approx(-0.42)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBuildGameSampleEdgeCases:
    """Edge case handling for _build_game_sample."""

    def test_empty_game_samples_produces_empty_steps(self):
        """Empty game_samples list produces perspectives with empty steps."""
        result = SelfPlayRunner._build_game_sample(
            game_samples=[],
            game_id=1,
            winner=None,
            rounds_played=0,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="a", ai2="b",
        )
        assert len(result["perspectives"]) == 2
        assert result["perspectives"][0]["steps"] == []
        assert result["perspectives"][1]["steps"] == []

    def test_single_round_produces_one_step_per_perspective(self):
        """One round produces exactly 1 step per perspective."""
        game_samples = _make_game_samples(1)
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=1,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        assert len(result["perspectives"][0]["steps"]) == 1
        assert len(result["perspectives"][1]["steps"]) == 1

    def test_all_player_1_entries_go_to_p1_perspective(self):
        """Entries with player='1' all go to P1 perspective, none to P2."""
        game_samples = [
            _make_step("1", [float(i)] + [0.0] * 13, "attack", 0.0, None)
            for i in range(5)
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="1",
            rounds_played=5,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="a", ai2="b",
        )
        assert len(result["perspectives"][0]["steps"]) == 5
        assert len(result["perspectives"][1]["steps"]) == 0

    def test_all_player_2_entries_go_to_p2_perspective(self):
        """Entries with player='2' all go to P2 perspective, none to P1."""
        game_samples = [
            _make_step("2", [0.0, float(i)] + [0.0] * 12, "defense", 0.0, None)
            for i in range(3)
        ]
        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner="2",
            rounds_played=3,
            p1_reward=-1.0,
            p2_reward=1.0,
            ai1="a", ai2="b",
        )
        assert len(result["perspectives"][0]["steps"]) == 0
        assert len(result["perspectives"][1]["steps"]) == 3
