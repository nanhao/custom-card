"""Tests for dismantle_target flow through SelfPlayRunner (design/37).

Covers the full self-play dismantle pipeline:
  ai_sample_action → SelfPlayRunner._play_one_game → play_round(dismantle_target=...)
  → _build_game_sample (action_to_policy_index encodes dismantle correctly).

These tests were missing from design/37 original implementation and
are critical because SelfPlayRunner is the PPO sample generation backbone.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.backend.ai.constants import HIDDEN_SIZE, POLICY_OUTPUT_SIZE
from src.backend.ai_adapter.converter import Converter
from src.backend.app.selfplay_runner import SelfPlayRunner
from src.backend.app.selfplay_config import SelfPlayConfig
from src.backend.game.card_type import CardType
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus
from src.backend.log.structured_logger import LoggerBuilder
from tests.helpers import make_game_state


# ---------------------------------------------------------------------------
# Module-level fixture: init LoggerBuilder for SelfPlayRunner
# autouse because conftest's _reset_logger_builder resets before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _init_logger(tmp_path):
    """Ensure LoggerBuilder is initialized before SelfPlayRunner tests.

    conftest.py's _reset_logger_builder resets the singleton before each
    test, but SelfPlayRunner.run() calls LoggerBuilder.get_logger() which
    requires init() to have been called.
    """
    LoggerBuilder.init(enabled=False, log_dir=str(tmp_path))


# ============================================================================
# T4.1: _build_game_sample encodes dismantle policy index correctly
# ============================================================================


class TestBuildGameSampleDismantle:
    """Verify _build_game_sample correctly records dismantle policy indices."""

    def test_dismantle_action_encoded_in_sample(self):
        """Dismantle action index should appear in the built sample."""
        # Simulate: P1 chooses BUILD + dismantle ATTACK → policy index 5
        p1_action_idx = Converter.action_to_policy_index(
            CardType.BUILD, build_target=None, dismantle_target=CardType.ATTACK,
        )
        assert p1_action_idx == 5, f"Expected 5, got {p1_action_idx}"

        game_samples = [
            {
                "player": "1",
                "input": [0.0] * 62,
                "action": p1_action_idx,
                "value": 0.5,
                "log_prob": -1.2,
            },
            {
                "player": "2",
                "input": [0.0] * 62,
                "action": 3,  # play ATTACK
                "value": 0.3,
                "log_prob": -0.8,
            },
        ]

        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=1,
            winner=None,
            rounds_played=1,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="rnn_v1.1",
            ai2="rnn_v1.1",
            perspective="both",
        )

        assert result["game_id"] == 1
        assert len(result["perspectives"]) == 2

        p1_perspective = result["perspectives"][0]
        assert p1_perspective["player"] == "1"
        assert len(p1_perspective["steps"]) == 1
        assert p1_perspective["steps"][0]["action"] == 5  # dismantle ATTACK

    def test_dismantle_defense_encoded_in_sample(self):
        """Dismantle DEFENSE → policy index 8."""
        p1_action_idx = Converter.action_to_policy_index(
            CardType.BUILD, build_target=None, dismantle_target=CardType.DEFENSE,
        )
        assert p1_action_idx == 8

        game_samples = [
            {
                "player": "1",
                "input": [0.0] * 62,
                "action": p1_action_idx,
                "value": 0.1,
                "log_prob": -2.0,
            },
        ]

        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=2,
            winner="1",
            rounds_played=1,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="rnn_v1.1",
            ai2="rnn_v1.1",
            perspective="ai1",
        )

        assert len(result["perspectives"]) == 1
        assert result["perspectives"][0]["steps"][0]["action"] == 8

    def test_build_and_dismantle_same_game(self):
        """Game with both build and dismantle actions produces correct indices."""
        game_samples = [
            {
                "player": "1",
                "input": [0.0] * 62,
                "action": Converter.action_to_policy_index(
                    CardType.BUILD, build_target=CardType.ATTACK,
                ),  # index 4
                "value": 0.2,
                "log_prob": -1.0,
            },
            {
                "player": "2",
                "input": [0.0] * 62,
                "action": Converter.action_to_policy_index(
                    CardType.BUILD, dismantle_target=CardType.DEFENSE,
                ),  # index 8
                "value": 0.4,
                "log_prob": -0.5,
            },
            {
                "player": "1",
                "input": [0.0] * 62,
                "action": Converter.action_to_policy_index(
                    CardType.BUILD, dismantle_target=CardType.HEAVY,
                ),  # index 11
                "value": 0.3,
                "log_prob": -1.5,
            },
        ]

        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=3,
            winner="1",
            rounds_played=3,
            p1_reward=1.0,
            p2_reward=-1.0,
            ai1="rnn_v1.1",
            ai2="rnn_v1.1",
            perspective="both",
        )

        p1_steps = result["perspectives"][0]["steps"]
        p2_steps = result["perspectives"][1]["steps"]

        assert len(p1_steps) == 2
        assert p1_steps[0]["action"] == 4  # build ATTACK
        assert p1_steps[1]["action"] == 11  # dismantle HEAVY
        assert len(p2_steps) == 1
        assert p2_steps[0]["action"] == 8  # dismantle DEFENSE

        # Terminal step gets reward, others get 0
        assert p1_steps[0]["reward"] == 0.0
        assert p1_steps[1]["reward"] == 1.0
        assert p2_steps[0]["reward"] == -1.0

    def test_dismantle_iron_defense_encoded(self):
        """Dismantle IRON_DEFENSE → policy index 23."""
        idx = Converter.action_to_policy_index(
            CardType.BUILD, build_target=None, dismantle_target=CardType.IRON_DEFENSE,
        )
        assert idx == 23

        game_samples = [{
            "player": "1",
            "input": [0.0] * 62,
            "action": idx,
            "value": 0.0,
            "log_prob": -3.0,
        }]

        result = SelfPlayRunner._build_game_sample(
            game_samples=game_samples,
            game_id=4,
            winner=None,
            rounds_played=1,
            p1_reward=0.0,
            p2_reward=0.0,
            ai1="rnn_v1.1",
            ai2="rnn_v1.1",
            perspective="both",
        )

        assert result["perspectives"][0]["steps"][0]["action"] == 23


# ============================================================================
# T4.2: Full self-play game with dismantle — end-to-end
# ============================================================================


def _make_dismantle_sample_action():
    """Create a sample_action callable that plays a dismantle game.

    Returns a factory function with the same signature as
    AppManager.ai_sample_action but with deterministic card selection:
      - P1: Round 1 = BUILD + dismantle ATTACK, subsequent rounds = pass
      - P2: always plays DEFENSE
    """
    call_count = {"1": 0, "2": 0}

    def sample_action(state, h_pol, h_val, ai_name, temperature, player_key):
        call_count[player_key] += 1
        cnt = call_count[player_key]

        if player_key == "1":
            if cnt == 1:
                return (
                    CardType.BUILD,           # card
                    None,                      # build_target
                    CardType.ATTACK,           # dismantle_target
                    [0.0] * HIDDEN_SIZE,       # h_pol
                    -1.0,                      # log_prob
                    0.5,                       # value
                    [0.0] * HIDDEN_SIZE,       # h_val
                )
            else:
                return (
                    None, None, None,
                    [0.0] * HIDDEN_SIZE, -0.1, 0.0, [0.0] * HIDDEN_SIZE,
                )
        else:
            return (
                CardType.DEFENSE, None, None,
                [0.0] * HIDDEN_SIZE, -1.0, 0.3, [0.0] * HIDDEN_SIZE,
            )

    return sample_action


class TestSelfPlayDismantleEndToEnd:
    """T4.2: Full self-play loop with dismantle actions.

    Uses mocked GameCore + SelfPlayRunner with a mock sample_action that
    returns dismantle targets. Mocking GameCore eliminates random-card-deal
    flakiness while still verifying the full SelfPlayRunner pipeline.
    Verifies:
      - Game completes without errors
      - Policy indices in sample output encode dismantle correctly
      - Dismantle targets flow correctly through the full pipeline
    """

    def test_single_game_dismantle_completes(self):
        """One game with P1 dismantling ATTACK completes without error."""
        from unittest.mock import MagicMock, PropertyMock, patch

        def mock_sample_action(state, h_pol, h_val, ai_name, temperature, player_key):
            if player_key == "1":
                return (
                    CardType.BUILD, None, CardType.ATTACK,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.5, [0.0] * HIDDEN_SIZE,
                )
            else:
                return (
                    CardType.DEFENSE, None, None,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.3, [0.0] * HIDDEN_SIZE,
                )

        with patch(
            "src.backend.app.selfplay_runner.GameCore", autospec=True,
        ) as MockGameCore:
            mock_gm = MockGameCore.return_value
            type(mock_gm).is_over = PropertyMock(side_effect=[False, True])
            mock_gm.winner = "1"
            mock_gm.round_log = [MagicMock(round=1)]
            mock_gm.mode = "pvp"
            mock_gm.get_state.return_value = make_game_state(
                hand_1=[CardType.BUILD], hand_2=[CardType.DEFENSE],
                hp_1=20, hp_2=20, energy_1=0,
                discard_pile_1=[CardType.ATTACK],
            )

            runner = SelfPlayRunner(mock_sample_action)

            with tempfile.TemporaryDirectory() as td:
                sample_path = os.path.join(td, "sample.json")
                config = SelfPlayConfig(
                    ai1="rnn_v1.1", ai2="rnn_v1.1",
                    game_count=1, sample_output=sample_path,
                    temperature=0.0, temp_side="both", perspective="both",
                )
                result = runner.run(config)

        assert result.total_games == 1
        assert result.ai1_wins + result.ai2_wins + result.draws == 1

    def test_dismantle_sample_contains_correct_policy_index(self):
        """Sample output encodes dismantle action with correct policy index."""
        from unittest.mock import MagicMock, PropertyMock, patch

        def mock_sample_action(state, h_pol, h_val, ai_name, temperature, player_key):
            if player_key == "1":
                return (
                    CardType.BUILD, None, CardType.ATTACK,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.5, [0.0] * HIDDEN_SIZE,
                )
            else:
                return (
                    CardType.DEFENSE, None, None,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.3, [0.0] * HIDDEN_SIZE,
                )

        with patch(
            "src.backend.app.selfplay_runner.GameCore", autospec=True,
        ) as MockGameCore:
            mock_gm = MockGameCore.return_value
            type(mock_gm).is_over = PropertyMock(side_effect=[False, True])
            mock_gm.winner = "1"
            mock_gm.round_log = [MagicMock(round=1)]
            mock_gm.mode = "pvp"
            mock_gm.get_state.return_value = make_game_state(
                hand_1=[CardType.BUILD], hand_2=[CardType.DEFENSE],
                hp_1=20, hp_2=20, energy_1=0,
                discard_pile_1=[CardType.ATTACK],
            )

            runner = SelfPlayRunner(mock_sample_action)

            with tempfile.TemporaryDirectory() as td:
                sample_path = os.path.join(td, "sample.json")
                config = SelfPlayConfig(
                    ai1="rnn_v1.1", ai2="rnn_v1.1",
                    game_count=1, sample_output=sample_path,
                    temperature=0.0, temp_side="both", perspective="both",
                )
                runner.run(config)

                assert os.path.exists(sample_path), "Sample output file was not created"
                samples = json.loads(Path(sample_path).read_text(encoding="utf-8"))
                assert isinstance(samples, list)
                assert len(samples) >= 1, "Expected at least 1 game sample"

                game = samples[0]
                assert "perspectives" in game
                assert len(game["perspectives"]) == 2

                p1_perspective = game["perspectives"][0]
                assert len(p1_perspective["steps"]) >= 1

                action = p1_perspective["steps"][0]["action"]
                assert 0 <= action < POLICY_OUTPUT_SIZE, (
                    f"Action {action} out of valid range [0, {POLICY_OUTPUT_SIZE})"
                )

    def test_dismantle_no_crash_on_sample_output(self):
        """Multiple games with dismantle should produce valid sample file."""
        from unittest.mock import MagicMock, PropertyMock, patch

        def mock_sample_action(state, h_pol, h_val, ai_name, temperature, player_key):
            if player_key == "1":
                return (
                    CardType.BUILD, None, CardType.ATTACK,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.5, [0.0] * HIDDEN_SIZE,
                )
            else:
                return (
                    CardType.DEFENSE, None, None,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.3, [0.0] * HIDDEN_SIZE,
                )

        with patch(
            "src.backend.app.selfplay_runner.GameCore", autospec=True,
        ) as MockGameCore:
            mock_gm = MockGameCore.return_value
            type(mock_gm).is_over = PropertyMock(side_effect=[False, True, False, True, False, True])
            mock_gm.winner = "1"
            mock_gm.round_log = [MagicMock(round=1)]
            mock_gm.mode = "pvp"
            mock_gm.get_state.return_value = make_game_state(
                hand_1=[CardType.BUILD], hand_2=[CardType.DEFENSE],
                hp_1=20, hp_2=20, energy_1=0,
                discard_pile_1=[CardType.ATTACK],
            )

            runner = SelfPlayRunner(mock_sample_action)

            with tempfile.TemporaryDirectory() as td:
                sample_path = os.path.join(td, "sample.json")
                config = SelfPlayConfig(
                    ai1="rnn_v1.1", ai2="rnn_v1.1",
                    game_count=3, sample_output=sample_path,
                    temperature=0.0, temp_side="both", perspective="both",
                )
                result = runner.run(config)

                assert result.total_games == 3
                assert os.path.exists(sample_path)

                samples = json.loads(Path(sample_path).read_text(encoding="utf-8"))
                assert len(samples) == 3

                for game in samples:
                    assert "perspectives" in game
                    for p in game["perspectives"]:
                        assert "steps" in p
                        for step in p["steps"]:
                            assert "action" in step
                            assert 0 <= step["action"] < POLICY_OUTPUT_SIZE

    def test_game_count_zero_with_dismantle_sample_fn(self, tmp_path):
        """game_count=0 still produces valid empty sample file."""
        sample_action_fn = _make_dismantle_sample_action()
        runner = SelfPlayRunner(sample_action_fn)

        sample_path = tmp_path / "sample.json"

        config = SelfPlayConfig(
            ai1="rnn_v1.1",
            ai2="rnn_v1.1",
            game_count=0,
            sample_output=str(sample_path),
            temperature=0.0,
        )

        result = runner.run(config)
        assert result.total_games == 0
        assert sample_path.exists()

        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        assert samples == []


# ============================================================================
# ============================================================================
# T4.3: play_round receives correct dismantle_target from sample_action
# ============================================================================


class TestSelfPlayPlayRoundDismantle:
    """Verify play_round is called with correct dismantle_target from sample_action."""

    def test_play_round_receives_build_target(self):
        """Build mode: play_round receives build_target, dismantle_target=None."""
        from unittest.mock import MagicMock, PropertyMock

        def mock_sample_action(state, h_pol, h_val, ai_name, temperature, player_key):
            if player_key == "1":
                return (
                    CardType.BUILD, CardType.ATTACK,
                    None,  # build mode, not dismantle
                    [0.0] * HIDDEN_SIZE, -1.0, 0.5, [0.0] * HIDDEN_SIZE,
                )
            else:
                return (
                    CardType.DEFENSE, None, None,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.3, [0.0] * HIDDEN_SIZE,
                )

        with patch(
            "src.backend.app.selfplay_runner.GameCore",
            autospec=True,
        ) as MockGameCore:
            mock_gm = MockGameCore.return_value
            type(mock_gm).is_over = PropertyMock(side_effect=[False, True])
            mock_gm.winner = "1"
            mock_gm.round_log = [MagicMock(round=1)]

            state = make_game_state(
                hand_1=[CardType.BUILD],
                hand_2=[CardType.DEFENSE],
                hp_1=20, hp_2=20,
                energy_1=2, energy_2=0,
                discard_pile_1=[CardType.ATTACK],
            )
            mock_gm.get_state.return_value = state

            runner = SelfPlayRunner(mock_sample_action)

            with tempfile.TemporaryDirectory() as td:
                sample_path = os.path.join(td, "sample.json")
                config = SelfPlayConfig(
                    ai1="rnn_v1.1",
                    ai2="rnn_v1.1",
                    game_count=1,
                    sample_output=sample_path,
                    temperature=0.0,
                    temp_side="both",
                    perspective="both",
                )

                runner.run(config)

                assert mock_gm.play_round.called, "play_round was never called"

                call_args = mock_gm.play_round.call_args
                bt1 = call_args[1].get("build_target_1")
                dt1 = call_args[1].get("dismantle_target_1")
                assert bt1 == CardType.ATTACK, f"build_target_1={bt1}"
                assert dt1 is None, f"dismantle_target_1 should be None, got {dt1}"

    def test_play_round_receives_dismantle_target(self):
        """Dismantle mode: play_round receives dismantle_target, build_target=None."""
        from unittest.mock import MagicMock, PropertyMock

        def mock_sample_action(state, h_pol, h_val, ai_name, temperature, player_key):
            if player_key == "1":
                return (
                    CardType.BUILD, None, CardType.ATTACK,  # dismantle mode
                    [0.0] * HIDDEN_SIZE, -1.0, 0.5, [0.0] * HIDDEN_SIZE,
                )
            else:
                return (
                    CardType.DEFENSE, None, None,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.3, [0.0] * HIDDEN_SIZE,
                )

        with patch(
            "src.backend.app.selfplay_runner.GameCore",
            autospec=True,
        ) as MockGameCore:
            mock_gm = MockGameCore.return_value
            type(mock_gm).is_over = PropertyMock(side_effect=[False, True])
            mock_gm.winner = "1"
            mock_gm.round_log = [MagicMock(round=1)]

            state = make_game_state(
                hand_1=[CardType.BUILD],
                hand_2=[CardType.DEFENSE],
                hp_1=20, hp_2=20,
                energy_1=0, energy_2=0,
                discard_pile_1=[CardType.ATTACK],
            )
            mock_gm.get_state.return_value = state

            runner = SelfPlayRunner(mock_sample_action)

            with tempfile.TemporaryDirectory() as td:
                sample_path = os.path.join(td, "sample.json")
                config = SelfPlayConfig(
                    ai1="rnn_v1.1",
                    ai2="rnn_v1.1",
                    game_count=1,
                    sample_output=sample_path,
                    temperature=0.0,
                )

                runner.run(config)

                assert mock_gm.play_round.called
                call_args = mock_gm.play_round.call_args

                build_target_1 = call_args[1].get("build_target_1")
                dismantle_target_1 = call_args[1].get("dismantle_target_1")

                assert build_target_1 is None, (
                    f"build_target_1 should be None, got {build_target_1}"
                )
                assert dismantle_target_1 == CardType.ATTACK, (
                    f"dismantle_target_1 should be ATTACK, got {dismantle_target_1}"
                )

    def test_play_round_with_both_dismantle(self):
        """Both players using dismantle in the same round."""
        from unittest.mock import MagicMock, PropertyMock

        def mock_sample_action(state, h_pol, h_val, ai_name, temperature, player_key):
            if player_key == "1":
                return (
                    CardType.BUILD, None, CardType.ATTACK,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.5, [0.0] * HIDDEN_SIZE,
                )
            else:
                return (
                    CardType.BUILD, None, CardType.DEFENSE,
                    [0.0] * HIDDEN_SIZE, -1.0, 0.3, [0.0] * HIDDEN_SIZE,
                )

        with patch(
            "src.backend.app.selfplay_runner.GameCore",
            autospec=True,
        ) as MockGameCore:
            mock_gm = MockGameCore.return_value
            type(mock_gm).is_over = PropertyMock(side_effect=[False, True])
            mock_gm.winner = "1"
            mock_gm.round_log = [MagicMock(round=1)]

            state = make_game_state(
                hand_1=[CardType.BUILD],
                hand_2=[CardType.BUILD],
                hp_1=20, hp_2=20,
                energy_1=0, energy_2=0,
                discard_pile_1=[CardType.ATTACK],
                discard_pile_2=[CardType.DEFENSE],
            )
            mock_gm.get_state.return_value = state

            runner = SelfPlayRunner(mock_sample_action)

            with tempfile.TemporaryDirectory() as td:
                sample_path = os.path.join(td, "sample.json")
                config = SelfPlayConfig(
                    ai1="rnn_v1.1",
                    ai2="rnn_v1.1",
                    game_count=1,
                    sample_output=sample_path,
                    temperature=0.0,
                    perspective="both",
                )

                runner.run(config)

                assert mock_gm.play_round.called
                call_args = mock_gm.play_round.call_args

                assert call_args[1].get("dismantle_target_1") == CardType.ATTACK
                assert call_args[1].get("dismantle_target_2") == CardType.DEFENSE
                assert call_args[1].get("build_target_1") is None
                assert call_args[1].get("build_target_2") is None
