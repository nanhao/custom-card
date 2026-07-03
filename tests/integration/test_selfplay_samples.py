"""Integration tests for PPO sample output — structure, rewards, content.

Covers TC-F1 through TC-F8 (structure), TC-R1 through TC-R4 (rewards),
and TC-S1 through TC-S7 (step content) from design/7 test_cases.md.

Tests exercise the new JSON array format (game → perspective → steps
hierarchy). Uses tmp_path for file output — no real data files affected.

For file output (TC-O1–O4) and boundary tests (TC-T1–T3), see
test_selfplay_samples_output.py.
"""

import pytest

from src.backend.app.selfplay_config import SelfPlayConfig
from src.backend.log.structured_logger import LoggerBuilder
from tests.helpers import read_sample_file


# ---------------------------------------------------------------------------
# LoggerBuilder must be initialized (even disabled) so SelfPlayRunner
# can call LoggerBuilder.get_logger() without raising RuntimeError.
# conftest.py resets the singleton before/after every test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _init_logger(tmp_path):
    """Initialize LoggerBuilder as disabled for sample output tests."""
    LoggerBuilder.init(enabled=False, log_dir=str(tmp_path / "logs"))


# ---------------------------------------------------------------------------
# TC-F1 through TC-F8: Sample File Structure
# ---------------------------------------------------------------------------


class TestSampleFileStructure:
    """TC-F1–F8: Sample file structure validation.

    Validates the game → perspective → step hierarchy and all field
    constraints defined in design/7 interfaces.md §2.
    """

    def test_output_is_valid_json_array_with_correct_count(
        self, app_manager, tmp_path,
    ):
        """TC-F1: File exists; top-level is JSON array with N elements."""
        sample_path = tmp_path / "samples.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        assert sample_path.exists()
        games = read_sample_file(sample_path)
        assert len(games) == 3

    def test_each_game_has_required_top_level_keys(
        self, app_manager, tmp_path,
    ):
        """TC-F2: Each game has keys game_id, winner, rounds, perspectives."""
        sample_path = tmp_path / "keys.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        required = {"game_id", "winner", "rounds", "perspectives"}
        for game in games:
            assert set(game.keys()) == required, (
                f"Expected keys {required}, got {set(game.keys())}"
            )

    def test_game_ids_are_sequential_from_one(self, app_manager, tmp_path):
        """TC-F3: game_id values are [1, 2, 3, ...]."""
        sample_path = tmp_path / "ids.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        ids = [g["game_id"] for g in games]
        assert ids == [1, 2, 3]

    def test_each_game_has_exactly_two_perspectives(
        self, app_manager, tmp_path,
    ):
        """TC-F4: Every game's perspectives array has length 2."""
        sample_path = tmp_path / "perspectives.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            assert len(game["perspectives"]) == 2, (
                f"Game {game['game_id']}: expected 2 perspectives, "
                f"got {len(game['perspectives'])}"
            )

    def test_perspectives_have_correct_player_assignments(
        self, app_manager, tmp_path,
    ):
        """TC-F5: perspectives[0].player == "1", perspectives[1].player == "2"."""
        sample_path = tmp_path / "player_assign.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            assert game["perspectives"][0]["player"] == "1"
            assert game["perspectives"][1]["player"] == "2"

    def test_perspectives_contain_ai_names(self, app_manager, tmp_path):
        """TC-F6: Player "1" has ai=ai1, player "2" has ai=ai2."""
        sample_path = tmp_path / "ai_names.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            assert game["perspectives"][0]["ai"] == "rnn_v1.1"
            assert game["perspectives"][1]["ai"] == "rnn_v1.1"

    def test_each_step_has_required_keys(self, app_manager, tmp_path):
        """TC-F7: Every step has keys state, action, reward, value, log_prob."""
        sample_path = tmp_path / "step_keys.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        required = {"state", "action", "reward", "value", "log_prob"}
        for game in games:
            for p in game["perspectives"]:
                for step in p["steps"]:
                    assert set(step.keys()) == required, (
                        f"Expected step keys {required}, "
                        f"got {set(step.keys())}"
                    )

    def test_state_vector_length_is_14_in_all_steps(
        self, app_manager, tmp_path,
    ):
        """TC-F8: Every step's state array has exactly 14 elements."""
        sample_path = tmp_path / "state_len.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=1.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for p in game["perspectives"]:
                for step in p["steps"]:
                    assert len(step["state"]) == 14, (
                        f"Expected 14 elements, got {len(step['state'])}"
                    )


# ---------------------------------------------------------------------------
# TC-R1 through TC-R4: Reward Assignment
# ---------------------------------------------------------------------------


class TestSampleRewards:
    """TC-R1–R4: Reward assignment matches game outcome.

    Rewards are assigned by game outcome, not by step-level action.
    All steps within a perspective share the same reward.
    """

    def test_rewards_match_game_outcome(self, app_manager, tmp_path):
        """Winning perspective gets reward=1.0, losing gets -1.0."""
        sample_path = tmp_path / "rewards.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=5, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            winner = game["winner"]
            p1_steps = game["perspectives"][0]["steps"]
            p2_steps = game["perspectives"][1]["steps"]

            if winner == "1":
                for s in p1_steps:
                    assert s["reward"] == 1.0
                for s in p2_steps:
                    assert s["reward"] == -1.0
            elif winner == "2":
                for s in p1_steps:
                    assert s["reward"] == -1.0
                for s in p2_steps:
                    assert s["reward"] == 1.0
            else:  # draw
                for s in p1_steps:
                    assert s["reward"] == 0.0
                for s in p2_steps:
                    assert s["reward"] == 0.0

    def test_reward_consistent_within_perspective(
        self, app_manager, tmp_path,
    ):
        """TC-R4: All steps within a perspective share the same reward."""
        sample_path = tmp_path / "consistent.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for p in game["perspectives"]:
                rewards = {s["reward"] for s in p["steps"]}
                assert len(rewards) == 1, (
                    f"Game {game['game_id']}, player {p['player']}: "
                    f"expected single reward, got {rewards}"
                )

    def test_reward_values_are_valid(self, app_manager, tmp_path):
        """All reward values are 1.0, -1.0, or 0.0."""
        sample_path = tmp_path / "valid_rewards.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for p in game["perspectives"]:
                for step in p["steps"]:
                    assert step["reward"] in (1.0, -1.0, 0.0), (
                        f"Invalid reward: {step['reward']}"
                    )


# ---------------------------------------------------------------------------
# TC-S1 through TC-S7: Step Content
# ---------------------------------------------------------------------------


class TestSampleContent:
    """TC-S1–S7: Step-level content validation.

    Validates step counts, action validity, log_prob behavior,
    value range, and round count accuracy.
    """

    def test_total_step_count_matches_rounds_times_two(
        self, app_manager, tmp_path,
    ):
        """TC-S1: Total steps = rounds_played × 2."""
        sample_path = tmp_path / "step_count.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            total_steps = sum(
                len(p["steps"]) for p in game["perspectives"]
            )
            expected = game["rounds"] * 2
            assert total_steps == expected, (
                f"Game {game['game_id']}: expected {expected} steps, "
                f"got {total_steps}"
            )

    def test_player1_actions_are_valid(self, app_manager, tmp_path):
        """TC-S2: P1 actions are from valid card set."""
        sample_path = tmp_path / "p1_actions.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=1.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        valid_actions = {"attack", "defense", "heavy", "counter"}
        games = read_sample_file(sample_path)
        for game in games:
            for step in game["perspectives"][0]["steps"]:
                assert step["action"] in valid_actions, (
                    f"Invalid P1 action: {step['action']}"
                )

    def test_player2_actions_are_valid(self, app_manager, tmp_path):
        """TC-S3: P2 actions are from valid card set."""
        sample_path = tmp_path / "p2_actions.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=1.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        valid_actions = {"attack", "defense", "heavy", "counter"}
        games = read_sample_file(sample_path)
        for game in games:
            for step in game["perspectives"][1]["steps"]:
                assert step["action"] in valid_actions, (
                    f"Invalid P2 action: {step['action']}"
                )

    def test_greedy_steps_have_null_log_prob(self, app_manager, tmp_path):
        """TC-S4: log_prob is null (None) when temperature=0."""
        sample_path = tmp_path / "greedy.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for p in game["perspectives"]:
                for step in p["steps"]:
                    assert step["log_prob"] is None, (
                        "Expected null log_prob for greedy step"
                    )

    def test_sampled_steps_have_float_log_prob(self, app_manager, tmp_path):
        """TC-S5: log_prob is a finite float when temperature>0 and both sides."""
        sample_path = tmp_path / "sampled.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=1.0,
            temp_step=None, temp_side="both",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for p in game["perspectives"]:
                for step in p["steps"]:
                    assert isinstance(step["log_prob"], float), (
                        f"Expected float log_prob, "
                        f"got {type(step['log_prob']).__name__}"
                    )
                    assert step["log_prob"] <= 0, (
                        f"log_prob should be ≤ 0, got {step['log_prob']}"
                    )

    def test_value_in_range(self, app_manager, tmp_path):
        """TC-S6: Every step's value satisfies -1.0 ≤ value ≤ 1.0."""
        sample_path = tmp_path / "value_range.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=1.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for p in game["perspectives"]:
                for step in p["steps"]:
                    assert -1.0 <= step["value"] <= 1.0, (
                        f"Value {step['value']} out of [-1.0, 1.0]"
                    )

    def test_round_count_is_accurate(self, app_manager, tmp_path):
        """TC-S7: Game's rounds field equals actual rounds played."""
        sample_path = tmp_path / "rounds.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            p1_step_count = len(game["perspectives"][0]["steps"])
            p2_step_count = len(game["perspectives"][1]["steps"])
            # Each player has one step per round
            assert game["rounds"] == p1_step_count, (
                f"Game {game['game_id']}: rounds={game['rounds']}, "
                f"P1 steps={p1_step_count}"
            )
            assert game["rounds"] == p2_step_count, (
                f"Game {game['game_id']}: rounds={game['rounds']}, "
                f"P2 steps={p2_step_count}"
            )
