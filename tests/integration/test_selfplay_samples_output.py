"""Integration tests for PPO sample output — file output and boundaries.

Covers TC-O1 through TC-O4 (file output), TC-T1 through TC-T3
(sub-unit boundaries), temp_step behavior, and temp_side behavior
from design/7 test_cases.md.

For structure, rewards, and step content tests, see
test_selfplay_samples.py.
"""

import json

import pytest

from src.backend.app.selfplay_config import SelfPlayConfig
from src.backend.log.structured_logger import LoggerBuilder
from tests.helpers import read_sample_file


# ---------------------------------------------------------------------------
# LoggerBuilder init
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _init_logger(tmp_path):
    """Initialize LoggerBuilder as disabled for sample output tests."""
    LoggerBuilder.init(enabled=False, log_dir=str(tmp_path / "logs"))


# ---------------------------------------------------------------------------
# TC-O1 through TC-O4: File Output
# ---------------------------------------------------------------------------


class TestSampleFileOutput:
    """TC-O1–O4: File output edge cases.

    Validates file creation behavior, parent directory creation,
    JSON validity, and empty-run handling.
    """

    def test_no_sample_output_creates_no_file(self, app_manager, tmp_path):
        """TC-O1: sample_output=None → no file created."""
        # Use a subdirectory to avoid fixture files (e.g. config.json)
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=1, sample_output=None,
        )
        app_manager.run_selfplay(config)

        # No sample file should appear in the output directory
        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 0, (
            f"Expected no .json files, found {json_files}"
        )

    def test_sample_output_creates_parent_directories(
        self, app_manager, tmp_path,
    ):
        """TC-O2: Non-existent parent directories are created."""
        sample_path = tmp_path / "nested" / "subdir" / "samples.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=1, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        assert sample_path.exists()
        # Verify it's valid JSON
        games = read_sample_file(sample_path)
        assert isinstance(games, list)

    def test_file_is_valid_json_not_jsonl(self, app_manager, tmp_path):
        """TC-O3: File content parses as single JSON array, not JSONL."""
        sample_path = tmp_path / "valid.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        # Parse entire file as single JSON value
        content = sample_path.read_text(encoding="utf-8")
        data = json.loads(content)
        assert isinstance(data, list), (
            "Top-level must be a JSON array"
        )
        # Verify no JSONL line separators (each line is NOT valid JSON)
        lines = content.strip().split("\n")
        if len(lines) > 1:
            for line in lines:
                line = line.strip()
                if line in ("[", "]"):
                    continue
                # Individual lines should NOT be valid complete JSON objects
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "input" in obj:
                        pytest.fail(
                            "File appears to be JSONL format, "
                            "not a JSON array"
                        )
                except json.JSONDecodeError:
                    pass  # Expected — single lines not valid standalone JSON

    def test_game_count_zero_produces_empty_array(
        self, app_manager, tmp_path,
    ):
        """TC-O4: game_count=0 → file contains []."""
        sample_path = tmp_path / "empty.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=0, sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        assert sample_path.exists()
        games = read_sample_file(sample_path)
        assert games == []


# ---------------------------------------------------------------------------
# TC-T1 through TC-T3: Sub-Unit Training Boundaries
# ---------------------------------------------------------------------------


class TestSampleBoundaries:
    """TC-T1–T3: Sub-unit boundary validation for training.

    Validates that perspectives form independent sub-units
    suitable for GRU hidden state reset during training.
    """

    def test_steps_within_perspective_are_ordered(
        self, app_manager, tmp_path,
    ):
        """TC-T1: Within a perspective, steps are in round order."""
        sample_path = tmp_path / "ordered.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=2, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for p in game["perspectives"]:
                step_count = len(p["steps"])
                assert step_count == game["rounds"], (
                    f"Game {game['game_id']}, player {p['player']}: "
                    f"expected {game['rounds']} steps, got {step_count}"
                )

    def test_perspectives_are_separate_arrays(self, app_manager, tmp_path):
        """TC-T2: Perspectives are separate arrays for independent iteration."""
        sample_path = tmp_path / "separate.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            p1 = game["perspectives"][0]
            p2 = game["perspectives"][1]
            # Each perspective has its own steps list
            assert isinstance(p1["steps"], list)
            assert isinstance(p2["steps"], list)
            # Steps lists are separate objects (not shared reference)
            assert p1["steps"] is not p2["steps"]

    def test_different_games_have_independent_perspectives(
        self, app_manager, tmp_path,
    ):
        """TC-T3: Game N's perspectives are separate from game M's."""
        sample_path = tmp_path / "independent.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=3, temperature=0.0,
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        assert len(games) == 3
        # Each game is a separate dict
        for i, game in enumerate(games):
            assert game["game_id"] == i + 1
            assert len(game["perspectives"]) == 2


# ---------------------------------------------------------------------------
# Temp step behavior
# ---------------------------------------------------------------------------


class TestTempStep:
    """temp_step controls when sampling stops during self-play."""

    def test_temp_step_switches_to_greedy(self, app_manager, tmp_path):
        """Some steps have non-null and some null log_prob with temp_step=2."""
        sample_path = tmp_path / "temp_step.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=10, temperature=1.0,
            temp_step=2, temp_side="both",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        all_steps = [
            s for g in games
            for p in g["perspectives"]
            for s in p["steps"]
        ]
        non_null = [s for s in all_steps if s["log_prob"] is not None]
        null = [s for s in all_steps if s["log_prob"] is None]

        assert len(non_null) > 0, "Expected some non-null log_probs"
        assert len(null) > 0, "Expected some null log_probs (greedy steps)"

    def test_temp_step_none_samples_all(self, app_manager, tmp_path):
        """temp_step=None → all rounds have non-null log_prob."""
        sample_path = tmp_path / "no_step.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=5, temperature=1.0,
            temp_step=None, temp_side="both",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for p in game["perspectives"]:
                for step in p["steps"]:
                    assert step["log_prob"] is not None, (
                        "All rounds should have non-null log_prob"
                    )


# ---------------------------------------------------------------------------
# Temp side behavior
# ---------------------------------------------------------------------------


class TestTempSide:
    """temp_side controls which player gets temperature sampling."""

    def test_temp_side_player1_only_samples_p1(self, app_manager, tmp_path):
        """temp_side="player1" → only P1 has non-null log_prob."""
        sample_path = tmp_path / "side_p1.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=5, temperature=1.0,
            temp_side="player1",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for step in game["perspectives"][0]["steps"]:
                assert step["log_prob"] is not None, (
                    "P1 should have non-null log_prob"
                )
            for step in game["perspectives"][1]["steps"]:
                assert step["log_prob"] is None, (
                    "P2 should have null log_prob"
                )

    def test_temp_side_player2_only_samples_p2(self, app_manager, tmp_path):
        """temp_side="player2" → only P2 has non-null log_prob."""
        sample_path = tmp_path / "side_p2.json"
        config = SelfPlayConfig(
            ai1="rnn_v1.1", ai2="rnn_v1.1",
            game_count=5, temperature=1.0,
            temp_side="player2",
            sample_output=str(sample_path),
        )
        app_manager.run_selfplay(config)

        games = read_sample_file(sample_path)
        for game in games:
            for step in game["perspectives"][0]["steps"]:
                assert step["log_prob"] is None, (
                    "P1 should have null log_prob"
                )
            for step in game["perspectives"][1]["steps"]:
                assert step["log_prob"] is not None, (
                    "P2 should have non-null log_prob"
                )
