"""Unit tests for validation-set mechanism — value network training (design/33).

Test cases per design/33 test_cases.md: TC-SL, TC-SPLIT, TC-VAL, TC-VSINT.
"""

import csv as _csv
import json
import math
import os
import random
import tempfile

import pytest
import torch

from src.backend.ai.rnn_v2.model import INPUT_SIZE, POLICY_OUTPUT_SIZE
from src.backend.training.rnn_v2.v2_training_runner import V2TrainingRunner
from src.backend.training.rnn_v2.sample_loader import SampleLoader
from src.backend.training.training_config import TrainingConfig
from src.backend.training.training_runner import BaseTrainingRunner


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_minimal_sample(num_games: int = 1, steps_per_game: int = 3) -> list[dict]:
    """Build minimal valid training sample JSON with deterministic data.

    Each game has 2 perspectives, each with steps_per_game steps.
    Input vectors are scaled monotonic floats for deterministic behavior.
    """
    games = []
    for g in range(num_games):
        perspectives = []
        for p in range(2):
            steps = []
            for t in range(steps_per_game):
                base = float((g + 1) * (p + 1) * (t + 1))
                input_vec = [base * (i + 1) / 42.0 for i in range(INPUT_SIZE)]
                steps.append({
                    "input": input_vec,
                    "action": (t + p) % POLICY_OUTPUT_SIZE,
                    "reward": base / 10.0,
                    "value": base / 5.0,
                    "log_prob": -(base / 100.0),
                })
            perspectives.append({
                "player": str(p + 1),
                "steps": steps,
            })
        games.append({
            "game_id": f"test_game_{g}",
            "winner": "1",
            "rounds": steps_per_game * 2,
            "perspectives": perspectives,
        })
    return games


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Create a fresh V2TrainingRunner instance."""
    return V2TrainingRunner()


@pytest.fixture
def sample_file(tmp_path):
    """Write a minimal sample file to a pytest temp directory and return the path."""
    games = _make_minimal_sample(num_games=2, steps_per_game=3)
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(games), encoding="utf-8")
    return str(path)


@pytest.fixture
def output_dir():
    """Create a temp output directory for training weights."""
    with tempfile.TemporaryDirectory() as td:
        yield td


# ===========================================================================
# TC-SL: SampleLoader game_id attachment
# ===========================================================================


class TestSampleLoaderGameId:
    """TC-SL: SampleLoader attaches game_id to perspective dicts (design/33)."""

    def test_perspective_has_game_id(self, tmp_path):
        """Each perspective dict should contain 'game_id' matching source game."""
        games = _make_minimal_sample(num_games=3, steps_per_game=2)
        path = tmp_path / "sample.json"
        path.write_text(json.dumps(games), encoding="utf-8")

        perspectives = SampleLoader.load(str(path))
        assert len(perspectives) == 6  # 3 games x 2 perspectives

        for p in perspectives:
            assert "game_id" in p, f"perspective missing 'game_id': {p}"
            assert p["game_id"] in ("test_game_0", "test_game_1", "test_game_2")

    def test_perspective_still_has_steps(self, tmp_path):
        """Perspective dict should still contain 'steps' key after adding game_id."""
        games = _make_minimal_sample(num_games=1, steps_per_game=3)
        path = tmp_path / "sample.json"
        path.write_text(json.dumps(games), encoding="utf-8")

        perspectives = SampleLoader.load(str(path))
        assert len(perspectives) == 2
        for p in perspectives:
            assert "steps" in p
            assert len(p["steps"]) == 3


# ===========================================================================
# TC-SPLIT: _split_train_val by game_id
# ===========================================================================


class TestSplitTrainVal:
    """TC-SPLIT: BaseTrainingRunner._split_train_val (design/33)."""

    @staticmethod
    def _make_perspectives(game_perspective_counts: list[int]) -> list[dict]:
        """Create perspectives grouped by game_id.

        game_perspective_counts[i] = number of perspectives for game i.
        """
        perspectives: list[dict] = []
        for g, count in enumerate(game_perspective_counts):
            for p in range(count):
                perspectives.append({
                    "steps": [{"input": [0.0] * 42, "action": 0,
                               "reward": 0.0, "value": 0.0, "log_prob": 0.0}],
                    "game_id": f"game_{g}",
                })
        return perspectives

    def test_normal_split_half(self):
        """TC-SPLIT-01: 6 games x 2 perspectives, split=0.5."""
        perspectives = self._make_perspectives([2, 2, 2, 2, 2, 2])
        train, val = BaseTrainingRunner._split_train_val(perspectives, 0.5)
        assert len(train) + len(val) == 12
        assert 2 <= len(val) <= 10  # at least 1 game, at most 5 games
        assert len(train) >= 2  # at least 1 game in train

        train_gids = {p["game_id"] for p in train}
        val_gids = {p["game_id"] for p in val}
        assert train_gids.isdisjoint(val_gids)

    def test_split_zero_returns_empty_val(self):
        """TC-SPLIT-02: split=0 -> val empty, train has all."""
        perspectives = self._make_perspectives([2, 2, 2])
        train, val = BaseTrainingRunner._split_train_val(perspectives, 0.0)
        assert len(val) == 0
        assert len(train) == 6

    def test_single_game_keeps_train(self):
        """TC-SPLIT-03: only 1 game -> train has all, val empty."""
        perspectives = self._make_perspectives([3])
        train, val = BaseTrainingRunner._split_train_val(perspectives, 0.2)
        assert len(train) == 3
        assert len(val) == 0

    def test_same_game_perspectives_not_split(self):
        """TC-SPLIT-04: same game perspectives all go to same set."""
        perspectives = self._make_perspectives([3, 2])
        train, val = BaseTrainingRunner._split_train_val(perspectives, 0.5)
        train_gids = {p["game_id"] for p in train}
        val_gids = {p["game_id"] for p in val}
        assert len(train_gids & val_gids) == 0
        assert len(train) + len(val) == 5


# ===========================================================================
# TC-VAL: _compute_val_loss
# ===========================================================================


class TestComputeValLoss:
    """TC-VAL: BaseTrainingRunner._compute_val_loss (design/33)."""

    def test_val_loss_is_finite(self, runner, tmp_path):
        """TC-VAL-01: val_loss on valid perspectives returns finite float."""
        torch.manual_seed(123)
        _, value_model, device = runner._build_models("rnn_v2.1")
        dummy_sample = tmp_path / "dummy.json"
        dummy_sample.write_text(json.dumps([]), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = TrainingConfig(
            sample_path=str(dummy_sample),
            output_dir=str(output_dir),
            model="rnn_v2.1",
            policy_epoch=1,
            value_epoch=1,
            policy_lr=0.001,
            value_lr=0.001,
            weight_decay=0.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            max_grad_norm=0.5,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=2,
        )
        val_perspectives = [
            {"steps": [{"input": [float(i) for i in range(INPUT_SIZE)], "action": 2,
                         "reward": 0.1, "value": 0.5, "log_prob": -0.1}],
             "game_id": "g0"},
        ]
        val_loss = runner._compute_val_loss(
            val_perspectives, value_model, config, device,
        )
        assert isinstance(val_loss, float)
        assert not math.isnan(val_loss)

    def test_val_loss_empty_returns_nan(self, runner, tmp_path):
        """TC-VAL-02: empty val set -> returns NaN."""
        torch.manual_seed(123)
        _, value_model, device = runner._build_models("rnn_v2.1")
        dummy_sample = tmp_path / "dummy.json"
        dummy_sample.write_text(json.dumps([]), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = TrainingConfig(
            sample_path=str(dummy_sample),
            output_dir=str(output_dir),
            model="rnn_v2.1",
            policy_epoch=1,
            value_epoch=1,
            policy_lr=0.001,
            value_lr=0.001,
            weight_decay=0.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            max_grad_norm=0.5,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=2,
        )
        val_loss = runner._compute_val_loss([], value_model, config, device)
        assert math.isnan(val_loss)

    def test_val_loss_all_none_returns_nan(self, runner, tmp_path):
        """TC-VAL-03: all perspectives return None loss -> returns NaN."""
        torch.manual_seed(123)
        _, value_model, device = runner._build_models("rnn_v2.1")
        dummy_sample = tmp_path / "dummy.json"
        dummy_sample.write_text(json.dumps([]), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = TrainingConfig(
            sample_path=str(dummy_sample),
            output_dir=str(output_dir),
            model="rnn_v2.1",
            policy_epoch=1,
            value_epoch=1,
            policy_lr=0.001,
            value_lr=0.001,
            weight_decay=0.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            max_grad_norm=0.5,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=2,
        )
        # Perspective with NaN values should cause _process_value_perspective
        # to return None (NaN guard)
        val_perspectives = [
            {"steps": [{"input": [float("nan")] * INPUT_SIZE, "action": 2,
                         "reward": float("nan"), "value": float("nan"),
                         "log_prob": 0.0}],
             "game_id": "g0"},
        ]
        val_loss = runner._compute_val_loss(
            val_perspectives, value_model, config, device,
        )
        assert math.isnan(val_loss)


# ===========================================================================
# TC-VSINT: Validation integration in run()
# ===========================================================================


class TestValidationIntegration:
    """TC-VSINT: validation integrated into training run (design/33)."""

    def test_run_with_validation_split_prints_val_loss(self, tmp_path, capsys):
        """TC-VSINT-01: run with validation_split=0.5 prints val_value_loss."""
        games = _make_minimal_sample(num_games=4, steps_per_game=3)
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(games), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = TrainingConfig(
            sample_path=str(sample_path),
            output_dir=str(output_dir),
            model="rnn_v2.1",
            policy_epoch=1,
            value_epoch=2,
            policy_lr=0.001,
            value_lr=0.001,
            weight_decay=0.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            max_grad_norm=0.5,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=4,
            validation_split=0.5,
        )

        random.seed(42)
        torch.manual_seed(42)
        runner = V2TrainingRunner()
        runner.run(config)

        captured = capsys.readouterr().out
        # Verify val_value_loss appears in Phase 1 output
        assert "val_value_loss:" in captured, (
            f"Expected 'val_value_loss:' in output, got:\n{captured}"
        )
        assert os.path.isfile(str(output_dir / "policy.pth"))
        assert os.path.isfile(str(output_dir / "value.pth"))

    def test_run_with_validation_zero_skips_val(self, sample_file, output_dir,
                                                 capsys):
        """TC-VSINT-02: run with validation_split=0 produces no val print."""
        config = TrainingConfig(
            sample_path=sample_file,
            output_dir=output_dir,
            model="rnn_v2.1",
            policy_epoch=1,
            value_epoch=1,
            policy_lr=0.001,
            value_lr=0.001,
            weight_decay=0.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            max_grad_norm=0.5,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=2,
            validation_split=0.0,
        )

        random.seed(42)
        torch.manual_seed(42)
        runner = V2TrainingRunner()
        runner.run(config)

        captured = capsys.readouterr().out
        assert "[Value] Epoch 1/1" in captured
        assert "val_value_loss:" not in captured

    def test_csv_output_includes_val_value_loss(self, tmp_path):
        """TC-VSINT-03: loss.csv includes val_value_loss column."""
        games = _make_minimal_sample(num_games=4, steps_per_game=3)
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(games), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = TrainingConfig(
            sample_path=str(sample_path),
            output_dir=str(output_dir),
            model="rnn_v2.1",
            policy_epoch=1,
            value_epoch=1,
            policy_lr=0.001,
            value_lr=0.001,
            weight_decay=0.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            max_grad_norm=0.5,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=4,
            validation_split=0.5,
        )

        random.seed(42)
        torch.manual_seed(42)
        runner = V2TrainingRunner()
        runner.run(config)

        loss_csv = os.path.join(os.getcwd(), "loss", "loss.csv")
        assert os.path.isfile(loss_csv), f"loss.csv not found at {loss_csv}"
        with open(loss_csv, "r", encoding="utf-8") as f:
            reader = _csv.reader(f)
            header = next(reader)
            assert "val_value_loss" in header, (
                f"missing val_value_loss in header: {header}"
            )
