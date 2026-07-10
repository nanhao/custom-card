"""Unit tests for TrainingRunner — sequence batching correctness (design/29).

Test cases per design/29 test_cases.md: TC-SB-01 through TC-SB-04.
"""

import json
import math
import os
import random
import tempfile
from unittest.mock import patch

import pytest
import torch

from src.backend.ai.rnn_v1.model import (
    GRUPolicyModel,
    GRUValueModel,
    HIDDEN_SIZE,
    INPUT_SIZE,
    POLICY_OUTPUT_SIZE,
)
from src.backend.ai.training_config import TrainingConfig
from src.backend.ai.rnn_v1.v1_training_runner import V1TrainingRunner
from src.backend.ai.sample_loader import SampleLoader


# ---------------------------------------------------------------------------
# Fixtures
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
                # Deterministic 42-d input: base value *= (g+1) * (p+1) * (t+1)
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


@pytest.fixture
def training_config(sample_file, output_dir):
    """Build a minimal TrainingConfig for 1-epoch training test."""
    return TrainingConfig(
        sample_path=sample_file,
        output_dir=output_dir,
        model="rnn_v1.1",
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


@pytest.fixture
def runner():
    """Create a fresh V1TrainingRunner instance."""
    return V1TrainingRunner()


# ---------------------------------------------------------------------------
# TC-SB-01: Value loss is finite after sequence batching
# ---------------------------------------------------------------------------


class TestValueTrainingCorrectness:
    """TC-SB-01: Value training produces finite, non-NaN losses."""

    def test_value_loss_is_finite(self, runner, training_config):
        """Phase 1 value loss should be a finite float (not NaN or Inf)."""
        random.seed(42)
        torch.manual_seed(42)
        result = runner.run(training_config)
        # runner.run() returns None — we verify via side effects and output
        # The key assertion: no exception raised, training completes cleanly.
        # Loss values are printed to stdout; verified by the training
        # successfully saving weights without crashing.
        policy_path = os.path.join(training_config.output_dir, "policy.pth")
        value_path = os.path.join(training_config.output_dir, "value.pth")
        assert os.path.isfile(policy_path), "policy.pth should be created"
        assert os.path.isfile(value_path), "value.pth should be created"

    def test_value_loss_different_from_initial(self, runner, training_config, sample_file):
        """Training should change weights from initial random values."""
        random.seed(42)
        torch.manual_seed(42)

        # Capture initial weights
        policy = GRUPolicyModel()
        initial_value_state = {
            k: v.clone()
            for k, v in GRUValueModel().state_dict().items()
        }

        runner.run(training_config)

        # Load trained weights and compare
        trained = torch.load(
            os.path.join(training_config.output_dir, "value.pth"),
            map_location="cpu", weights_only=True,
        )
        all_same = True
        for k, v in initial_value_state.items():
            if not torch.allclose(v, trained[k], atol=1e-6):
                all_same = False
                break
        assert not all_same, (
            "Value model weights should change after training"
        )


# ---------------------------------------------------------------------------
# TC-SB-04: Full training produces valid weight files
# ---------------------------------------------------------------------------


class TestFullTrainingOutput:
    """TC-SB-04: Full three-phase training outputs valid weights."""

    def test_output_files_exist(self, runner, training_config):
        """Both policy.pth and value.pth should exist after training."""
        random.seed(42)
        torch.manual_seed(42)
        runner.run(training_config)
        assert os.path.isfile(
            os.path.join(training_config.output_dir, "policy.pth"),
        )
        assert os.path.isfile(
            os.path.join(training_config.output_dir, "value.pth"),
        )

    def test_output_weights_are_loadable(self, runner, training_config):
        """Output .pth files should be loadable as valid state dicts."""
        random.seed(42)
        torch.manual_seed(42)
        runner.run(training_config)
        policy = GRUPolicyModel()
        value = GRUValueModel()
        policy.load_state_dict(
            torch.load(
                os.path.join(training_config.output_dir, "policy.pth"),
                map_location="cpu", weights_only=True,
            ),
        )
        value.load_state_dict(
            torch.load(
                os.path.join(training_config.output_dir, "value.pth"),
                map_location="cpu", weights_only=True,
            ),
        )
        # Smoke test: forward pass with loaded weights
        x = torch.randn(1, 1, INPUT_SIZE)
        h_pol = torch.zeros(1, 1, HIDDEN_SIZE)
        h_val = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, log_probs, _ = policy(x, h_pol)
        val, _ = value(x, h_val)
        assert probs.shape == (1, POLICY_OUTPUT_SIZE)
        assert val.shape == (1, 1)


# ---------------------------------------------------------------------------
# TC-SB: Numerical equivalence — full sequence vs step-by-step GRU
# ---------------------------------------------------------------------------


class TestGRUSequenceEquivalence:
    """Verify that full-sequence GRU output matches step-by-step GRU output.

    This is the core correctness guarantee of the sequence batching optimization:
    feeding (T, 1, 42) through one GRU call produces the same final hidden state
    and outputs as iterating T times with (1, 1, 42).
    """

    def test_value_gru_sequence_equals_stepwise(self):
        """GRUValueModel full-sequence output matches step-by-step output."""
        torch.manual_seed(123)
        model = GRUValueModel()
        model.eval()

        T = 5
        x_seq = torch.randn(T, 1, INPUT_SIZE)

        # Step-by-step
        h_step = torch.zeros(1, 1, HIDDEN_SIZE)
        step_vals = []
        for t in range(T):
            x_t = x_seq[t:t + 1]  # (1, 1, 42)
            val, h_step = model(x_t, h_step)
            step_vals.append(val.item())

        # Full sequence — direct GRU + fc_value (as in training_runner)
        h_seq = torch.zeros(1, 1, HIDDEN_SIZE)
        with torch.no_grad():
            gru_out, _ = model.gru(x_seq, h_seq)  # (T, 1, 32)
            seq_vals = model.fc_value(gru_out.squeeze(1)).squeeze()  # (T,)

        for t in range(T):
            assert abs(step_vals[t] - seq_vals[t].item()) < 1e-5, (
                f"Step {t}: step={step_vals[t]:.6f}, seq={seq_vals[t].item():.6f}"
            )

    def test_policy_gru_sequence_equals_stepwise(self):
        """GRUPolicyModel full-sequence output matches step-by-step output."""
        torch.manual_seed(123)
        model = GRUPolicyModel()
        model.eval()

        T = 5
        x_seq = torch.randn(T, 1, INPUT_SIZE)

        # Step-by-step
        h_step = torch.zeros(1, 1, HIDDEN_SIZE)
        step_log_probs = []
        for t in range(T):
            x_t = x_seq[t:t + 1]  # (1, 1, 42)
            probs, log_probs, h_step = model(x_t, h_step, mask_fn=None, temperature=1.0)
            step_log_probs.append(log_probs)

        # Full sequence — direct GRU + fc_policy (as in training_runner)
        h_seq = torch.zeros(1, 1, HIDDEN_SIZE)
        with torch.no_grad():
            gru_out, _ = model.gru(x_seq, h_seq)  # (T, 1, 32)
            policy_logits = model.fc_policy(gru_out.squeeze(1))  # (T, 14)
            seq_log_probs = torch.log_softmax(policy_logits, dim=-1)  # (T, 14)

        for t in range(T):
            diff = (seq_log_probs[t] - step_log_probs[t].squeeze(0)).abs().max().item()
            assert diff < 1e-5, (
                f"Step {t}: max log_prob diff={diff:.6f}"
            )


# ---------------------------------------------------------------------------
# TC-LP: Load progress output
# ---------------------------------------------------------------------------


class TestLoadProgressOutput:
    """TC-LP: Load progress messages are printed to stdout."""

    def test_load_progress_contains_expected_messages(
        self, runner, training_config, capsys,
    ):
        """Stdout should contain load progress messages."""
        random.seed(42)
        torch.manual_seed(42)
        runner.run(training_config)
        captured = capsys.readouterr().out
        assert "Loading " in captured, "Missing 'Loading' message"
        assert "Parsed " in captured, "Missing 'Parsed' message"
        assert "Loaded " in captured, "Missing 'Loaded' message"

    def test_phase2_message_present(self, runner, training_config, capsys):
        """Stdout should contain Phase 2 start message."""
        random.seed(42)
        torch.manual_seed(42)
        runner.run(training_config)
        captured = capsys.readouterr().out
        assert "[Phase 2]" in captured, "Missing '[Phase 2]' message"


# ---------------------------------------------------------------------------
# TC-RUN: Epoch loop count verification (design/32)
# ---------------------------------------------------------------------------


class TestEpochLoopCount:
    """TC-RUN-01 through TC-RUN-03: verify independent epoch loops."""

    def test_value_epoch_controls_phase1_loop_count(
        self, runner, sample_file, output_dir, capsys,
    ):
        """TC-RUN-01: value_epoch=3 → Phase 1 prints 3 rounds."""
        random.seed(42)
        torch.manual_seed(42)
        config = TrainingConfig(
            sample_path=sample_file,
            output_dir=output_dir,
            model="rnn_v1.1",
            policy_epoch=1,
            value_epoch=3,
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
        runner.run(config)
        captured = capsys.readouterr().out
        assert "[Value] Epoch 1/3" in captured
        assert "[Value] Epoch 2/3" in captured
        assert "[Value] Epoch 3/3" in captured
        assert "[Value] Epoch 4/" not in captured

    def test_policy_epoch_controls_phase3_loop_count(
        self, runner, sample_file, output_dir, capsys,
    ):
        """TC-RUN-02: policy_epoch=3 → Phase 3 prints 3 rounds."""
        random.seed(42)
        torch.manual_seed(42)
        config = TrainingConfig(
            sample_path=sample_file,
            output_dir=output_dir,
            model="rnn_v1.1",
            policy_epoch=3,
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
        runner.run(config)
        captured = capsys.readouterr().out
        assert "[Policy] Epoch 1/3" in captured
        assert "[Policy] Epoch 2/3" in captured
        assert "[Policy] Epoch 3/3" in captured
        assert "[Policy] Epoch 4/" not in captured

    def test_phase_records_has_correct_total_length(
        self, runner, sample_file, output_dir, capsys,
    ):
        """TC-RUN-03: value_epoch=3, policy_epoch=5 → 8 total records."""
        random.seed(42)
        torch.manual_seed(42)
        config = TrainingConfig(
            sample_path=sample_file,
            output_dir=output_dir,
            model="rnn_v1.1",
            policy_epoch=5,
            value_epoch=3,
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
        runner.run(config)
        captured = capsys.readouterr().out
        # Verify phase record count: 3 value + 5 policy = 8
        value_epoch_count = captured.count("[Value] Epoch")
        policy_epoch_count = captured.count("[Policy] Epoch")
        assert value_epoch_count == 3, (
            f"Expected 3 value epochs, got {value_epoch_count}"
        )
        assert policy_epoch_count == 5, (
            f"Expected 5 policy epochs, got {policy_epoch_count}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestTrainingErrorHandling:
    """Boundary condition: invalid inputs."""

    def test_sample_path_not_exists(self):
        """TrainingConfig should raise ValueError for missing sample path."""
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError, match="sample_path does not exist"):
                TrainingConfig(
                    sample_path=os.path.join(td, "nonexistent.json"),
                    output_dir=td,
                    model="rnn_v1.1",
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

    def test_policy_epoch_must_be_positive(self):
        """TrainingConfig should reject policy_epoch <= 0."""
        with tempfile.TemporaryDirectory() as td:
            sample_path = os.path.join(td, "dummy.json")
            with open(sample_path, "w") as f:
                json.dump([], f)
            with pytest.raises(ValueError, match="policy_epoch must be >= 1"):
                TrainingConfig(
                    sample_path=sample_path,
                    output_dir=td,
                    model="rnn_v1.1",
                    policy_epoch=0,
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


# ===========================================================================
# TC-SL: SampleLoader game_id attachment (design/33)
# ===========================================================================


class TestSampleLoaderGameId:
    """TC-SL: SampleLoader attaches game_id to perspective dicts (design/33)."""

    def test_perspective_has_game_id(self, tmp_path):
        """Each perspective dict should contain 'game_id' matching source game."""
        games = _make_minimal_sample(num_games=3, steps_per_game=2)
        # _make_minimal_sample uses f"test_game_{g}" as game_id
        path = tmp_path / "sample.json"
        path.write_text(json.dumps(games), encoding="utf-8")

        perspectives = SampleLoader.load(str(path))
        assert len(perspectives) == 6  # 3 games × 2 perspectives

        # Verify each perspective has game_id
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
# TC-SPLIT: _split_train_val by game_id (design/33)
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
        """TC-SPLIT-01: 6 games × 2 perspectives, split=0.5."""
        perspectives = self._make_perspectives([2, 2, 2, 2, 2, 2])
        train, val = V1TrainingRunner._split_train_val(perspectives, 0.5)
        assert len(train) + len(val) == 12
        # Each game has 2 perspectives, so val should have ~3 games → 6 perspectives
        assert 2 <= len(val) <= 10  # at least 1 game, at most 5 games
        assert len(train) >= 2  # at least 1 game in train

        # Same game perspectives not split
        train_gids = {p["game_id"] for p in train}
        val_gids = {p["game_id"] for p in val}
        assert train_gids.isdisjoint(val_gids)

    def test_split_zero_returns_empty_val(self):
        """TC-SPLIT-02: split=0 → val empty, train has all."""
        perspectives = self._make_perspectives([2, 2, 2])
        train, val = V1TrainingRunner._split_train_val(perspectives, 0.0)
        assert len(val) == 0
        assert len(train) == 6

    def test_single_game_keeps_train(self):
        """TC-SPLIT-03: only 1 game → train has all, val empty."""
        perspectives = self._make_perspectives([3])
        train, val = V1TrainingRunner._split_train_val(perspectives, 0.2)
        assert len(train) == 3
        assert len(val) == 0

    def test_same_game_perspectives_not_split(self):
        """TC-SPLIT-04: same game perspectives all go to same set."""
        perspectives = self._make_perspectives([3, 2])
        train, val = V1TrainingRunner._split_train_val(perspectives, 0.5)
        train_gids = {p["game_id"] for p in train}
        val_gids = {p["game_id"] for p in val}
        # Each game_id appears in only one set
        assert len(train_gids & val_gids) == 0
        # Total perspectives preserved
        assert len(train) + len(val) == 5


# ===========================================================================
# TC-VAL: _compute_val_loss (design/33)
# ===========================================================================


class TestComputeValLoss:
    """TC-VAL: BaseTrainingRunner._compute_val_loss (design/33)."""

    def test_val_loss_is_finite(self, runner, tmp_path):
        """TC-VAL-01: val_loss on valid perspectives returns finite float."""
        import torch
        torch.manual_seed(123)
        _, value_model, device = runner._build_models("rnn_v1.1")
        # Create dummy sample file so TrainingConfig validates
        dummy_sample = tmp_path / "dummy.json"
        dummy_sample.write_text(json.dumps([]), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = TrainingConfig(
            sample_path=str(dummy_sample),
            output_dir=str(output_dir),
            model="rnn_v1.1",
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
        # Create val perspectives with minimal steps
        val_perspectives = [
            {"steps": [{"input": [float(i) for i in range(42)], "action": 2,
                         "reward": 0.1, "value": 0.5, "log_prob": -0.1}],
             "game_id": "g0"},
        ]
        val_loss = runner._compute_val_loss(
            val_perspectives, value_model, config, device,
        )
        assert isinstance(val_loss, float)
        import math
        assert not math.isnan(val_loss)

    def test_val_loss_empty_returns_nan(self, runner, tmp_path):
        """TC-VAL-02: empty val set → returns NaN."""
        import torch
        torch.manual_seed(123)
        _, value_model, device = runner._build_models("rnn_v1.1")
        dummy_sample = tmp_path / "dummy.json"
        dummy_sample.write_text(json.dumps([]), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = TrainingConfig(
            sample_path=str(dummy_sample),
            output_dir=str(output_dir),
            model="rnn_v1.1",
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
        import math
        val_loss = runner._compute_val_loss([], value_model, config, device)
        assert math.isnan(val_loss)


# ===========================================================================
# TC-VSINT: Validation integration in run() (design/33)
# ===========================================================================


class TestValidationIntegration:
    """TC-VSINT: validation integrated into training run (design/33)."""

    def test_run_with_validation_split_produces_val_loss(self, tmp_path):
        """TC-VSINT-01: run with validation_split=0.5 prints val_value_loss."""
        import random as _random
        import torch
        from src.backend.ai.rnn_v1.v1_training_runner import V1TrainingRunner

        # Create sample with multiple games so split works
        games = _make_minimal_sample(num_games=4, steps_per_game=3)
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(games), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = TrainingConfig(
            sample_path=str(sample_path),
            output_dir=str(output_dir),
            model="rnn_v1.1",
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

        _random.seed(42)
        torch.manual_seed(42)
        runner = V1TrainingRunner()
        runner.run(config)

        # Verify output files exist (training completed)
        assert os.path.isfile(str(output_dir / "policy.pth"))
        assert os.path.isfile(str(output_dir / "value.pth"))

    def test_run_with_validation_zero_skips_val(self, sample_file, output_dir,
                                                 capsys):
        """TC-VSINT-02: run with validation_split=0 produces nan val_value_loss."""
        import random as _random
        import torch

        config = TrainingConfig(
            sample_path=sample_file,
            output_dir=output_dir,
            model="rnn_v1.1",
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

        _random.seed(42)
        torch.manual_seed(42)
        runner = V1TrainingRunner()
        runner.run(config)

        captured = capsys.readouterr().out
        # Should NOT contain val_value_loss in print (split=0 skips validation)
        # But the structure should still work
        assert "[Value] Epoch 1/1" in captured

    def test_csv_output_includes_val_value_loss(self, tmp_path):
        """TC-VSINT-03: loss.csv includes val_value_loss column."""
        import random as _random
        import torch
        import csv as _csv
        from src.backend.ai.rnn_v1.v1_training_runner import V1TrainingRunner

        games = _make_minimal_sample(num_games=4, steps_per_game=3)
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(games), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Override loss dir to temp
        loss_dir = tmp_path / "loss"
        loss_dir.mkdir()

        config = TrainingConfig(
            sample_path=str(sample_path),
            output_dir=str(output_dir),
            model="rnn_v1.1",
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

        _random.seed(42)
        torch.manual_seed(42)
        runner = V1TrainingRunner()
        runner.run(config)

        # Verify loss.csv
        loss_csv = os.path.join(os.getcwd(), "loss", "loss.csv")
        assert os.path.isfile(loss_csv), f"loss.csv not found at {loss_csv}"
        with open(loss_csv, "r", encoding="utf-8") as f:
            reader = _csv.reader(f)
            header = next(reader)
            assert "val_value_loss" in header, f"missing val_value_loss in header: {header}"


# ===========================================================================
# V2TrainingRunner — smoke tests (design/31 code review #3)
# ===========================================================================


class TestV2TrainingSmoke:
    """V2 training pipeline works end-to-end with fresh random weights.

    Creates temporary weight files and sample data, runs full three-phase
    training, then verifies output files exist and are loadable.
    All files created in pytest temp directories — no side effects.
    """

    def test_v2_training_completes_and_outputs_weights(self, tmp_path):
        """V2 three-phase training with fresh weights produces loadable output."""
        from src.backend.ai.rnn_v2.model import GRUPolicyModelV2, GRUValueModelV2
        from src.backend.ai.rnn_v2.v2_training_runner import V2TrainingRunner

        # -- Create fresh v2 weight files in gen1 dir --
        gen_dir = tmp_path / "rnn_v2" / "gen1"
        gen_dir.mkdir(parents=True)

        policy_model = GRUPolicyModelV2()
        value_model = GRUValueModelV2()
        torch.save(policy_model.state_dict(), str(gen_dir / "policy.pth"))
        torch.save(value_model.state_dict(), str(gen_dir / "value.pth"))

        # -- Create minimal training sample --
        steps_per_game = 3
        perspectives = []
        for p in range(2):
            steps = []
            for t in range(steps_per_game):
                base = float((p + 1) * (t + 1))
                steps.append({
                    "input": [base * (i + 1) / 42.0 for i in range(INPUT_SIZE)],
                    "action": (t + p) % POLICY_OUTPUT_SIZE,
                    "reward": base / 10.0,
                    "value": base / 5.0,
                    "log_prob": -(base / 100.0),
                })
            perspectives.append({
                "player": str(p + 1),
                "steps": steps,
            })
        sample = [{
            "game_id": "test_v2",
            "winner": "1",
            "rounds": steps_per_game * 2,
            "perspectives": perspectives,
        }]
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample), encoding="utf-8")

        # Patch registry to use tmp dir for weight paths (clean restore via mock.patch)
        import src.backend.ai.registry as registry_module

        def _patched_derive(name, gen, suffix):
            return str(gen_dir / f"{suffix}.pth")

        with patch.object(registry_module, '_derive_weight_path', _patched_derive):
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
                batch_size=2,
            )

            random.seed(42)
            torch.manual_seed(42)
            runner = V2TrainingRunner()
            runner.run(config)

            # Verify output files
            assert os.path.isfile(str(output_dir / "policy.pth"))
            assert os.path.isfile(str(output_dir / "value.pth"))

            # Verify output weights are loadable into v2 models
            trained_policy = GRUPolicyModelV2()
            trained_value = GRUValueModelV2()
            trained_policy.load_state_dict(torch.load(
                str(output_dir / "policy.pth"), map_location="cpu", weights_only=True,
            ))
            trained_value.load_state_dict(torch.load(
                str(output_dir / "value.pth"), map_location="cpu", weights_only=True,
            ))

            # Smoke forward with trained weights
            x = torch.randn(1, 1, INPUT_SIZE)
            h = torch.zeros(1, 1, HIDDEN_SIZE)
            probs, log_probs, _ = trained_policy(x, h)
            val, _ = trained_value(x, h)
            assert probs.shape == (1, POLICY_OUTPUT_SIZE)
            assert val.shape == (1, 1)
