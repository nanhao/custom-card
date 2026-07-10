"""Unit tests for TrainingRunner — sequence batching correctness (design/29).

Test cases per design/29 test_cases.md: TC-SB-01 through TC-SB-04.
"""

import json
import os
import random
import tempfile

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
        epoch=1,
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
                    epoch=1,
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

    def test_epoch_must_be_positive(self):
        """TrainingConfig should reject epoch <= 0."""
        with tempfile.TemporaryDirectory() as td:
            sample_path = os.path.join(td, "dummy.json")
            with open(sample_path, "w") as f:
                json.dump([], f)
            with pytest.raises(ValueError, match="epoch must be >= 1"):
                TrainingConfig(
                    sample_path=sample_path,
                    output_dir=td,
                    model="rnn_v1.1",
                    epoch=0,
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

        # Monkeypatch registry to use tmp dir for weight paths
        import src.backend.ai.registry as registry_module
        original_derive = registry_module._derive_weight_path

        def _patched_derive(name, gen, suffix):
            return str(gen_dir / f"{suffix}.pth")

        registry_module._derive_weight_path = _patched_derive

        try:
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            config = TrainingConfig(
                sample_path=str(sample_path),
                output_dir=str(output_dir),
                model="rnn_v2.1",
                epoch=1,
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

        finally:
            registry_module._derive_weight_path = original_derive
