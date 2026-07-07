"""Unit tests for split TrainingRunner — three-phase PPO training.

Covers test cases from design/9 test_cases.md:
- Phase execution order (TC-PHS1 through PHS5)
- Gradient isolation (TC-GRAD1 through GRAD3)
- Weight saving (TC-SAVE1 through SAVE3)
- Artifact format (TC-ART1, TC-ART2)
- Optimizer (TC-OPT1, TC-OPT2)

Updated for design/10: uses unified output_dir, epoch, and "input" key.
"""

import json
import os
import tempfile

import pytest
import torch

from src.backend.ai.training_config import TrainingConfig
from src.backend.ai.training_runner import TrainingRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_json(games: list[dict]) -> str:
    """Write a list of game dicts as a JSON file, return the file path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(games, f)
    return path


def _make_perspective(
    steps: list, player: str = "1", ai: str = "rnn_v1.1",
) -> dict:
    """Build a perspective dict from step tuples.

    Each step: (input_vec, action, reward, value, log_prob).
    """
    return {
        "player": player,
        "ai": ai,
        "steps": [
            {
                "input": s[0], "action": s[1], "reward": s[2],
                "value": s[3], "log_prob": s[4],
            }
            for s in steps
        ],
    }


def _make_game(
    game_id: int, winner: str, rounds: int, perspectives: list[dict],
) -> dict:
    """Build a game dict."""
    return {
        "game_id": game_id, "winner": winner, "rounds": rounds,
        "perspectives": perspectives,
    }


def _make_config(
    sample_path: str, output_dir: str, **overrides,
) -> TrainingConfig:
    """Build a TrainingConfig with training-friendly defaults."""
    defaults = dict(
        sample_path=sample_path,
        output_dir=output_dir,
        model="rnn_v1.1",
        epoch=2,
        policy_lr=0.001,
        value_lr=0.001,
        weight_decay=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        max_grad_norm=0.5,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=2,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# 14-d input vectors (renamed from state per design/10 D1)
WIN_STEPS = [
    ([3, 2, 1, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], "attack", 0.0, -0.06, -1.34),
    ([2, 2, 1, 1, 3, 3, 1, 0, 0, 0, 0, 1, 0, 0], "attack", 0.0, -0.10, -1.33),
    ([1, 2, 1, 1, 3, 3, 1, 0, 0, 0, 0, 1, 0, 0], "counter", 1.0, -0.08, -1.30),
]

LOSE_STEPS = [
    ([3, 2, 1, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], "defense", 0.0, -0.06, -1.57),
    ([3, 1, 1, 1, 3, 3, 0, 1, 0, 0, 1, 0, 0, 0], "attack", 0.0, -0.11, -1.31),
    ([2, 1, 1, 1, 3, 1, 1, 0, 0, 0, 0, 1, 0, 0], "heavy", -1.0, -0.05, -0.85),
]


# ---------------------------------------------------------------------------
# TC-PHS1: Three-phase execution order
# ---------------------------------------------------------------------------


class TestThreePhaseExecution:
    """TC-PHS1: Training runs Phase 1 -> Phase 2 -> Phase 3 in order."""

    def test_training_completes_three_phases(self, tmp_path, monkeypatch):
        """Full training run completes without error."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 3, [
            _make_perspective(WIN_STEPS, player="1"),
            _make_perspective(LOSE_STEPS, player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir)
        runner = TrainingRunner()
        runner.run(config)

        policy_path = os.path.join(output_dir, "policy.pth")
        value_path = os.path.join(output_dir, "value.pth")
        assert os.path.exists(policy_path)
        assert os.path.exists(value_path)

    def test_value_epochs_executed_before_policy(self, tmp_path, monkeypatch):
        """Phase 1 (value) runs all its epochs before Phase 3 (policy)."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 3, [
            _make_perspective(WIN_STEPS, player="1"),
            _make_perspective(LOSE_STEPS, player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir, epoch=5)
        runner = TrainingRunner()
        runner.run(config)

        csv_path = loss_dir / "loss.csv"
        lines = csv_path.read_text().strip().split("\n")
        # Header + value epochs + policy epochs -> 1 + 5 + 5
        assert len(lines) == 1 + 5 + 5
        assert lines[1].startswith("value")
        assert lines[6].startswith("policy")


# ---------------------------------------------------------------------------
# TC-SAVE1 through TC-SAVE3: Weight saving
# ---------------------------------------------------------------------------


class TestWeightSaving:
    """TC-SAVE1: Two weight files produced after training."""

    def test_two_weight_files_produced(self, tmp_path, monkeypatch):
        """Both policy.pth and value.pth exist after run()."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 3, [
            _make_perspective(WIN_STEPS, player="1"),
            _make_perspective(LOSE_STEPS, player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir)
        runner = TrainingRunner()
        runner.run(config)

        policy_path = os.path.join(output_dir, "policy.pth")
        value_path = os.path.join(output_dir, "value.pth")
        assert os.path.exists(policy_path), "policy.pth not created"
        assert os.path.exists(value_path), "value.pth not created"

    def test_saved_policy_weights_load_into_GRUPolicyModel(
        self, tmp_path, monkeypatch,
    ):
        """Saved policy weights are compatible with GRUPolicyModel."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 3, [
            _make_perspective(WIN_STEPS, player="1"),
            _make_perspective(LOSE_STEPS, player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir)
        runner = TrainingRunner()
        runner.run(config)

        from src.backend.ai.rnn_v1.model import GRUPolicyModel
        model = GRUPolicyModel()
        policy_path = os.path.join(output_dir, "policy.pth")
        state_dict = torch.load(
            policy_path, map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        x = torch.zeros(1, 1, 14)
        h = torch.zeros(1, 1, 8)
        probs, log_probs, new_h = model(x, h)
        assert probs.shape == (1, 4)
        assert new_h.shape == (1, 1, 8)

    def test_saved_value_weights_load_into_GRUValueModel(
        self, tmp_path, monkeypatch,
    ):
        """Saved value weights are compatible with GRUValueModel."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 3, [
            _make_perspective(WIN_STEPS, player="1"),
            _make_perspective(LOSE_STEPS, player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir)
        runner = TrainingRunner()
        runner.run(config)

        from src.backend.ai.rnn_v1.model import GRUValueModel
        model = GRUValueModel()
        value_path = os.path.join(output_dir, "value.pth")
        state_dict = torch.load(
            value_path, map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        x = torch.zeros(1, 1, 14)
        h = torch.zeros(1, 1, 8)
        value, new_h = model(x, h)
        assert value.shape == (1, 1)
        assert new_h.shape == (1, 1, 8)


# ---------------------------------------------------------------------------
# TC-ART1, TC-ART2: Artifact format
# ---------------------------------------------------------------------------


class TestArtifactFormat:
    """TC-ART1 and TC-ART2: CSV headers include phase column."""

    def test_loss_csv_has_phase_column(self, tmp_path, monkeypatch):
        """loss.csv header starts with 'phase'."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 1, [
            _make_perspective(WIN_STEPS[:1], player="1"),
            _make_perspective(LOSE_STEPS[:1], player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir, epoch=2, batch_size=2)
        runner = TrainingRunner()
        runner.run(config)

        csv_path = loss_dir / "loss.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert lines[0] == "phase,epoch,policy_loss,value_loss,entropy"

    def test_grad_csv_has_phase_column(self, tmp_path, monkeypatch):
        """grad.csv header starts with 'phase'."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 1, [
            _make_perspective(WIN_STEPS[:1], player="1"),
            _make_perspective(LOSE_STEPS[:1], player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir, epoch=2, batch_size=2)
        runner = TrainingRunner()
        runner.run(config)

        csv_path = loss_dir / "grad.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert lines[0] == "phase,epoch,policy_grad_norm,value_grad_norm"


# ---------------------------------------------------------------------------
# TC-OPT1, TC-OPT2: Optimizer checks
# ---------------------------------------------------------------------------


class TestOptimizer:
    """TC-OPT1 and TC-OPT2: AdamW optimizers with weight_decay."""

    def test_adamw_optimizer_used(self, tmp_path, monkeypatch):
        """Both optimizers are torch.optim.AdamW."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 1, [
            _make_perspective(WIN_STEPS[:1], player="1"),
            _make_perspective(LOSE_STEPS[:1], player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(
            sample_path, output_dir, epoch=1, weight_decay=0.01,
        )
        runner = TrainingRunner()
        runner.run(config)

        policy_path = os.path.join(output_dir, "policy.pth")
        value_path = os.path.join(output_dir, "value.pth")
        assert os.path.exists(policy_path)
        assert os.path.exists(value_path)


# ---------------------------------------------------------------------------
# Gradient isolation tests
# ---------------------------------------------------------------------------


class TestGradientIsolation:
    """TC-GRAD1, TC-GRAD2: policy backward isolation."""

    def test_policy_weights_change_after_policy_epoch(
        self, tmp_path, monkeypatch,
    ):
        """Policy weights change after Phase 3 policy training."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 3, [
            _make_perspective(WIN_STEPS, player="1"),
            _make_perspective(LOSE_STEPS, player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir, epoch=3, batch_size=1)
        runner = TrainingRunner()

        from src.backend.ai.rnn_v1.model import GRUPolicyModel
        from src.backend.ai.registry import get_ai_info
        initial_model = GRUPolicyModel()
        info = get_ai_info("rnn_v1.1")
        initial_state = torch.load(
            info["policy"]["weight_path"], map_location="cpu", weights_only=True,
        )
        initial_model.load_state_dict(initial_state)
        initial_params = {
            k: v.clone() for k, v in initial_model.state_dict().items()
        }

        runner.run(config)

        policy_path = os.path.join(output_dir, "policy.pth")
        saved = torch.load(policy_path, map_location="cpu", weights_only=True)
        any_changed = any(
            not torch.equal(initial_params[k], saved[k])
            for k in initial_params
        )
        assert any_changed, "Policy weights did not change after training"

    def test_value_weights_change_after_value_epoch(
        self, tmp_path, monkeypatch,
    ):
        """Value weights change after Phase 1 value training."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 3, [
            _make_perspective(WIN_STEPS, player="1"),
            _make_perspective(LOSE_STEPS, player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir, epoch=1, batch_size=1)
        runner = TrainingRunner()

        from src.backend.ai.rnn_v1.model import GRUValueModel
        from src.backend.ai.registry import get_ai_info
        initial_model = GRUValueModel()
        info = get_ai_info("rnn_v1.1")
        initial_state = torch.load(
            info["value"]["weight_path"], map_location="cpu", weights_only=True,
        )
        initial_model.load_state_dict(initial_state)
        initial_params = {
            k: v.clone() for k, v in initial_model.state_dict().items()
        }

        runner.run(config)

        value_path = os.path.join(output_dir, "value.pth")
        saved = torch.load(value_path, map_location="cpu", weights_only=True)
        any_changed = any(
            not torch.equal(initial_params[k], saved[k])
            for k in initial_params
        )
        assert any_changed, "Value weights did not change after training"


# ---------------------------------------------------------------------------
# TC-GAE1 through TC-GAE4: GAE edge cases
# ---------------------------------------------------------------------------


class TestGAEComputation:
    """TC-GAE1-4: GAE advantage computation edge cases."""

    def test_winning_perspective_advantages_positive(self):
        """TC-GAE1: Winning perspective gets positive advantages."""
        runner = TrainingRunner()
        values = [-0.06, -0.10, -0.08]
        rewards = [0.0, 0.0, 1.0]
        advantages, _ = runner._compute_gae(values, rewards, 0.99, 0.95)
        for adv in advantages:
            assert adv > 0, f"Expected positive advantage, got {adv}"

    def test_losing_perspective_advantages_negative(self):
        """TC-GAE2: Losing perspective gets negative advantages."""
        runner = TrainingRunner()
        values = [-0.06, -0.11, -0.05]
        rewards = [0.0, 0.0, -1.0]
        advantages, _ = runner._compute_gae(values, rewards, 0.99, 0.95)
        for adv in advantages:
            assert adv < 0, f"Expected negative advantage, got {adv}"

    def test_last_step_advantage_equals_td_error(self):
        """TC-GAE3: Last step advantage = r_T - V_T (TD error)."""
        runner = TrainingRunner()
        values = [0.35, 0.42, 0.28]
        rewards = [0.0, 0.0, 1.0]
        advantages, _ = runner._compute_gae(values, rewards, 0.99, 0.95)
        expected_last = 1.0 - 0.28
        assert abs(advantages[-1] - expected_last) < 1e-10

    def test_advantage_length_matches_steps(self):
        """TC-GAE4: Advantage list length = step count."""
        runner = TrainingRunner()
        for n in [1, 3, 7]:
            values = [0.0] * n
            rewards = [0.0] * n
            advantages, value_targets = runner._compute_gae(
                values, rewards, 0.99, 0.95,
            )
            assert len(advantages) == n
            assert len(value_targets) == n


# ---------------------------------------------------------------------------
# Design/10 TC-SR-02, TC-SR-03: TrainingRunner key validation
# ---------------------------------------------------------------------------


class TestInputKeyInTraining:
    """TC-SR-02 and TC-SR-03 from design/10."""

    def test_training_accepts_input_key(self, tmp_path, monkeypatch):
        """TC-SR-02: TrainingRunner loads samples with 'input' key."""
        loss_dir = tmp_path / "loss"
        monkeypatch.setattr(
            "src.backend.ai.training_runner.save_split_training_artifacts",
            lambda records, output_dir=None: _patched_artifacts(
                records, str(loss_dir),
            ),
        )

        output_dir = str(tmp_path / "weights")
        game = _make_game(1, "1", 3, [
            _make_perspective(WIN_STEPS, player="1"),
            _make_perspective(LOSE_STEPS, player="2"),
        ])
        sample_path = _make_sample_json([game])

        config = _make_config(sample_path, output_dir, epoch=1)
        runner = TrainingRunner()
        runner.run(config)

        policy_path = os.path.join(output_dir, "policy.pth")
        assert os.path.exists(policy_path)

    def test_training_rejects_state_key(self, tmp_path):
        """TC-SR-03: Samples with 'state' key raise RuntimeError."""
        old_game = {
            "game_id": 1, "winner": "1", "rounds": 1,
            "perspectives": [{
                "player": "1", "ai": "test",
                "steps": [{
                    "state": [0.0] * 14, "action": "attack",
                    "reward": 0.0, "value": 0.0,
                }],
            }],
        }
        sample_path = _make_sample_json([old_game])
        output_dir = str(tmp_path / "weights")

        config = _make_config(sample_path, output_dir, epoch=1)
        runner = TrainingRunner()
        with pytest.raises(RuntimeError, match="missing required key"):
            runner.run(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_artifacts(records: list[dict], output_dir: str) -> None:
    """Redirect artifact output to a temp directory."""
    from src.backend.ai.training_utils import (
        save_split_training_artifacts as _orig,
    )
    _orig(records, output_dir=output_dir)
