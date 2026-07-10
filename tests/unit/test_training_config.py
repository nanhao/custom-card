"""Unit tests for TrainingConfig — policy_epoch / value_epoch validation (design/32).

Test cases per design/32 test_cases.md: TC-EC-01 through TC-EC-07.
"""

import json
import os
import tempfile

import pytest

from src.backend.ai.training_config import TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tmp_sample(tmp_dir: str) -> str:
    """Create a minimal dummy JSON file for TrainingConfig validation."""
    path = os.path.join(tmp_dir, "dummy.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([], f)
    return path


def _base_kwargs(sample_path: str, output_dir: str, **overrides) -> dict:
    """Build minimal valid kwargs for TrainingConfig, overridable."""
    kwargs = {
        "sample_path": sample_path,
        "output_dir": output_dir,
        "model": "rnn_v1.1",
        "policy_epoch": 1,
        "value_epoch": 1,
        "policy_lr": 0.001,
        "value_lr": 0.001,
        "weight_decay": 0.0,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "max_grad_norm": 0.5,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "batch_size": 2,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# TC-EC-01 ~ TC-EC-07
# ---------------------------------------------------------------------------


class TestTrainingConfigEpochValidation:
    """TC-EC-01 through TC-EC-07: policy_epoch / value_epoch field validation."""

    def test_same_policy_and_value_epoch_creates_successfully(self):
        """TC-EC-01: policy_epoch=10, value_epoch=10 → success."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_tmp_sample(td)
            cfg = TrainingConfig(**_base_kwargs(sample, td, policy_epoch=10, value_epoch=10))
            assert cfg.policy_epoch == 10
            assert cfg.value_epoch == 10

    def test_different_policy_and_value_epoch_creates_successfully(self):
        """TC-EC-02: policy_epoch=10, value_epoch=5 → success."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_tmp_sample(td)
            cfg = TrainingConfig(**_base_kwargs(sample, td, policy_epoch=10, value_epoch=5))
            assert cfg.policy_epoch == 10
            assert cfg.value_epoch == 5

    def test_policy_epoch_zero_raises_value_error(self):
        """TC-EC-03: policy_epoch=0 → ValueError with 'policy_epoch must be >= 1'."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_tmp_sample(td)
            with pytest.raises(ValueError, match="policy_epoch must be >= 1"):
                TrainingConfig(**_base_kwargs(sample, td, policy_epoch=0))

    def test_value_epoch_zero_raises_value_error(self):
        """TC-EC-04: value_epoch=0 → ValueError with 'value_epoch must be >= 1'."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_tmp_sample(td)
            with pytest.raises(ValueError, match="value_epoch must be >= 1"):
                TrainingConfig(**_base_kwargs(sample, td, value_epoch=0))

    def test_policy_epoch_negative_raises_value_error(self):
        """TC-EC-05: policy_epoch=-1 → ValueError."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_tmp_sample(td)
            with pytest.raises(ValueError):
                TrainingConfig(**_base_kwargs(sample, td, policy_epoch=-1))

    def test_value_epoch_negative_raises_value_error(self):
        """TC-EC-06: value_epoch=-1 → ValueError."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_tmp_sample(td)
            with pytest.raises(ValueError):
                TrainingConfig(**_base_kwargs(sample, td, value_epoch=-1))

    def test_old_epoch_field_removed(self):
        """TC-EC-07: passing 'epoch' → TypeError (unexpected keyword)."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_tmp_sample(td)
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                # Intentionally pass the old 'epoch' parameter
                TrainingConfig(
                    sample_path=sample,
                    output_dir=td,
                    model="rnn_v1.1",
                    epoch=10,  # type: ignore[call-arg]
                    policy_epoch=1,  # type: ignore[call-arg]
                    value_epoch=1,  # type: ignore[call-arg]
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
