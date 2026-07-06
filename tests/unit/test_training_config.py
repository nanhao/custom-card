"""Unit tests for ai/training_config.py — TrainingConfig dataclass.

Covers TC-C1 through TC-C12 from design/8 test_cases.md.
Tests validation logic in __post_init__ for all fields.
"""

import os
import tempfile

import pytest

from src.backend.ai.training_config import TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_temp_file(content: str) -> str:
    """Create a temp file with given content, return its path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _valid_config(sample_path: str) -> TrainingConfig:
    """Build a valid TrainingConfig for testing."""
    return TrainingConfig(
        sample_path=sample_path,
        output_path="output.pth",
        model="rnn_v1.1",
        epoch=10,
        lr=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        max_grad_norm=0.5,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=64,
    )


# ---------------------------------------------------------------------------
# TC-C1: valid config passes validation
# ---------------------------------------------------------------------------


def test_valid_config_passes_validation():
    """TC-C1: all fields valid — no exception raised."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        config = _valid_config(sample_path)
        assert config.epoch == 10
        assert config.batch_size == 64
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-C2 through TC-C12: individual field validation
# ---------------------------------------------------------------------------


def test_epoch_lt_1_raises_valueerror():
    """TC-C2: epoch < 1 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="epoch must be >= 1"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=0, lr=0.001, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_batch_size_lt_1_raises_valueerror():
    """TC-C3: batch_size < 1 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=0,
            )
    finally:
        os.unlink(sample_path)


def test_gamma_le_0_raises_valueerror():
    """TC-C4: gamma <= 0 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="gamma must be in \\(0, 1\\]"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=0.0, gae_lambda=0.95, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_gamma_gt_1_raises_valueerror():
    """TC-C5: gamma > 1 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="gamma must be in \\(0, 1\\]"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=1.1, gae_lambda=0.95, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_gae_lambda_lt_0_raises_valueerror():
    """TC-C6: gae_lambda < 0 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="gae_lambda must be in \\[0, 1\\]"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=0.99, gae_lambda=-0.1, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_gae_lambda_gt_1_raises_valueerror():
    """TC-C7: gae_lambda > 1 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="gae_lambda must be in \\[0, 1\\]"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=0.99, gae_lambda=1.1, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_clip_eps_le_0_raises_valueerror():
    """TC-C8: clip_eps <= 0 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="clip_eps must be > 0"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=0.99, gae_lambda=0.95, clip_eps=0.0,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_lr_le_0_raises_valueerror():
    """TC-C9: lr <= 0 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="lr must be > 0"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.0, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_value_coef_lt_0_raises_valueerror():
    """TC-C10: value_coef < 0 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="value_coef must be >= 0"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=-0.1, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_entropy_coef_lt_0_raises_valueerror():
    """TC-C11: entropy_coef < 0 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="entropy_coef must be >= 0"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                max_grad_norm=0.5, value_coef=0.5, entropy_coef=-0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_max_grad_norm_le_0_raises_valueerror():
    """TC-C12: max_grad_norm <= 0 raises ValueError."""
    sample_path = _make_temp_file('[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]')
    try:
        with pytest.raises(ValueError, match="max_grad_norm must be > 0"):
            TrainingConfig(
                sample_path=sample_path, output_path="o.pth", model="rnn_v1.1",
                epoch=10, lr=0.001, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                max_grad_norm=0.0, value_coef=0.5, entropy_coef=0.01, batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_nonexistent_sample_path_raises_valueerror():
    """TC-L4: nonexistent sample_path raises ValueError."""
    with pytest.raises(ValueError, match="sample_path does not exist"):
        TrainingConfig(
            sample_path="/nonexistent/path/xyz.json",
            output_path="o.pth", model="rnn_v1.1",
            epoch=10, lr=0.001, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
            max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01, batch_size=64,
        )
