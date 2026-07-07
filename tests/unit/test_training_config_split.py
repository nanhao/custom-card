"""Unit tests for split TrainingConfig dataclass.

Covers TC-CFG1 through TC-CFG12 from design/9 test_cases.md,
and new TC-TC-* from design/10 test_cases.md.
Tests validation logic for all split-training fields,
including the unified output_dir and epoch (design/10 D3).
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
        output_dir="output_dir",
        model="rnn_v1.1",
        epoch=10,
        policy_lr=0.001,
        value_lr=0.002,
        weight_decay=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        max_grad_norm=0.5,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=64,
    )


# ---------------------------------------------------------------------------
# TC-CFG1: valid config passes validation
# ---------------------------------------------------------------------------


def test_valid_config_passes_validation():
    """TC-CFG1: all fields valid — no exception raised."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        config = _valid_config(sample_path)
        assert config.epoch == 10
        assert config.batch_size == 64
        assert config.policy_lr == 0.001
        assert config.value_lr == 0.002
        assert config.weight_decay == 0.01
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-CFG2, TC-CFG3: Learning rate validation
# ---------------------------------------------------------------------------


def test_policy_lr_le_0_raises_valueerror():
    """TC-CFG2: policy_lr <= 0 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="policy_lr must be > 0"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.0,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_value_lr_le_0_raises_valueerror():
    """TC-CFG3: value_lr <= 0 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="value_lr must be > 0"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=-0.1,
                weight_decay=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-CFG4, TC-CFG5: gamma validation
# ---------------------------------------------------------------------------


def test_gamma_le_0_raises_valueerror():
    """TC-CFG4: gamma <= 0 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="gamma must be in \\(0, 1\\]"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=0.0,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_gamma_gt_1_raises_valueerror():
    """TC-CFG5: gamma > 1 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="gamma must be in \\(0, 1\\]"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=1.1,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-CFG6: gae_lambda validation
# ---------------------------------------------------------------------------


def test_gae_lambda_out_of_range_raises_valueerror():
    """TC-CFG6: gae_lambda outside [0,1] raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        for bad_val in [-0.1, 1.1]:
            with pytest.raises(ValueError, match="gae_lambda must be in \\[0, 1\\]"):
                TrainingConfig(
                    sample_path=sample_path,
                    output_dir="output_dir",
                    model="rnn_v1.1",
                    epoch=10,
                    policy_lr=0.001,
                    value_lr=0.001,
                    weight_decay=0.01,
                    gamma=0.99,
                    gae_lambda=bad_val,
                    clip_eps=0.2,
                    max_grad_norm=0.5,
                    value_coef=0.5,
                    entropy_coef=0.01,
                    batch_size=64,
                )
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-CFG7: clip_eps validation
# ---------------------------------------------------------------------------


def test_clip_eps_le_0_raises_valueerror():
    """TC-CFG7: clip_eps <= 0 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="clip_eps must be > 0"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.0,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-CFG8: value_coef/entropy_coef < 0 validation
# ---------------------------------------------------------------------------


def test_value_coef_lt_0_raises_valueerror():
    """TC-CFG8: value_coef < 0 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="value_coef must be >= 0"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=-0.1,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_entropy_coef_lt_0_raises_valueerror():
    """TC-CFG8: entropy_coef < 0 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="entropy_coef must be >= 0"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=-0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-CFG9: max_grad_norm validation
# ---------------------------------------------------------------------------


def test_max_grad_norm_le_0_raises_valueerror():
    """TC-CFG9: max_grad_norm <= 0 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="max_grad_norm must be > 0"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.0,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-CFG10: sample_path nonexistent
# ---------------------------------------------------------------------------


def test_nonexistent_sample_path_raises_valueerror():
    """TC-CFG10: nonexistent sample_path raises ValueError."""
    with pytest.raises(ValueError, match="sample_path does not exist"):
        TrainingConfig(
            sample_path="/nonexistent/path/xyz.json",
            output_dir="output_dir",
            model="rnn_v1.1",
            epoch=10,
            policy_lr=0.001,
            value_lr=0.001,
            weight_decay=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            max_grad_norm=0.5,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=64,
        )


# ---------------------------------------------------------------------------
# TC-CFG12: weight_decay validation
# ---------------------------------------------------------------------------


def test_weight_decay_lt_0_raises_valueerror():
    """TC-CFG12: weight_decay < 0 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=-0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# TC-CFG: epoch < 1 validation
# ---------------------------------------------------------------------------


def test_epoch_lt_1_raises_valueerror():
    """epoch < 1 raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="epoch must be >= 1"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="output_dir",
                model="rnn_v1.1",
                epoch=0,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


# ---------------------------------------------------------------------------
# Design/10 TC-TC-03, TC-TC-04: New validations
# ---------------------------------------------------------------------------


def test_output_dir_empty_raises_valueerror():
    """TC-TC-04: output_dir="" raises ValueError."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        with pytest.raises(ValueError, match="output_dir must be non-empty"):
            TrainingConfig(
                sample_path=sample_path,
                output_dir="",
                model="rnn_v1.1",
                epoch=10,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=64,
            )
    finally:
        os.unlink(sample_path)


def test_training_config_no_longer_has_old_fields():
    """TC-TC-01 + TC-TC-02: output_dir replaces old fields, no value_epochs."""
    sample_path = _make_temp_file(
        '[{"game_id":1,"winner":"1","rounds":1,"perspectives":[]}]'
    )
    try:
        config = _valid_config(sample_path)
        # New field exists
        assert config.output_dir == "output_dir"
        # Old fields are gone
        assert not hasattr(config, "policy_output_path")
        assert not hasattr(config, "value_output_path")
        assert not hasattr(config, "value_epochs")
    finally:
        os.unlink(sample_path)
