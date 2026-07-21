"""B7: Training artifacts saved to wrong directory.

Bug: save_split_training_artifacts() is called with output_dir=None, so
artifacts (loss.csv, grad.csv) go to cwd/loss/ instead of
config.output_dir/. Weights go to config.output_dir/ correctly.

Fix: Pass config.output_dir to save_split_training_artifacts.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

from src.backend.training.training_utils import save_split_training_artifacts


class TestB7TrainingArtifactsDirectory:
    """B7: Artifacts must be saved to config.output_dir, not cwd/loss/."""

    def test_artifacts_go_to_output_dir_not_cwd(self):
        """When output_dir is specified, loss.csv goes there, not cwd/loss/."""
        # Create a temp directory as our "output_dir"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "training_output")
            os.makedirs(output_dir, exist_ok=True)

            # Record current cwd
            cwd_loss_dir = os.path.join(os.getcwd(), "loss")
            cwd_before = os.path.exists(cwd_loss_dir)

            # Minimal phase records
            records = [
                {
                    "phase": "value", "epoch": 1,
                    "policy_loss": 0.5, "value_loss": 0.3,
                    "entropy": 1.0, "policy_grad_norm": 0.1,
                    "value_grad_norm": 0.2, "val_value_loss": float("nan"),
                },
            ]

            save_split_training_artifacts(records, output_dir=output_dir)

            # B7 CHECK: loss.csv should exist in output_dir
            loss_csv = os.path.join(output_dir, "loss.csv")
            assert os.path.exists(loss_csv), (
                f"B7 FAIL: loss.csv not found in {output_dir}. "
                f"Directory contents: {os.listdir(output_dir)}"
            )

            # B7 CHECK: loss.csv should NOT exist in cwd/loss/
            cwd_after = os.path.exists(cwd_loss_dir)
            if cwd_after and not cwd_before:
                # Only fail if we created it (wasn't there before)
                assert not os.path.exists(os.path.join(cwd_loss_dir, "loss.csv")), (
                    "B7 FAIL: loss.csv was created in cwd/loss/ instead of output_dir"
                )

    def test_none_output_dir_uses_cwd_loss(self):
        """When output_dir is None, artifacts go to cwd/loss/ (default behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                records = [
                    {
                        "phase": "policy", "epoch": 1,
                        "policy_loss": 0.5, "value_loss": float("nan"),
                        "entropy": 1.0, "policy_grad_norm": 0.1,
                        "value_grad_norm": float("nan"),
                    },
                ]
                save_split_training_artifacts(records, output_dir=None)
                expected = os.path.join(tmpdir, "loss", "loss.csv")
                assert os.path.exists(expected), (
                    f"B7 FAIL: default None should use cwd/loss/. "
                    f"Expected: {expected}"
                )
            finally:
                os.chdir(original_cwd)
