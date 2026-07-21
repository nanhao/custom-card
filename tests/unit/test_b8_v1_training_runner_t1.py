"""B8: V1TrainingRunner crashes when T=1.

Bug: .squeeze().tolist() produces a scalar float (not a list) when T=1,
then _compute_gae calls len() on it → TypeError.

Fix: Change .squeeze() to .squeeze(-1) to maintain (T,) shape.
"""

import torch
from unittest.mock import MagicMock, patch

from src.backend.ai.rnn_v1.v1_training_runner import V1TrainingRunner


class TestB8V1TrainingRunnerT1:
    """B8: V1TrainingRunner._recompute_advantages must handle T=1."""

    def test_recompute_advantages_with_t1_does_not_crash(self):
        """A single-step perspective (T=1) should not cause TypeError."""
        runner = V1TrainingRunner.__new__(V1TrainingRunner)

        # Mock _compute_gae to skip GAE logic and just verify it's called
        # with a list (not a scalar float)
        runner._compute_gae = MagicMock(return_value=([0.5], [0.5]))

        # Create a single-step perspective
        perspective = {
            "steps": [
                {
                    "reward": 1.0,
                    "input": [0.0] * 62,  # INPUT_SIZE = 62
                }
            ]
        }

        # Mock value model
        value_model = MagicMock()
        # Simulate GRU output: shape (1, 1, 128)
        fake_gru_out = torch.zeros(1, 1, 32)  # HIDDEN_SIZE = 32
        value_model.gru = MagicMock(return_value=(fake_gru_out, None))
        # Simulate fc_value output: shape (1, 1)
        fake_value_out = torch.tensor([[0.5]])
        value_model.fc_value = MagicMock(return_value=fake_value_out)

        config = MagicMock()
        config.gamma = 0.99
        config.gae_lambda = 0.95
        device = torch.device("cpu")

        # This should NOT raise TypeError
        runner._recompute_advantages(
            [perspective], value_model, config, device
        )

        # Verify _compute_gae was called with a list, not a scalar
        call_args = runner._compute_gae.call_args
        new_values_arg = call_args[0][0]  # first positional arg
        assert isinstance(new_values_arg, list), (
            f"B8 FAIL: new_values should be a list, got {type(new_values_arg)}: "
            f"{new_values_arg}"
        )
        assert len(new_values_arg) == 1, (
            f"B8 FAIL: expected list of length 1, got length {len(new_values_arg)}"
        )
