"""V1 training runner — GRU bypass forward strategy for rnn_v1 models (design/31 D7).

Uses direct GRU internal access (model.gru + model.fc_*) for efficient
full-sequence forward passes during training.
"""

import torch
import torch.nn as nn

from src.backend.ai.rnn_v1.model import (
    GRUPolicyModel,
    GRUValueModel,
    HIDDEN_SIZE,
    INPUT_SIZE,
)
from src.backend.ai.training_config import TrainingConfig
from src.backend.ai.training_runner import BaseTrainingRunner


class V1TrainingRunner(BaseTrainingRunner):
    """Three-phase split PPO training with v1 GRU bypass forward strategy.

    Inherits all shared logic from BaseTrainingRunner.
    Overrides only the three phase methods that differ in forward strategy.
    """

    # ------------------------------------------------------------------
    # Phase 1: Value Training — v1 bypass
    # ------------------------------------------------------------------

    def _process_value_perspective(
        self,
        perspective: dict,
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Compute value MSE loss for one perspective using GRU bypass.

        Directly accesses value_model.gru + value_model.fc_value
        for GPU-efficient full-sequence forward.
        """
        steps = perspective["steps"]
        T = len(steps)

        old_values: list[float] = []
        rewards: list[float] = []
        states: list[list[float]] = []

        for step in steps:
            states.append(step["input"])
            old_values.append(step["value"])
            rewards.append(step["reward"])

        _advantages, value_targets = self._compute_gae(
            old_values, rewards, config.gamma, config.gae_lambda,
        )

        # Full sequence GRU → fc_value bypass
        x = torch.tensor(states, dtype=torch.float32, device=device).view(
            T, 1, INPUT_SIZE,
        )
        h = torch.tensor(
            GRUValueModel.INITIAL_HIDDEN_STATE,
            dtype=torch.float32, device=device,
        ).view(1, 1, HIDDEN_SIZE)
        gru_out, _ = value_model.gru(x, h)
        v_new = value_model.fc_value(gru_out.squeeze(1)).squeeze()

        v_target = torch.tensor(value_targets, dtype=torch.float32, device=device)
        value_loss = config.value_coef * nn.functional.mse_loss(v_new, v_target)

        if torch.isnan(value_loss):
            return None
        return value_loss

    # ------------------------------------------------------------------
    # Phase 2: Advantage Re-computation — v1 bypass
    # ------------------------------------------------------------------

    def _recompute_advantages(
        self,
        perspectives: list[dict],
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Re-compute GAE advantages using v1 GRU bypass (no_grad)."""
        print(
            f"[Phase 2] Recomputing advantages for {len(perspectives)} perspectives..."
        )
        with torch.no_grad():
            for perspective in perspectives:
                steps = perspective["steps"]
                T = len(steps)

                rewards = [s["reward"] for s in steps]

                # Full sequence GRU → fc_value bypass
                inputs = [s["input"] for s in steps]
                x = torch.tensor(
                    inputs, dtype=torch.float32, device=device,
                ).view(T, 1, INPUT_SIZE)
                h = torch.tensor(
                    GRUValueModel.INITIAL_HIDDEN_STATE,
                    dtype=torch.float32, device=device,
                ).view(1, 1, HIDDEN_SIZE)
                gru_out, _ = value_model.gru(x, h)
                new_values = (
                    value_model.fc_value(gru_out.squeeze(1)).squeeze().tolist()
                )

                advantages, value_targets = self._compute_gae(
                    new_values, rewards, config.gamma, config.gae_lambda,
                )

                perspective["new_advantages"] = advantages
                perspective["new_value_targets"] = value_targets

    # ------------------------------------------------------------------
    # Phase 3: Policy Training — v1 bypass
    # ------------------------------------------------------------------

    def _process_policy_perspective(
        self,
        perspective: dict,
        policy_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> tuple | None:
        """Compute PPO policy loss for one perspective using GRU bypass.

        Directly accesses policy_model.gru + policy_model.fc_policy
        for GPU-efficient full-sequence forward.
        """
        steps = perspective["steps"]
        T = len(steps)
        advantages = perspective["new_advantages"]

        old_log_probs: list[float] = []
        states: list[list[float]] = []
        actions: list[int] = []

        for step in steps:
            states.append(step["input"])
            actions.append(step["action"])
            old_log_probs.append(step["log_prob"])

        # Full sequence GRU → fc_policy bypass
        x = torch.tensor(states, dtype=torch.float32, device=device).view(
            T, 1, INPUT_SIZE,
        )
        h = torch.tensor(
            GRUPolicyModel.INITIAL_HIDDEN_STATE,
            dtype=torch.float32, device=device,
        ).view(1, 1, HIDDEN_SIZE)
        gru_out, _ = policy_model.gru(x, h)
        policy_logits = policy_model.fc_policy(gru_out.squeeze(1))
        log_probs = torch.log_softmax(policy_logits, dim=-1)

        action_indices = torch.tensor(actions, dtype=torch.long, device=device)
        new_log = log_probs[torch.arange(T, device=device), action_indices]

        old_log = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=device)

        ratio = torch.exp(new_log - old_log)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps,
        )
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

        entropy = (
            -(torch.exp(log_probs) * log_probs)
            .nan_to_num(0.0)
            .sum(dim=-1)
            .mean()
        )

        if torch.isnan(policy_loss) or torch.isnan(entropy):
            return None
        return policy_loss, entropy
