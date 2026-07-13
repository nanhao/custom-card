"""V2 training runner — skip-connection forward strategy for rnn_v2 models (design/31 D7).

Uses internal layer access (model.gru + model.fc1 + model.fc2 + model.fc_*)
with skip connection (concat GRU output with raw input) for efficient
full-sequence forward passes during training.
"""

import torch
import torch.nn as nn

from src.backend.ai.rnn_v2.model import (
    GRUPolicyModelV2,
    GRUValueModelV2,
    HIDDEN_SIZE,
    INPUT_SIZE,
)
from src.backend.training.training_config import TrainingConfig
from src.backend.training.training_runner import BaseTrainingRunner


class V2TrainingRunner(BaseTrainingRunner):
    """Three-phase split PPO training with v2 skip-connection forward strategy."""

    # ------------------------------------------------------------------
    # Shared forward helpers (eliminate duplicate skip-connection logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_value_sequence(
        value_model: nn.Module, x: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        """Batch value forward: GRU → concat(input) → fc1→ReLU→fc2→ReLU→fc_value."""
        gru_out, _ = value_model.gru(x, h)  # (T, 1, 32)
        combined = torch.cat(
            [gru_out.squeeze(1), x.squeeze(1)], dim=-1,
        )  # (T, 74)
        h1 = torch.relu(value_model.fc1(combined))   # (T, 64)
        h2 = torch.relu(value_model.fc2(h1))          # (T, 32)
        return value_model.fc_value(h2).squeeze(-1)   # (T,)

    @staticmethod
    def _forward_policy_sequence(
        policy_model: nn.Module, x: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        """Batch policy forward: GRU → concat(input) → fc1→ReLU→fc2→ReLU→fc_policy."""
        gru_out, _ = policy_model.gru(x, h)  # (T, 1, 32)
        combined = torch.cat(
            [gru_out.squeeze(1), x.squeeze(1)], dim=-1,
        )  # (T, 74)
        h1 = torch.relu(policy_model.fc1(combined))   # (T, 64)
        h2 = torch.relu(policy_model.fc2(h1))          # (T, 32)
        return policy_model.fc_policy(h2)              # (T, 14)

    # ------------------------------------------------------------------
    # Phase 1: Value Training
    # ------------------------------------------------------------------

    def _process_value_perspective(
        self,
        perspective: dict,
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Compute value MSE loss using v2 skip-connection forward.

        GRU on full sequence → concat(GRU_out, input) → fc1 → ReLU →
        fc2 → ReLU → fc_value → MSE loss.
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

        # Full sequence GRU → skip-connection → MLP head
        x = torch.tensor(states, dtype=torch.float32, device=device).view(
            T, 1, INPUT_SIZE,
        )
        h = torch.tensor(
            GRUValueModelV2.INITIAL_HIDDEN_STATE,
            dtype=torch.float32, device=device,
        ).view(1, 1, HIDDEN_SIZE)
        v_new = self._forward_value_sequence(value_model, x, h)

        v_target = torch.tensor(value_targets, dtype=torch.float32, device=device)
        value_loss = config.value_coef * nn.functional.mse_loss(v_new, v_target)

        if torch.isnan(value_loss):
            return None
        return value_loss

    # ------------------------------------------------------------------
    # Phase 2: Advantage Re-computation — v2 skip-connection forward
    # ------------------------------------------------------------------

    def _recompute_advantages(
        self,
        perspectives: list[dict],
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Re-compute GAE advantages using v2 skip-connection forward (no_grad)."""
        print(
            f"[Phase 2] Recomputing advantages for {len(perspectives)} perspectives..."
        )
        with torch.no_grad():
            for perspective in perspectives:
                steps = perspective["steps"]
                T = len(steps)
                rewards = [s["reward"] for s in steps]

                # Full sequence GRU → skip-connection → MLP head
                inputs = [s["input"] for s in steps]
                x = torch.tensor(
                    inputs, dtype=torch.float32, device=device,
                ).view(T, 1, INPUT_SIZE)
                h = torch.tensor(
                    GRUValueModelV2.INITIAL_HIDDEN_STATE,
                    dtype=torch.float32, device=device,
                ).view(1, 1, HIDDEN_SIZE)
                new_values = self._forward_value_sequence(
                    value_model, x, h,
                ).tolist()

                advantages, value_targets = self._compute_gae(
                    new_values, rewards, config.gamma, config.gae_lambda,
                )

                perspective["new_advantages"] = advantages
                perspective["new_value_targets"] = value_targets

    # ------------------------------------------------------------------
    # Phase 3: Policy Training — v2 skip-connection forward
    # ------------------------------------------------------------------

    def _process_policy_perspective(
        self,
        perspective: dict,
        policy_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> tuple | None:
        """Compute PPO policy loss using v2 skip-connection forward.

        GRU on full sequence → concat(GRU_out, input) → fc1 → ReLU →
        fc2 → ReLU → fc_policy → log_softmax → PPO loss.
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

        # Full sequence GRU → skip-connection → MLP head
        x = torch.tensor(states, dtype=torch.float32, device=device).view(
            T, 1, INPUT_SIZE,
        )
        h = torch.tensor(
            GRUPolicyModelV2.INITIAL_HIDDEN_STATE,
            dtype=torch.float32, device=device,
        ).view(1, 1, HIDDEN_SIZE)
        policy_logits = self._forward_policy_sequence(policy_model, x, h)
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
