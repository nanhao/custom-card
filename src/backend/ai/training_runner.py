"""PPO training orchestrator for GRU-based card game AI models.

Per design/8 architecture D1: pure AI concern, no game or app imports.
Handles sample loading, PPO training loop with gradient accumulation,
GAE advantage estimation, weight saving, and loss/gradient chart generation.

Public interface (per design/8 interfaces.md §2):
    TrainingRunner.run(config: TrainingConfig) -> None
"""

import importlib
import json
import math
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from src.backend.ai.registry import get_ai_info
from src.backend.ai.rnn_v1.model import (
    GRUCardModel,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CARD_TYPES,
)
from src.backend.ai.training_config import TrainingConfig
from src.backend.ai.training_utils import compute_grad_norm, save_training_artifacts


# ---------------------------------------------------------------------------
# Action-to-index mapping (per design D13)
# Must match Converter.card_to_index() in app/.
# ---------------------------------------------------------------------------

ACTION_TO_INDEX: dict[str, int] = {
    "attack": 0,
    "defense": 1,
    "heavy": 2,
    "counter": 3,
}


# ---------------------------------------------------------------------------
# TrainingRunner
# ---------------------------------------------------------------------------


class TrainingRunner:
    """PPO training orchestrator for GRU-based models.

    Handles sample loading, training loop with gradient accumulation,
    GAE advantage estimation, weight saving, and loss/gradient charts.

    Public method:
        run(config: TrainingConfig) -> None
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: TrainingConfig) -> None:
        """Execute PPO training end-to-end."""
        model, device = self._build_model(config.model)
        all_perspectives = self._load_samples(config.sample_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # --- Training loop ---
        epoch_losses: list[float] = []
        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_entropies: list[float] = []
        epoch_grad_norms: list[float] = []

        for epoch in range(1, config.epoch + 1):
            # Shuffle perspectives
            random.shuffle(all_perspectives)

            # Process in gradient accumulation groups
            group_metrics = self._train_one_epoch(
                all_perspectives, model, optimizer, config, device
            )

            epoch_losses.append(group_metrics["avg_loss"])
            epoch_policy_losses.append(group_metrics["avg_policy_loss"])
            epoch_value_losses.append(group_metrics["avg_value_loss"])
            epoch_entropies.append(group_metrics["avg_entropy"])
            epoch_grad_norms.append(group_metrics["avg_grad_norm"])

            print(
                f"Epoch {epoch}/{config.epoch} | "
                f"avg_loss: {group_metrics['avg_loss']:.4f} | "
                f"policy_loss: {group_metrics['avg_policy_loss']:.4f} | "
                f"value_loss: {group_metrics['avg_value_loss']:.4f} | "
                f"entropy: {group_metrics['avg_entropy']:.4f} | "
                f"grad_norm: {group_metrics['avg_grad_norm']:.4f}"
            )

        # --- Save and generate artifacts ---
        self._save_weights(model, config.output_path)
        save_training_artifacts(
            epoch_losses, epoch_policy_losses, epoch_value_losses,
            epoch_entropies, epoch_grad_norms,
        )

    # --- Model building ---

    def _build_model(self, model_name: str) -> tuple[nn.Module, torch.device]:
        """Resolve model from registry, instantiate, load weights.

        Returns (model, device) tuple. Model is in training mode,
        moved to the detected device.

        Raises:
            ValueError: If model_name not in registry.
        """
        # Resolve from registry
        info = get_ai_info(model_name)
        module_name = info["module"]
        class_name = info["class"]
        weight_path = info["weight_path"]

        # Import module, instantiate class
        ai_module = importlib.import_module(module_name)
        model_class = getattr(ai_module, class_name)
        model = model_class()

        # Load weights
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        # Detect device (per design D6)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("CUDA not available, falling back to CPU", RuntimeWarning)

        model.to(device)
        model.train()  # Training mode (per TC-M3)
        return model, device

    # --- Sample loading ---

    def _load_samples(self, sample_path: str) -> list[dict]:
        """Load perspectives from sample file(s). Directory loads all .json.

        Raises RuntimeError if no valid samples found.
        """
        json_files: list[str] = []
        if os.path.isfile(sample_path):
            json_files = [sample_path]
        elif os.path.isdir(sample_path):
            for entry in sorted(os.listdir(sample_path)):
                if entry.lower().endswith(".json"):
                    json_files.append(os.path.join(sample_path, entry))

        all_perspectives: list[dict] = []
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    games = json.load(f)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse sample file {file_path!r}: {exc}"
                ) from exc

            for game in games:
                for perspective in game.get("perspectives", []):
                    # Deep-copy perspective to avoid mutation across epochs
                    all_perspectives.append({
                        "steps": [
                            {
                                "state": list(step["state"]),
                                "action": step["action"],
                                "reward": float(step["reward"]),
                                "value": float(step["value"]),
                                "log_prob": (
                                    float(step["log_prob"])
                                    if step.get("log_prob") is not None
                                    else 0.0
                                ),
                            }
                            for step in perspective["steps"]
                        ],
                    })

        if not all_perspectives:
            raise RuntimeError(
                f"no valid training samples found in {sample_path!r}"
            )
        return all_perspectives

    # --- Training helpers ---

    def _train_one_epoch(
        self,
        perspectives: list[dict],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        device: torch.device,
    ) -> dict:
        """Run one training epoch with gradient accumulation.

        Returns dict with keys: avg_loss, avg_policy_loss, avg_value_loss,
        avg_entropy, avg_grad_norm (averages across all groups).
        """
        # Accumulators for group-level averages
        all_losses: list[float] = []
        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_entropies: list[float] = []
        all_grad_norms: list[float] = []

        total = len(perspectives)
        idx = 0

        while idx < total:
            # Group of batch_size perspectives
            group_end = min(idx + config.batch_size, total)
            group = perspectives[idx:group_end]
            idx = group_end

            optimizer.zero_grad()

            group_loss_total = 0.0
            group_policy_total = 0.0
            group_value_total = 0.0
            group_entropy_total = 0.0
            group_has_nan = False

            for perspective in group:
                metrics = self._process_perspective(
                    perspective, model, config, device
                )
                if metrics is None:
                    # NaN detected — skip this perspective
                    group_has_nan = True
                    continue

                # Scale loss by batch_size for gradient accumulation (D8)
                scaled_loss = metrics["total_loss"] / len(group)
                scaled_loss.backward()

                group_loss_total += metrics["total_loss"].item()
                group_policy_total += metrics["policy_loss"].item()
                group_value_total += metrics["value_loss"].item()
                group_entropy_total += metrics["entropy"].item()

            if group_has_nan:
                warnings.warn(
                    f"NaN detected in group, skipping optimizer step"
                )
                continue

            # Gradient clipping (D11)
            nn_utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Compute gradient norm before stepping
            grad_norm = compute_grad_norm(model)

            # Check for NaN gradient (D12)
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                warnings.warn(
                    f"NaN/Inf gradient norm detected, skipping optimizer step"
                )
                continue

            optimizer.step()

            n = len(group)
            all_losses.append(group_loss_total / n)
            all_policy_losses.append(group_policy_total / n)
            all_value_losses.append(group_value_total / n)
            all_entropies.append(group_entropy_total / n)
            all_grad_norms.append(grad_norm)

        if not all_losses:
            # Fallback: all groups had NaN
            return {
                "avg_loss": float("nan"),
                "avg_policy_loss": float("nan"),
                "avg_value_loss": float("nan"),
                "avg_entropy": float("nan"),
                "avg_grad_norm": float("nan"),
            }

        return {
            "avg_loss": sum(all_losses) / len(all_losses),
            "avg_policy_loss": sum(all_policy_losses) / len(all_policy_losses),
            "avg_value_loss": sum(all_value_losses) / len(all_value_losses),
            "avg_entropy": sum(all_entropies) / len(all_entropies),
            "avg_grad_norm": sum(all_grad_norms) / len(all_grad_norms),
        }

    # --- Perspective processing ---

    def _process_perspective(
        self,
        perspective: dict,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> dict | None:
        """Process one perspective: GAE + forward + PPO loss. Returns None if NaN."""
        steps = perspective["steps"]
        T = len(steps)

        # --- Step 1: Load old data from sample ---
        old_log_probs: list[float] = []
        old_values: list[float] = []
        rewards: list[float] = []
        states_14d: list[list[float]] = []
        actions: list[str] = []

        for step in steps:
            states_14d.append(step["state"])
            actions.append(step["action"])
            old_values.append(step["value"])
            rewards.append(step["reward"])
            # Replace null with 0.0 (from greedy sampling during self-play)
            old_log_probs.append(step["log_prob"])

        # --- Step 2: Compute GAE advantages (D10) ---
        advantages, value_targets = self._compute_gae(
            old_values, rewards, config.gamma, config.gae_lambda, device
        )

        # --- Step 3: Forward pass to get new policy ---
        new_log_probs: list[float] = []
        new_values: list[float] = []
        all_probs: list[torch.Tensor] = []  # For entropy computation

        h = torch.zeros(1, 1, HIDDEN_SIZE, device=device)

        for t in range(T):
            state_t = states_14d[t]
            x = torch.tensor(state_t, dtype=torch.float32, device=device).view(1, 1, INPUT_SIZE)

            # Forward with temperature=1.0, mask_fn=None (D9)
            probs, log_probs, value, h = model(x, h, mask_fn=None, temperature=1.0)

            # Extract new_log_prob for the action that was taken (D13)
            action_idx = ACTION_TO_INDEX[actions[t]]
            new_log_probs.append(float(log_probs[0, action_idx].item()))
            new_values.append(float(value.item()))
            all_probs.append(probs)

        # --- Step 4: Compute PPO clipped loss ---
        loss_result = self._compute_ppo_loss(
            old_log_probs, new_log_probs,
            old_values, new_values,
            advantages, value_targets,
            all_probs,
            config,
        )

        # NaN detection (D12)
        if loss_result is None:
            warnings.warn("NaN loss detected in perspective, skipping")
            return None

        return loss_result

    # ------------------------------------------------------------------
    # Internal: GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        old_values: list[float],
        rewards: list[float],
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> tuple[list[float], list[float]]:
        """Compute GAE. V_{t+1}=0 for terminal step. Returns (advantages, value_targets)."""
        T = len(old_values)
        advantages = [0.0] * T
        value_targets = [0.0] * T

        gae = 0.0
        for t in reversed(range(T)):
            # Terminal bootstrap: V_{t+1} = 0 for last step
            next_value = old_values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + gamma * next_value - old_values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            value_targets[t] = gae + old_values[t]

        return advantages, value_targets

    # ------------------------------------------------------------------
    # Internal: PPO loss computation
    # ------------------------------------------------------------------

    def _compute_ppo_loss(
        self,
        old_log_probs: list[float],
        new_log_probs: list[float],
        old_values: list[float],
        new_values: list[float],
        advantages: list[float],
        value_targets: list[float],
        all_probs: list[torch.Tensor],
        config: TrainingConfig,
    ) -> dict | None:
        """Compute PPO clipped loss, averaged over all T steps.

        Loss components:
            L_clip = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
            L_value = mean((V_target - V_new)^2)
            L_entropy = -mean(sum(p * log(p)))
            L_total = L_clip + cv * L_value - ce * L_entropy

        Returns dict with total_loss, policy_loss, value_loss, entropy.
        Returns None if any component is NaN.
        """
        T = len(old_log_probs)
        device = all_probs[0].device

        # Convert to tensors
        old_log = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        new_log = torch.tensor(new_log_probs, dtype=torch.float32, device=device)
        new_val = torch.tensor(new_values, dtype=torch.float32, device=device)
        old_val = torch.tensor(old_values, dtype=torch.float32, device=device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=device)
        v_target = torch.tensor(value_targets, dtype=torch.float32, device=device)

        # Policy loss: PPO clipped objective
        ratio = torch.exp(new_log - old_log)
        clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

        # Value loss: MSE
        value_loss = torch.nn.functional.mse_loss(new_val, v_target)

        # Entropy bonus (regularization)
        entropy_sum = torch.tensor(0.0, device=device)
        for probs in all_probs:
            # probs shape: (1, 4), avoid log(0)
            p = probs.clamp(min=1e-8)
            entropy_sum = entropy_sum - (p * torch.log(p)).sum()
        entropy = entropy_sum / T

        # Total loss (averaged over T steps per §6)
        total_loss = (
            policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy
        )

        # NaN detection (D12)
        if (
            torch.isnan(total_loss)
            or torch.isnan(policy_loss)
            or torch.isnan(value_loss)
            or torch.isnan(entropy)
        ):
            return None

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------
    # Internal: weight saving
    # ------------------------------------------------------------------

    def _save_weights(self, model: nn.Module, output_path: str) -> None:
        """Save trained model weights to output_path (D4).

        Creates parent directories if needed (TC-S3).
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), output_path)
