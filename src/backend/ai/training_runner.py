"""Base PPO training runner — shared logic for v1/v2 (design/31 D7).

Phase 1: value training, Phase 2: advantage re-computation, Phase 3: policy training.

Subclasses:
    V1TrainingRunner: rnn_v1/v1_training_runner.py (GRU bypass)
    V2TrainingRunner: rnn_v2/v2_training_runner.py (skip-connection forward)
"""

import abc
import importlib
import math
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from src.backend.ai.registry import get_ai_info
from src.backend.ai.sample_loader import SampleLoader
from src.backend.ai.training_config import TrainingConfig
from src.backend.ai.training_utils import (
    compute_grad_norm,
    save_split_training_artifacts,
)


# ---------------------------------------------------------------------------
# TrainingRunner
# ---------------------------------------------------------------------------


class BaseTrainingRunner(abc.ABC):
    """Abstract base class for three-phase split PPO training.

    Shared: run(), _build_models(), _load_samples(), _compute_gae(), _save_weights(),
            _train_value_epoch(), _train_policy_epoch().
    Abstract: _recompute_advantages(), _process_value_perspective(), _process_policy_perspective().
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: TrainingConfig) -> None:
        """Execute three-phase split PPO training."""
        random.seed(0)
        policy_model, value_model, device = self._build_models(config.model)
        all_perspectives = self._load_samples(config.sample_path)

        # Split train/val by game_id (design/33)
        train_perspectives, val_perspectives = self._split_train_val(
            all_perspectives, config.validation_split,
        )
        if val_perspectives:
            print(
                f"Split: {len(train_perspectives)} train, "
                f"{len(val_perspectives)} val perspectives"
            )

        policy_optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.policy_lr,
            weight_decay=config.weight_decay,
        )
        value_optimizer = torch.optim.AdamW(
            value_model.parameters(),
            lr=config.value_lr,
            weight_decay=config.weight_decay,
        )

        phase_records: list[dict] = []

        # Phase 1: Value Training
        for epoch in range(1, config.value_epoch + 1):
            random.shuffle(train_perspectives)
            metrics = self._train_value_epoch(
                train_perspectives, val_perspectives,
                value_model, policy_model,
                value_optimizer, config, device,
            )
            phase_records.append({
                "phase": "value",
                "epoch": epoch,
                "policy_loss": float("nan"),
                "value_loss": metrics["avg_value_loss"],
                "val_value_loss": metrics["val_value_loss"],
                "entropy": float("nan"),
                "policy_grad_norm": float("nan"),
                "value_grad_norm": metrics["avg_grad_norm"],
            })
            val_str = (
                f"val_value_loss: {metrics['val_value_loss']:.4f} | "
                if not math.isnan(metrics["val_value_loss"])
                else ""
            )
            print(
                f"[Value] Epoch {epoch}/{config.value_epoch} | "
                f"value_loss: {metrics['avg_value_loss']:.4f} | "
                f"{val_str}"
                f"grad_norm: {metrics['avg_grad_norm']:.4f}"
            )

        # Phase 2: Advantage Re-computation
        self._recompute_advantages(
            all_perspectives, value_model, config, device,
        )

        # Phase 3: Policy Training
        for epoch in range(1, config.policy_epoch + 1):
            random.shuffle(all_perspectives)
            metrics = self._train_policy_epoch(
                all_perspectives, policy_model, value_model,
                policy_optimizer, config, device,
            )
            phase_records.append({
                "phase": "policy",
                "epoch": epoch,
                "policy_loss": metrics["avg_policy_loss"],
                "value_loss": float("nan"),
                "val_value_loss": float("nan"),
                "entropy": metrics["avg_entropy"],
                "policy_grad_norm": metrics["avg_grad_norm"],
                "value_grad_norm": float("nan"),
            })
            print(
                f"[Policy] Epoch {epoch}/{config.policy_epoch} | "
                f"policy_loss: {metrics['avg_policy_loss']:.4f} | "
                f"entropy: {metrics['avg_entropy']:.4f} | "
                f"grad_norm: {metrics['avg_grad_norm']:.4f}"
            )

        policy_path = os.path.join(config.output_dir, "policy.pth")
        value_path = os.path.join(config.output_dir, "value.pth")
        self._save_weights(policy_model, policy_path)
        self._save_weights(value_model, value_path)
        save_split_training_artifacts(phase_records, output_dir=None)

    # -- Model building --

    def _build_models(
        self, model_name: str
    ) -> tuple[nn.Module, nn.Module, torch.device]:
        """Resolve registry entry, instantiate both networks, move to device."""
        info = get_ai_info(model_name)

        policy_info = info["policy"]
        policy_module = importlib.import_module(policy_info["module"])
        policy_class = getattr(policy_module, policy_info["class"])
        policy_model = policy_class()
        policy_state = torch.load(
            policy_info["weight_path"], map_location="cpu", weights_only=True
        )
        policy_model.load_state_dict(policy_state)

        value_info = info["value"]
        value_module = importlib.import_module(value_info["module"])
        value_class = getattr(value_module, value_info["class"])
        value_model = value_class()
        value_state = torch.load(
            value_info["weight_path"], map_location="cpu", weights_only=True
        )
        value_model.load_state_dict(value_state)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("CUDA not available, falling back to CPU", RuntimeWarning)

        policy_model.to(device)
        policy_model.train()
        value_model.to(device)
        value_model.train()

        return policy_model, value_model, device

    # -- Sample loading --

    def _load_samples(self, sample_path: str) -> list[dict]:
        """Load perspectives from sample file(s). Delegates to SampleLoader."""
        return SampleLoader.load(sample_path)

    # -- Train/val split --

    @staticmethod
    def _split_train_val(
        perspectives: list[dict],
        validation_split: float,
    ) -> tuple[list[dict], list[dict]]:
        """Split perspectives into train/val by game_id.

        Groups perspectives by game_id, randomly shuffles game IDs,
        then assigns the first (validation_split * num_games) games to val.
        At least 1 game is kept in train when possible.

        Args:
            perspectives: List of perspective dicts, each with "game_id" key.
            validation_split: Fraction of games for validation (0 <= split < 1).

        Returns:
            (train_perspectives, val_perspectives) tuple.
        """
        # Group perspectives by game_id
        game_groups: dict[str, list[dict]] = {}
        for p in perspectives:
            raw_gid = p.get("game_id")
            if raw_gid is None:
                warnings.warn(
                    "perspective missing 'game_id', grouping as 'unknown'. "
                    "This may cause data leakage across games."
                )
                raw_gid = "unknown"
            gid = str(raw_gid)
            if gid not in game_groups:
                game_groups[gid] = []
            game_groups[gid].append(p)

        game_ids = list(game_groups.keys())

        # validation_split=0 or only 1 game: all to train
        if validation_split <= 0 or len(game_ids) <= 1:
            return list(perspectives), []

        random.shuffle(game_ids)
        n_val = max(1, int(len(game_ids) * validation_split))
        # Ensure at least 1 game in train
        n_val = min(n_val, len(game_ids) - 1)
        val_gids = set(game_ids[:n_val])

        train: list[dict] = []
        val: list[dict] = []
        for gid, perps in game_groups.items():
            if gid in val_gids:
                val.extend(perps)
            else:
                train.extend(perps)
        return train, val

    # -- Validation loss --

    def _compute_val_loss(
        self,
        val_perspectives: list[dict],
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> float:
        """Compute validation loss on val set (no gradient).

        Sets model to eval mode, computes MSE loss for each perspective
        via _process_value_perspective, then restores train mode.

        Args:
            val_perspectives: Validation perspectives.
            value_model: Value network to evaluate.
            config: Training configuration.
            device: Torch device.

        Returns:
            Average val value loss, or NaN if val set is empty.
        """
        if not val_perspectives:
            return float("nan")

        value_model.eval()
        total_loss = 0.0
        count = 0
        try:
            with torch.no_grad():
                for perspective in val_perspectives:
                    value_loss = self._process_value_perspective(
                        perspective, value_model, config, device,
                    )
                    if value_loss is not None:
                        total_loss += value_loss.item()
                        count += 1
        finally:
            value_model.train()

        if count == 0:
            return float("nan")
        return total_loss / count

    # -- Phase 1: Value Training (batching loop) --

    def _train_value_epoch(
        self,
        perspectives: list[dict],
        val_perspectives: list[dict],
        value_model: nn.Module,
        policy_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        device: torch.device,
    ) -> dict:
        """Run one value-training epoch with gradient accumulation.

        Only value_model is updated. policy_model is unused (frozen).
        After training, computes validation loss on val_perspectives.
        Returns dict with avg_value_loss, val_value_loss, and avg_grad_norm.
        """
        all_value_losses: list[float] = []
        all_grad_norms: list[float] = []

        total = len(perspectives)
        idx = 0

        while idx < total:
            group_end = min(idx + config.batch_size, total)
            group = perspectives[idx:group_end]
            idx = group_end

            optimizer.zero_grad()

            group_value_total = 0.0
            group_has_nan = False

            for perspective in group:
                value_loss = self._process_value_perspective(
                    perspective, value_model, config, device,
                )
                if value_loss is None:
                    group_has_nan = True
                    continue

                scaled = value_loss / len(group)
                scaled.backward()

                group_value_total += value_loss.item()

            if group_has_nan:
                warnings.warn("NaN detected in value group, skipping")
                continue

            # Gradient clipping
            pre_clip_norm = compute_grad_norm(value_model)
            if math.isnan(pre_clip_norm) or math.isinf(pre_clip_norm):
                warnings.warn("NaN/Inf gradient in value model, skipping")
                continue

            nn_utils.clip_grad_norm_(
                value_model.parameters(), config.max_grad_norm,
            )
            optimizer.step()

            n = len(group)
            all_value_losses.append(group_value_total / n)
            all_grad_norms.append(pre_clip_norm)

        if not all_value_losses:
            return {
                "avg_value_loss": float("nan"),
                "avg_grad_norm": float("nan"),
                "val_value_loss": float("nan"),
            }

        # Compute validation loss after training (design/33)
        val_loss = self._compute_val_loss(
            val_perspectives, value_model, config, device,
        )

        return {
            "avg_value_loss": sum(all_value_losses) / len(all_value_losses),
            "avg_grad_norm": sum(all_grad_norms) / len(all_grad_norms),
            "val_value_loss": val_loss,
        }

    @abc.abstractmethod
    def _process_value_perspective(
        self,
        perspective: dict,
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Compute value MSE loss for one perspective. Returns None if NaN.

        Must be implemented by subclass — forward strategy differs per version.
        """
        raise NotImplementedError

    # -- Phase 2: Advantage Re-computation (abstract) --

    @abc.abstractmethod
    def _recompute_advantages(
        self,
        perspectives: list[dict],
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Re-compute GAE advantages using trained value network (no_grad).

        Stores new_advantages and new_value_targets in each perspective dict.
        Must be implemented by subclass — forward strategy differs per version.
        """
        raise NotImplementedError

    # -- Phase 3: Policy Training (batching loop) --

    def _train_policy_epoch(
        self,
        perspectives: list[dict],
        policy_model: nn.Module,
        value_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        device: torch.device,
    ) -> dict:
        """Run one policy-training epoch with gradient accumulation.

        Only policy_model is updated. value_model is frozen.
        Returns dict with avg_policy_loss, avg_entropy, avg_grad_norm.
        """
        all_policy_losses: list[float] = []
        all_entropies: list[float] = []
        all_grad_norms: list[float] = []

        total = len(perspectives)
        idx = 0

        while idx < total:
            group_end = min(idx + config.batch_size, total)
            group = perspectives[idx:group_end]
            idx = group_end

            optimizer.zero_grad()

            group_policy_total = 0.0
            group_entropy_total = 0.0
            group_has_nan = False

            for perspective in group:
                result = self._process_policy_perspective(
                    perspective, policy_model, config, device,
                )
                if result is None:
                    group_has_nan = True
                    continue

                policy_loss, entropy = result
                total_loss = policy_loss - config.entropy_coef * entropy
                scaled = total_loss / len(group)
                scaled.backward()

                group_policy_total += policy_loss.item()
                group_entropy_total += entropy.item()

            if group_has_nan:
                warnings.warn("NaN detected in policy group, skipping")
                continue

            pre_clip_norm = compute_grad_norm(policy_model)
            if math.isnan(pre_clip_norm) or math.isinf(pre_clip_norm):
                warnings.warn("NaN/Inf gradient in policy model, skipping")
                continue

            nn_utils.clip_grad_norm_(
                policy_model.parameters(), config.max_grad_norm,
            )
            optimizer.step()

            n = len(group)
            all_policy_losses.append(group_policy_total / n)
            all_entropies.append(group_entropy_total / n)
            all_grad_norms.append(pre_clip_norm)

        if not all_policy_losses:
            return {
                "avg_policy_loss": float("nan"),
                "avg_entropy": float("nan"),
                "avg_grad_norm": float("nan"),
            }

        return {
            "avg_policy_loss": sum(all_policy_losses) / len(all_policy_losses),
            "avg_entropy": sum(all_entropies) / len(all_entropies),
            "avg_grad_norm": sum(all_grad_norms) / len(all_grad_norms),
        }

    @abc.abstractmethod
    def _process_policy_perspective(
        self,
        perspective: dict,
        policy_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> tuple | None:
        """Compute PPO policy loss for one perspective. Returns None if NaN.

        Must be implemented by subclass — forward strategy differs per version.
        """
        raise NotImplementedError

    # -- GAE computation (shared) --

    def _compute_gae(
        self,
        values: list[float],
        rewards: list[float],
        gamma: float,
        gae_lambda: float,
    ) -> tuple[list[float], list[float]]:
        """Compute GAE. V_{t+1}=0 for terminal step.

        Returns (advantages, value_targets).
        """
        T = len(values)
        advantages = [0.0] * T
        value_targets = [0.0] * T

        gae = 0.0
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            value_targets[t] = gae + values[t]

        return advantages, value_targets

    # -- Weight saving --

    def _save_weights(self, model: nn.Module, output_path: str) -> None:
        """Save model weights, creating parent directories if needed."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), output_path)
