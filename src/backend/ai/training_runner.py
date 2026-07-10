"""Base PPO training runner — shared logic for v1/v2 (design/31 D7).

Phase 1: value training, Phase 2: advantage re-computation, Phase 3: policy training.

Subclasses:
    V1TrainingRunner: rnn_v1/v1_training_runner.py (GRU bypass)
    V2TrainingRunner: rnn_v2/v2_training_runner.py (skip-connection forward)
"""

import abc
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
    HIDDEN_SIZE,
    INPUT_SIZE,
    POLICY_OUTPUT_SIZE,
)
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
        for epoch in range(1, config.epoch + 1):
            random.shuffle(all_perspectives)
            metrics = self._train_value_epoch(
                all_perspectives, value_model, policy_model,
                value_optimizer, config, device,
            )
            phase_records.append({
                "phase": "value",
                "epoch": epoch,
                "policy_loss": float("nan"),
                "value_loss": metrics["avg_value_loss"],
                "entropy": float("nan"),
                "policy_grad_norm": float("nan"),
                "value_grad_norm": metrics["avg_grad_norm"],
            })
            print(
                f"[Value] Epoch {epoch}/{config.epoch} | "
                f"value_loss: {metrics['avg_value_loss']:.4f} | "
                f"grad_norm: {metrics['avg_grad_norm']:.4f}"
            )

        # Phase 2: Advantage Re-computation
        self._recompute_advantages(
            all_perspectives, value_model, config, device,
        )

        # Phase 3: Policy Training
        for epoch in range(1, config.epoch + 1):
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
                "entropy": metrics["avg_entropy"],
                "policy_grad_norm": metrics["avg_grad_norm"],
                "value_grad_norm": float("nan"),
            })
            print(
                f"[Policy] Epoch {epoch}/{config.epoch} | "
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
            print(f"Loading {file_path}...")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    games = json.load(f)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse sample file {file_path!r}: {exc}"
                ) from exc

            print(f"  Parsed {len(games)} games, validating steps...")
            total_games = len(games)
            for game_idx, game in enumerate(games):
                # Print progress every 10% of games
                if game_idx % max(1, total_games // 10) == 0:
                    pct = int(game_idx / total_games * 100)
                    print(f"  Validating... {pct}% ({game_idx}/{total_games} games)")
                for p_idx, perspective in enumerate(game.get("perspectives", [])):
                    for s_idx, step in enumerate(perspective.get("steps", [])):
                        for key in ("input", "action", "reward", "value"):
                            if key not in step:
                                raise RuntimeError(
                                    f"Step {s_idx} in perspective {p_idx} "
                                    f"of game {game.get('game_id', '?')} "
                                    f"in {file_path!r} is missing required key {key!r}"
                                )
                        try:
                            float(step["reward"])
                            float(step["value"])
                            log_prob_raw = step.get("log_prob")
                            float(log_prob_raw) if log_prob_raw is not None else 0.0
                            # Validate action is an integer policy index (design/14)
                            action = step["action"]
                            if not isinstance(action, int) or action < 0 or action >= POLICY_OUTPUT_SIZE:
                                raise RuntimeError(
                                    f"Step {s_idx} in perspective {p_idx} "
                                    f"of game {game.get('game_id', '?')} "
                                    f"in {file_path!r}: action must be int in [0, {POLICY_OUTPUT_SIZE - 1}], "
                                    f"got {action!r}"
                                )
                        except (ValueError, TypeError) as exc:
                            raise RuntimeError(
                                f"Step {s_idx} in perspective {p_idx} "
                                f"of game {game.get('game_id', '?')} "
                                f"in {file_path!r}: invalid numeric value — {exc}"
                            ) from exc
                        if len(step["input"]) != INPUT_SIZE:
                            raise RuntimeError(
                                f"Step {s_idx} in perspective {p_idx} "
                                f"of game {game.get('game_id', '?')} "
                                f"in {file_path!r}: input has {len(step['input'])} elements, "
                                f"expected {INPUT_SIZE}"
                            )
                        # Deep-copy perspective
                        all_perspectives.append({
                            "steps": [
                                {
                                    "input": list(step["input"]),
                                    "action": step["action"],
                                    "reward": float(step["reward"]),
                                    "value": float(step["value"]),
                                    "log_prob": (
                                        float(step["log_prob"])
                                        if step.get("log_prob") is not None
                                        else 0.0
                                    ),
                                    "build_target": step.get("build_target"),
                                }
                                for step in perspective["steps"]
                            ],
                        })

        if not all_perspectives:
            raise RuntimeError(
                f"no valid training samples found in {sample_path!r}"
            )

        total_steps = sum(len(p["steps"]) for p in all_perspectives)
        print(f"  Loaded {len(all_perspectives)} perspectives ({total_steps} steps)")
        return all_perspectives

    # -- Phase 1: Value Training (batching loop) --

    def _train_value_epoch(
        self,
        perspectives: list[dict],
        value_model: nn.Module,
        policy_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        device: torch.device,
    ) -> dict:
        """Run one value-training epoch with gradient accumulation.

        Only value_model is updated. policy_model is unused (frozen).
        Returns dict with avg_value_loss and avg_grad_norm.
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
            return {"avg_value_loss": float("nan"), "avg_grad_norm": float("nan")}

        return {
            "avg_value_loss": sum(all_value_losses) / len(all_value_losses),
            "avg_grad_norm": sum(all_grad_norms) / len(all_grad_norms),
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
