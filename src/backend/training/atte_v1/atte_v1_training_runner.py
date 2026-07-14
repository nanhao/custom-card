"""AtteV1TrainingRunner — three-phase PPO training for atte_v1 model (design/40).

Inherits BaseTrainingRunner.run() three-phase loop.
Overrides _load_samples, _process_value_perspective, _recompute_advantages,
_process_policy_perspective for atte_v1 variable-length sequence input format.
"""

import torch
import torch.nn as nn

from src.backend.ai.atte_v1.constants import HIDDEN_SIZE
from src.backend.training.training_config import TrainingConfig
from src.backend.training.training_runner import BaseTrainingRunner


class AtteV1TrainingRunner(BaseTrainingRunner):
    """Three-phase split PPO training for atte_v1 model."""

    def _load_samples(self, sample_path: str) -> list[dict]:
        """Load atte_v1 format samples from JSON.

        Overrides base to handle AtteV1Input format (variable-length sequences).
        """
        return super()._load_samples(sample_path)

    def _process_value_perspective(
        self,
        perspective: dict,
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Phase 1: value MSE loss for one perspective."""
        steps = perspective.get("steps", [])
        if not steps:
            return None

        T = len(steps)
        old_values = [s.get("value", 0.0) for s in steps]
        rewards = [s.get("reward", 0.0) for s in steps]

        _advantages, value_targets = self._compute_gae(
            old_values, rewards, config.gamma, config.gae_lambda,
        )

        v_target = torch.tensor(value_targets, dtype=torch.float32, device=device)
        v_new_list: list[torch.Tensor] = []
        h = torch.tensor(
            [0.0] * HIDDEN_SIZE, dtype=torch.float32, device=device,
        ).view(1, 1, HIDDEN_SIZE)

        for s in steps:
            inp = s.get("input")
            if inp is None:
                v_new_list.append(torch.tensor(0.0, device=device))
                continue
            kwargs = self._build_value_kwargs(inp, h)
            kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            value, h = value_model(**kwargs)
            v_new_list.append(value.squeeze())

        v_new_t = torch.stack(v_new_list)
        value_loss = config.value_coef * nn.functional.mse_loss(v_new_t, v_target)
        if torch.isnan(value_loss):
            return None
        return value_loss

    def _recompute_advantages(
        self,
        perspectives: list[dict],
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Phase 2: recompute GAE advantages."""
        with torch.no_grad():
            for perspective in perspectives:
                steps = perspective.get("steps", [])
                if not steps:
                    continue
                rewards = [s.get("reward", 0.0) for s in steps]
                h = torch.tensor(
                    [0.0] * HIDDEN_SIZE, dtype=torch.float32, device=device,
                ).view(1, 1, HIDDEN_SIZE)
                new_values: list[float] = []
                for s in steps:
                    inp = s.get("input")
                    if inp is None:
                        new_values.append(0.0)
                        continue
                    kwargs = self._build_value_kwargs(inp, h)
                    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                    value, h = value_model(**kwargs)
                    new_values.append(float(value.item()))
                advantages, value_targets = self._compute_gae(
                    new_values, rewards, config.gamma, config.gae_lambda,
                )
                perspective["new_advantages"] = advantages
                perspective["new_value_targets"] = value_targets

    def _process_policy_perspective(
        self,
        perspective: dict,
        policy_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> tuple | None:
        """Phase 3: PPO policy loss for one perspective."""
        steps = perspective.get("steps", [])
        if not steps:
            return None

        T = len(steps)
        advantages = perspective.get("new_advantages", [0.0] * T)
        old_log_probs = [s.get("log_prob", 0.0) for s in steps]
        actions = [s.get("action", 0) for s in steps]

        new_logs: list[torch.Tensor] = []
        h = torch.tensor(
            [0.0] * HIDDEN_SIZE, dtype=torch.float32, device=device,
        ).view(1, 1, HIDDEN_SIZE)

        for s in steps:
            inp = s.get("input")
            if inp is None:
                new_logs.append(torch.tensor(-1e9, device=device))
                continue
            kwargs = self._build_policy_kwargs(inp, h)
            kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            _probs, log_probs_all, h = policy_model(**kwargs)
            action_idx = s.get("action", 0)
            new_logs.append(log_probs_all[0, action_idx])

        new_log_t = torch.stack(new_logs)
        old_log_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        ratio = torch.exp(new_log_t - old_log_t)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps,
        )
        policy_loss = -torch.min(ratio * adv_t, clipped_ratio * adv_t).mean()

        entropy = 0.0  # Simplified; full entropy requires full log_probs
        if torch.isnan(policy_loss):
            return None
        return policy_loss, torch.tensor(entropy)

    @staticmethod
    def _build_value_kwargs(inp, hidden: torch.Tensor) -> dict:
        """Build kwargs for value model forward from AtteV1Input or dict."""
        from src.backend.ai_adapter.atte_v1_ai import _build_model_kwargs
        from src.backend.ai_adapter.atte_v1_converter import AtteV1Input, AtteV1PlayerInput
        if isinstance(inp, dict):
            own_d = inp["own"]
            opp_d = inp["opp"]
            inp = AtteV1Input(
                own=AtteV1PlayerInput(
                    hand_ids=own_d["hand_ids"],
                    discard_ids=own_d["discard_ids"],
                    draw_ids=own_d["draw_ids"],
                    last_action=own_d["last_action"],
                    hp=own_d["hp"],
                    energy=own_d["energy"],
                    status_ids=own_d["status_ids"],
                    status_durations=own_d["status_durations"],
                    opp_hand_ids=own_d["opp_hand_ids"],
                    opp_discard_ids=own_d["opp_discard_ids"],
                    opp_draw_ids=own_d["opp_draw_ids"],
                ),
                opp=AtteV1PlayerInput(
                    hand_ids=opp_d["hand_ids"],
                    discard_ids=opp_d["discard_ids"],
                    draw_ids=opp_d["draw_ids"],
                    last_action=opp_d["last_action"],
                    hp=opp_d["hp"],
                    energy=opp_d["energy"],
                    status_ids=opp_d["status_ids"],
                    status_durations=opp_d["status_durations"],
                    opp_hand_ids=opp_d["opp_hand_ids"],
                    opp_discard_ids=opp_d["opp_discard_ids"],
                    opp_draw_ids=opp_d["opp_draw_ids"],
                ),
                round_in_cycle=inp["round_in_cycle"],
            )
        elif isinstance(inp, list):
            raise RuntimeError(
                "Sample input is old flat-list format. "
                "Delete old samples and regenerate with the updated self-play."
            )
        return _build_model_kwargs(inp, hidden.squeeze(1).squeeze(0).tolist())

    @staticmethod
    def _build_policy_kwargs(inp, hidden: torch.Tensor) -> dict:
        """Build kwargs for policy model forward from AtteV1Input."""
        return AtteV1TrainingRunner._build_value_kwargs(inp, hidden)
