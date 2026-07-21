"""AtteV1TrainingRunner — three-phase PPO training for atte_v1 model (design/40).

Inherits BaseTrainingRunner.run() three-phase loop.
Design/43: batch GRU optimization — pre-compute per-step features, then
one GRU call for the entire perspective sequence.
"""

import torch
import torch.nn as nn

from src.backend.ai.atte_v1.constants import HIDDEN_SIZE
from src.backend.training.training_config import TrainingConfig
from src.backend.training.training_runner import BaseTrainingRunner


class AtteV1TrainingRunner(BaseTrainingRunner):
    """Three-phase split PPO training for atte_v1 model with batch GRU (design/43)."""

    # ------------------------------------------------------------------
    # Public override
    # ------------------------------------------------------------------

    def _load_samples(self, sample_path: str) -> list[dict]:
        """Load atte_v1 format samples from JSON.

        Overrides base to handle AtteV1Input format (variable-length sequences).
        """
        return super()._load_samples(sample_path)

    # ------------------------------------------------------------------
    # New: batch GRU helpers (design/43)
    # ------------------------------------------------------------------

    def _compute_all_pre_gru(
        self,
        perspective: dict,
        device: torch.device,
        body: torch.nn.Module | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        """Compute forward_pre_gru() for all steps in a perspective.

        Args:
            perspective: dict with "steps" key.
            device: torch device.
            body: _AtteV1Body instance. Created lazily if None.

        Returns list of (gru_input, intermediates) tuples, one per step.
        Steps with input=None produce a zero gru_input tensor and empty dict.
        """
        from src.backend.ai.atte_v1.model import _AtteV1Body
        if body is None:
            body = _AtteV1Body().to(device)
        steps = perspective.get("steps", [])
        results: list[tuple[torch.Tensor, dict]] = []
        for s in steps:
            inp = s.get("input")
            if inp is None:
                zero = torch.zeros(1, 80, device=device)
                results.append((zero, {}))
                continue
            kwargs = self._build_pre_gru_kwargs(inp, device)
            gru_input, intermediates = body.forward_pre_gru(**kwargs)
            results.append((gru_input, intermediates))
        return results

    def _forward_value_sequence(
        self,
        pre_gru_results: list[tuple[torch.Tensor, dict]],
        value_model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Batch value forward: stack gru_inputs → one GRU → per-step post_gru + value head.

        Args:
            pre_gru_results: list of (gru_input, intermediates) from _compute_all_pre_gru.
            value_model: AtteV1ValueModel instance.
            device: torch device.

        Returns:
            values: (T,) float32 tensor of value estimates.
        """
        if not pre_gru_results:
            return torch.empty(0, device=device)

        # Stack all gru_inputs → (T, 1, 80)
        gru_inputs = [r[0].unsqueeze(0) for r in pre_gru_results]
        stacked = torch.cat(gru_inputs, dim=0)  # (T, 1, 80)  or (T, batch, 80)
        # Determine batch dim from first result
        batch_size = pre_gru_results[0][0].shape[0]

        h0 = torch.zeros(
            1, batch_size, HIDDEN_SIZE,
            dtype=torch.float32, device=device,
        )
        gru_out, _new_h = value_model.body.gru(stacked, h0)  # (T, batch, 32)

        # Per-step post_gru + value head
        values: list[torch.Tensor] = []
        for t_idx, (_gru_input, intermediates) in enumerate(pre_gru_results):
            if not intermediates:
                values.append(torch.tensor(0.0, device=device))
                continue
            h_out_t = gru_out[t_idx]  # (batch, 32)
            out = value_model.body.forward_post_gru(h_out_t, intermediates)
            h1 = torch.relu(value_model.fc1(out))
            v = value_model.fc_value(h1).squeeze(-1)  # (batch,) or (batch, 1)
            values.append(v.squeeze())

        return torch.stack(values)  # (T,)

    def _forward_policy_sequence(
        self,
        pre_gru_results: list[tuple[torch.Tensor, dict]],
        policy_model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Batch policy forward: stack gru_inputs → one GRU → per-step post_gru + policy head.

        Args:
            pre_gru_results: list of (gru_input, intermediates) from _compute_all_pre_gru.
            policy_model: AtteV1PolicyModel instance.
            device: torch device.

        Returns:
            log_probs: (T, 24) float32 tensor.
        """
        if not pre_gru_results:
            return torch.empty(0, 24, device=device)

        gru_inputs = [r[0].unsqueeze(0) for r in pre_gru_results]
        stacked = torch.cat(gru_inputs, dim=0)  # (T, 1, 80)
        batch_size = pre_gru_results[0][0].shape[0]

        h0 = torch.zeros(
            1, batch_size, HIDDEN_SIZE,
            dtype=torch.float32, device=device,
        )
        gru_out, _new_h = policy_model.body.gru(stacked, h0)  # (T, batch, 32)

        log_probs_list: list[torch.Tensor] = []
        for t_idx, (_gru_input, intermediates) in enumerate(pre_gru_results):
            if not intermediates:
                log_probs_list.append(
                    torch.full((batch_size, 24), -1e9, device=device),
                )
                continue
            h_out_t = gru_out[t_idx]
            out = policy_model.body.forward_post_gru(h_out_t, intermediates)
            h1 = torch.relu(policy_model.fc1(out))
            logits = policy_model.fc_policy(h1)
            log_probs_list.append(torch.log_softmax(logits, dim=-1))

        return torch.cat(log_probs_list, dim=0)  # (T, 24)

    # ------------------------------------------------------------------
    # Phase 1: Value Training (design/43: batch GRU)
    # ------------------------------------------------------------------

    def _process_value_perspective(
        self,
        perspective: dict,
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Phase 1: value MSE loss using batch GRU forward."""
        steps = perspective.get("steps", [])
        if not steps:
            return None

        T = len(steps)
        old_values = [s.get("value", 0.0) for s in steps]
        rewards = [s.get("reward", 0.0) for s in steps]

        _advantages, value_targets = self._compute_gae(
            old_values, rewards, config.gamma, config.gae_lambda,
        )

        # Compute pre-GRU for all steps, then batched GRU + post-GRU
        pre_gru_results = self._compute_all_pre_gru(
            perspective, device, body=value_model.body,
        )
        v_new = self._forward_value_sequence(pre_gru_results, value_model, device)

        v_target = torch.tensor(value_targets, dtype=torch.float32, device=device)
        value_loss = config.value_coef * nn.functional.mse_loss(v_new, v_target)
        if torch.isnan(value_loss):
            return None
        return value_loss

    # ------------------------------------------------------------------
    # Phase 2: Advantage Re-computation (design/43: batch GRU)
    # ------------------------------------------------------------------

    def _recompute_advantages(
        self,
        perspectives: list[dict],
        value_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Phase 2: recompute GAE advantages using batch GRU forward."""
        with torch.no_grad():
            for perspective in perspectives:
                steps = perspective.get("steps", [])
                if not steps:
                    continue
                rewards = [s.get("reward", 0.0) for s in steps]

                pre_gru_results = self._compute_all_pre_gru(
                    perspective, device, body=value_model.body,
                )
                v_new = self._forward_value_sequence(
                    pre_gru_results, value_model, device,
                )
                new_values = v_new.tolist()

                advantages, value_targets = self._compute_gae(
                    new_values, rewards, config.gamma, config.gae_lambda,
                )
                perspective["new_advantages"] = advantages
                perspective["new_value_targets"] = value_targets

    # ------------------------------------------------------------------
    # Phase 3: Policy Training (design/43: batch GRU)
    # ------------------------------------------------------------------

    def _process_policy_perspective(
        self,
        perspective: dict,
        policy_model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> tuple | None:
        """Phase 3: PPO policy loss using batch GRU forward."""
        steps = perspective.get("steps", [])
        if not steps:
            return None

        T = len(steps)
        advantages = perspective.get("new_advantages", [0.0] * T)
        old_log_probs = [s.get("log_prob", 0.0) for s in steps]
        actions = [s.get("action", 0) for s in steps]

        pre_gru_results = self._compute_all_pre_gru(
            perspective, device, body=policy_model.body,
        )
        log_probs_all = self._forward_policy_sequence(
            pre_gru_results, policy_model, device,
        )  # (T, 24)

        action_indices = torch.tensor(actions, dtype=torch.long, device=device)
        new_log = log_probs_all[torch.arange(T, device=device), action_indices]

        old_log = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=device)

        ratio = torch.exp(new_log - old_log)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps,
        )
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

        entropy = (
            -(torch.exp(log_probs_all) * log_probs_all)
            .nan_to_num(0.0)
            .sum(dim=-1)
            .mean()
        )
        if torch.isnan(policy_loss) or torch.isnan(entropy):
            return None
        return policy_loss, entropy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pre_gru_kwargs(inp: dict, device: torch.device) -> dict:
        """Build kwargs for body.forward_pre_gru() from a dict-format step input.

        Converts Python list/dict fields to PyTorch tensors in the format
        expected by _AtteV1Body.forward_pre_gru() (no hidden arg, no mask_fn).
        """
        from src.backend.ai.atte_v1.model import _AtteV1Body

        own = inp["own"]
        opp = inp["opp"]

        def _tensor(ids: list[int]) -> torch.Tensor:
            if not ids:
                return torch.zeros(1, 0, dtype=torch.long)
            return torch.tensor([ids], dtype=torch.long)

        def _scalar(val: float) -> torch.Tensor:
            return torch.tensor([[val]], dtype=torch.float32)

        return {
            "own_hand_ids": _tensor(own["hand_ids"]),
            "own_discard_ids": _tensor(own["discard_ids"]),
            "own_draw_ids": _tensor(own["draw_ids"]),
            "own_last_action": torch.tensor(
                [[own["last_action"]]], dtype=torch.long,
            ),
            "own_status_ids": _tensor(own["status_ids"]),
            "own_status_durations": torch.tensor(
                [own["status_durations"]], dtype=torch.float32,
            ),
            "own_opp_hand_ids": _tensor(own["opp_hand_ids"]),
            "own_opp_discard_ids": _tensor(own["opp_discard_ids"]),
            "own_opp_draw_ids": _tensor(own["opp_draw_ids"]),
            "opp_hand_ids": _tensor(opp["hand_ids"]),
            "opp_discard_ids": _tensor(opp["discard_ids"]),
            "opp_draw_ids": _tensor(opp["draw_ids"]),
            "opp_last_action": torch.tensor(
                [[opp["last_action"]]], dtype=torch.long,
            ),
            "opp_status_ids": _tensor(opp["status_ids"]),
            "opp_status_durations": torch.tensor(
                [opp["status_durations"]], dtype=torch.float32,
            ),
            "own_hp": _scalar(own["hp"]),
            "own_energy": _scalar(own["energy"]),
            "opp_hp": _scalar(opp["hp"]),
            "opp_energy": _scalar(opp["energy"]),
            "round_in_cycle": _scalar(inp["round_in_cycle"]),
        }
