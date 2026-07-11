"""Split policy/value GRU-based AI models for the custom card game (design/37).

Per design/37: INPUT_SIZE 56→62 (8 card types), POLICY_OUTPUT_SIZE 14→24
(adds dismantle dimension per card type).
Input layout per intention/input.md.

Architecture:
    GRUPolicyModel: Input(62) -> GRU(32, 1 layer) -> Linear(32->24) -> softmax
        Output 24-d: pass(0) + 8×(play, build, dismantle) triplets.
    GRUValueModel:  Input(62) -> GRU(32, 1 layer) -> Linear(32->1)

Input vector layout (0-based, per intention/input.md, design/37):
    index 0:     Own HP
    index 1:     Opponent HP
    index 2:     Own energy (0-5)
    index 3:     Opponent energy (0-5)
    index 4:     round_in_cycle (1-3, design/26 D1)
    index 5:     cycle number
    indices 6-13:   Own hand card counts (8 types)
    indices 14-21:  Opponent hand card counts (8 types)
    indices 22-24:  Own status lifetimes (DEFENSE, GUARD, FRAGILE; 0 if absent)
    indices 25-27:  Opponent status lifetimes
    indices 28-35:  Own discard pile card counts (8 types)
    indices 36-43:  Opponent discard pile card counts (8 types)
    indices 44-51:  Own draw pile card counts (8 types)
    indices 52-59:  Opponent draw pile card counts (8 types)
    index 60:    Own last action policy index (-1 if first round)
    index 61:    Opponent last action policy index (-1 if first round)

mask_fn supports 24-d triplet masking:
    - Pass (index 0): always legal
    - Play indices: block cards not in hand
    - Build indices: block if can't build or insufficient energy
    - Dismantle indices: block if BUILD not in hand or discard lacks target
"""

from collections.abc import Callable

import torch
import torch.nn as nn

from src.backend.ai.constants import (  # shared constants (design/31)
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CARD_TYPES,
    POLICY_OUTPUT_SIZE,
)


# ---------------------------------------------------------------------------
# GRUPolicyModel
# ---------------------------------------------------------------------------


class GRUPolicyModel(nn.Module):
    """GRU-based policy network with 14-d output (design/14, design/34, design/28).

    Architecture: Input(56) -> GRU(32, 1 layer) -> Linear(32->14) -> softmax.

    Input (56-d, per intention/input.md, design/34):
        [HP_own, HP_opp, energy_own, energy_opp, round, cycle |
         own_hand_counts(7) | opp_hand_counts(7) |
         own_status_lifetimes(3) | opp_status_lifetimes(3) |
         own_discard_counts(7) | opp_discard_counts(7) |
         own_draw_counts(7) | opp_draw_counts(7) |
         last_own, last_opp]

    Output (14-d, design/28):
        index 0:                     pass (no card played)
        odd indices (1,3,5,7,9,11,13): build + build target
        even indices (2,4,6,8,10,12):  pure card play
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        """Initialize the policy GRU model with random weights."""
        super().__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=False,
        )
        self.fc_policy = nn.Linear(HIDDEN_SIZE, POLICY_OUTPUT_SIZE)  # 14 outputs (design/28)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one GRU step, apply optional two-segment mask and temperature.

        Args:
            x: Input tensor of shape (1, 1, 56).
            hidden: Hidden state tensor of shape (1, 1, 32).
            mask_fn: Optional callable that takes policy logits (1,14) and
                     returns masked logits (1,14). If None, no masking.
            temperature: Temperature for softmax. 0 = greedy argmax.

        Returns:
            (probs, log_probs, new_hidden):
                probs: (1, 14) masked probability distribution.
                log_probs: (1, 14) log-probabilities.
                new_hidden: (1, 1, 32) updated GRU hidden state.
        """
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(0)  # (1, 32)

        policy_logits = self.fc_policy(gru_out)  # (1, 14)
        if mask_fn is not None:
            policy_logits = mask_fn(policy_logits)

        if temperature == 0.0:
            argmax_idx = torch.argmax(policy_logits, dim=-1, keepdim=True)
            probs = torch.zeros_like(policy_logits)
            probs.scatter_(1, argmax_idx, 1.0)
            log_probs = torch.full_like(policy_logits, float("-inf"))
            log_probs.scatter_(1, argmax_idx, 0.0)
        else:
            scaled_logits = policy_logits / temperature
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            probs = torch.exp(log_probs)

        return probs, log_probs, new_hidden


# ---------------------------------------------------------------------------
# GRUValueModel
# ---------------------------------------------------------------------------


class GRUValueModel(nn.Module):
    """GRU-based value network for state evaluation.

    Architecture: Input(56) -> GRU(32, 1 layer) -> Linear(32->1).
    No tanh — value can exceed [-1, 1].

    Input (56-d, per intention/input.md, design/34):
        [HP_own, HP_opp, energy_own, energy_opp, round, cycle |
         own_hand_counts(7) | opp_hand_counts(7) |
         own_status_lifetimes(3) | opp_status_lifetimes(3) |
         own_discard_counts(7) | opp_discard_counts(7) |
         own_draw_counts(7) | opp_draw_counts(7) |
         last_own, last_opp]
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        """Initialize the value GRU model with random weights."""
        super().__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=False,
        )
        self.fc_value = nn.Linear(HIDDEN_SIZE, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one GRU step, return scalar value estimate.

        Args:
            x: Input tensor of shape (1, 1, 56).
            hidden: Hidden state tensor of shape (1, 1, 32).

        Returns:
            (value, new_hidden):
                value: (1, 1) scalar value estimate.
                new_hidden: (1, 1, 32) updated GRU hidden state.
        """
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(0)  # (1, 32)
        value = self.fc_value(gru_out)  # (1, 1), raw — no tanh
        return value, new_hidden
