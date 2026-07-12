"""Skip-connection GRU-based AI models for the custom card game (design/37).

Per design/37: INPUT_SIZE 56→62, POLICY_OUTPUT_SIZE 14→24.
SKIP_INPUT_SIZE = 62 + 32 = 94.

Architecture:
    GRUPolicyModelV2: Input(62) -> GRU(32, 1 layer) ─┐
                       Input(62) ─────────────────────┤-> Concat(94)
                         -> Linear(94->64) -> ReLU
                         -> Linear(64->32) -> ReLU
                         -> Linear(32->24) -> softmax
        Output 24-d: pass(0) + 8×(play, build, dismantle) triplets.

    GRUValueModelV2:  Input(62) -> GRU(32, 1 layer) ─┐
                       Input(62) ─────────────────────┤-> Concat(94)
                         -> Linear(94->64) -> ReLU
                         -> Linear(64->32) -> ReLU
                         -> Linear(32->1)

Per design/31 D5: forward() signatures are identical to v1, allowing
transparent replacement in AppManager, SelfPlayRunner, and TrainingRunner.
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

SKIP_INPUT_SIZE = INPUT_SIZE + HIDDEN_SIZE  # 94: GRU_out(32) + raw_input(62)

# ---------------------------------------------------------------------------
# GRUPolicyModelV2
# ---------------------------------------------------------------------------


class GRUPolicyModelV2(nn.Module):
    """GRU-based policy network with skip connection and 3-layer MLP head (design/34).

    Architecture:
        x -> GRU(62->32) -> gru_out
        concat(gru_out(32), x(62)) = 94
        -> Linear(94->64) -> ReLU
        -> Linear(64->32) -> ReLU
        -> Linear(32->24) -> softmax

    Input (62-d, per intention/input.md, design/37):
        [HP_own, HP_opp, energy_own, energy_opp, round, cycle |
         own_hand_counts(7) | opp_hand_counts(7) |
         own_status_lifetimes(3) | opp_status_lifetimes(3) |
         own_discard_counts(7) | opp_discard_counts(7) |
         own_draw_counts(7) | opp_draw_counts(7) |
         last_own, last_opp]

    Output (24-d, design/37):
        index 0:                     pass (no card played)
        odd indices: build + build target
        even indices: card play or dismantle
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        """Initialize the v2 policy model with random weights.

        GRU layer is identical to v1. Head is a 3-layer MLP that receives
        both GRU output and raw input via skip connection (design/31 D1).
        """
        super().__init__()
        # GRU — same as v1 (design/31 D2)
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=False,
        )
        # 3-layer MLP head (design/31 D3)
        self.fc1 = nn.Linear(SKIP_INPUT_SIZE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_policy = nn.Linear(32, POLICY_OUTPUT_SIZE)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one GRU step with skip connection, apply mask and temperature.

        Args:
            x: Input tensor of shape (1, 1, 62).
            hidden: Hidden state tensor of shape (1, 1, 32).
            mask_fn: Optional callable that takes policy logits (1,24) and
                     returns masked logits (1,24). If None, no masking.
            temperature: Temperature for softmax. 0 = greedy argmax.

        Returns:
            (probs, log_probs, new_hidden):
                probs: (1, 24) masked probability distribution.
                log_probs: (1, 24) log-probabilities.
                new_hidden: (1, 1, 32) updated GRU hidden state.
        """
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(0)  # (1, 32)
        x_squeezed = x.squeeze(0)     # (1, 62)

        # Skip connection: concat GRU output with raw input (design/31 D1)
        combined = torch.cat([gru_out, x_squeezed], dim=-1)  # (1, 94)

        # 3-layer MLP head (design/31 D3)
        h1 = torch.relu(self.fc1(combined))   # (1, 64)
        h2 = torch.relu(self.fc2(h1))         # (1, 32)
        policy_logits = self.fc_policy(h2)    # (1, 14)

        if mask_fn is not None:
            policy_logits = mask_fn(policy_logits)

        # Temperature-based sampling (same logic as v1)
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
# GRUValueModelV2
# ---------------------------------------------------------------------------


class GRUValueModelV2(nn.Module):
    """GRU-based value network with skip connection and 3-layer MLP head (design/34).

    Architecture:
        x -> GRU(62->32) -> gru_out
        concat(gru_out(32), x(62)) = 94
        -> Linear(94->64) -> ReLU
        -> Linear(64->32) -> ReLU
        -> Linear(32->1)

    No tanh — value can exceed [-1, 1], same as v1.

    Input (62-d, per intention/input.md, design/37):
        [HP_own, HP_opp, energy_own, energy_opp, round, cycle |
         own_hand_counts(7) | opp_hand_counts(7) |
         own_status_lifetimes(3) | opp_status_lifetimes(3) |
         own_discard_counts(7) | opp_discard_counts(7) |
         own_draw_counts(7) | opp_draw_counts(7) |
         last_own, last_opp]
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        """Initialize the v2 value model with random weights."""
        super().__init__()
        # GRU — same as v1 (design/31 D2)
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=False,
        )
        # 3-layer MLP head (design/31 D3)
        self.fc1 = nn.Linear(SKIP_INPUT_SIZE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_value = nn.Linear(32, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one GRU step with skip connection, return scalar value estimate.

        Args:
            x: Input tensor of shape (1, 1, 62).
            hidden: Hidden state tensor of shape (1, 1, 32).

        Returns:
            (value, new_hidden):
                value: (1, 1) scalar value estimate.
                new_hidden: (1, 1, 32) updated GRU hidden state.
        """
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(0)  # (1, 32)
        x_squeezed = x.squeeze(0)     # (1, 62)

        # Skip connection: concat GRU output with raw input (design/31 D1)
        combined = torch.cat([gru_out, x_squeezed], dim=-1)  # (1, 94)

        # 3-layer MLP head (design/31 D3)
        h1 = torch.relu(self.fc1(combined))   # (1, 64)
        h2 = torch.relu(self.fc2(h1))         # (1, 32)
        value = self.fc_value(h2)             # (1, 1), raw — no tanh

        return value, new_hidden
