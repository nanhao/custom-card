"""Split policy/value GRU-based AI models for the custom card game (design/13).

Per design/13: input vector reordered — metadata (HP, round, cycle) at front,
card-type-specific features at tail for future card type extensibility.

Architecture:
    GRUPolicyModel: Input(25) -> GRU(32, 1 layer) -> Linear(32->14) -> softmax
        Output 14-d: indices 0-6 for card action, indices 7-13 for build target.
    GRUValueModel:  Input(25) -> GRU(32, 1 layer) -> Linear(32->1)

Input vector layout (0-based):
    index 0:     Own HP
    index 1:     Opponent HP
    index 2:     round_in_cycle (1-5)
    index 3:     cycle number
    indices 4-10:   Card counts (attack..flex, 7 types)
    indices 11-17:  One-hot of own last move
    indices 18-24:  One-hot of opponent's last move

mask_fn supports two-segment masking:
    - indices 0-6: block cards not in hand
    - indices 7-13: block build targets exceeding available energy
"""

from collections.abc import Callable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE = 25    # HP_own(1) + HP_opp(1) + round(1) + cycle(1) + 7 counts + 7 onehot(own) + 7 onehot(opp)
HIDDEN_SIZE = 32   # GRU hidden state dimension
NUM_CARD_TYPES = 7  # attack, defense, heavy, counter, energy, build, flex


# ---------------------------------------------------------------------------
# GRUPolicyModel
# ---------------------------------------------------------------------------


class GRUPolicyModel(nn.Module):
    """GRU-based policy network with dual-output head (14-d).

    Architecture: Input(25) -> GRU(32, 1 layer) -> Linear(32->14) -> softmax.

    Input (25-d, per design/13):
        [HP_own, HP_opp, round, cycle | card_counts(7) | onehot_own(7) | onehot_opp(7)]

    Output interpretation:
        indices 0-6:  card action probabilities (attack..flex)
        indices 7-13: build target probabilities (attack..flex),
                       only meaningful when action == BUILD
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
        self.fc_policy = nn.Linear(HIDDEN_SIZE, NUM_CARD_TYPES * 2)  # 14 outputs

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one GRU step, apply optional two-segment mask and temperature.

        Args:
            x: Input tensor of shape (1, 1, 25).
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

    Architecture: Input(25) -> GRU(32, 1 layer) -> Linear(32->1).
    No tanh — value can exceed [-1, 1].

    Input (25-d, per design/13):
        [HP_own, HP_opp, round, cycle | card_counts(7) | onehot_own(7) | onehot_opp(7)]
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
            x: Input tensor of shape (1, 1, 25).
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
