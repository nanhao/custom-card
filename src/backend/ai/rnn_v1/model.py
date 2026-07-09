"""Split policy/value GRU-based AI models for the custom card game (design/14, design/27, design/28).

Per design/14: input vector redesigned — metadata (HP, energy, round, cycle) at
front, card counts in middle, policy-index last actions at tail.
Per design/27: INPUT_SIZE expanded from 15 to 42 — added opponent hand (7 dims),
status lifetimes for both players (6 dims), own discard pile (7 dims), and own
draw pile (7 dims).
Per design/28: POLICY_OUTPUT_SIZE expanded from 13 to 14 — added pass at index 0;
even indices pure play, odd indices build+target.

Architecture:
    GRUPolicyModel: Input(42) -> GRU(32, 1 layer) -> Linear(32->14) -> softmax
        Output 14-d: index 0=pass, even(2,4,6,8,10,12)=pure play, odd(1,3,5,7,9,11,13)=build.
    GRUValueModel:  Input(42) -> GRU(32, 1 layer) -> Linear(32->1)

Input vector layout (0-based, design/27):
    index 0:     Own HP
    index 1:     Opponent HP
    index 2:     Own energy (0-5)
    index 3:     Opponent energy (0-5)
    index 4:     round_in_cycle (1-3, design/26 D1)
    index 5:     cycle number
    indices 6-12:   Own hand card counts (attack..flex, 7 types)
    indices 13-19:  Opponent hand card counts (attack..flex, 7 types)
    indices 20-22:  Own status lifetimes (DEFENSE, GUARD, FRAGILE; 0 if absent)
    indices 23-25:  Opponent status lifetimes (DEFENSE, GUARD, FRAGILE; 0 if absent)
    indices 26-32:  Own discard pile card counts (attack..flex, 7 types)
    indices 33-39:  Own draw pile card counts (attack..flex, 7 types)
    index 40:    Own last action policy index (-1 if first round)
    index 41:    Opponent last action policy index (-1 if first round)

mask_fn supports 13-d interleaved masking:
    - Odd indices (1,3,5,7,9,11): block cards not in hand
    - Even indices (0,2,4,6,8,10,12): block when build not available or insufficient energy
"""

from collections.abc import Callable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE = 42       # From Converter.INPUT_SIZE (design/27): expanded 15→42 with opponent hand, status lifetimes, pile info
HIDDEN_SIZE = 32      # GRU hidden state dimension
NUM_CARD_TYPES = 7    # attack, defense, heavy, counter, energy, build, flex
POLICY_OUTPUT_SIZE = 14  # 14-d: pass(0) + interleaved build/play pairs (design/28)


# ---------------------------------------------------------------------------
# GRUPolicyModel
# ---------------------------------------------------------------------------


class GRUPolicyModel(nn.Module):
    """GRU-based policy network with 14-d output (design/14, design/27, design/28).

    Architecture: Input(42) -> GRU(32, 1 layer) -> Linear(32->14) -> softmax.

    Input (42-d, per design/27):
        [HP_own, HP_opp, energy_own, energy_opp, round, cycle |
         own_hand_counts(7) | opp_hand_counts(7) |
         own_status_lifetimes(3) | opp_status_lifetimes(3) |
         own_discard_counts(7) | own_draw_counts(7) |
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
            x: Input tensor of shape (1, 1, 42).
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

    Architecture: Input(42) -> GRU(32, 1 layer) -> Linear(32->1).
    No tanh — value can exceed [-1, 1].

    Input (42-d, per design/27):
        [HP_own, HP_opp, energy_own, energy_opp, round, cycle |
         own_hand_counts(7) | opp_hand_counts(7) |
         own_status_lifetimes(3) | opp_status_lifetimes(3) |
         own_discard_counts(7) | own_draw_counts(7) |
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
            x: Input tensor of shape (1, 1, 42).
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
