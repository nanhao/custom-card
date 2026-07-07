"""GRU-based card-selection AI model for the custom card game.

Per design/3 architecture D1, this module contains ONLY the GRUCardModel class.
All game knowledge (card types, hand state, input semantics) lives in
app/Converter. All orchestration (loading, inference, masking) lives in
app/AppManager.

Architecture:
    Input(14) → GRU(8) → ┬→ FC(8→4) → (optional mask_fn) → softmax   (policy head)
                          └→ FC(8→1) → tanh                           (value head)

Public interface (per design/3 interfaces.md §1, updated per design/5):
    GRUCardModel.INITIAL_HIDDEN_STATE: list[float]
    model.forward(x, hidden, mask_fn=None, temperature=1.0) → tuple[Tensor, Tensor, Tensor, Tensor]
        Returns (probs, log_probs, value, new_hidden)
"""

from collections.abc import Callable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE = 14   # 4 card counts + 2 HP + 4 one-hot (AI last move) + 4 one-hot (human last move)
HIDDEN_SIZE = 8   # GRU hidden state dimension
NUM_CARD_TYPES = 4  # attack, defense, heavy, counter
WEIGHT_PATH = "src/backend/ai/rnn_v1/gen1/rnn_v1.pth"


# ---------------------------------------------------------------------------
# PyTorch Model
# ---------------------------------------------------------------------------


class GRUPolicyModel(nn.Module):
    """GRU-based policy network for card selection.

    Architecture: Input(14) → GRU(8, 1 layer) → Linear(8→4) → softmax.
    No value head — this network outputs only action probabilities.

    The GRU is not batch-first — inputs are (seq_len=1, batch=1, features=14).

    Attributes:
        gru: nn.GRU layer (input=14, hidden=8, 1 layer).
        fc_policy: nn.Linear policy head (8→4).
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        """Initialize the policy GRU model with random weights.

        Layers:
            self.gru: nn.GRU(14, 8, num_layers=1, batch_first=False)
            self.fc_policy: nn.Linear(8, 4)
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=False,
        )
        self.fc_policy = nn.Linear(HIDDEN_SIZE, NUM_CARD_TYPES)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one GRU step, apply optional mask and temperature.

        Args:
            x: Input tensor of shape (1, 1, 14).
            hidden: Hidden state tensor of shape (1, 1, 8).
            mask_fn: Optional callable that takes policy logits (1,4) and
                     returns masked logits (1,4). If None, no masking.
            temperature: Temperature for softmax scaling. 0 = greedy argmax.
                         Default 1.0 (standard softmax).

        Returns:
            (probs, log_probs, new_hidden):
                probs: (1, 4) masked probability distribution.
                log_probs: (1, 4) log-probabilities.
                new_hidden: (1, 1, 8) updated GRU hidden state.
        """
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(0)  # (1, 8)

        policy_logits = self.fc_policy(gru_out)  # (1, 4)
        if mask_fn is not None:
            policy_logits = mask_fn(policy_logits)

        if temperature == 0.0:
            # Greedy: one-hot at argmax
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


class GRUValueModel(nn.Module):
    """GRU-based value network for state evaluation.

    Architecture: Input(14) → GRU(8, 1 layer) → Linear(8→1).
    No tanh — value can exceed [-1, 1] (per PPO lesson learned).

    The GRU is not batch-first — inputs are (seq_len=1, batch=1, features=14).

    Attributes:
        gru: nn.GRU layer (input=14, hidden=8, 1 layer).
        fc_value: nn.Linear value head (8→1).
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        """Initialize the value GRU model with random weights.

        Layers:
            self.gru: nn.GRU(14, 8, num_layers=1, batch_first=False)
            self.fc_value: nn.Linear(8, 1)
        """
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
            x: Input tensor of shape (1, 1, 14).
            hidden: Hidden state tensor of shape (1, 1, 8).

        Returns:
            (value, new_hidden):
                value: (1, 1) scalar value estimate.
                new_hidden: (1, 1, 8) updated GRU hidden state.
        """
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(0)  # (1, 8)
        value = self.fc_value(gru_out)  # (1, 1), raw — no tanh
        return value, new_hidden

