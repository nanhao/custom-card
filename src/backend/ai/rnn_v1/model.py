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


class GRUCardModel(nn.Module):
    """GRU-based neural network for card selection.

    Architecture:
        Input(14) → GRU(8, 1 layer) → policy_head(8→4) + value_head(8→1)

    The GRU is not batch-first — inputs are (seq_len=1, batch=1, features=14).

    Attributes:
        gru: nn.GRU layer (input=14, hidden=8, 1 layer).
        fc_policy: nn.Linear policy head (8→4).
        fc_value: nn.Linear value head (8→1).
    """

    # Per design D6: initial GRU hidden state is a class constant.
    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        """Initialize the GRU model with random weights.

        Layers:
            self.gru: nn.GRU(14, 8, num_layers=1, batch_first=False)
            self.fc_policy: nn.Linear(8, 4)
            self.fc_value: nn.Linear(8, 1)
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=False,
        )
        self.fc_policy = nn.Linear(HIDDEN_SIZE, NUM_CARD_TYPES)
        self.fc_value = nn.Linear(HIDDEN_SIZE, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one GRU step, apply optional mask and temperature, return results.

        Per design D3, the mask function is passed as a parameter. The caller
        (AppManager) constructs it as a closure with domain knowledge of the
        AI's hand. The model applies it internally before softmax.

        Per design/5 D4: temperature parameter controls softmax scaling.
        temperature=0 → greedy (one-hot probs).
        temperature>0 → softmax(logits / temperature).
        Also returns log_probs for PPO sample generation.

        Args:
            x: Input tensor of shape (1, 1, 14).
            hidden: Hidden state tensor of shape (1, 1, 8).
            mask_fn: Optional callable that takes policy logits (1,4) and
                     returns masked logits (1,4). If None, no masking applied.
            temperature: Temperature for softmax scaling. 0 = greedy argmax.
                         Default 1.0 (standard softmax).

        Returns:
            (probs, log_probs, value, new_hidden) tuple:
                probs: (1, 4) probability distribution over 4 card types.
                       Already masked if mask_fn is provided.
                log_probs: (1, 4) log-probabilities over 4 card types.
                value: (1, 1) win-rate estimate in [-1, 1].
                new_hidden: (1, 1, 8) updated GRU hidden state.
        """
        # GRU forward step
        gru_out, new_hidden = self.gru(x, hidden)
        # gru_out shape: (1, 1, 8) → squeeze to (1, 8)
        gru_out = gru_out.squeeze(0)

        # Policy head: raw logits → optional mask → temperature → softmax
        policy_logits = self.fc_policy(gru_out)  # (1, 4)
        if mask_fn is not None:
            policy_logits = mask_fn(policy_logits)

        if temperature == 0.0:
            # Greedy mode: one-hot at argmax position
            argmax_idx = torch.argmax(policy_logits, dim=-1, keepdim=True)  # (1, 1)
            probs = torch.zeros_like(policy_logits)
            probs.scatter_(1, argmax_idx, 1.0)
            # log_probs: 0 at argmax, -inf elsewhere
            log_probs = torch.full_like(policy_logits, float("-inf"))
            log_probs.scatter_(1, argmax_idx, 0.0)
        else:
            # Apply temperature scaling: softmax(logits / T)
            scaled_logits = policy_logits / temperature
            log_probs = torch.log_softmax(scaled_logits, dim=-1)  # (1, 4)
            probs = torch.exp(log_probs)  # (1, 4)

        # Value head: scalar (no tanh — V_target may exceed [-1,1])
        value = self.fc_value(gru_out)  # (1, 1)

        return probs, log_probs, value, new_hidden
