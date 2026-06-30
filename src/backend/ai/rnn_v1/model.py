"""GRU-based card-selection AI model for the custom card game.

Architecture (per ai_module.md §2.1):
    Input(14) → GRU(8) → ┬→ FC(8→4) → mask → softmax    (policy head)
                          └→ FC(8→1) → tanh               (value head)

The model takes a 14-dimensional input vector representing the current
game state from the AI's perspective and produces a probability distribution
over available card types via illegal-action masking.

Public interface (per ai_module.md §1):
    init_hidden_state() → list[float]
    select_action(state, hidden_state) → tuple[str, list[float]]
"""

import torch
import torch.nn as nn

from src.backend.game.card import CardType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE = 14   # 4 card counts + 2 HP + 4 one-hot (AI last move) + 4 one-hot (human last move)
HIDDEN_SIZE = 8   # GRU hidden state dimension
NUM_CARD_TYPES = 4  # attack, defense, heavy, counter
WEIGHT_PATH = "src/backend/ai/rnn_v1/gen1/rnn_v1.pth"

# Mapping from CardType to input vector index for card counts.
_CARD_TYPE_TO_INDEX: dict[str, int] = {
    "attack": 0,
    "defense": 1,
    "heavy": 2,
    "counter": 3,
}

# Mapping from card string to one-hot index.
_CARD_TO_ONEHOT_INDEX: dict[str, int] = {
    "attack": 0,
    "defense": 1,
    "heavy": 2,
    "counter": 3,
}

# Valid card strings for validation.
_VALID_CARD_STRINGS = {"attack", "defense", "heavy", "counter"}


# ---------------------------------------------------------------------------
# PyTorch Model
# ---------------------------------------------------------------------------


class GRUCardModel(nn.Module):
    """GRU-based neural network for card selection.

    Architecture:
        Input(14) → GRU(8, 1 layer) → policy_head(8→4) + value_head(8→1)

    The GRU is not batch-first — inputs are (seq_len=1, batch=1, features=14).
    """

    def __init__(self) -> None:
        """Initialize the GRU model with random weights."""
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
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one step of the GRU and produce policy + value outputs.

        Args:
            x: Input tensor of shape (1, 1, 14).
            hidden: Hidden state tensor of shape (1, 1, 8).

        Returns:
            (policy_logits, value, new_hidden) tuple:
                policy_logits: (1, 4) raw logits over card types.
                value: (1, 1) win-rate estimate in [-1, 1].
                new_hidden: (1, 1, 8) updated GRU hidden state.
        """
        gru_out, new_hidden = self.gru(x, hidden)
        # gru_out shape: (1, 1, 8) → squeeze to (1, 8)
        gru_out = gru_out.squeeze(0)
        policy_logits = self.fc_policy(gru_out)  # (1, 4)
        value = torch.tanh(self.fc_value(gru_out))  # (1, 1)
        return policy_logits, value, new_hidden


# ---------------------------------------------------------------------------
# Module-level cached model instance
# ---------------------------------------------------------------------------

# The model is loaded eagerly at module import time (per ai_module.md §2.5).
# If eager loading fails (e.g., missing weight file), the variable stays None
# and the first select_action() call will attempt to load it.
_model: GRUCardModel | None = None


def _load_model() -> GRUCardModel:
    """Load the GRU model and its weights from disk.

    Weights are loaded with weights_only=True and map_location='cpu'
    for safety and portability. The model instance is cached globally.

    Returns:
        The loaded GRUCardModel in eval mode.

    Raises:
        RuntimeError: If the weight file cannot be loaded.
    """
    global _model
    if _model is not None:
        return _model

    model = GRUCardModel()
    try:
        state_dict = torch.load(
            WEIGHT_PATH,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        raise RuntimeError(
            f"Failed to load model weights: file not found at {WEIGHT_PATH!r}"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model weights from {WEIGHT_PATH!r}: {exc}"
        ) from exc

    model.eval()
    _model = model
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_hidden_state() -> list[float]:
    """Return the initial GRU hidden state: list of 8 zeros.

    This function always returns the same value regardless of model state,
    so it does not require the model to be loaded.
    """
    return [0.0] * HIDDEN_SIZE


def select_action(
    state: dict, hidden_state: list[float]
) -> tuple[str, list[float]]:
    """Select the AI's card for the current round.

    Builds a 14-d input vector from the game state dict, runs a GRU
    forward pass with illegal-action masking, and returns the selected
    card string and the updated hidden state.

    Args:
        state: Game state dict from GameManager.get_state(), with keys:
            round, players, is_over, winner, mode, round_log.
        hidden_state: Current GRU hidden state as a list of 8 floats.

    Returns:
        (card_string, new_hidden_state) tuple where card_string is one of
        "attack", "defense", "heavy", "counter".

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    model = _load_model()

    # ---- Build 14-d input vector ----
    input_vector = _build_input_vector(state)

    # ---- Prepare tensors ----
    # x: (1, 1, 14) — seq_len=1, batch=1, features=14
    x = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, INPUT_SIZE)
    # h: (1, 1, 8) — num_layers=1, batch=1, hidden=8
    h = torch.tensor(hidden_state, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)

    # ---- Forward pass ----
    with torch.inference_mode():
        policy_logits, _value, new_hidden = model(x, h)
        # policy_logits shape: (1, 4)

        # ---- Illegal action masking (per ai_module.md §2.3) ----
        hand = state["players"]["2"]["hand"]  # AI is Player 2
        _mask_illegal_actions(policy_logits, hand)

        # ---- Select action via argmax ----
        probs = torch.softmax(policy_logits, dim=-1)
        action_idx = int(torch.argmax(probs, dim=-1).item())

    # ---- Convert index to card string ----
    card_str = _index_to_card(action_idx)

    # ---- Extract new hidden state as flat list ----
    new_hidden_list = new_hidden.squeeze(1).squeeze(0).tolist()

    return card_str, new_hidden_list


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_input_vector(state: dict) -> list[float]:
    """Build the 14-dimensional input vector from the game state dict.

    Per ai_module.md §2.2:
        indices 0–3: card counts (AI's remaining hand)
        indices 4–5: HP values (AI HP, human HP)
        indices 6–9: one-hot of AI's last move
        indices 10–13: one-hot of human's last move

    HP values are clamped to [0, 3] per ai_module.md §2.2.
    Round 1: both one-hot vectors are all zeros.
    """
    ai_player = state["players"]["2"]
    human_player = state["players"]["1"]

    # ---- Card counts (indices 0–3) ----
    counts = [0.0] * NUM_CARD_TYPES
    for card_str in ai_player["hand"]:
        idx = _CARD_TYPE_TO_INDEX.get(card_str)
        if idx is not None:
            counts[idx] += 1.0

    # ---- HP values (indices 4–5), clamped to [0, 3] ----
    ai_hp = max(0.0, min(3.0, float(ai_player["hp"])))
    human_hp = max(0.0, min(3.0, float(human_player["hp"])))

    # ---- One-hot of last moves (indices 6–13) ----
    ai_last = [0.0] * NUM_CARD_TYPES
    human_last = [0.0] * NUM_CARD_TYPES

    if ai_player["history"]:
        last_card = ai_player["history"][-1]
        onehot_idx = _CARD_TO_ONEHOT_INDEX.get(last_card)
        if onehot_idx is not None:
            ai_last[onehot_idx] = 1.0

    if human_player["history"]:
        last_card = human_player["history"][-1]
        onehot_idx = _CARD_TO_ONEHOT_INDEX.get(last_card)
        if onehot_idx is not None:
            human_last[onehot_idx] = 1.0

    return counts + [ai_hp, human_hp] + ai_last + human_last


def _mask_illegal_actions(
    logits: torch.Tensor, hand: list[str]
) -> None:
    """Set logits to -inf for card types not in the AI's hand (in-place).

    This ensures illegal cards have probability 0 after softmax.

    Args:
        logits: Policy logits tensor of shape (1, 4). Modified in-place.
        hand: List of card strings currently in the AI's hand.
    """
    legal_cards = set(hand)
    for card_str, idx in _CARD_TO_ONEHOT_INDEX.items():
        if card_str not in legal_cards:
            logits[0, idx] = float("-inf")


def _index_to_card(index: int) -> str:
    """Convert a policy head index (0–3) back to a card string.

    Mapping: 0→attack, 1→defense, 2→heavy, 3→counter.
    """
    mapping = {0: "attack", 1: "defense", 2: "heavy", 3: "counter"}
    return mapping[index]


def _card_to_index(card_str: str) -> int:
    """Convert a card string to its policy head index.

    Returns the index (0–3), or -1 if unrecognized.
    """
    return _CARD_TO_ONEHOT_INDEX.get(card_str, -1)
