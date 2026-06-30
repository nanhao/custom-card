"""Unit tests for ai/rnn_v1/model.py — GRU model and card selection.

Covers TC-AI1 through TC-AI16 from test_cases.md.
Tests model instantiation, forward pass shapes, card selection with
illegal action masking, input vector construction, determinism,
inference mode, and weight file loading.
"""

import pytest
import torch

from src.backend.game.card import CardType, get_initial_hand
from src.backend.game.game_manager import GameManager
from src.backend.ai.rnn_v1.model import (
    GRUCardModel,
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_CARD_TYPES,
    init_hidden_state,
    select_action,
    _build_input_vector,
    _mask_illegal_actions,
    _index_to_card,
    _card_to_index,
    _load_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(hand_1=None, hand_2=None, hp_1=3, hp_2=3,
                history_1=None, history_2=None, round_num=1,
                is_over=False, winner=None):
    """Build a minimal game state dict matching GameManager.get_state() format.

    All parameters are optional with sensible defaults for a fresh game.
    """
    if hand_1 is None:
        hand_1 = ["attack", "attack", "attack", "defense", "defense", "heavy", "counter"]
    if hand_2 is None:
        hand_2 = ["attack", "attack", "attack", "defense", "defense", "heavy", "counter"]
    if history_1 is None:
        history_1 = []
    if history_2 is None:
        history_2 = []

    return {
        "round": round_num,
        "players": {
            "1": {"hp": hp_1, "hand": list(hand_1), "history": list(history_1)},
            "2": {"hp": hp_2, "hand": list(hand_2), "history": list(history_2)},
        },
        "is_over": is_over,
        "winner": winner,
        "mode": "pve",
        "round_log": [],
    }


# ---------------------------------------------------------------------------
# TC-AI1: Model instantiation
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    """TC-AI1: Model instantiation and architecture verification."""

    def test_model_has_gru_layer(self):
        """Model must have a GRU layer with input=14, hidden=8."""
        model = GRUCardModel()
        assert hasattr(model, "gru")
        assert model.gru.input_size == INPUT_SIZE  # 14
        assert model.gru.hidden_size == HIDDEN_SIZE  # 8

    def test_model_has_fc_policy_layer(self):
        """Model must have fc_policy Linear(8→4)."""
        model = GRUCardModel()
        assert hasattr(model, "fc_policy")
        assert model.fc_policy.in_features == HIDDEN_SIZE  # 8
        assert model.fc_policy.out_features == NUM_CARD_TYPES  # 4

    def test_model_has_fc_value_layer(self):
        """Model must have fc_value Linear(8→1)."""
        model = GRUCardModel()
        assert hasattr(model, "fc_value")
        assert model.fc_value.in_features == HIDDEN_SIZE  # 8
        assert model.fc_value.out_features == 1

    def test_model_has_parameters(self):
        """Model state_dict must be non-empty (all parameters exist)."""
        model = GRUCardModel()
        state_dict = model.state_dict()
        assert len(state_dict) > 0
        # Should have at least GRU weights + policy weights + value weights
        assert any("gru" in k for k in state_dict)
        assert any("fc_policy" in k for k in state_dict)
        assert any("fc_value" in k for k in state_dict)


# ---------------------------------------------------------------------------
# TC-AI2: init_hidden_state
# ---------------------------------------------------------------------------


class TestInitHiddenState:
    """TC-AI2: init_hidden_state returns list of 8 zeros."""

    def test_returns_list_of_8_zeros(self):
        """init_hidden_state() must return [0.0] * 8."""
        result = init_hidden_state()
        assert isinstance(result, list)
        assert len(result) == HIDDEN_SIZE  # 8
        assert result == [0.0] * HIDDEN_SIZE

    def test_all_elements_are_float(self):
        """All elements must be float type, not int."""
        result = init_hidden_state()
        for val in result:
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# TC-AI3: Forward pass shapes
# ---------------------------------------------------------------------------


class TestForwardPass:
    """TC-AI3: Forward pass produces correct output shapes."""

    def test_policy_output_shape(self):
        """Policy output must be (1, 4) — logits over 4 card types."""
        model = GRUCardModel()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        policy, value, new_h = model(x, h)
        assert policy.shape == (1, NUM_CARD_TYPES)

    def test_value_output_shape(self):
        """Value output must be (1, 1)."""
        model = GRUCardModel()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        policy, value, new_h = model(x, h)
        assert value.shape == (1, 1)

    def test_value_in_range(self):
        """Value output must be in [-1, 1] after tanh."""
        model = GRUCardModel()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        policy, value, new_h = model(x, h)
        assert -1.0 <= value.item() <= 1.0


# ---------------------------------------------------------------------------
# TC-AI4: select_action returns legal card
# ---------------------------------------------------------------------------


class TestSelectActionLegal:
    """TC-AI4–TC-AI6: select_action returns only legal cards."""

    def test_returns_legal_card_no_counter(self):
        """TC-AI4: When AI has [attack, defense, heavy], never returns counter."""
        state = _make_state(
            hand_2=["attack", "attack", "defense", "defense", "heavy", "defense", "attack"],
        )
        h0 = init_hidden_state()
        card_str, new_h = select_action(state, h0)
        assert card_str in ("attack", "defense", "heavy")

    def test_returns_tuple_of_str_and_list(self):
        """Return type must be (str, list[float])."""
        state = _make_state()
        h0 = init_hidden_state()
        card_str, new_h = select_action(state, h0)
        assert isinstance(card_str, str)
        assert isinstance(new_h, list)
        assert len(new_h) == HIDDEN_SIZE

    def test_returns_all_cards_possible_with_full_hand(self):
        """TC-AI5: With full 7-card hand, all 4 types are possible."""
        state = _make_state()  # Standard 7-card hand includes all types
        h0 = init_hidden_state()
        card_str, _ = select_action(state, h0)
        assert card_str in ("attack", "defense", "heavy", "counter")

    def test_returns_only_card_when_one_left(self):
        """TC-AI6: When AI has only one card, must select it."""
        state = _make_state(hand_2=["defense"])
        h0 = init_hidden_state()
        card_str, _ = select_action(state, h0)
        assert card_str == "defense"


# ---------------------------------------------------------------------------
# TC-AI7: Hidden state updates
# ---------------------------------------------------------------------------


class TestHiddenStateUpdates:
    """TC-AI7: Hidden state changes across calls."""

    def test_hidden_state_changes_between_calls(self):
        """Hidden state must evolve across select_action calls."""
        state = _make_state()
        h0 = init_hidden_state()
        card1, h1 = select_action(state, h0)
        card2, h2 = select_action(state, h1)
        assert h1 != h0, "Hidden state must change after first call"
        assert h2 != h1, "Hidden state must change after second call"

    def test_both_cards_are_legal(self):
        """Both selected cards must be legal."""
        state = _make_state()
        h0 = init_hidden_state()
        card1, h1 = select_action(state, h0)
        card2, _ = select_action(state, h1)
        assert card1 in ("attack", "defense", "heavy", "counter")
        assert card2 in ("attack", "defense", "heavy", "counter")


# ---------------------------------------------------------------------------
# TC-AI8–TC-AI10: Input vector construction
# ---------------------------------------------------------------------------


class TestInputVectorConstruction:
    """TC-AI8–TC-AI10: Verify the 14-d input vector is built correctly."""

    def test_round_1_both_onehot_are_zeros(self):
        """TC-AI8: Round 1 → both one-hot vectors are all zeros."""
        state = _make_state(history_1=[], history_2=[], round_num=1)
        vec = _build_input_vector(state)
        # Indices 6–9: AI last move one-hot
        assert vec[6:10] == [0.0, 0.0, 0.0, 0.0]
        # Indices 10–13: human last move one-hot
        assert vec[10:14] == [0.0, 0.0, 0.0, 0.0]

    def test_round_2_onehot_reflects_history(self):
        """TC-AI9: Round 2 → one-hot reflects prior round's cards."""
        state = _make_state(
            history_1=["defense"],  # human played defense last round
            history_2=["attack"],   # AI played attack last round
            round_num=2,
        )
        vec = _build_input_vector(state)
        # AI last move = attack → indices 6–9 = [1, 0, 0, 0]
        assert vec[6:10] == [1.0, 0.0, 0.0, 0.0]
        # Human last move = defense → indices 10–13 = [0, 1, 0, 0]
        assert vec[10:14] == [0.0, 1.0, 0.0, 0.0]

    def test_card_counts_reflect_hand(self):
        """TC-AI10: Card counts reflect AI's remaining hand."""
        state = _make_state(
            hand_2=["attack", "attack", "defense"],  # 2 atk, 1 def
            hp_2=2,  # AI HP
            hp_1=1,  # Human HP
        )
        vec = _build_input_vector(state)
        assert vec[0] == 2.0  # attack count
        assert vec[1] == 1.0  # defense count
        assert vec[2] == 0.0  # heavy count
        assert vec[3] == 0.0  # counter count
        assert vec[4] == 2.0  # AI HP
        assert vec[5] == 1.0  # Human HP

    def test_hp_clamped_to_0_3(self):
        """HP values are clamped to [0, 3] range."""
        state = _make_state(hp_1=5, hp_2=-2)
        vec = _build_input_vector(state)
        assert vec[4] == 0.0  # AI HP: -2 → 0
        assert vec[5] == 3.0  # Human HP: 5 → 3


# ---------------------------------------------------------------------------
# TC-AI11, TC-AI14, TC-AI15: Determinism and inference mode
# ---------------------------------------------------------------------------


class TestDeterminismAndInferenceMode:
    """TC-AI11, TC-AI14, TC-AI15: Deterministic output and eval mode."""

    def test_deterministic_with_fixed_seed(self):
        """TC-AI11: Same seed + same state → same card."""
        torch.manual_seed(42)
        state = _make_state()
        h0 = init_hidden_state()
        card1, _ = select_action(state, h0)

        # Reset seed and try again
        torch.manual_seed(42)
        card2, _ = select_action(state, h0)
        assert card1 == card2

    def test_deterministic_without_per_call_seed(self):
        """TC-AI15: Fixed weights + eval mode = deterministic without per-call seed."""
        state = _make_state()
        h0 = init_hidden_state()
        card1, _ = select_action(state, h0)
        card2, _ = select_action(state, h0)
        assert card1 == card2, (
            "Same model instance with same weights should be deterministic "
            "under eval+inference_mode"
        )

    def test_model_is_in_eval_mode_after_load(self):
        """TC-AI14: Model operates in eval mode during inference."""
        model = _load_model()
        assert not model.training, "Model should be in eval mode"


# ---------------------------------------------------------------------------
# TC-AI12: Weight file loading
# ---------------------------------------------------------------------------


class TestWeightFileLoading:
    """TC-AI12: Weight file loads successfully and produces usable model."""

    def test_model_loads_without_exception(self):
        """Loading the weight file should not raise an exception."""
        model = _load_model()
        assert model is not None

    def test_loaded_model_selects_legal_card(self):
        """After loading weights, select_action must produce a legal card."""
        state = _make_state()
        h0 = init_hidden_state()
        card_str, _ = select_action(state, h0)
        assert card_str in ("attack", "defense", "heavy", "counter")


# ---------------------------------------------------------------------------
# TC-AI13: Different models produce different outputs
# ---------------------------------------------------------------------------


class TestDifferentModels:
    """TC-AI13: Different random seeds produce potentially different models."""

    def test_different_seeds_produce_different_weights(self):
        """Two models with different seeds have different parameters."""
        torch.manual_seed(42)
        m1 = GRUCardModel()
        torch.manual_seed(99)
        m2 = GRUCardModel()

        # Compare a key parameter
        w1 = m1.fc_policy.weight.data.clone()
        w2 = m2.fc_policy.weight.data.clone()
        assert not torch.allclose(w1, w2), (
            "Different seeds should produce different weights"
        )


# ---------------------------------------------------------------------------
# TC-AI16: Corrupted weight file
# ---------------------------------------------------------------------------


class TestCorruptedWeightFile:
    """TC-AI16: Corrupted weight files produce clear error messages."""

    def test_raises_on_nonexistent_file(self, monkeypatch):
        """Loading from a nonexistent path raises RuntimeError."""
        import src.backend.ai.rnn_v1.model as model_module

        monkeypatch.setattr(model_module, "WEIGHT_PATH", "/nonexistent/path.pth")
        # Reset the cached model to force a reload attempt
        monkeypatch.setattr(model_module, "_model", None)

        with pytest.raises(RuntimeError, match="file not found"):
            model_module._load_model()


# ---------------------------------------------------------------------------
# Illegal action masking unit test
# ---------------------------------------------------------------------------


class TestIllegalActionMasking:
    """Verify illegal action masking sets correct logits to -inf."""

    def test_masks_absent_cards(self):
        """Cards not in hand must have -inf logit."""
        logits = torch.zeros(1, 4)
        _mask_illegal_actions(logits, ["attack"])  # Only attack is legal
        assert logits[0, 0] == 0.0  # Attack: legal
        assert logits[0, 1] == float("-inf")  # Defense: illegal
        assert logits[0, 2] == float("-inf")  # Heavy: illegal
        assert logits[0, 3] == float("-inf")  # Counter: illegal

    def test_all_cards_legal_when_all_present(self):
        """When all 4 card types are in hand, no masks are applied."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        hand = ["attack", "defense", "heavy", "counter"]
        _mask_illegal_actions(logits, hand)
        # No values should be -inf
        assert all(logits[0, i].item() != float("-inf") for i in range(4))


# ---------------------------------------------------------------------------
# Card index mapping
# ---------------------------------------------------------------------------


class TestCardIndexMapping:
    """Verify card_string ↔ index conversion."""

    def test_index_to_card_valid(self):
        """0→attack, 1→defense, 2→heavy, 3→counter."""
        assert _index_to_card(0) == "attack"
        assert _index_to_card(1) == "defense"
        assert _index_to_card(2) == "heavy"
        assert _index_to_card(3) == "counter"

    def test_card_to_index_valid(self):
        """attack→0, defense→1, heavy→2, counter→3."""
        assert _card_to_index("attack") == 0
        assert _card_to_index("defense") == 1
        assert _card_to_index("heavy") == 2
        assert _card_to_index("counter") == 3

    def test_card_to_index_invalid(self):
        """Unknown card returns -1."""
        assert _card_to_index("unknown") == -1
