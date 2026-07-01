"""Unit tests for app/converter.py — Converter class.

Covers TC-CV1 through TC-CV8 from test_cases.md §3.
Tests card↔index conversion, round-trip consistency, and input vector
construction from game state dicts.
"""

import pytest

from src.backend.app.converter import Converter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(hand_2=None, hp_1=3, hp_2=3,
                history_1=None, history_2=None, round_num=1):
    """Build a minimal game state dict matching GameManager.get_state() format.

    All parameters are optional with sensible defaults for a fresh game.
    """
    if hand_2 is None:
        hand_2 = ["attack", "attack", "attack", "defense", "defense", "heavy", "counter"]
    if history_1 is None:
        history_1 = []
    if history_2 is None:
        history_2 = []

    return {
        "round": round_num,
        "players": {
            "1": {"hp": hp_1, "hand": ["attack"] * 7, "history": list(history_1)},
            "2": {"hp": hp_2, "hand": list(hand_2), "history": list(history_2)},
        },
        "is_over": False,
        "winner": None,
        "mode": "pve",
        "round_log": [],
    }


# ---------------------------------------------------------------------------
# TC-CV1: card_to_index valid cards
# ---------------------------------------------------------------------------


class TestCardToIndex:
    """TC-CV1–TC-CV2: Card string → index conversion."""

    def test_attack_maps_to_0(self):
        """'attack' must map to index 0."""
        assert Converter.card_to_index("attack") == 0

    def test_defense_maps_to_1(self):
        """'defense' must map to index 1."""
        assert Converter.card_to_index("defense") == 1

    def test_heavy_maps_to_2(self):
        """'heavy' must map to index 2."""
        assert Converter.card_to_index("heavy") == 2

    def test_counter_maps_to_3(self):
        """'counter' must map to index 3."""
        assert Converter.card_to_index("counter") == 3

    def test_invalid_card_returns_minus_1(self):
        """Unrecognized card strings must return -1."""
        assert Converter.card_to_index("unknown") == -1
        assert Converter.card_to_index("") == -1
        assert Converter.card_to_index("ATTACK") == -1  # Case-sensitive


# ---------------------------------------------------------------------------
# TC-CV3: index_to_card valid indices
# ---------------------------------------------------------------------------


class TestIndexToCard:
    """TC-CV3: Index → card string conversion."""

    def test_0_maps_to_attack(self):
        assert Converter.index_to_card(0) == "attack"

    def test_1_maps_to_defense(self):
        assert Converter.index_to_card(1) == "defense"

    def test_2_maps_to_heavy(self):
        assert Converter.index_to_card(2) == "heavy"

    def test_3_maps_to_counter(self):
        assert Converter.index_to_card(3) == "counter"


# ---------------------------------------------------------------------------
# TC-CV4: round-trip: card → index → card
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """TC-CV4: Round-trip conversion preserves card identity."""

    def test_attack_round_trip(self):
        assert Converter.index_to_card(Converter.card_to_index("attack")) == "attack"

    def test_defense_round_trip(self):
        assert Converter.index_to_card(Converter.card_to_index("defense")) == "defense"

    def test_heavy_round_trip(self):
        assert Converter.index_to_card(Converter.card_to_index("heavy")) == "heavy"

    def test_counter_round_trip(self):
        assert Converter.index_to_card(Converter.card_to_index("counter")) == "counter"


# ---------------------------------------------------------------------------
# TC-CV5–TC-CV8: Input vector construction
# ---------------------------------------------------------------------------


class TestBuildInputVector:
    """TC-CV5–TC-CV8: Build the 14-d input vector from game state."""

    def test_round_1_onehot_are_zeros(self):
        """TC-CV5: Round 1 → both one-hot vectors are all zeros."""
        state = _make_state(history_1=[], history_2=[], round_num=1)
        vec = Converter.build_input_vector(state)
        # Indices 6–9: AI last move one-hot
        assert vec[6:10] == [0.0, 0.0, 0.0, 0.0]
        # Indices 10–13: human last move one-hot
        assert vec[10:14] == [0.0, 0.0, 0.0, 0.0]

    def test_round_2_onehot_reflects_history(self):
        """TC-CV6: Round 2 → one-hot reflects prior round's cards."""
        state = _make_state(
            history_1=["defense"],  # human played defense last round
            history_2=["attack"],   # AI played attack last round
            round_num=2,
        )
        vec = Converter.build_input_vector(state)
        # AI last move = attack → indices 6–9 = [1, 0, 0, 0]
        assert vec[6:10] == [1.0, 0.0, 0.0, 0.0]
        # Human last move = defense → indices 10–13 = [0, 1, 0, 0]
        assert vec[10:14] == [0.0, 1.0, 0.0, 0.0]

    def test_card_counts_reflect_hand(self):
        """TC-CV7: Card counts reflect AI's remaining hand."""
        state = _make_state(
            hand_2=["attack", "attack", "defense"],
            hp_2=2,  # AI HP
            hp_1=1,  # Human HP
        )
        vec = Converter.build_input_vector(state)
        assert vec[0] == 2.0  # attack count
        assert vec[1] == 1.0  # defense count
        assert vec[2] == 0.0  # heavy count
        assert vec[3] == 0.0  # counter count
        assert vec[4] == 2.0  # AI HP
        assert vec[5] == 1.0  # Human HP

    def test_hp_clamped_to_0_3(self):
        """TC-CV8: HP values are clamped to [0, 3] range."""
        state = _make_state(hp_1=5, hp_2=-2)
        vec = Converter.build_input_vector(state)
        assert vec[4] == 0.0  # AI HP: -2 → 0
        assert vec[5] == 3.0  # Human HP: 5 → 3

    def test_returns_list_of_14_floats(self):
        """Output must be list[float] of length 14."""
        state = _make_state()
        vec = Converter.build_input_vector(state)
        assert isinstance(vec, list)
        assert len(vec) == 14
        for val in vec:
            assert isinstance(val, float)

    def test_class_constants(self):
        """Verify Converter has the expected class constants."""
        assert Converter.NUM_CARD_TYPES == 4
        assert Converter.INPUT_SIZE == 14
