"""Unit tests for app/converter.py — Converter class.

Covers TC-CV1 through TC-CV8 from design/3 test_cases.md §3, plus
new TC-C1 through TC-C3 from design/5 test_cases.md §2 (player_key).
Tests card↔index conversion, round-trip consistency, and input vector
construction from game state dicts.
"""

import pytest

from src.backend.app.converter import Converter
from tests.helpers import make_game_state


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
        state = make_game_state(history_1=[], history_2=[], round_num=1)
        vec = Converter.build_input_vector(state)
        # Indices 6–9: AI last move one-hot
        assert vec[6:10] == [0.0, 0.0, 0.0, 0.0]
        # Indices 10–13: human last move one-hot
        assert vec[10:14] == [0.0, 0.0, 0.0, 0.0]

    def test_round_2_onehot_reflects_history(self):
        """TC-CV6: Round 2 → one-hot reflects prior round's cards."""
        state = make_game_state(
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
        state = make_game_state(
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
        state = make_game_state(hp_1=5, hp_2=-2)
        vec = Converter.build_input_vector(state)
        assert vec[4] == 0.0  # AI HP: -2 → 0
        assert vec[5] == 3.0  # Human HP: 5 → 3

    def test_returns_list_of_14_floats(self):
        """Output must be list[float] of length 14."""
        state = make_game_state()
        vec = Converter.build_input_vector(state)
        assert isinstance(vec, list)
        assert len(vec) == 14
        for val in vec:
            assert isinstance(val, float)

    def test_class_constants(self):
        """Verify Converter has the expected class constants."""
        assert Converter.NUM_CARD_TYPES == 4
        assert Converter.INPUT_SIZE == 14
        assert Converter.ALL_CARD_TYPES == frozenset({"attack", "defense", "heavy", "counter"})


# ---------------------------------------------------------------------------
# Boundary: malformed state dicts
# ---------------------------------------------------------------------------


class TestBuildInputVectorBoundary:
    """Boundary tests for build_input_vector with invalid inputs."""

    def test_missing_players_key_raises_key_error(self):
        """State dict missing 'players' key should raise KeyError."""
        with pytest.raises(KeyError):
            Converter.build_input_vector({"round": 1})

    def test_missing_player_2_raises_key_error(self):
        """State dict missing player '2' should raise KeyError."""
        with pytest.raises(KeyError):
            Converter.build_input_vector({"players": {"1": {"hp": 3, "hand": [], "history": []}}})

    def test_index_to_card_raises_on_negative(self):
        """index_to_card with negative index raises IndexError."""
        with pytest.raises(IndexError, match="-1"):
            Converter.index_to_card(-1)

    def test_index_to_card_raises_on_out_of_range(self):
        """index_to_card with index >= NUM_CARD_TYPES raises IndexError."""
        with pytest.raises(IndexError, match="4"):
            Converter.index_to_card(4)


# ---------------------------------------------------------------------------
# New tests from design/5: Player perspective in build_input_vector (TC-C1–C3)
# ---------------------------------------------------------------------------


class TestBuildInputVectorPlayerKey:
    """TC-C1–TC-C3: player_key parameter in build_input_vector()."""

    def test_default_player_key_2_matches_original(self):
        """TC-C1: build_input_vector(state) == build_input_vector(state, player_key='2')."""
        state = make_game_state(
            hand_1=["attack", "attack", "defense"],
            hand_2=["heavy", "counter"],
            history_1=["defense"],
            history_2=["attack"],
            hp_1=2,
            hp_2=1,
            round_num=2,
        )
        vec_default = Converter.build_input_vector(state)
        vec_explicit = Converter.build_input_vector(state, player_key="2")
        assert vec_default == vec_explicit
        # Also verify it matches the expected P2 perspective:
        # indices 0–3: P2 hand counts (heavy=1, counter=1)
        assert vec_default[0:4] == [0.0, 0.0, 1.0, 1.0]
        # indices 4–5: P2 HP=1, P1 HP=2
        assert vec_default[4:6] == [1.0, 2.0]
        # indices 6–9: P2 last move (attack) → [1,0,0,0]
        assert vec_default[6:10] == [1.0, 0.0, 0.0, 0.0]
        # indices 10–13: P1 last move (defense) → [0,1,0,0]
        assert vec_default[10:14] == [0.0, 1.0, 0.0, 0.0]

    def test_player_key_1_swaps_perspectives(self):
        """TC-C2: player_key='1' builds from player 1's perspective."""
        state = make_game_state(
            hand_1=["attack", "attack", "defense"],
            hand_2=["heavy", "counter"],
            history_1=["defense"],
            history_2=["attack"],
            hp_1=2,
            hp_2=1,
            round_num=2,
        )
        vec = Converter.build_input_vector(state, player_key="1")
        # indices 0–3: P1 hand counts (attack=2, defense=1)
        assert vec[0:4] == [2.0, 1.0, 0.0, 0.0]
        # indices 4–5: P1 HP=2, P2 HP=1 (self, opponent)
        assert vec[4:6] == [2.0, 1.0]
        # indices 6–9: P1 last move (defense) → [0,1,0,0]
        assert vec[6:10] == [0.0, 1.0, 0.0, 0.0]
        # indices 10–13: P2 last move (attack) → [1,0,0,0]
        assert vec[10:14] == [1.0, 0.0, 0.0, 0.0]

    def test_invalid_player_key_raises_value_error(self):
        """TC-C3: player_key='3' raises ValueError."""
        state = make_game_state()
        with pytest.raises(ValueError, match="player_key"):
            Converter.build_input_vector(state, player_key="3")

    def test_invalid_player_key_empty_string_raises(self):
        """player_key='' raises ValueError."""
        state = make_game_state()
        with pytest.raises(ValueError, match="player_key"):
            Converter.build_input_vector(state, player_key="")
