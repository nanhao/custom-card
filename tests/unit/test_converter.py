"""Unit tests for Converter.build_input_vector — design/13 input reordering.

Per design/13 §2, the 25-d input vector follows the new order:
    index 0:    Own HP
    index 1:    Opponent HP
    index 2:    round_in_cycle (1-5)
    index 3:    cycle number
    indices 4-10:   Card counts (attack..flex)
    indices 11-17:  One-hot of own last move
    indices 18-24:  One-hot of opponent's last move
"""

import pytest

from src.backend.app.converter import Converter
from tests.helpers import make_game_state


class TestBuildInputVectorNewOrder:
    """Verify the new index ordering defined by design/13."""

    # ------------------------------------------------------------------
    # TC-IV-01: Full ordering correctness (player_key="2")
    # ------------------------------------------------------------------

    def test_full_ordering_player2(self):
        """All 25 indices in correct positions when viewing from player 2."""
        # Given: P1 HP=8, P2 HP=15, round_in_cycle=3, cycle=2,
        #        P2 has all 7 card types, no history
        state = make_game_state(
            hand_1=[],
            hand_2=["attack", "defense", "heavy", "counter", "energy", "build", "flex"],
            hp_1=8, hp_2=15,
            round_in_cycle=3, cycle=2,
        )

        # When
        vec = Converter.build_input_vector(state, player_key="2")

        # Then: indices 0-3 (metadata)
        assert len(vec) == Converter.INPUT_SIZE
        assert vec[0] == 15.0  # own HP (P2)
        assert vec[1] == 8.0   # opponent HP (P1)
        assert vec[2] == 3.0   # round_in_cycle
        assert vec[3] == 2.0   # cycle

        # indices 4-10: card counts — one of each type
        for i in range(7):
            assert vec[4 + i] == 1.0, f"index {4 + i}: expected 1.0"

        # indices 11-24: one-hot — all zeros (no history)
        for i in range(11, 25):
            assert vec[i] == 0.0, f"index {i}: expected 0.0"

    # ------------------------------------------------------------------
    # TC-IV-02: Perspective swap (player_key="1")
    # ------------------------------------------------------------------

    def test_perspective_swap_player1(self):
        """HP and counts swap when viewing from player 1's perspective."""
        state = make_game_state(
            hand_1=["attack", "defense"],
            hand_2=["heavy", "counter", "energy"],
            hp_1=20, hp_2=10,
            round_in_cycle=1, cycle=1,
        )

        vec = Converter.build_input_vector(state, player_key="1")

        # metadata swapped
        assert vec[0] == 20.0  # own HP = P1
        assert vec[1] == 10.0  # opponent HP = P2

        # card counts from P1's hand
        assert vec[4] == 1.0   # attack
        assert vec[5] == 1.0   # defense
        assert vec[6] == 0.0   # heavy (not in P1 hand)
        assert vec[7] == 0.0   # counter
        assert vec[8] == 0.0   # energy

    # ------------------------------------------------------------------
    # TC-IV-03: One-hot history encoding
    # ------------------------------------------------------------------

    def test_onehot_history(self):
        """Last played cards are encoded in indices 11-24."""
        state = make_game_state(
            hand_1=["attack"],
            hand_2=["defense"],
            hp_1=20, hp_2=20,
            history_1=["attack"],
            history_2=["defense"],
        )

        vec = Converter.build_input_vector(state, player_key="2")

        # Own last move (P2) = defense -> index 12 = 1.0
        assert vec[11] == 0.0   # attack
        assert vec[12] == 1.0   # defense
        for i in range(13, 18):
            assert vec[i] == 0.0, f"index {i}: expected 0.0"

        # Opponent last move (P1) = attack -> index 18 = 1.0
        assert vec[18] == 1.0   # attack
        for i in range(19, 25):
            assert vec[i] == 0.0, f"index {i}: expected 0.0"

    # ------------------------------------------------------------------
    # TC-IV-04: Partial / empty hand
    # ------------------------------------------------------------------

    def test_partial_hand(self):
        """Only present cards are counted; missing types are 0.0."""
        state = make_game_state(
            hand_2=["attack", "attack"],
            hp_1=20, hp_2=20,
        )

        vec = Converter.build_input_vector(state, player_key="2")

        assert vec[4] == 2.0   # attack x2
        for i in range(5, 11):
            assert vec[i] == 0.0, f"index {i}: expected 0.0"

    def test_empty_hand(self):
        """Empty hand produces all-zero card counts."""
        state = make_game_state(
            hand_2=[],
            hp_1=20, hp_2=20,
        )

        vec = Converter.build_input_vector(state, player_key="2")

        for i in range(4, 11):
            assert vec[i] == 0.0, f"index {i}: expected 0.0"

    # ------------------------------------------------------------------
    # TC-IV-05: Invalid player_key
    # ------------------------------------------------------------------

    def test_invalid_player_key_raises(self):
        """ValueError raised for player_key not '1' or '2'."""
        state = make_game_state()
        with pytest.raises(ValueError):
            Converter.build_input_vector(state, player_key="3")
        with pytest.raises(ValueError):
            Converter.build_input_vector(state, player_key="0")
        with pytest.raises(ValueError):
            Converter.build_input_vector(state, player_key="")

    # ------------------------------------------------------------------
    # TC-IV-06: HP boundary (zero)
    # ------------------------------------------------------------------

    def test_hp_zero_boundary(self):
        """HP=0 is passed through without normalization."""
        state = make_game_state(
            hp_1=1, hp_2=0,
            hand_2=["attack"],
        )

        vec = Converter.build_input_vector(state, player_key="2")

        assert vec[0] == 0.0   # own HP
        assert vec[1] == 1.0   # opponent HP

    # ------------------------------------------------------------------
    # TC-IV-07: Default player_key
    # ------------------------------------------------------------------

    def test_default_player_key_is_2(self):
        """When player_key is omitted, defaults to player 2's perspective."""
        state = make_game_state(
            hand_2=["attack"], hand_1=["defense"],
            hp_1=10, hp_2=20,
        )

        vec_default = Converter.build_input_vector(state)
        vec_explicit = Converter.build_input_vector(state, player_key="2")

        assert vec_default == vec_explicit
        assert vec_default[0] == 20.0  # P2 HP
