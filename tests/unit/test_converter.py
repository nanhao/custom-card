"""Unit tests for Converter — 15-d input + action encoding (design/14).

Test cases per design/14 test_cases.md §1 (TC-IV), §2 (TC-AP), §3 (TC-PI).
"""

import pytest

from src.backend.app.converter import Converter
from src.backend.game.card import CardType
from src.backend.game.player import PlayerHistoryItem
from tests.helpers import make_game_state


def _phi(card: CardType, bt: CardType | None = None) -> PlayerHistoryItem:
    """Shorthand for creating PlayerHistoryItem in tests."""
    return PlayerHistoryItem(card=card, build_target=bt)


class TestBuildInputVector15D:
    """TC-IV-01~09: build_input_vector with 15-d format."""

    # ------------------------------------------------------------------
    # TC-IV-01: Full ordering (player_key="2", with history)
    # ------------------------------------------------------------------

    def test_full_ordering_player2_with_history(self):
        """All 15 indices correct with attack/defense history."""
        state = make_game_state(
            hand_1=[],
            hand_2=[CardType.ATTACK, CardType.DEFENSE, CardType.HEAVY, CardType.COUNTER,
                     CardType.ENERGY, CardType.BUILD, CardType.FLEX],
            hp_1=8, hp_2=15,
            energy_1=1, energy_2=3,
            round_in_cycle=3, cycle=2,
            history_1=[_phi(CardType.DEFENSE)],
            history_2=[_phi(CardType.ATTACK)],
        )

        vec = Converter.build_input_vector(state, player_key="2")

        assert len(vec) == Converter.INPUT_SIZE  # 15
        # Metadata (indices 0-5)
        assert vec[0] == 15.0   # own HP (P2)
        assert vec[1] == 8.0    # opponent HP (P1)
        assert vec[2] == 3.0    # own energy (P2)
        assert vec[3] == 1.0    # opponent energy (P1)
        assert vec[4] == 3.0    # round_in_cycle
        assert vec[5] == 2.0    # cycle
        # Card counts (indices 6-12) — one of each type
        for i in range(7):
            assert vec[6 + i] == 1.0, f"index {6 + i}: expected 1.0"
        # Last action (indices 13-14)
        assert vec[13] == 1.0   # P2 last: attack → policy index 1
        assert vec[14] == 3.0   # P1 last: defense → policy index 3

    # ------------------------------------------------------------------
    # TC-IV-02: Perspective swap (player_key="1")
    # ------------------------------------------------------------------

    def test_perspective_swap_player1(self):
        """HP, energy, and last-action indices swap for player 1."""
        state = make_game_state(
            hand_1=[CardType.ATTACK, CardType.DEFENSE],
            hand_2=[CardType.HEAVY, CardType.COUNTER, CardType.ENERGY],
            hp_1=20, hp_2=10,
            energy_1=2, energy_2=4,
            round_in_cycle=1, cycle=1,
            history_1=[_phi(CardType.ATTACK)],
            history_2=[_phi(CardType.DEFENSE)],
        )

        vec = Converter.build_input_vector(state, player_key="1")

        assert vec[0] == 20.0   # own HP = P1
        assert vec[1] == 10.0   # opponent HP = P2
        assert vec[2] == 2.0    # own energy = P1
        assert vec[3] == 4.0    # opponent energy = P2
        # Card counts from P1's hand
        assert vec[6] == 1.0    # attack
        assert vec[7] == 1.0    # defense
        assert vec[8] == 0.0    # heavy (not in P1 hand)
        # Last action swapped
        assert vec[13] == 1.0   # P1 last: attack → 1
        assert vec[14] == 3.0   # P2 last: defense → 3

    # ------------------------------------------------------------------
    # TC-IV-03: No history (first round)
    # ------------------------------------------------------------------

    def test_first_round_no_history(self):
        """Empty history → last_action indices = -1."""
        state = make_game_state(
            hand_2=[CardType.ATTACK],
            hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
        )

        vec = Converter.build_input_vector(state, player_key="2")

        assert vec[13] == -1.0
        assert vec[14] == -1.0

    # ------------------------------------------------------------------
    # TC-IV-04: Build actions in history
    # ------------------------------------------------------------------

    def test_build_action_history(self):
        """Build+attack (index 2) and build+defense (index 4)."""
        state = make_game_state(
            hand_2=[CardType.ATTACK],
            hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            history_2=[_phi(CardType.BUILD, CardType.ATTACK)],
            history_1=[_phi(CardType.BUILD, CardType.DEFENSE)],
        )

        vec = Converter.build_input_vector(state, player_key="2")

        assert vec[13] == 2.0   # build+attack
        assert vec[14] == 4.0   # build+defense

    # ------------------------------------------------------------------
    # TC-IV-05: Empty hand
    # ------------------------------------------------------------------

    def test_empty_hand_all_zero_counts(self):
        """Empty hand → indices 6-12 all 0.0."""
        state = make_game_state(
            hand_2=[],
            hp_1=20, hp_2=20,
        )

        vec = Converter.build_input_vector(state, player_key="2")

        for i in range(6, 13):
            assert vec[i] == 0.0, f"index {i}: expected 0.0"

    # ------------------------------------------------------------------
    # TC-IV-06: Invalid player_key
    # ------------------------------------------------------------------

    def test_invalid_player_key_raises(self):
        """ValueError for non-'1'/'2' player_key."""
        state = make_game_state()
        for bad_key in ("3", "0", ""):
            with pytest.raises(ValueError):
                Converter.build_input_vector(state, player_key=bad_key)

    # ------------------------------------------------------------------
    # TC-IV-07: HP=0 boundary
    # ------------------------------------------------------------------

    def test_hp_zero_boundary(self):
        """HP=0 passes through without normalization."""
        state = make_game_state(
            hp_1=1, hp_2=0,
            hand_2=[CardType.ATTACK],
        )

        vec = Converter.build_input_vector(state, player_key="2")

        assert vec[0] == 0.0    # own HP
        assert vec[1] == 1.0    # opponent HP

    # ------------------------------------------------------------------
    # TC-IV-08: Energy boundary
    # ------------------------------------------------------------------

    def test_energy_boundaries(self):
        """Energy 0 and 5 pass through correctly."""
        state = make_game_state(
            hp_1=20, hp_2=20,
            energy_1=0, energy_2=5,
        )

        vec = Converter.build_input_vector(state, player_key="2")

        assert vec[2] == 5.0    # own energy (P2)
        assert vec[3] == 0.0    # opponent energy (P1)

    # ------------------------------------------------------------------
    # TC-IV-09: Large cycle/round
    # ------------------------------------------------------------------

    def test_large_cycle_and_round(self):
        """Large cycle and round values pass through."""
        state = make_game_state(
            hp_1=20, hp_2=20,
            cycle=100, round_in_cycle=5,
        )

        vec = Converter.build_input_vector(state, player_key="2")

        assert vec[4] == 5.0
        assert vec[5] == 100.0


class TestActionToPolicyIndex:
    """TC-AP-01~05: action_to_policy_index."""

    # ------------------------------------------------------------------
    # TC-AP-01: Pure play actions (odd indices)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("card_str,expected", [
        ("attack", 1),
        ("defense", 3),
        ("heavy", 5),
        ("counter", 7),
        ("energy", 9),
        ("flex", 11),
    ])
    def test_pure_play_actions(self, card_str, expected):
        """Pure card plays map to odd indices."""
        assert Converter.action_to_policy_index(card_str, None) == expected

    # ------------------------------------------------------------------
    # TC-AP-02: Build actions (even indices)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("build_target,expected", [
        ("build", 0),
        ("attack", 2),
        ("defense", 4),
        ("heavy", 6),
        ("counter", 8),
        ("energy", 10),
        ("flex", 12),
    ])
    def test_build_actions(self, build_target, expected):
        """Build + target map to even indices."""
        assert Converter.action_to_policy_index("build", build_target) == expected

    # ------------------------------------------------------------------
    # TC-AP-03: Invalid card_str
    # ------------------------------------------------------------------

    def test_invalid_card_str_raises(self):
        """Unknown card string raises ValueError."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index("invalid", None)

    # ------------------------------------------------------------------
    # TC-AP-04: Non-build with build_target
    # ------------------------------------------------------------------

    def test_non_build_with_target_raises(self):
        """attack with build_target='defense' is illegal."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index("attack", "defense")

    # ------------------------------------------------------------------
    # TC-AP-05: Build without build_target
    # ------------------------------------------------------------------

    def test_build_without_target_raises(self):
        """build with build_target=None is illegal."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index("build", None)


class TestPolicyIndexToAction:
    """TC-PI-01~04: policy_index_to_action."""

    # ------------------------------------------------------------------
    # TC-PI-01: Odd indices → pure play
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("index,card_str", [
        (1, "attack"),
        (3, "defense"),
        (5, "heavy"),
        (7, "counter"),
        (9, "energy"),
        (11, "flex"),
    ])
    def test_odd_indices_pure_play(self, index, card_str):
        """Odd indices decode to (card_str, None)."""
        result = Converter.policy_index_to_action(index)
        assert result == (card_str, None)

    # ------------------------------------------------------------------
    # TC-PI-02: Even indices → build + target
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("index,build_target", [
        (0, "build"),
        (2, "attack"),
        (4, "defense"),
        (6, "heavy"),
        (8, "counter"),
        (10, "energy"),
        (12, "flex"),
    ])
    def test_even_indices_build_action(self, index, build_target):
        """Even indices decode to ('build', target)."""
        result = Converter.policy_index_to_action(index)
        assert result == ("build", build_target)

    # ------------------------------------------------------------------
    # TC-PI-03: Out-of-range indices
    # ------------------------------------------------------------------

    def test_negative_index_raises(self):
        """Index -1 raises IndexError."""
        with pytest.raises(IndexError):
            Converter.policy_index_to_action(-1)

    def test_index_13_raises(self):
        """Index 13 (beyond max) raises IndexError."""
        with pytest.raises(IndexError):
            Converter.policy_index_to_action(13)

    # ------------------------------------------------------------------
    # TC-PI-04: Round-trip consistency
    # ------------------------------------------------------------------

    def test_roundtrip_consistency(self):
        """action_to_policy_index → policy_index_to_action is identity."""
        test_cases = [
            ("attack", None),
            ("defense", None),
            ("heavy", None),
            ("counter", None),
            ("energy", None),
            ("flex", None),
            ("build", "build"),
            ("build", "attack"),
            ("build", "defense"),
            ("build", "heavy"),
            ("build", "counter"),
            ("build", "energy"),
            ("build", "flex"),
        ]
        for card_str, bt in test_cases:
            idx = Converter.action_to_policy_index(card_str, bt)
            result = Converter.policy_index_to_action(idx)
            assert result == (card_str, bt), f"Failed for ({card_str}, {bt})"


class TestBuildCost:
    """Converter.build_cost() edge cases (C8)."""

    @pytest.mark.parametrize("card_str,expected", [
        ("attack", 1),
        ("defense", 1),
        ("heavy", 1),
        ("counter", 2),
        ("energy", 1),
        ("build", 1),
        ("flex", 2),
    ])
    def test_known_card_costs(self, card_str, expected):
        """All known card types return correct build cost."""
        assert Converter.build_cost(card_str) == expected

    def test_unknown_card_returns_999(self):
        """Unknown card string returns 999 as sentinel."""
        assert Converter.build_cost("nonexistent") == 999
