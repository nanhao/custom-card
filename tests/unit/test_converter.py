"""Unit tests for Converter — 42-d input + action encoding (design/27).

Test cases per design/27 test_cases.md.
"""

import pytest

from src.backend.ai.rnn_v1.model import INPUT_SIZE, POLICY_OUTPUT_SIZE
from src.backend.app.converter import Converter
from src.backend.game.card_type import CardType
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus
from tests.helpers import make_game_state


def _phi(card: CardType, bt: CardType | None = None) -> PlayerHistoryItem:
    """Shorthand for creating PlayerHistoryItem in tests."""
    return PlayerHistoryItem(card=card, build_target=bt)


class TestBuildInputVector42D:
    """TC-IV: build_input_vector 42-d format (design/27)."""

    def test_vector_length_is_42(self):
        """Returned vector length is 42."""
        state = make_game_state(
            hand_2=[CardType.ATTACK, CardType.DEFENSE],
            hand_1=[CardType.ATTACK, CardType.DEFENSE],
            hp_1=20, hp_2=20,
            history_2=[_phi(CardType.ATTACK)],
            history_1=[_phi(CardType.DEFENSE)],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert len(vec) == INPUT_SIZE  # 42

    def test_metadata_segment(self):
        """Indices 0-5: HP, energy, round_in_cycle, cycle."""
        state = make_game_state(
            hp_2=15, hp_1=8,
            energy_2=3, energy_1=1,
            round_in_cycle=2, cycle=3,
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[0] == 15.0
        assert vec[1] == 8.0
        assert vec[2] == 3.0
        assert vec[3] == 1.0
        assert vec[4] == 2.0
        assert vec[5] == 3.0

    def test_own_hand_counts(self):
        """One of each card type in own hand, attack x2."""
        state = make_game_state(
            hand_2=[CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE,
                     CardType.HEAVY, CardType.COUNTER, CardType.ENERGY,
                     CardType.BUILD, CardType.FLEX],
            hand_1=[],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[6] == 2.0
        assert vec[7] == 1.0
        assert vec[8] == 1.0
        assert vec[9] == 1.0
        assert vec[10] == 1.0
        assert vec[11] == 1.0
        assert vec[12] == 1.0

    def test_opponent_hand_counts(self):
        """Opponent has defense x2 and heavy."""
        state = make_game_state(
            hand_1=[CardType.DEFENSE, CardType.DEFENSE, CardType.HEAVY],
            hand_2=[CardType.ATTACK],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[13] == 0.0
        assert vec[14] == 2.0
        assert vec[15] == 1.0
        assert vec[16] == 0.0
        assert vec[17] == 0.0
        assert vec[18] == 0.0
        assert vec[19] == 0.0

    def test_perspective_swap(self):
        """P1 hand maps to indices 6-12, P2 hand to 13-19."""
        state = make_game_state(
            hand_1=[CardType.ATTACK],
            hand_2=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="1")
        assert vec[6] == 1.0
        for i in range(7, 13):
            assert vec[i] == 0.0
        assert vec[13] == 0.0
        assert vec[14] == 1.0
        for i in range(15, 20):
            assert vec[i] == 0.0

    def test_own_status_lifetimes(self):
        """DEFENSE=2, FRAGILE=1, no GUARD."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            statuses_2=[
                PlayerStatus(StatusType.DEFENSE, lifetime=2),
                PlayerStatus(StatusType.FRAGILE, lifetime=1),
            ],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[20] == 2.0
        assert vec[21] == 0.0
        assert vec[22] == 1.0

    def test_opponent_status_lifetimes(self):
        """Opponent has GUARD=3."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            statuses_1=[PlayerStatus(StatusType.GUARD, lifetime=3)],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[23] == 0.0
        assert vec[24] == 3.0
        assert vec[25] == 0.0

    def test_no_statuses_all_zero(self):
        """Indices 20-25 all 0.0 when no statuses."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="2")
        for i in range(20, 26):
            assert vec[i] == 0.0

    def test_own_discard_pile_counts(self):
        """P2 discard: attack x2, defense x3, flex x1."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            discard_pile_2=[CardType.ATTACK, CardType.ATTACK,
                            CardType.DEFENSE, CardType.DEFENSE,
                            CardType.DEFENSE, CardType.FLEX],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[26] == 2.0
        assert vec[27] == 3.0
        assert vec[28] == 0.0
        assert vec[29] == 0.0
        assert vec[30] == 0.0
        assert vec[31] == 0.0
        assert vec[32] == 1.0

    def test_own_draw_pile_counts(self):
        """P2 draw: energy x2, build x1."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            draw_pile_2=[CardType.ENERGY, CardType.ENERGY, CardType.BUILD],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[33] == 0.0
        assert vec[34] == 0.0
        assert vec[35] == 0.0
        assert vec[36] == 0.0
        assert vec[37] == 2.0
        assert vec[38] == 1.0
        assert vec[39] == 0.0

    def test_empty_piles_all_zero(self):
        """Indices 26-39 all 0.0 when piles are empty."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            discard_pile_2=[], draw_pile_2=[],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        for i in range(26, 40):
            assert vec[i] == 0.0

    def test_last_action_segment(self):
        """P2 played ATTACK, P1 played BUILD+DEFENSE."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            history_2=[_phi(CardType.ATTACK)],
            history_1=[_phi(CardType.BUILD, CardType.DEFENSE)],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[40] == 1.0
        assert vec[41] == 4.0

    def test_first_round_no_history(self):
        """Indices 40-41 = -1.0 when no history."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[40] == -1.0
        assert vec[41] == -1.0

    def test_invalid_player_key_raises(self):
        """ValueError for non-'1'/'2' player_key."""
        state = make_game_state()
        for bad_key in ("3", "0", ""):
            with pytest.raises(ValueError):
                Converter.build_input_vector(state, player_key=bad_key)

    def test_empty_hands_all_zero(self):
        """Indices 6-19 all 0.0 when both hands empty."""
        state = make_game_state(hand_2=[], hand_1=[], hp_1=20, hp_2=20)
        vec = Converter.build_input_vector(state, player_key="2")
        for i in range(6, 20):
            assert vec[i] == 0.0

    def test_hp_zero_boundary(self):
        """HP=0 passes through correctly."""
        state = make_game_state(
            hp_2=0, hp_1=1,
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[0] == 0.0
        assert vec[1] == 1.0

    def test_energy_boundaries(self):
        """Energy 0 and 5 pass through correctly."""
        state = make_game_state(
            hp_1=20, hp_2=20, energy_2=5, energy_1=0,
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[2] == 5.0
        assert vec[3] == 0.0

    def test_large_cycle_and_round(self):
        """Large cycle and round values pass through."""
        state = make_game_state(
            hp_1=20, hp_2=20, cycle=100, round_in_cycle=3,
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[4] == 3.0
        assert vec[5] == 100.0

    def test_many_copies_of_one_card(self):
        """5 ATTACK in hand."""
        state = make_game_state(hand_2=[CardType.ATTACK] * 5, hand_1=[], hp_1=20, hp_2=20)
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[6] == 5.0
        for i in range(7, 13):
            assert vec[i] == 0.0

    def test_full_perspective_swap(self):
        """Comprehensive validation when player_key='1'."""
        state = make_game_state(
            hp_1=10, hp_2=20, energy_1=2, energy_2=4,
            hand_1=[CardType.HEAVY, CardType.COUNTER],
            hand_2=[CardType.ATTACK],
            statuses_1=[PlayerStatus(StatusType.DEFENSE, lifetime=1)],
            statuses_2=[PlayerStatus(StatusType.GUARD, lifetime=2),
                        PlayerStatus(StatusType.FRAGILE, lifetime=3)],
            history_1=[_phi(CardType.ATTACK)],
            history_2=[_phi(CardType.DEFENSE)],
        )
        vec = Converter.build_input_vector(state, player_key="1")
        assert vec[0] == 10.0
        assert vec[1] == 20.0
        assert vec[2] == 2.0
        assert vec[3] == 4.0
        assert vec[6] == 0.0
        assert vec[7] == 0.0
        assert vec[8] == 1.0
        assert vec[9] == 1.0
        assert vec[10] == 0.0
        assert vec[11] == 0.0
        assert vec[12] == 0.0
        assert vec[13] == 1.0
        for i in range(14, 20):
            assert vec[i] == 0.0
        assert vec[20] == 1.0
        assert vec[21] == 0.0
        assert vec[22] == 0.0
        assert vec[23] == 0.0
        assert vec[24] == 2.0
        assert vec[25] == 3.0
        assert vec[40] == 1.0
        assert vec[41] == 3.0


class TestCountByCardType:
    """TC-CC: _count_by_card_type internal method."""

    def test_normal_counting(self):
        """Two ATTACK, one DEFENSE, one FLEX."""
        cards = [CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE, CardType.FLEX]
        result = Converter._count_by_card_type(cards)
        assert result == [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def test_empty_list(self):
        """Empty list -> seven zeros."""
        result = Converter._count_by_card_type([])
        assert result == [0.0] * 7

    def test_unknown_item_raises(self):
        """String in card list raises KeyError."""
        with pytest.raises(KeyError):
            Converter._count_by_card_type([CardType.ATTACK, "invalid"])  # type: ignore[list-item]


class TestExtractStatusLifetime:
    """TC-ES: _extract_status_lifetime internal method."""

    def test_found_status(self):
        """DEFENSE with lifetime=3 -> 3.0."""
        statuses = [PlayerStatus(StatusType.DEFENSE, lifetime=3)]
        result = Converter._extract_status_lifetime(statuses, StatusType.DEFENSE)
        assert result == 3.0

    def test_not_found(self):
        """GUARD not in list -> 0.0."""
        statuses = [PlayerStatus(StatusType.DEFENSE, lifetime=2)]
        result = Converter._extract_status_lifetime(statuses, StatusType.GUARD)
        assert result == 0.0

    def test_empty_list(self):
        """Empty status list -> 0.0."""
        result = Converter._extract_status_lifetime([], StatusType.FRAGILE)
        assert result == 0.0

    def test_multiple_statuses(self):
        """Find FRAGILE among DEFENSE, GUARD, FRAGILE."""
        statuses = [
            PlayerStatus(StatusType.DEFENSE, lifetime=1),
            PlayerStatus(StatusType.GUARD, lifetime=2),
            PlayerStatus(StatusType.FRAGILE, lifetime=3),
        ]
        result = Converter._extract_status_lifetime(statuses, StatusType.FRAGILE)
        assert result == 3.0


class TestInputSizeConstant:
    """TC-IS: INPUT_SIZE = 42, model accepts 42-d input."""

    def test_input_size_is_42(self):
        """INPUT_SIZE from model module equals 42."""
        assert INPUT_SIZE == 42

    def test_policy_model_accepts_42d_input(self):
        """GRUPolicyModel forward accepts (1, 1, 42) tensor."""
        import torch
        from src.backend.ai.rnn_v1.model import GRUPolicyModel, HIDDEN_SIZE
        model = GRUPolicyModel()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.randn(1, 1, HIDDEN_SIZE)
        probs, log_probs, new_hidden = model.forward(x, hidden)
        assert probs.shape == (1, POLICY_OUTPUT_SIZE)

    def test_value_model_accepts_42d_input(self):
        """GRUValueModel forward accepts (1, 1, 42) tensor."""
        import torch
        from src.backend.ai.rnn_v1.model import GRUValueModel, HIDDEN_SIZE
        model = GRUValueModel()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.randn(1, 1, HIDDEN_SIZE)
        value, new_hidden = model.forward(x, hidden)
        assert value.shape == (1, 1)


class TestActionToPolicyIndex:
    """TC-AP: action_to_policy_index (unchanged from design/14)."""

    @pytest.mark.parametrize("card_type,expected", [
        (CardType.ATTACK, 1),
        (CardType.DEFENSE, 3),
        (CardType.HEAVY, 5),
        (CardType.COUNTER, 7),
        (CardType.ENERGY, 9),
        (CardType.FLEX, 11),
    ])
    def test_pure_play_actions(self, card_type, expected):
        assert Converter.action_to_policy_index(card_type, None) == expected

    @pytest.mark.parametrize("build_target,expected", [
        (CardType.BUILD, 0),
        (CardType.ATTACK, 2),
        (CardType.DEFENSE, 4),
        (CardType.HEAVY, 6),
        (CardType.COUNTER, 8),
        (CardType.ENERGY, 10),
        (CardType.FLEX, 12),
    ])
    def test_build_actions(self, build_target, expected):
        assert Converter.action_to_policy_index(CardType.BUILD, build_target) == expected

    def test_string_input_raises_valueerror(self):
        with pytest.raises(ValueError):
            Converter.action_to_policy_index("attack", None)  # type: ignore[arg-type]

    def test_non_build_with_target_raises(self):
        with pytest.raises(ValueError):
            Converter.action_to_policy_index(CardType.ATTACK, CardType.DEFENSE)

    def test_build_without_target_raises(self):
        with pytest.raises(ValueError):
            Converter.action_to_policy_index(CardType.BUILD, None)


class TestPolicyIndexToAction:
    """TC-PI: policy_index_to_action (unchanged from design/14)."""

    @pytest.mark.parametrize("index,card_type", [
        (1, CardType.ATTACK),
        (3, CardType.DEFENSE),
        (5, CardType.HEAVY),
        (7, CardType.COUNTER),
        (9, CardType.ENERGY),
        (11, CardType.FLEX),
    ])
    def test_odd_indices_pure_play(self, index, card_type):
        result = Converter.policy_index_to_action(index)
        assert result == (card_type, None)

    @pytest.mark.parametrize("index,build_target", [
        (0, CardType.BUILD),
        (2, CardType.ATTACK),
        (4, CardType.DEFENSE),
        (6, CardType.HEAVY),
        (8, CardType.COUNTER),
        (10, CardType.ENERGY),
        (12, CardType.FLEX),
    ])
    def test_even_indices_build_action(self, index, build_target):
        result = Converter.policy_index_to_action(index)
        assert result == (CardType.BUILD, build_target)

    def test_negative_index_raises(self):
        with pytest.raises(IndexError):
            Converter.policy_index_to_action(-1)

    def test_index_13_raises(self):
        with pytest.raises(IndexError):
            Converter.policy_index_to_action(13)

    def test_roundtrip_consistency(self):
        test_cases = [
            (CardType.ATTACK, None),
            (CardType.DEFENSE, None),
            (CardType.HEAVY, None),
            (CardType.COUNTER, None),
            (CardType.ENERGY, None),
            (CardType.FLEX, None),
            (CardType.BUILD, CardType.BUILD),
            (CardType.BUILD, CardType.ATTACK),
            (CardType.BUILD, CardType.DEFENSE),
            (CardType.BUILD, CardType.HEAVY),
            (CardType.BUILD, CardType.COUNTER),
            (CardType.BUILD, CardType.ENERGY),
            (CardType.BUILD, CardType.FLEX),
        ]
        for card_type, bt in test_cases:
            idx = Converter.action_to_policy_index(card_type, bt)
            result = Converter.policy_index_to_action(idx)
            assert result == (card_type, bt)


class TestBuildCost:
    """Converter.build_cost() (unchanged)."""

    @pytest.mark.parametrize("card_type,expected", [
        (CardType.ATTACK, 1),
        (CardType.DEFENSE, 1),
        (CardType.HEAVY, 2),
        (CardType.COUNTER, 2),
        (CardType.ENERGY, 1),
        (CardType.BUILD, 1),
        (CardType.FLEX, 2),
    ])
    def test_known_card_costs(self, card_type, expected):
        assert Converter.build_cost(card_type) == expected

    def test_unknown_card_returns_999(self):
        assert Converter.build_cost("nonexistent") == 999  # type: ignore[arg-type]


class TestConstantsRemoved:
    """INPUT_SIZE / POLICY_OUTPUT_SIZE removed from Converter."""

    def test_input_size_removed_from_converter(self):
        assert not hasattr(Converter, "INPUT_SIZE")

    def test_policy_output_size_removed_from_converter(self):
        assert not hasattr(Converter, "POLICY_OUTPUT_SIZE")
