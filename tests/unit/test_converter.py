"""Unit tests for Converter — 62-d input + action encoding (design/37).

Per design/37: INPUT_SIZE 56→62, POLICY_OUTPUT_SIZE 14→24, 8 card types.
Test cases per design/34 test_cases.md, updated for design/37.
"""

import pytest

from src.backend.ai.rnn_v1.model import INPUT_SIZE, POLICY_OUTPUT_SIZE
from src.backend.app.converter import Converter
from src.backend.game.card_type import CardType
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus
from tests.helpers import make_game_state


def _phi(
    card: CardType,
    bt: CardType | None = None,
    dt: CardType | None = None,
) -> PlayerHistoryItem:
    """Shorthand for creating PlayerHistoryItem in tests (design/37: 3-field)."""
    return PlayerHistoryItem(card=card, build_target=bt, dismantle_target=dt)


class TestBuildInputVector62D:
    """TC-IV: build_input_vector 62-d format (design/37)."""

    def test_vector_length_is_62(self):
        """Returned vector length is 62."""
        state = make_game_state(
            hand_2=[CardType.ATTACK, CardType.DEFENSE],
            hand_1=[CardType.ATTACK, CardType.DEFENSE],
            hp_1=20, hp_2=20,
            history_2=[_phi(CardType.ATTACK)],
            history_1=[_phi(CardType.DEFENSE)],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert len(vec) == INPUT_SIZE  # 62

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
        """One of each card type in own hand, attack x2. 8 types: BUILD,ATTACK,DEFENSE,HEAVY,COUNTER,ENERGY,FLEX,IRON_DEFENSE."""
        state = make_game_state(
            hand_2=[CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE,
                     CardType.HEAVY, CardType.COUNTER, CardType.ENERGY,
                     CardType.BUILD, CardType.FLEX],
            hand_1=[],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="2")
        # Order: BUILD, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
        assert vec[6] == 1.0   # BUILD
        assert vec[7] == 2.0   # ATTACK x2
        assert vec[8] == 1.0   # DEFENSE
        assert vec[9] == 1.0   # HEAVY
        assert vec[10] == 1.0  # COUNTER
        assert vec[11] == 1.0  # ENERGY
        assert vec[12] == 1.0  # FLEX
        assert vec[13] == 0.0  # IRON_DEFENSE

    def test_opponent_hand_counts(self):
        """Opponent has defense x2 and heavy. 8 types (design/37: indices 14-21)."""
        state = make_game_state(
            hand_1=[CardType.DEFENSE, CardType.DEFENSE, CardType.HEAVY],
            hand_2=[CardType.ATTACK],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="2")
        # Order: BUILD, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
        assert vec[14] == 0.0  # BUILD
        assert vec[15] == 0.0  # ATTACK
        assert vec[16] == 2.0  # DEFENSE
        assert vec[17] == 1.0  # HEAVY
        assert vec[18] == 0.0  # COUNTER
        assert vec[19] == 0.0  # ENERGY
        assert vec[20] == 0.0  # FLEX
        assert vec[21] == 0.0  # IRON_DEFENSE

    def test_perspective_swap(self):
        """P1 hand maps to indices 6-13, P2 hand to 14-21. BUILD is position 0."""
        state = make_game_state(
            hand_1=[CardType.ATTACK],
            hand_2=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="1")
        # Own hand (P1): indices 6-13
        assert vec[6] == 0.0   # BUILD
        assert vec[7] == 1.0   # ATTACK
        for i in range(8, 14):
            assert vec[i] == 0.0
        # Opp hand (P2): indices 14-21
        assert vec[14] == 0.0
        assert vec[15] == 0.0  # ATTACK
        assert vec[16] == 1.0  # DEFENSE
        for i in range(17, 22):
            assert vec[i] == 0.0

    def test_own_status_lifetimes(self):
        """DEFENSE=2, FRAGILE=1, no GUARD. Per design/37: indices 22-24."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            statuses_2=[
                PlayerStatus(StatusType.DEFENSE, lifetime=2),
                PlayerStatus(StatusType.FRAGILE, lifetime=1),
            ],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[22] == 2.0
        assert vec[23] == 0.0
        assert vec[24] == 1.0

    def test_opponent_status_lifetimes(self):
        """Opponent has GUARD=3. Per design/37: indices 25-27."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            statuses_1=[PlayerStatus(StatusType.GUARD, lifetime=3)],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[25] == 0.0
        assert vec[26] == 3.0
        assert vec[27] == 0.0

    def test_no_statuses_all_zero(self):
        """Indices 22-27 all 0.0 when no statuses (design/37)."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="2")
        for i in range(22, 28):
            assert vec[i] == 0.0

    def test_own_discard_pile_counts(self):
        """P2 discard: attack x2, defense x3, flex x1. Per design/37: indices 28-35."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            discard_pile_2=[CardType.ATTACK, CardType.ATTACK,
                            CardType.DEFENSE, CardType.DEFENSE,
                            CardType.DEFENSE, CardType.FLEX],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        # Order: BUILD, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
        assert vec[28] == 0.0  # BUILD
        assert vec[29] == 2.0  # ATTACK
        assert vec[30] == 3.0  # DEFENSE
        assert vec[31] == 0.0  # HEAVY
        assert vec[32] == 0.0  # COUNTER
        assert vec[33] == 0.0  # ENERGY
        assert vec[34] == 1.0  # FLEX
        assert vec[35] == 0.0  # IRON_DEFENSE

    def test_own_draw_pile_counts(self):
        """P2 draw: energy x2, build x1. Per design/37: indices 44-51."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            draw_pile_2=[CardType.ENERGY, CardType.ENERGY, CardType.BUILD],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        # Order: BUILD, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
        assert vec[44] == 1.0  # BUILD
        assert vec[45] == 0.0  # ATTACK
        assert vec[46] == 0.0  # DEFENSE
        assert vec[47] == 0.0  # HEAVY
        assert vec[48] == 0.0  # COUNTER
        assert vec[49] == 2.0  # ENERGY
        assert vec[50] == 0.0  # FLEX
        assert vec[51] == 0.0  # IRON_DEFENSE

    def test_opponent_discard_pile_counts(self):
        """P1 discard: heavy x1, energy x2, build x1. Per design/37: indices 36-43."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            discard_pile_1=[CardType.HEAVY, CardType.ENERGY, CardType.ENERGY, CardType.BUILD],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        # Order: BUILD, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
        assert vec[36] == 1.0  # BUILD
        assert vec[37] == 0.0  # ATTACK
        assert vec[38] == 0.0  # DEFENSE
        assert vec[39] == 1.0  # HEAVY
        assert vec[40] == 0.0  # COUNTER
        assert vec[41] == 2.0  # ENERGY
        assert vec[42] == 0.0  # FLEX
        assert vec[43] == 0.0  # IRON_DEFENSE

    def test_opponent_discard_pile_empty(self):
        """Opponent discard empty → indices 36-43 all 0.0 (design/37)."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            discard_pile_1=[],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        for i in range(36, 44):
            assert vec[i] == 0.0

    def test_opponent_draw_pile_counts(self):
        """P1 draw: counter x2, flex x3. Per design/37: indices 52-59."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            draw_pile_1=[CardType.COUNTER, CardType.COUNTER, CardType.FLEX, CardType.FLEX, CardType.FLEX],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        # Order: BUILD, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
        assert vec[52] == 0.0  # BUILD
        assert vec[53] == 0.0  # ATTACK
        assert vec[54] == 0.0  # DEFENSE
        assert vec[55] == 0.0  # HEAVY
        assert vec[56] == 2.0  # COUNTER
        assert vec[57] == 0.0  # ENERGY
        assert vec[58] == 3.0  # FLEX
        assert vec[59] == 0.0  # IRON_DEFENSE

    def test_opponent_draw_pile_empty(self):
        """Opponent draw empty → indices 52-59 all 0.0 (design/37)."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            draw_pile_1=[],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        for i in range(52, 60):
            assert vec[i] == 0.0

    def test_empty_piles_all_zero(self):
        """Indices 28-59 all 0.0 when all piles are empty (design/37: expanded range)."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            discard_pile_2=[], draw_pile_2=[],
            discard_pile_1=[], draw_pile_1=[],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        for i in range(28, 60):
            assert vec[i] == 0.0

    def test_piles_non_none_no_error(self):
        """build_input_vector operates normally when piles are non-None."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            discard_pile_2=[], draw_pile_2=[],
            discard_pile_1=[], draw_pile_1=[],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert len(vec) == INPUT_SIZE

    def test_last_action_segment(self):
        """P2 played ATTACK, P1 played BUILD+DEFENSE. Per design/37: indices 60-61."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
            history_2=[_phi(CardType.ATTACK)],
            history_1=[_phi(CardType.BUILD, CardType.DEFENSE)],
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[60] == 3.0  # ATTACK play → index 3 (design/37)
        assert vec[61] == 7.0  # BUILD+DEFENSE → index 7 (design/37)

    def test_first_round_no_history(self):
        """Indices 60-61 = -1.0 when no history."""
        state = make_game_state(
            hand_2=[CardType.ATTACK], hand_1=[CardType.DEFENSE],
            hp_1=20, hp_2=20,
        )
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[60] == -1.0
        assert vec[61] == -1.0

    def test_invalid_player_key_raises(self):
        """ValueError for non-'1'/'2' player_key."""
        state = make_game_state()
        for bad_key in ("3", "0", ""):
            with pytest.raises(ValueError):
                Converter.build_input_vector(state, player_key=bad_key)

    def test_empty_hands_all_zero(self):
        """Indices 6-21 all 0.0 when both hands empty (design/37: 8+8=16 slots)."""
        state = make_game_state(hand_2=[], hand_1=[], hp_1=20, hp_2=20)
        vec = Converter.build_input_vector(state, player_key="2")
        for i in range(6, 22):
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
        """5 ATTACK in hand. ATTACK is position 1 (index 7). 8 types → check indices 6-13."""
        state = make_game_state(hand_2=[CardType.ATTACK] * 5, hand_1=[], hp_1=20, hp_2=20)
        vec = Converter.build_input_vector(state, player_key="2")
        assert vec[6] == 0.0   # BUILD
        assert vec[7] == 5.0   # ATTACK
        for i in range(8, 14):
            assert vec[i] == 0.0

    def test_full_perspective_swap(self):
        """Comprehensive validation when player_key='1' (design/37: 62-d indices)."""
        state = make_game_state(
            hp_1=10, hp_2=20, energy_1=2, energy_2=4,
            hand_1=[CardType.HEAVY, CardType.COUNTER],
            hand_2=[CardType.ATTACK],
            statuses_1=[PlayerStatus(StatusType.DEFENSE, lifetime=1)],
            statuses_2=[PlayerStatus(StatusType.GUARD, lifetime=2),
                        PlayerStatus(StatusType.FRAGILE, lifetime=3)],
            history_1=[_phi(CardType.ATTACK)],
            history_2=[_phi(CardType.DEFENSE)],
            discard_pile_1=[CardType.ATTACK], draw_pile_1=[CardType.ENERGY],
            discard_pile_2=[CardType.DEFENSE, CardType.DEFENSE], draw_pile_2=[CardType.FLEX],
        )
        vec = Converter.build_input_vector(state, player_key="1")
        # Metadata (indices 0-5)
        assert vec[0] == 10.0
        assert vec[1] == 20.0
        assert vec[2] == 2.0
        assert vec[3] == 4.0
        # P1 hand (indices 6-13): BUILD=0, ATTACK=0, DEFENSE=0, HEAVY=1, COUNTER=1, ENERGY=0, FLEX=0, IRON_DEFENSE=0
        assert vec[6] == 0.0   # BUILD
        assert vec[7] == 0.0   # ATTACK
        assert vec[8] == 0.0   # DEFENSE
        assert vec[9] == 1.0   # HEAVY
        assert vec[10] == 1.0  # COUNTER
        assert vec[11] == 0.0  # ENERGY
        assert vec[12] == 0.0  # FLEX
        assert vec[13] == 0.0  # IRON_DEFENSE
        # P2 hand (indices 14-21): BUILD=0, ATTACK=1, rest=0
        assert vec[14] == 0.0  # BUILD
        assert vec[15] == 1.0  # ATTACK
        for i in range(16, 22):
            assert vec[i] == 0.0
        # Own status (indices 22-24)
        assert vec[22] == 1.0
        assert vec[23] == 0.0
        assert vec[24] == 0.0
        # Opp status (indices 25-27)
        assert vec[25] == 0.0
        assert vec[26] == 2.0
        assert vec[27] == 3.0
        # P1 discard (indices 28-35): attack=1 at position 1
        assert vec[29] == 1.0  # ATTACK in own discard
        # P2 discard (indices 36-43): defense=2 at position 2
        assert vec[38] == 2.0  # DEFENSE in opp discard
        # P1 draw (indices 44-51): energy=1 at position 5
        assert vec[49] == 1.0  # ENERGY in own draw
        # P2 draw (indices 52-59): flex=1 at position 6
        assert vec[58] == 1.0  # FLEX in opp draw
        # Last actions (indices 60-61): ATTACK→3, DEFENSE→6 (design/37)
        assert vec[60] == 3.0
        assert vec[61] == 6.0

    def test_build_adds_to_opponent_draw(self):
        """After play_round with BUILD+ATTACK, opponent draw has ATTACK (design/37: draw_pile)."""
        import random
        from src.backend.game.game_core import GameCore
        gc = GameCore()
        # Seed RNG so shuffle is deterministic (initial draw pile order known).
        # With seed=0: remaining draw=[FLEX, IRON_DEFENSE] after dealing 5.
        random.seed(0)
        gc.start_game()
        # Ensure BUILD is in P1's hand and energy is sufficient
        gc.player_1.hand = [CardType.BUILD]
        gc.player_2.hand = [CardType.DEFENSE]
        gc.player_1.energy = 3
        gc.play_round([CardType.BUILD], [CardType.DEFENSE], build_target_1=CardType.ATTACK)
        gs = gc.get_state()
        vec = Converter.build_input_vector(gs, player_key="2")
        # P1 draw pile (opponent's draw from P2 perspective): should contain the built ATTACK
        # ATTACK is position 1 → index 52+1=53
        assert vec[53] == 2.0  # ATTACK in opponent (P1) draw_pile (design/38: 3 initial ATTACK)
        # P2 discard pile (own): contains DEFENSE at position 2 → index 28+2=30
        assert vec[30] == 1.0  # DEFENSE in own (P2) discard


class TestCountByCardType:
    """TC-CC: _count_by_card_type internal method. 8 types: BUILD,ATTACK,DEFENSE,HEAVY,COUNTER,ENERGY,FLEX,IRON_DEFENSE."""

    def test_normal_counting(self):
        """Two ATTACK, one DEFENSE, one FLEX → 8-element result."""
        cards = [CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE, CardType.FLEX]
        result = Converter._count_by_card_type(cards)
        # Order: BUILD, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
        assert result == [0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    def test_empty_list(self):
        """Empty list → eight zeros."""
        result = Converter._count_by_card_type([])
        assert result == [0.0] * 8

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
    """TC-IS: INPUT_SIZE = 62, model accepts 62-d input (design/37)."""

    def test_input_size_is_62(self):
        """INPUT_SIZE from model module equals 62."""
        assert INPUT_SIZE == 62

    def test_policy_model_accepts_62d_input(self):
        """GRUPolicyModel forward accepts (1, 1, 62) tensor."""
        import torch
        from src.backend.ai.rnn_v1.model import GRUPolicyModel, HIDDEN_SIZE
        model = GRUPolicyModel()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.randn(1, 1, HIDDEN_SIZE)
        probs, log_probs, new_hidden = model.forward(x, hidden)
        assert probs.shape == (1, POLICY_OUTPUT_SIZE)

    def test_value_model_accepts_62d_input(self):
        """GRUValueModel forward accepts (1, 1, 62) tensor."""
        import torch
        from src.backend.ai.rnn_v1.model import GRUValueModel, HIDDEN_SIZE
        model = GRUValueModel()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.randn(1, 1, HIDDEN_SIZE)
        value, new_hidden = model.forward(x, hidden)
        assert value.shape == (1, 1)


class TestActionToPolicyIndex:
    """TC-AP: action_to_policy_index (design/37: 24-d, triplet: play/build/dismantle)."""

    def test_pass_action(self):
        """card=None, build_target=None → index 0 (pass)."""
        assert Converter.action_to_policy_index(None) == 0

    def test_pass_with_build_target_raises(self):
        """pass (card=None) with build_target raises ValueError."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index(None, CardType.ATTACK)

    def test_pass_with_dismantle_target_raises(self):
        """pass (card=None) with dismantle_target raises ValueError."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index(None, dismantle_target=CardType.ATTACK)

    @pytest.mark.parametrize("card_type,expected", [
        (CardType.ATTACK, 3),
        (CardType.DEFENSE, 6),
        (CardType.HEAVY, 9),
        (CardType.COUNTER, 12),
        (CardType.ENERGY, 15),
        (CardType.FLEX, 18),
        (CardType.IRON_DEFENSE, 21),
    ])
    def test_pure_play_actions(self, card_type, expected):
        """Pure card plays map to play indices (3,6,9,12,15,18,21). Design/37 triplet layout."""
        assert Converter.action_to_policy_index(card_type) == expected

    @pytest.mark.parametrize("build_target,expected", [
        (CardType.BUILD, 1),
        (CardType.ATTACK, 4),
        (CardType.DEFENSE, 7),
        (CardType.HEAVY, 10),
        (CardType.COUNTER, 13),
        (CardType.ENERGY, 16),
        (CardType.FLEX, 19),
        (CardType.IRON_DEFENSE, 22),
    ])
    def test_build_actions(self, build_target, expected):
        """Build + target map to build indices. Design/37 triplet layout."""
        assert Converter.action_to_policy_index(CardType.BUILD, build_target) == expected

    @pytest.mark.parametrize("dismantle_target,expected", [
        (CardType.BUILD, 2),
        (CardType.ATTACK, 5),
        (CardType.DEFENSE, 8),
        (CardType.HEAVY, 11),
        (CardType.COUNTER, 14),
        (CardType.ENERGY, 17),
        (CardType.FLEX, 20),
        (CardType.IRON_DEFENSE, 23),
    ])
    def test_dismantle_actions(self, dismantle_target, expected):
        """Build + dismantle_target map to dismantle indices. Design/37."""
        assert Converter.action_to_policy_index(CardType.BUILD, dismantle_target=dismantle_target) == expected

    def test_string_input_raises_valueerror(self):
        """String input 'attack' raises ValueError (keys are CardType)."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index("attack")  # type: ignore[arg-type]

    def test_non_build_with_build_target_raises(self):
        """ATTACK with build_target=DEFENSE is illegal."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index(CardType.ATTACK, CardType.DEFENSE)

    def test_non_build_with_dismantle_target_raises(self):
        """ATTACK with dismantle_target=DEFENSE is illegal."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index(CardType.ATTACK, dismantle_target=CardType.DEFENSE)

    def test_build_without_target_raises(self):
        """BUILD with build_target=None and dismantle_target=None is illegal."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index(CardType.BUILD)

    def test_build_with_both_targets_raises(self):
        """BUILD with both build_target and dismantle_target raises."""
        with pytest.raises(ValueError):
            Converter.action_to_policy_index(CardType.BUILD, CardType.ATTACK, CardType.DEFENSE)


class TestPolicyIndexToAction:
    """TC-PI: policy_index_to_action (design/37: 24-d, 3-tuple return, pass=(None,None,None))."""

    def test_pass_index(self):
        """Index 0 decodes to (None, None, None) = pass."""
        assert Converter.policy_index_to_action(0) == (None, None, None)

    @pytest.mark.parametrize("index,card_type", [
        (3, CardType.ATTACK),
        (6, CardType.DEFENSE),
        (9, CardType.HEAVY),
        (12, CardType.COUNTER),
        (15, CardType.ENERGY),
        (18, CardType.FLEX),
        (21, CardType.IRON_DEFENSE),
    ])
    def test_play_indices(self, index, card_type):
        """Play indices decode to (CardType, None, None). Design/37 triplet layout."""
        result = Converter.policy_index_to_action(index)
        assert result == (card_type, None, None)

    @pytest.mark.parametrize("index,build_target", [
        (1, CardType.BUILD),
        (4, CardType.ATTACK),
        (7, CardType.DEFENSE),
        (10, CardType.HEAVY),
        (13, CardType.COUNTER),
        (16, CardType.ENERGY),
        (19, CardType.FLEX),
        (22, CardType.IRON_DEFENSE),
    ])
    def test_build_indices(self, index, build_target):
        """Build indices decode to (BUILD, target, None). Design/37."""
        result = Converter.policy_index_to_action(index)
        assert result == (CardType.BUILD, build_target, None)

    @pytest.mark.parametrize("index,dismantle_target", [
        (2, CardType.BUILD),
        (5, CardType.ATTACK),
        (8, CardType.DEFENSE),
        (11, CardType.HEAVY),
        (14, CardType.COUNTER),
        (17, CardType.ENERGY),
        (20, CardType.FLEX),
        (23, CardType.IRON_DEFENSE),
    ])
    def test_dismantle_indices(self, index, dismantle_target):
        """Dismantle indices decode to (BUILD, None, target). Design/37."""
        result = Converter.policy_index_to_action(index)
        assert result == (CardType.BUILD, None, dismantle_target)

    def test_negative_index_raises(self):
        """Index -1 raises IndexError."""
        with pytest.raises(IndexError):
            Converter.policy_index_to_action(-1)

    def test_index_24_raises(self):
        """Index 24 (beyond max) raises IndexError (design/37: 24-d range 0-23)."""
        with pytest.raises(IndexError):
            Converter.policy_index_to_action(24)

    def test_roundtrip_consistency(self):
        """action_to_policy_index → policy_index_to_action is identity (design/37: 3-tuple)."""
        test_cases = [
            # (card, build_target, dismantle_target)
            (None, None, None),   # pass
            (CardType.ATTACK, None, None),
            (CardType.DEFENSE, None, None),
            (CardType.HEAVY, None, None),
            (CardType.COUNTER, None, None),
            (CardType.ENERGY, None, None),
            (CardType.FLEX, None, None),
            (CardType.IRON_DEFENSE, None, None),
            (CardType.BUILD, CardType.BUILD, None),
            (CardType.BUILD, CardType.ATTACK, None),
            (CardType.BUILD, CardType.DEFENSE, None),
            (CardType.BUILD, CardType.HEAVY, None),
            (CardType.BUILD, CardType.COUNTER, None),
            (CardType.BUILD, CardType.ENERGY, None),
            (CardType.BUILD, CardType.FLEX, None),
            (CardType.BUILD, CardType.IRON_DEFENSE, None),
            (CardType.BUILD, None, CardType.BUILD),
            (CardType.BUILD, None, CardType.ATTACK),
            (CardType.BUILD, None, CardType.DEFENSE),
            (CardType.BUILD, None, CardType.HEAVY),
            (CardType.BUILD, None, CardType.COUNTER),
            (CardType.BUILD, None, CardType.ENERGY),
            (CardType.BUILD, None, CardType.FLEX),
            (CardType.BUILD, None, CardType.IRON_DEFENSE),
        ]
        for card_type, bt, dt in test_cases:
            idx = Converter.action_to_policy_index(card_type, bt, dt)
            result = Converter.policy_index_to_action(idx)
            assert result == (card_type, bt, dt), f"Roundtrip failed: ({card_type}, {bt}, {dt}) → idx={idx} → {result}"


class TestBuildCost:
    """Converter.build_cost() — design/37: includes IRON_DEFENSE."""

    @pytest.mark.parametrize("card_type,expected", [
        (CardType.ATTACK, 1),
        (CardType.DEFENSE, 1),
        (CardType.HEAVY, 2),
        (CardType.COUNTER, 2),
        (CardType.ENERGY, 1),
        (CardType.BUILD, 1),
        (CardType.FLEX, 2),
        (CardType.IRON_DEFENSE, 2),
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
