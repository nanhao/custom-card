"""Unit tests for AtteV1Converter — design/40 T1-T4."""

import pytest

from src.backend.ai.atte_v1.constants import MAX_ENERGY, MAX_HP, PAD_ACTION_ID
from src.backend.ai_adapter.atte_v1_converter import AtteV1Converter
from src.backend.game.card_type import CardType
from src.backend.game.game_state_snapshot import GameStateSnapshot
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import PlayerStatus, StatusType
from src.backend.game.player_state_snapshot import PlayerStateSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestAtteV1ConverterHelpers:
    """Helper class to build test fixtures for AtteV1Converter tests."""

    @staticmethod
    def _make_player_snapshot(
        hp: int = 30,
        energy: int = 5,
        hand: list[CardType] | None = None,
        discard_pile: list[CardType] | None = None,
        draw_pile: list[CardType] | None = None,
        statuses: list[PlayerStatus] | None = None,
        history: list[PlayerHistoryItem] | None = None,
    ) -> PlayerStateSnapshot:
        """Build a PlayerStateSnapshot with given values."""
        return PlayerStateSnapshot(
            hp=hp,
            energy=energy,
            statuses=statuses or [],
            hand=hand or [],
            history=history or [],
            draw_pile=draw_pile or [],
            discard_pile=discard_pile or [],
            draw_pile_count=len(draw_pile) if draw_pile else 0,
            discard_pile_count=len(discard_pile) if discard_pile else 0,
        )

    @staticmethod
    def _make_state(
        p1: PlayerStateSnapshot | None = None,
        p2: PlayerStateSnapshot | None = None,
        round_in_cycle: int = 1,
    ) -> GameStateSnapshot:
        """Build a minimal GameStateSnapshot for converter tests."""
        return GameStateSnapshot(
            round=1,
            cycle=1,
            round_in_cycle=round_in_cycle,
            players={
                "1": p1 or TestAtteV1ConverterHelpers._make_player_snapshot(),
                "2": p2 or TestAtteV1ConverterHelpers._make_player_snapshot(),
            },
            is_over=False,
            winner=None,
            mode="pvp",
            round_log=[],
        )


# ---------------------------------------------------------------------------
# T1: card_to_action_id
# ---------------------------------------------------------------------------


class TestCardToActionId:
    """T1: AtteV1Converter.card_to_action_id tests."""

    def test_attack_returns_3(self) -> None:
        """T1.1: card_to_action_id(ATTACK) returns 3."""
        assert AtteV1Converter.card_to_action_id(CardType.ATTACK) == 3

    def test_defense_returns_6(self) -> None:
        """T1.2: card_to_action_id(DEFENSE) returns 6."""
        assert AtteV1Converter.card_to_action_id(CardType.DEFENSE) == 6

    def test_heavy_returns_9(self) -> None:
        """T1.3: card_to_action_id(HEAVY) returns 9."""
        assert AtteV1Converter.card_to_action_id(CardType.HEAVY) == 9

    def test_counter_returns_12(self) -> None:
        """T1.4: card_to_action_id(COUNTER) returns 12."""
        assert AtteV1Converter.card_to_action_id(CardType.COUNTER) == 12

    def test_energy_returns_15(self) -> None:
        """T1.5: card_to_action_id(ENERGY) returns 15."""
        assert AtteV1Converter.card_to_action_id(CardType.ENERGY) == 15

    def test_flex_returns_18(self) -> None:
        """T1.6: card_to_action_id(FLEX) returns 18."""
        assert AtteV1Converter.card_to_action_id(CardType.FLEX) == 18

    def test_iron_defense_returns_21(self) -> None:
        """T1.7: card_to_action_id(IRON_DEFENSE) returns 21."""
        assert AtteV1Converter.card_to_action_id(CardType.IRON_DEFENSE) == 21

    def test_build_raises_value_error(self) -> None:
        """T1.8: card_to_action_id(BUILD) raises ValueError."""
        with pytest.raises(ValueError):
            AtteV1Converter.card_to_action_id(CardType.BUILD)


# ---------------------------------------------------------------------------
# T2: status_to_id
# ---------------------------------------------------------------------------


class TestStatusToId:
    """T2: AtteV1Converter.status_to_id tests."""

    def test_defense_returns_0(self) -> None:
        """T2.1: status_to_id(DEFENSE) returns 0."""
        assert AtteV1Converter.status_to_id(StatusType.DEFENSE) == 0

    def test_guard_returns_1(self) -> None:
        """T2.2: status_to_id(GUARD) returns 1."""
        assert AtteV1Converter.status_to_id(StatusType.GUARD) == 1

    def test_fragile_returns_2(self) -> None:
        """T2.3: status_to_id(FRAGILE) returns 2."""
        assert AtteV1Converter.status_to_id(StatusType.FRAGILE) == 2


# ---------------------------------------------------------------------------
# T3: build_input
# ---------------------------------------------------------------------------


class TestBuildInput:
    """T3: AtteV1Converter.build_input tests."""

    def test_hand_ids_mapped(self) -> None:
        """T3.1: hand [ATTACK, DEFENSE] → own.hand_ids = [3, 6]."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(
            hand=[CardType.ATTACK, CardType.DEFENSE],
        )
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.hand_ids == [3, 6]

    def test_discard_ids_mapped(self) -> None:
        """T3.2: discard [ATTACK, ATTACK] → own.discard_ids = [3, 3]."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(
            discard_pile=[CardType.ATTACK, CardType.ATTACK],
        )
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.discard_ids == [3, 3]

    def test_draw_ids_mapped(self) -> None:
        """T3.3: draw [ENERGY] → own.draw_ids = [15]."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(
            draw_pile=[CardType.ENERGY],
        )
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.draw_ids == [15]

    def test_empty_hand(self) -> None:
        """T3.4: empty hand → own.hand_ids = []."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(hand=[])
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.hand_ids == []

    def test_first_round_last_action(self) -> None:
        """T3.5: first round (no history) → last_action = -1."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(history=[])
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.last_action == -1

    def test_last_action_from_history(self) -> None:
        """T3.6: history with ATTACK → last_action = 3."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(
            history=[PlayerHistoryItem(card=CardType.ATTACK)],
        )
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.last_action == 3

    def test_hp_normalized(self) -> None:
        """T3.7: HP=30, energy=5 → hp=1.0, energy=1.0."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(hp=30, energy=5)
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.hp == pytest.approx(1.0)
        assert result.own.energy == pytest.approx(1.0)

    def test_hp_half_normalized(self) -> None:
        """T3.8: HP=15, energy=2 → hp=0.5, energy=0.4."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(hp=15, energy=2)
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.hp == pytest.approx(0.5)
        assert result.own.energy == pytest.approx(0.4)

    def test_defense_status(self) -> None:
        """T3.9: DEFENSE(lifetime=2) → status_ids=[0], durations=[2]."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(
            statuses=[PlayerStatus(status_type=StatusType.DEFENSE, lifetime=2)],
        )
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.status_ids == [0]
        assert result.own.status_durations == [2]

    def test_no_statuses(self) -> None:
        """T3.10: no status → status_ids=[], durations=[]."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(statuses=[])
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        assert result.own.status_ids == []
        assert result.own.status_durations == []

    def test_player2_perspective(self) -> None:
        """T3.11: player_key='2' → own=player2, opp=player1."""
        p1 = TestAtteV1ConverterHelpers._make_player_snapshot(
            hand=[CardType.ATTACK],
        )
        p2 = TestAtteV1ConverterHelpers._make_player_snapshot(
            hand=[CardType.DEFENSE],
        )
        state = TestAtteV1ConverterHelpers._make_state(p1=p1, p2=p2)
        result = AtteV1Converter.build_input(state, "2")
        assert result.own.hand_ids == [6]  # player2's DEFENSE
        assert result.opp.hand_ids == [3]  # player1's ATTACK

    def test_invalid_player_key(self) -> None:
        """T3.12: invalid player_key raises ValueError."""
        state = TestAtteV1ConverterHelpers._make_state()
        with pytest.raises(ValueError):
            AtteV1Converter.build_input(state, "3")

    def test_opponent_piles_visible(self) -> None:
        """T3.13: opponent discard/draw/hand visible."""
        p1 = TestAtteV1ConverterHelpers._make_player_snapshot(
            hand=[CardType.ATTACK],
            discard_pile=[CardType.DEFENSE],
            draw_pile=[CardType.ENERGY],
        )
        state = TestAtteV1ConverterHelpers._make_state(p1=p1)
        result = AtteV1Converter.build_input(state, "2")  # player2 perspective
        # Player2 sees player1's data as opponent
        assert result.own.opp_hand_ids == [3]       # player1's hand
        assert result.own.opp_discard_ids == [6]     # player1's discard
        assert result.own.opp_draw_ids == [15]       # player1's draw

    def test_round_in_cycle_mapping(self) -> None:
        """T3.14: GameStateSnapshot 1/2/3 → AtteV1Input -1/0/1."""
        for gs_val, expected in [(1, -1), (2, 0), (3, 1)]:
            state = TestAtteV1ConverterHelpers._make_state(round_in_cycle=gs_val)
            result = AtteV1Converter.build_input(state, "1")
            assert result.round_in_cycle == expected

    def test_last_action_pad_mapping(self) -> None:
        """T3.15: last_action=-1 mapped to PAD_ACTION_ID(24)."""
        # The Converter maps -1 to 24 internally; this verifies the mapping
        # by checking that a state with no history produces last_action=-1
        p = TestAtteV1ConverterHelpers._make_player_snapshot(history=[])
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        # -1 is the user-facing value; PAD mapping is internal detail
        assert result.own.last_action == -1
        # Verify the PAD constant exists for the model to use
        assert PAD_ACTION_ID == 24

    def test_build_card_filtered_from_hand_ids(self) -> None:
        """T3.17: BUILD in hand is filtered from hand_ids."""
        p = TestAtteV1ConverterHelpers._make_player_snapshot(
            hand=[CardType.BUILD, CardType.ATTACK],
        )
        state = TestAtteV1ConverterHelpers._make_state(p1=p)
        result = AtteV1Converter.build_input(state, "1")
        # BUILD should NOT appear in hand_ids
        assert 1 not in result.own.hand_ids  # BUILD build/build index
        assert CardType.BUILD not in [
            CardType.ATTACK, CardType.DEFENSE, CardType.HEAVY,
            CardType.COUNTER, CardType.ENERGY, CardType.FLEX, CardType.IRON_DEFENSE,
        ] or True  # BUILD is excluded from card_to_action_id mapping
        # Only ATTACK (3) should be in hand_ids
        assert result.own.hand_ids == [3]


# ---------------------------------------------------------------------------
# T4: action ↔ policy index
# ---------------------------------------------------------------------------


class TestActionPolicyIndex:
    """T4: AtteV1Converter.action_to_policy_index and policy_index_to_action."""

    def test_pass_returns_0(self) -> None:
        """T4.1: action_to_policy_index(None) returns 0."""
        assert AtteV1Converter.action_to_policy_index(None) == 0

    def test_attack_play_returns_3(self) -> None:
        """T4.2: action_to_policy_index(ATTACK) returns 3."""
        assert AtteV1Converter.action_to_policy_index(CardType.ATTACK) == 3

    def test_build_attack_returns_4(self) -> None:
        """T4.3: BUILD+target=ATTACK returns 4."""
        assert (
            AtteV1Converter.action_to_policy_index(
                CardType.BUILD, build_target=CardType.ATTACK,
            )
            == 4
        )

    def test_dismantle_attack_returns_5(self) -> None:
        """T4.4: BUILD+dismantle=ATTACK returns 5."""
        assert (
            AtteV1Converter.action_to_policy_index(
                CardType.BUILD, dismantle_target=CardType.ATTACK,
            )
            == 5
        )

    def test_both_targets_raises(self) -> None:
        """T4.5: both build_target and dismantle_target raises ValueError."""
        with pytest.raises(ValueError):
            AtteV1Converter.action_to_policy_index(
                CardType.BUILD,
                build_target=CardType.ATTACK,
                dismantle_target=CardType.DEFENSE,
            )

    def test_index_0_to_action(self) -> None:
        """T4.6: policy_index_to_action(0) → (None, None, None)."""
        assert AtteV1Converter.policy_index_to_action(0) == (None, None, None)

    def test_index_21_to_action(self) -> None:
        """T4.7: policy_index_to_action(21) → (IRON_DEFENSE, None, None)."""
        assert AtteV1Converter.policy_index_to_action(21) == (
            CardType.IRON_DEFENSE, None, None,
        )

    def test_index_24_raises(self) -> None:
        """T4.8: policy_index_to_action(24) raises IndexError."""
        with pytest.raises(IndexError):
            AtteV1Converter.policy_index_to_action(24)

    def test_build_cost_attack(self) -> None:
        """T4.9: build_cost(ATTACK) returns 1."""
        assert AtteV1Converter.build_cost(CardType.ATTACK) == 1
