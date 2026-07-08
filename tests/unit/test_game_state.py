"""Unit tests for src/backend/game/game_state.py — PlayerState / GameState dataclass.

Test cases per design/19 test_cases.md §1-3 (TC-PS-01~05, TC-GS-01~03, TC-TD-01~07).
"""

import pytest

from src.backend.game.card import CardType
from src.backend.game.game_core import RoundResult, BeforeResolve, AfterResolve, EndActions, AfterEndActions
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus


# Import after game_state.py is created — test will fail until then.
# We write the import now so the test runner can reason about the dependency.
from src.backend.game.game_state import GameState, PlayerState  # noqa: E402


class TestPlayerState:
    """Tests for PlayerState dataclass construction and field types."""

    def test_basic_construction(self):
        """TC-PS-01: Basic construction with empty fields."""
        ps = PlayerState(hp=20, energy=3, statuses=[], hand=[], history=[])
        assert ps.hp == 20
        assert ps.energy == 3
        assert ps.statuses == []
        assert ps.hand == []
        assert ps.history == []
        assert ps.draw_pile is None
        assert ps.discard_pile is None
        assert ps.draw_pile_count == 0
        assert ps.discard_pile_count == 0

    def test_full_construction(self):
        """TC-PS-02: Construction with cards, statuses, history, and piles."""
        ps = PlayerState(
            hp=15,
            energy=2,
            statuses=[PlayerStatus(StatusType.DEFENSE, 1)],
            hand=[CardType.ATTACK, CardType.DEFENSE],
            history=[PlayerHistoryItem(CardType.ATTACK, None)],
            draw_pile=[CardType.BUILD],
            discard_pile=[CardType.ENERGY],
            draw_pile_count=5,
            discard_pile_count=1,
        )
        assert ps.hp == 15
        assert ps.energy == 2
        assert ps.statuses == [PlayerStatus(StatusType.DEFENSE, 1)]
        assert ps.hand == [CardType.ATTACK, CardType.DEFENSE]
        assert isinstance(ps.hand[0], CardType)
        assert isinstance(ps.hand[1], CardType)
        assert ps.history[0].card == CardType.ATTACK
        assert ps.history[0].build_target is None
        assert ps.draw_pile == [CardType.BUILD]
        assert ps.discard_pile == [CardType.ENERGY]
        assert ps.draw_pile_count == 5
        assert ps.discard_pile_count == 1

    def test_hand_elements_are_cardtype_not_str(self):
        """TC-PS-03: Hand elements are CardType instances, not strings."""
        ps = PlayerState(hp=20, energy=0, statuses=[], hand=[CardType.ATTACK], history=[])
        assert isinstance(ps.hand[0], CardType)
        assert not isinstance(ps.hand[0], str)

    def test_statuses_elements_are_playerstatus_not_dict(self):
        """TC-PS-04: Statuses elements are PlayerStatus instances, not dicts."""
        ps = PlayerState(
            hp=20, energy=0,
            statuses=[PlayerStatus(StatusType.DEFENSE, 1)],
            hand=[], history=[],
        )
        assert isinstance(ps.statuses[0], PlayerStatus)
        assert not isinstance(ps.statuses[0], dict)

    def test_default_piles_none(self):
        """TC-PS-05: Default draw_pile / discard_pile are None, counts are 0."""
        ps = PlayerState(hp=20, energy=0, statuses=[], hand=[], history=[])
        assert ps.draw_pile is None
        assert ps.discard_pile is None
        assert ps.draw_pile_count == 0
        assert ps.discard_pile_count == 0


class TestGameState:
    """Tests for GameState dataclass construction."""

    def test_basic_construction(self):
        """TC-GS-01: Basic construction with two PlayerStates."""
        ps1 = PlayerState(hp=20, energy=0, statuses=[], hand=[], history=[])
        ps2 = PlayerState(hp=20, energy=0, statuses=[], hand=[], history=[])
        gs = GameState(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        assert gs.round == 1
        assert gs.cycle == 1
        assert gs.round_in_cycle == 1
        assert isinstance(gs.players["1"], PlayerState)
        assert isinstance(gs.players["2"], PlayerState)
        assert gs.is_over is False
        assert gs.winner is None
        assert gs.mode == "pvp"
        assert gs.round_log == []

    def test_game_over_state(self):
        """TC-GS-02: Game over with winner."""
        ps1 = PlayerState(hp=0, energy=0, statuses=[], hand=[], history=[])
        ps2 = PlayerState(hp=20, energy=0, statuses=[], hand=[], history=[])
        gs = GameState(
            round=10, cycle=2, round_in_cycle=5,
            players={"1": ps1, "2": ps2},
            is_over=True, winner="1", mode="pvp", round_log=[],
        )
        assert gs.is_over is True
        assert gs.winner == "1"

    def test_draw_state(self):
        """TC-GS-03: Draw state."""
        ps1 = PlayerState(hp=0, energy=0, statuses=[], hand=[], history=[])
        ps2 = PlayerState(hp=0, energy=0, statuses=[], hand=[], history=[])
        gs = GameState(
            round=5, cycle=1, round_in_cycle=5,
            players={"1": ps1, "2": ps2},
            is_over=True, winner="draw", mode="pvp", round_log=[],
        )
        assert gs.winner == "draw"


class TestGameStateToDict:
    """Tests for GameState.to_dict() serialization."""

    @staticmethod
    def _make_empty_player(hp=20):
        """Helper: create an empty PlayerState for testing."""
        return PlayerState(hp=hp, energy=0, statuses=[], hand=[], history=[])

    def test_empty_state_serialization(self):
        """TC-TD-01: Serialize an empty game state."""
        ps1 = self._make_empty_player(20)
        ps2 = self._make_empty_player(20)
        gs = GameState(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = GameState.to_dict(gs)
        assert result["round"] == 1
        assert result["cycle"] == 1
        assert result["round_in_cycle"] == 1
        assert result["players"]["1"]["hp"] == 20
        assert result["players"]["1"]["energy"] == 0
        assert result["players"]["1"]["statuses"] == []
        assert result["players"]["1"]["hand"] == []
        assert result["players"]["1"]["history"] == []
        assert result["players"]["1"]["draw_pile_count"] == 0
        assert result["players"]["1"]["discard_pile_count"] == 0
        assert result["players"]["2"]["hp"] == 20
        assert result["is_over"] is False
        assert result["winner"] is None
        assert result["mode"] == "pvp"
        assert result["round_log"] == []

    def test_cards_serialized_as_strings(self):
        """TC-TD-02: Cards serialized as strings."""
        ps1 = PlayerState(
            hp=20, energy=0, statuses=[], history=[],
            hand=[CardType.ATTACK, CardType.DEFENSE],
            draw_pile=[CardType.BUILD],
        )
        ps2 = self._make_empty_player(20)
        gs = GameState(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = GameState.to_dict(gs)
        assert result["players"]["1"]["hand"] == ["attack", "defense"]
        assert result["players"]["1"]["draw_pile"] == ["build"]
        assert all(isinstance(s, str) for s in result["players"]["1"]["hand"])

    def test_statuses_serialized_as_dicts(self):
        """TC-TD-03: PlayerStatus serialized via to_dict()."""
        ps1 = PlayerState(
            hp=20, energy=0, hand=[], history=[],
            statuses=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        ps2 = self._make_empty_player(20)
        gs = GameState(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = GameState.to_dict(gs)
        assert result["players"]["1"]["statuses"] == [
            {"type": "defense", "lifetime": 1}
        ]

    def test_history_serialized_as_dicts(self):
        """TC-TD-04: PlayerHistoryItem serialized as dicts."""
        ps1 = PlayerState(
            hp=20, energy=0, hand=[], statuses=[],
            history=[
                PlayerHistoryItem(CardType.ATTACK, None),
                PlayerHistoryItem(CardType.BUILD, CardType.COUNTER),
            ],
        )
        ps2 = self._make_empty_player(20)
        gs = GameState(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = GameState.to_dict(gs)
        assert result["players"]["1"]["history"] == [
            {"card": "attack", "build_target": None},
            {"card": "build", "build_target": "counter"},
        ]

    def test_draw_pile_none_omitted(self):
        """TC-TD-05: When draw_pile is None, it is omitted from dict."""
        ps1 = PlayerState(
            hp=20, energy=0, statuses=[], hand=[], history=[],
            draw_pile=None, draw_pile_count=5,
        )
        ps2 = self._make_empty_player(20)
        gs = GameState(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = GameState.to_dict(gs)
        assert "draw_pile" not in result["players"]["1"]
        assert result["players"]["1"]["draw_pile_count"] == 5

    def test_round_log_serialization(self):
        """TC-TD-06: round_log contains serialized RoundResult."""
        # Build a minimal RoundResult
        before = BeforeResolve(
            statuses_1=[], statuses_2=[],
            energy_1=0, energy_2=0, hp_1=20, hp_2=20,
        )
        after_resolve = AfterResolve(
            statuses_1=[], statuses_2=[],
            energy_1=0, energy_2=0, hp_1=20, hp_2=20,
            details=["P1 takes 0 damage"], is_game_over=False, winner=None,
        )
        end_actions = EndActions(cycle_ended=False, bleed_damage_1=0, bleed_damage_2=0)
        after_end = AfterEndActions(
            is_game_over=False, winner=None,
            round=2, cycle=1, round_in_cycle=2,
            hand_cards_1=[], hand_cards_2=[],
            statuses_1=[], statuses_2=[],
            energy_1=0, energy_2=0, hp_1=20, hp_2=20,
        )
        rr = RoundResult(
            round=1, cycle=1, round_in_cycle=1,
            cards_1=[CardType.ATTACK], cards_2=[CardType.DEFENSE],
            before=before, after_resolve=after_resolve,
            end_actions=end_actions, after_end_actions=after_end,
        )
        ps1 = self._make_empty_player(20)
        ps2 = self._make_empty_player(20)
        gs = GameState(
            round=2, cycle=1, round_in_cycle=2,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[rr],
        )
        result = GameState.to_dict(gs)
        assert len(result["round_log"]) == 1
        log_entry = result["round_log"][0]
        assert log_entry["round"] == 1
        assert "before" in log_entry
        assert "after_resolve" in log_entry
        assert "end_actions" in log_entry
        assert "after_end_actions" in log_entry

    def test_to_dict_does_not_mutate_original(self):
        """TC-TD-07: to_dict() does not modify the original GameState."""
        ps1 = PlayerState(
            hp=20, energy=3, statuses=[], hand=[CardType.ATTACK], history=[],
        )
        ps2 = self._make_empty_player(20)
        gs = GameState(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result1 = GameState.to_dict(gs)
        result2 = GameState.to_dict(gs)
        assert result1 == result2
        # Original unchanged
        assert gs.players["1"].hp == 20
        assert isinstance(gs.players["1"].hand[0], CardType)
