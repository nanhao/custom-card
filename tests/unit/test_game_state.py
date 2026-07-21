"""Unit tests for src/backend/game/game_state.py — PlayerStateSnapshot / GameStateSnapshot dataclass.

Test cases per design/34 test_cases.md §3 (TC-GS-01~04).
"""

import pytest

from src.backend.game.card_type import CardType
from src.backend.game.game_core import RoundResult, BeforeResolve, AfterResolve, EndActions, AfterEndActions
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus


# Import after game_state.py is created — test will fail until then.
# We write the import now so the test runner can reason about the dependency.
from src.backend.game.game_state_snapshot import GameStateSnapshot
from src.backend.game.player_state_snapshot import PlayerStateSnapshot  # noqa: E402


class TestPlayerStateSnapshot:
    """Tests for PlayerStateSnapshot dataclass construction and field types."""

    def test_basic_construction(self):
        """TC-PS-01: Basic construction with empty fields (design/34: piles default to [])."""
        ps = PlayerStateSnapshot(hp=20, energy=3, statuses=[], hand=[], history=[])
        assert ps.hp == 20
        assert ps.energy == 3
        assert ps.statuses == []
        assert ps.hand == []
        assert ps.history == []
        # design/34: piles always list, not None
        assert ps.draw_pile == []
        assert ps.discard_pile == []
        assert ps.draw_pile_count == 0
        assert ps.discard_pile_count == 0

    def test_full_construction(self):
        """TC-PS-02: Construction with cards, statuses, history, and piles."""
        ps = PlayerStateSnapshot(
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
        ps = PlayerStateSnapshot(hp=20, energy=0, statuses=[], hand=[CardType.ATTACK], history=[])
        assert isinstance(ps.hand[0], CardType)
        assert not isinstance(ps.hand[0], str)

    def test_statuses_elements_are_playerstatus_not_dict(self):
        """TC-PS-04: Statuses elements are PlayerStatus instances, not dicts."""
        ps = PlayerStateSnapshot(
            hp=20, energy=0,
            statuses=[PlayerStatus(StatusType.DEFENSE, 1)],
            hand=[], history=[],
        )
        assert isinstance(ps.statuses[0], PlayerStatus)
        assert not isinstance(ps.statuses[0], dict)

    def test_default_piles_empty_list(self):
        """TC-PS-05: Default draw_pile / discard_pile are empty lists (design/34: never None)."""
        ps = PlayerStateSnapshot(hp=20, energy=0, statuses=[], hand=[], history=[])
        assert ps.draw_pile == []
        assert ps.discard_pile == []
        assert isinstance(ps.draw_pile, list)
        assert isinstance(ps.discard_pile, list)
        assert ps.draw_pile_count == 0
        assert ps.discard_pile_count == 0


class TestGameState:
    """Tests for GameStateSnapshot dataclass construction."""

    def test_basic_construction(self):
        """TC-GS-01: Basic construction with two PlayerStateSnapshots."""
        ps1 = PlayerStateSnapshot(hp=20, energy=0, statuses=[], hand=[], history=[])
        ps2 = PlayerStateSnapshot(hp=20, energy=0, statuses=[], hand=[], history=[])
        gs = GameStateSnapshot(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        assert gs.round == 1
        assert gs.cycle == 1
        assert gs.round_in_cycle == 1
        assert isinstance(gs.players["1"], PlayerStateSnapshot)
        assert isinstance(gs.players["2"], PlayerStateSnapshot)
        assert gs.is_over is False
        assert gs.winner is None
        assert gs.mode == "pvp"
        assert gs.round_log == []

    def test_game_over_state(self):
        """TC-GS-02: Game over with winner."""
        ps1 = PlayerStateSnapshot(hp=0, energy=0, statuses=[], hand=[], history=[])
        ps2 = PlayerStateSnapshot(hp=20, energy=0, statuses=[], hand=[], history=[])
        gs = GameStateSnapshot(
            round=10, cycle=4, round_in_cycle=3,
            players={"1": ps1, "2": ps2},
            is_over=True, winner="1", mode="pvp", round_log=[],
        )
        assert gs.is_over is True
        assert gs.winner == "1"

    def test_draw_state(self):
        """TC-GS-03: Draw state."""
        ps1 = PlayerStateSnapshot(hp=0, energy=0, statuses=[], hand=[], history=[])
        ps2 = PlayerStateSnapshot(hp=0, energy=0, statuses=[], hand=[], history=[])
        gs = GameStateSnapshot(
            round=3, cycle=1, round_in_cycle=3,
            players={"1": ps1, "2": ps2},
            is_over=True, winner="draw", mode="pvp", round_log=[],
        )
        assert gs.winner == "draw"


class TestGameStateToDict:
    """Tests for GameStateSnapshot.to_dict() serialization."""

    @staticmethod
    def _make_empty_player(hp=20):
        """Helper: create an empty PlayerStateSnapshot for testing."""
        return PlayerStateSnapshot(hp=hp, energy=0, statuses=[], hand=[], history=[])

    def test_empty_state_serialization(self):
        """TC-TD-01: Serialize an empty game state."""
        ps1 = self._make_empty_player(20)
        ps2 = self._make_empty_player(20)
        gs = GameStateSnapshot(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = gs.to_dict()
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
        ps1 = PlayerStateSnapshot(
            hp=20, energy=0, statuses=[], history=[],
            hand=[CardType.ATTACK, CardType.DEFENSE],
            draw_pile=[CardType.BUILD],
        )
        ps2 = self._make_empty_player(20)
        gs = GameStateSnapshot(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = gs.to_dict()
        assert result["players"]["1"]["hand"] == ["attack", "defense"]
        assert result["players"]["1"]["draw_pile"] == ["build"]
        assert all(isinstance(s, str) for s in result["players"]["1"]["hand"])

    def test_statuses_serialized_as_dicts(self):
        """TC-TD-03: PlayerStatus serialized via to_dict()."""
        ps1 = PlayerStateSnapshot(
            hp=20, energy=0, hand=[], history=[],
            statuses=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        ps2 = self._make_empty_player(20)
        gs = GameStateSnapshot(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = gs.to_dict()
        assert result["players"]["1"]["statuses"] == [
            {"type": "defense", "lifetime": 1}
        ]

    def test_history_serialized_as_dicts(self):
        """TC-TD-04: PlayerHistoryItem serialized as dicts."""
        ps1 = PlayerStateSnapshot(
            hp=20, energy=0, hand=[], statuses=[],
            history=[
                PlayerHistoryItem(CardType.ATTACK, None),
                PlayerHistoryItem(CardType.BUILD, CardType.COUNTER),
            ],
        )
        ps2 = self._make_empty_player(20)
        gs = GameStateSnapshot(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = gs.to_dict()
        assert result["players"]["1"]["history"] == [
            {"card": "attack", "build_target": None, "dismantle_target": None, "had_defense_after": False},
            {"card": "build", "build_target": "counter", "dismantle_target": None, "had_defense_after": False},
        ]

    def test_draw_pile_always_serialized(self):
        """TC-GS-03: When draw_pile is empty list, it is serialized (design/34: always included)."""
        ps1 = PlayerStateSnapshot(
            hp=20, energy=0, statuses=[], hand=[], history=[],
            draw_pile=[], draw_pile_count=0,
        )
        ps2 = self._make_empty_player(20)
        gs = GameStateSnapshot(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result = gs.to_dict()
        # design/34: draw_pile always serialized, even when empty
        assert "draw_pile" in result["players"]["1"]
        assert result["players"]["1"]["draw_pile"] == []
        assert result["players"]["1"]["draw_pile_count"] == 0
        # P2 should also have draw_pile/discard_pile
        assert "draw_pile" in result["players"]["2"]
        assert "discard_pile" in result["players"]["2"]

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
        gs = GameStateSnapshot(
            round=2, cycle=1, round_in_cycle=2,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[rr],
        )
        result = gs.to_dict()
        assert len(result["round_log"]) == 1
        log_entry = result["round_log"][0]
        assert log_entry["round"] == 1
        assert "before" in log_entry
        assert "after_resolve" in log_entry
        assert "end_actions" in log_entry
        assert "after_end_actions" in log_entry

    def test_to_dict_does_not_mutate_original(self):
        """TC-TD-07: to_dict() does not modify the original GameStateSnapshot."""
        ps1 = PlayerStateSnapshot(
            hp=20, energy=3, statuses=[], hand=[CardType.ATTACK], history=[],
        )
        ps2 = self._make_empty_player(20)
        gs = GameStateSnapshot(
            round=1, cycle=1, round_in_cycle=1,
            players={"1": ps1, "2": ps2},
            is_over=False, winner=None, mode="pvp", round_log=[],
        )
        result1 = gs.to_dict()
        result2 = gs.to_dict()
        assert result1 == result2
        # Original unchanged
        assert gs.players["1"].hp == 20
        assert isinstance(gs.players["1"].hand[0], CardType)
