"""Round result dataclasses for the custom card game v2 (design/11, design/21).

Extracted from game_core.py per design/21 review — keeps game_core.py under
the 500-line limit. These are pure data containers with no game logic.

Per architecture D9, this module is pure data — no UI, no I/O.
"""

from dataclasses import dataclass, field

from src.backend.game.card_type import CardType
from src.backend.game.player_state import PlayerStatus


@dataclass
class BeforeResolve:
    """State snapshot taken BEFORE card resolution.

    Captures HP, energy and statuses as they were at the start of the round,
    before any card effects are applied.
    """
    statuses_1: list[PlayerStatus]
    statuses_2: list[PlayerStatus]
    energy_1: int
    energy_2: int
    hp_1: int
    hp_2: int


@dataclass
class AfterResolve:
    """State immediately AFTER card resolution (Phase 1).

    Includes resolution details and game-over check triggered by
    HP reaching 0 during card resolution.
    """
    statuses_1: list[PlayerStatus]
    statuses_2: list[PlayerStatus]
    energy_1: int
    energy_2: int
    hp_1: int
    hp_2: int
    details: list[str]
    is_game_over: bool = False
    winner: str | None = None


@dataclass
class EndActions:
    """Round-end actions applied after card resolution (Phase 2).

    cycle_ended is True when the round is the last round of the cycle
    (round_in_cycle reaches ROUNDS_PER_CYCLE).
    bleed_damage values are 0 when cycle_ended is False.
    """
    cycle_ended: bool = False
    bleed_damage_1: int = 0
    bleed_damage_2: int = 0


@dataclass
class AfterEndActions:
    """Final state snapshot after all round-end processing (Phase 3).

    Includes round/counter advancement, deck reshuffling (if cycle ended),
    and a second game-over check for bleed damage.
    hand_cards_1/hand_cards_2 reflect the NEW hand after cycle processing.
    """
    is_game_over: bool = False
    winner: str | None = None
    round: int = 0
    cycle: int = 0
    round_in_cycle: int = 0
    hand_cards_1: list[CardType] = field(default_factory=list)
    hand_cards_2: list[CardType] = field(default_factory=list)
    statuses_1: list[PlayerStatus] = field(default_factory=list)
    statuses_2: list[PlayerStatus] = field(default_factory=list)
    energy_1: int = 0
    energy_2: int = 0
    hp_1: int = 0
    hp_2: int = 0


@dataclass
class RoundResult:
    """Value object returned by GameCore.play_round().

    Top-level: round identifiers and cards played.
    before: snapshot before card resolution.
    after_resolve: resolution outcome and immediate post-resolve state.
    end_actions: round-end actions (status tick, cycle bleed).
    after_end_actions: final state after all processing (next round start).
    """
    round: int
    cycle: int
    round_in_cycle: int
    cards_1: list[CardType]
    cards_2: list[CardType]
    before: BeforeResolve
    after_resolve: AfterResolve
    end_actions: EndActions
    after_end_actions: AfterEndActions

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(rr: "RoundResult") -> dict:
        """Serialize a RoundResult to a JSON-safe dict.

        Converts all CardType lists to strings via .value and
        PlayerStatus to dict via .to_dict().
        """
        return {
            "round": rr.round,
            "cycle": rr.cycle,
            "round_in_cycle": rr.round_in_cycle,
            "cards_1": [c.value for c in rr.cards_1],
            "cards_2": [c.value for c in rr.cards_2],
            "before": {
                "statuses_1": [ps.to_dict() for ps in rr.before.statuses_1],
                "statuses_2": [ps.to_dict() for ps in rr.before.statuses_2],
                "energy_1": rr.before.energy_1,
                "energy_2": rr.before.energy_2,
                "hp_1": rr.before.hp_1,
                "hp_2": rr.before.hp_2,
            },
            "after_resolve": {
                "statuses_1": [ps.to_dict() for ps in rr.after_resolve.statuses_1],
                "statuses_2": [ps.to_dict() for ps in rr.after_resolve.statuses_2],
                "energy_1": rr.after_resolve.energy_1,
                "energy_2": rr.after_resolve.energy_2,
                "hp_1": rr.after_resolve.hp_1,
                "hp_2": rr.after_resolve.hp_2,
                "details": rr.after_resolve.details,
                "is_game_over": rr.after_resolve.is_game_over,
                "winner": rr.after_resolve.winner,
            },
            "end_actions": {
                "cycle_ended": rr.end_actions.cycle_ended,
                "bleed_damage_1": rr.end_actions.bleed_damage_1,
                "bleed_damage_2": rr.end_actions.bleed_damage_2,
            },
            "after_end_actions": {
                "is_game_over": rr.after_end_actions.is_game_over,
                "winner": rr.after_end_actions.winner,
                "round": rr.after_end_actions.round,
                "cycle": rr.after_end_actions.cycle,
                "round_in_cycle": rr.after_end_actions.round_in_cycle,
                "hand_cards_1": [c.value for c in rr.after_end_actions.hand_cards_1],
                "hand_cards_2": [c.value for c in rr.after_end_actions.hand_cards_2],
                "statuses_1": [ps.to_dict() for ps in rr.after_end_actions.statuses_1],
                "statuses_2": [ps.to_dict() for ps in rr.after_end_actions.statuses_2],
                "energy_1": rr.after_end_actions.energy_1,
                "energy_2": rr.after_end_actions.energy_2,
                "hp_1": rr.after_end_actions.hp_1,
                "hp_2": rr.after_end_actions.hp_2,
            },
        }
