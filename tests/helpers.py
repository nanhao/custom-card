"""Shared test helpers for unit and integration tests.

Provides factory functions for constructing GameState instances
and utility functions for reading test output files.
"""

import json

from src.backend.game.card_type import CardType
from src.backend.game.game_state import GameState, PlayerState
from src.backend.game.player import PlayerHistoryItem


def read_sample_file(path):
    """Read a sample output file as a JSON array.

    Validates that the file contains a valid JSON array before returning.
    """
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    assert isinstance(data, list), (
        f"Expected JSON array (list), got {type(data).__name__}"
    )
    return data


def make_game_state(hand_1=None, hand_2=None, hp_1=20, hp_2=20,
                    history_1=None, history_2=None, round_num=1,
                    energy_1=0, energy_2=0, statuses_1=None, statuses_2=None,
                    cycle=1, round_in_cycle=1):
    """Build a minimal GameState instance for testing.

    Per design/19: returns a typed GameState, not a dict.
    Hand entries are CardType enum members.

    Args:
        hand_1, hand_2: Player hands (list of CardType). Default 5x ATTACK.
        hp_1, hp_2: Player HP. Default 20.
        history_1, history_2: Played card histories as list[PlayerHistoryItem].
                              Default empty.
        round_num: Global round number.
        energy_1, energy_2: Current energy. Default 0.
        statuses_1, statuses_2: Current statuses (list of PlayerStatus).
        cycle: Current cycle number.
        round_in_cycle: Position within cycle.

    Returns:
        GameState instance matching GameCore.get_state() output format.
    """
    if hand_1 is None:
        hand_1 = [CardType.ATTACK] * 5
    if hand_2 is None:
        hand_2 = [CardType.ATTACK] * 5
    if history_1 is None:
        history_1 = []
    if history_2 is None:
        history_2 = []
    if statuses_1 is None:
        statuses_1 = []
    if statuses_2 is None:
        statuses_2 = []

    return GameState(
        round=round_num,
        cycle=cycle,
        round_in_cycle=round_in_cycle,
        players={
            "1": PlayerState(
                hp=hp_1, energy=energy_1,
                statuses=list(statuses_1),
                hand=list(hand_1), history=list(history_1),
            ),
            "2": PlayerState(
                hp=hp_2, energy=energy_2,
                statuses=list(statuses_2),
                hand=list(hand_2), history=list(history_2),
            ),
        },
        is_over=False,
        winner=None,
        mode="pvp",
        round_log=[],
    )
