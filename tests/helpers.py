"""Shared test helpers for unit and integration tests.

Provides factory functions for constructing realistic game state dicts
used across multiple test modules.
"""


def make_game_state(hand_1=None, hand_2=None, hp_1=3, hp_2=3,
                    history_1=None, history_2=None, round_num=1):
    """Build a minimal game state dict matching GameCore.get_state() format.

    All parameters are optional with sensible defaults for a fresh game.
    The AI is always Player "2" (index "2" in the players dict).

    Args:
        hand_1: Player 1's hand (list of card strings). Defaults to standard 7-card hand.
        hand_2: Player 2's hand (list of card strings). Defaults to standard 7-card hand.
        hp_1: Player 1 HP. Defaults to 3.
        hp_2: Player 2 HP. Defaults to 3.
        history_1: Player 1's played card history. Defaults to empty list.
        history_2: Player 2's played card history. Defaults to empty list.
        round_num: Current round number. Defaults to 1.

    Returns:
        dict matching GameCore.get_state() output format.
    """
    _default_hand = ["attack", "attack", "attack", "defense", "defense", "heavy", "counter"]
    if hand_1 is None:
        hand_1 = list(_default_hand)
    if hand_2 is None:
        hand_2 = list(_default_hand)
    if history_1 is None:
        history_1 = []
    if history_2 is None:
        history_2 = []

    return {
        "round": round_num,
        "players": {
            "1": {"hp": hp_1, "hand": list(hand_1), "history": list(history_1)},
            "2": {"hp": hp_2, "hand": list(hand_2), "history": list(history_2)},
        },
        "is_over": False,
        "winner": None,
        "mode": "pve",
        "round_log": [],
    }
