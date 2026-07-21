"""Tests for B1: Cycle-end bleed causes game state corruption.

When cycle-end bleed (round 3 HP loss) kills a player, the code must NOT
advance cycle_number, reset round_in_cycle, call _start_cycle(), or
advance round_number. This test verifies that game state remains consistent
when a player dies from cycle-end bleed damage.
"""

import pytest

from src.backend.game.card_type import CardType
from src.backend.game.game_core import GameCore


class TestB1BleedGameOverNoStateAdvance:
    """B1: Verify no state advancement when bleed causes game over."""

    def test_bleed_kills_both_players_no_counter_advance(self) -> None:
        """When both players die from cycle-end bleed, counters stay frozen.

        Pre-condition: Both players at 1 HP, round 3 (cycle end).
        After bleed: both die (0 HP) → game over (draw).
        Expected:
          - cycle_number stays 1 (NOT advanced to 2)
          - round_number stays 3 (NOT advanced to 4)
          - round_in_cycle stays 3 (NOT reset to 0 then advanced to 1)
          - _start_cycle() is NOT called → hands empty after discard
          - Game is properly over with winner="draw"
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        # Set up round 3, end of cycle 1
        gm.round_in_cycle = 3
        gm.cycle_number = 1
        gm.round_number = 3
        # Give each player a DEFENSE card for the round
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.DEFENSE]
        gm.player_1.hp = 1
        gm.player_2.hp = 1

        result = gm.play_round([CardType.DEFENSE], [CardType.DEFENSE])

        # Game is over — bleed killed both players
        assert gm.is_over is True
        assert gm.winner == "draw"
        assert gm.player_1.hp == 0
        assert gm.player_2.hp == 0

        # CRITICAL: cycle_number must NOT have advanced.
        # Bug behaviour: cycle_number would be 2.
        assert gm.cycle_number == 1, (
            f"B1 FAIL: Expected cycle_number=1 (not advanced), got {gm.cycle_number}"
        )

        # CRITICAL: round_number must NOT have advanced.
        # Bug behaviour: round_number would be 4.
        assert gm.round_number == 3, (
            f"B1 FAIL: Expected round_number=3 (not advanced), got {gm.round_number}"
        )

        # CRITICAL: round_in_cycle must NOT have changed.
        # Bug behaviour: round_in_cycle would be 1 (reset to 0 then +1).
        assert gm.round_in_cycle == 3, (
            f"B1 FAIL: Expected round_in_cycle=3 (not changed), got {gm.round_in_cycle}"
        )

        # CRITICAL: _start_cycle() was NOT called — hands not reclaimed.
        # After playing DEFENSE, hand should be empty (card was discarded).
        # If _start_cycle() was wrongly called, hand would have 5 fresh cards.
        assert len(gm.player_1.hand) == 0, (
            f"B1 FAIL: Expected empty hand (DEFENSE discarded, no _start_cycle), "
            f"got {len(gm.player_1.hand)} cards"
        )
        assert len(gm.player_2.hand) == 0, (
            f"B1 FAIL: Expected empty hand (DEFENSE discarded, no _start_cycle), "
            f"got {len(gm.player_2.hand)} cards"
        )

        # Verify discard_pile has the played DEFENSE (not reclaimed by _start_cycle)
        assert CardType.DEFENSE in gm.player_1.discard_pile, (
            "B1 FAIL: P1 DEFENSE not in discard_pile — may have been reclaimed by _start_cycle"
        )
        assert CardType.DEFENSE in gm.player_2.discard_pile, (
            "B1 FAIL: P2 DEFENSE not in discard_pile — may have been reclaimed by _start_cycle"
        )

        # Verify RoundResult captures the correct final state
        assert result.after_end_actions.is_game_over is True
        assert result.after_end_actions.winner == "draw"

    def test_bleed_kills_one_player_no_counter_advance(self) -> None:
        """When only one player dies from cycle-end bleed, counters stay frozen.

        Pre-condition: P1 at 2 HP, P2 at 1 HP, round 3.
        After DEFENSE plays (no damage): P1 stays 2 HP, P2 stays 1 HP.
        After bleed (1 damage each): P1 → 1 HP, P2 → 0 HP → P1 wins.
        Expected: counters NOT advanced, hands NOT refreshed.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.round_in_cycle = 3
        gm.cycle_number = 1
        gm.round_number = 3
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.DEFENSE]
        gm.player_1.hp = 2
        gm.player_2.hp = 1

        result = gm.play_round([CardType.DEFENSE], [CardType.DEFENSE])

        # Game over: P2 died from bleed, P1 wins
        assert gm.is_over is True
        assert gm.winner == "1"
        assert gm.player_1.hp == 1  # 2 - 1 = 1
        assert gm.player_2.hp == 0  # 1 - 1 = 0

        # Counters must NOT advance
        assert gm.cycle_number == 1
        assert gm.round_number == 3
        assert gm.round_in_cycle == 3

        # Hands must be empty (cards discarded, no _start_cycle)
        assert len(gm.player_1.hand) == 0
        assert len(gm.player_2.hand) == 0

        # Verify RoundResult
        assert result.after_end_actions.is_game_over is True
        assert result.after_end_actions.winner == "1"
