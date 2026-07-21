"""Tests for B2: BUILD energy validation ignores same-round ENERGY cards.

When a player selects BUILD + ENERGY in the same round, the validation
must account for energy provided by ENERGY cards during resolution.
"""

import pytest

from src.backend.game.card_type import CardType
from src.backend.game.game_core import GameCore


class TestB2BuildWithEnergySameRound:
    """B2: BUILD validation accounts for same-round ENERGY cards."""

    def test_build_heavy_with_energy_same_round_succeeds(self) -> None:
        """Player has energy=1, plays BUILD(HEAVY, cost=2) + ENERGY.

        ENERGY provides +1 energy minimum during resolution, so total
        available energy is 1 + 1 = 2 which covers the HEAVY cost of 2.
        Previously this would raise ValueError incorrectly.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 1
        gm.player_1.hand = [CardType.BUILD, CardType.ENERGY]
        gm.player_2.hand = [CardType.DEFENSE]

        # This should succeed — ENERGY card provides the missing energy
        result = gm.play_round(
            [CardType.BUILD, CardType.ENERGY],
            [CardType.DEFENSE],
            build_target_1=CardType.HEAVY,
        )

        # Verify the round completed and BUILD was processed
        assert result is not None
        # HEAVY was built into draw_pile
        assert CardType.HEAVY in gm.player_1.draw_pile

    def test_build_heavy_with_energy_still_insufficient(self) -> None:
        """Player has energy=0, plays BUILD(HEAVY, cost=2) + ENERGY.

        ENERGY provides +1 energy minimum, total = 1 < cost 2.
        Should still raise ValueError.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 0
        gm.player_1.hand = [CardType.BUILD, CardType.ENERGY]
        gm.player_2.hand = [CardType.DEFENSE]

        with pytest.raises(ValueError):
            gm.play_round(
                [CardType.BUILD, CardType.ENERGY],
                [CardType.DEFENSE],
                build_target_1=CardType.HEAVY,
            )

    def test_build_heavy_with_two_energy_same_round(self) -> None:
        """Player has energy=0, plays BUILD(HEAVY, cost=2) + ENERGY + ENERGY.

        Two ENERGY cards provide +2 energy minimum, total = 0 + 2 = 2 = cost.
        Should succeed.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 0
        gm.player_1.hand = [CardType.BUILD, CardType.ENERGY, CardType.ENERGY]
        gm.player_2.hand = [CardType.DEFENSE]

        result = gm.play_round(
            [CardType.BUILD, CardType.ENERGY, CardType.ENERGY],
            [CardType.DEFENSE],
            build_target_1=CardType.HEAVY,
        )

        assert result is not None
        assert CardType.HEAVY in gm.player_1.draw_pile

    def test_build_attack_cost_1_no_energy_needed(self) -> None:
        """Player has energy=0, plays BUILD(ATTACK, cost=1) + ENERGY.

        ENERGY provides +1, total = 1 = cost 1. Should succeed.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 0
        gm.player_1.hand = [CardType.BUILD, CardType.ENERGY]
        gm.player_2.hand = [CardType.DEFENSE]

        result = gm.play_round(
            [CardType.BUILD, CardType.ENERGY],
            [CardType.DEFENSE],
            build_target_1=CardType.ATTACK,
        )

        assert result is not None
        assert CardType.ATTACK in gm.player_1.draw_pile

    def test_player_two_build_with_energy_same_round(self) -> None:
        """Same fix works for Player 2: energy=1, BUILD(HEAVY, cost=2) + ENERGY."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.energy = 1
        gm.player_2.hand = [CardType.BUILD, CardType.ENERGY]

        result = gm.play_round(
            [CardType.DEFENSE],
            [CardType.BUILD, CardType.ENERGY],
            build_target_2=CardType.HEAVY,
        )

        assert result is not None
        assert CardType.HEAVY in gm.player_2.draw_pile
