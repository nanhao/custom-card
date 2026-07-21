"""B6: AfterEndActions missing energy/hand_cards on game-over (damage) path.

Bug: When game ends during card resolution (damage kills a player), the
AfterEndActions on the game-over path doesn't include energy_1/2 and
hand_cards_1/2 fields. The normal path includes them, causing inconsistent
RoundResult structures.

Fix: Add energy_1/2 and hand_cards_1/2 to the AfterEndActions constructor
call in the game-over path.
"""

from src.backend.game.card_type import CardType
from src.backend.game.game_core import GameCore


class TestB6GameOverAfterEndActions:
    """B6: AfterEndActions must include energy and hand_cards on game-over path."""

    def test_game_over_damage_after_end_actions_has_energy(self):
        """When game ends during card resolution, energy fields are populated.

        Scenario: P2 has 1 HP, P1 attacks. P2 plays ENERGY (no defense).
        ATTACK deals 1 damage, P2 dies. Game over during resolution.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()

        gm.player_2.hp = 1
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.ENERGY]
        gm.player_1.energy = 3
        gm.player_2.energy = 3

        result = gm.play_round([CardType.ATTACK], [CardType.ENERGY])

        # Game should be over (P2 killed by ATTACK damage)
        assert result.after_resolve.is_game_over, (
            f"Expected game over, hp_2={result.after_resolve.hp_2}"
        )
        assert result.after_end_actions.is_game_over

        # B6 CHECK: energy fields must NOT be default (0)
        assert result.after_end_actions.energy_1 != 0, (
            f"B6 FAIL: energy_1 is 0 (default). Expected actual energy value."
        )
        assert result.after_end_actions.energy_2 != 0, (
            f"B6 FAIL: energy_2 is 0 (default). Expected actual energy value."
        )

    def test_game_over_damage_after_end_actions_has_hand_cards(self):
        """When game ends during card resolution, hand_cards fields are populated.

        Scenario: P2 has 1 HP. P1 attacks + builds. P2 plays ENERGY only.
        ATTACK kills P2 (no defense to block). hand_cards should reflect
        remaining cards after played cards were discarded.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()

        gm.player_2.hp = 1
        gm.player_1.hand = [CardType.ATTACK, CardType.ENERGY, CardType.BUILD]
        gm.player_2.hand = [CardType.ENERGY, CardType.ATTACK]  # 2 cards, play 1
        gm.player_1.energy = 5
        gm.player_2.energy = 5

        result = gm.play_round(
            [CardType.ATTACK, CardType.BUILD],
            [CardType.ENERGY],  # P2 just ENERGY, no blocker
            build_target_1=CardType.ENERGY,
        )

        assert result.after_resolve.is_game_over

        # B6 CHECK: hand_cards must NOT be empty (default)
        assert len(result.after_end_actions.hand_cards_1) > 0, (
            f"B6 FAIL: hand_cards_1 is empty (default). Expected cards in hand."
        )
        assert len(result.after_end_actions.hand_cards_2) > 0, (
            f"B6 FAIL: hand_cards_2 is empty (default). Expected cards in hand."
        )
