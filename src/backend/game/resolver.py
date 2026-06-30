"""Stateless card resolution engine for the custom card game.

Per architecture D1, CardResolver is stateless — all methods are pure functions
with no side effects. Per architecture D7, Defense never deals damage.
Per architecture D8, when both cards imply damage to the same player, use max().
Per architecture D9, this module is pure logic — no UI, no I/O.
"""

from dataclasses import dataclass, field

from src.backend.game.card import CardType


@dataclass
class ResolutionResult:
    """Value object returned by CardResolver.resolve().

    Attributes:
        damage_1: Damage dealt to Player 1 this round.
        damage_2: Damage dealt to Player 2 this round.
        details: Human-readable resolution steps (for logging/debugging).
    """

    damage_1: int
    damage_2: int
    details: list[str] = field(default_factory=list)


class CardResolver:
    """Stateless card-vs-card resolution engine.

    All methods are static — no instantiation needed.
    Resolves any combination of two cards given each player's history.
    """

    @staticmethod
    def get_consecutive_attack(history: list[CardType]) -> int:
        """Count consecutive Attack cards from the end of history.

        Used to determine the damage multiplier for the Attack card.
        Traverses the history backwards; stops at the first non-Attack card.

        Per architecture D4: empty history returns 0 (round 1 edge case).
        """
        count = 0
        for card in reversed(history):
            if card == CardType.ATTACK:
                count += 1
            else:
                break
        return count

    @staticmethod
    def played_defense_last_round(history: list[CardType]) -> bool:
        """Check if the last card in history was Defense.

        Used by Heavy Strike and Defense resolution to determine
        whether the "consecutive Defense" condition is met.

        Per architecture D4: empty history returns False (round 1 edge case).
        """
        return len(history) > 0 and history[-1] == CardType.DEFENSE

    @staticmethod
    def resolve(
        cards_1: list[CardType],
        cards_2: list[CardType],
        history_1: list[CardType],
        history_2: list[CardType],
    ) -> ResolutionResult:
        """Compute damage for both players given their played cards and history.

        Accepts lists for future multi-card support (architecture D12).
        Current version expects exactly 1 card per list.

        Per architecture D3: both card effects are computed independently,
        then combined with max() per D8 to avoid double-counting.

        Per architecture D7: Defense never deals damage — any damage from
        a Heavy Strike hitting Defense comes from the HS card, not the Defense.

        Args:
            cards_1: Cards played by Player 1 this round.
            cards_2: Cards played by Player 2 this round.
            history_1: Cards played by 1 in previous rounds (chronological).
            history_2: Cards played by 2 in previous rounds (chronological).

        Returns:
            ResolutionResult with damage_1, damage_2, and resolution details.

        Raises:
            ValueError: If either cards list is not exactly 1 element
                       (current version limitation).
        """
        if len(cards_1) != 1 or len(cards_2) != 1:
            raise ValueError(
                f"Current version supports exactly 1 card per player per round. "
                f"Got {len(cards_1)} for 1, {len(cards_2)} for 2."
            )
        card_1 = cards_1[0]
        card_2 = cards_2[0]
        # ---- Step 1: Compute damage from each card independently ----
        d1_from_1, d2_from_1 = CardResolver._card_effect(
            card_1, card_2, history_1, history_2, is_player_1=True
        )
        # For player 2: _card_effect returns (dmg_to_2, dmg_to_1).
        # We need (dmg_to_1, dmg_to_2), so swap the unpacking.
        d2_from_2, d1_from_2 = CardResolver._card_effect(
            card_2, card_1, history_2, history_1, is_player_1=False
        )

        # ---- Step 2: Combine with max() (D8) ----
        damage_1 = max(d1_from_1, d1_from_2)
        damage_2 = max(d2_from_1, d2_from_2)

        # ---- Step 3: Build human-readable details ----
        details = []
        # Card 1's perspective
        details.append(CardResolver._describe_effect(
            card_1, card_2, history_1, history_2, "1", "2"
        ))
        # Card 2's perspective
        details.append(CardResolver._describe_effect(
            card_2, card_1, history_2, history_1, "2", "1"
        ))

        return ResolutionResult(
            damage_1=damage_1,
            damage_2=damage_2,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal helpers (public for testability per interface design)
    # ------------------------------------------------------------------

    @staticmethod
    def _card_effect(
        my_card: CardType,
        opp_card: CardType,
        my_history: list[CardType],
        opp_history: list[CardType],
        is_player_1: bool,
    ) -> tuple[int, int]:
        """Compute (damage_to_me, damage_to_opponent) for one card's effect.

        This is called twice per resolution — once for each player's card.

        Args:
            my_card: The card being evaluated.
            opp_card: The opponent's card.
            my_history: This player's history (for consecutive checks).
            opp_history: Opponent's history (for consecutive checks).
            is_player_1: Whether this is Player 1's card.

        Returns:
            (damage_to_self, damage_to_opponent) tuple.
        """
        dmg_self = 0
        dmg_opp = 0

        if my_card == CardType.ATTACK:
            # Attack: deals X damage where X = consecutive attack count + 1
            # Blocked by Defense or Counter
            if opp_card != CardType.DEFENSE and opp_card != CardType.COUNTER:
                consecutive = CardResolver.get_consecutive_attack(my_history) + 1
                dmg_opp = consecutive

        elif my_card == CardType.DEFENSE:
            # Defense is pure-block (D7): never deals damage.
            # When HS hits a non-consecutive Defense, the damage comes
            # from the HS card's effect, not from the Defense card.
            pass  # No damage assignment

        elif my_card == CardType.HEAVY_STRIKE:
            # Heavy Strike: deals 1 damage unless opponent played Defense
            # in BOTH the current round AND the previous round.
            if opp_card == CardType.DEFENSE and CardResolver.played_defense_last_round(opp_history):
                pass  # Blocked by consecutive Defense
            else:
                dmg_opp = 1

        elif my_card == CardType.COUNTER:
            # Counter:
            # - vs Attack: opponent takes 1, self takes 0
            # - vs Heavy Strike: both take 1
            # - otherwise: no effect
            if opp_card == CardType.ATTACK:
                dmg_opp = 1  # Opponent takes 1, self takes 0
            elif opp_card == CardType.HEAVY_STRIKE:
                dmg_opp = 1  # Opponent takes 1 (self damage comes from opponent's HS)
                # Note: self damage (dmg_self) from this interaction is set by
                # the opponent's HS effect when it's evaluated separately.
                # The max() in resolve() combines them correctly per D8.

        return dmg_self, dmg_opp

    @staticmethod
    def _describe_effect(
        my_card: CardType,
        opp_card: CardType,
        my_history: list[CardType],
        opp_history: list[CardType],
        my_name: str,
        opp_name: str,
    ) -> str:
        """Produce a human-readable description of one card's effect."""
        cn = {
            CardType.ATTACK: "Attack",
            CardType.DEFENSE: "Defense",
            CardType.HEAVY_STRIKE: "Heavy Strike",
            CardType.COUNTER: "Counter",
        }

        if my_card == CardType.ATTACK:
            if opp_card in (CardType.DEFENSE, CardType.COUNTER):
                return f"{my_name} Attack blocked by {opp_name}'s {cn[opp_card]} → {opp_name} takes 0"
            x = CardResolver.get_consecutive_attack(my_history) + 1
            return f"{my_name} Attack (×{x}) → {opp_name} takes {x}"

        if my_card == CardType.DEFENSE:
            if opp_card == CardType.HEAVY_STRIKE:
                if CardResolver.played_defense_last_round(my_history):
                    return f"{my_name} consecutive Defense blocks Heavy Strike → {my_name} takes 0"
                return f"{my_name} Defense fails vs Heavy Strike → {my_name} takes 1"
            return f"{my_name} Defense vs {cn[opp_card]} → {my_name} takes 0"

        if my_card == CardType.HEAVY_STRIKE:
            if opp_card == CardType.DEFENSE and CardResolver.played_defense_last_round(opp_history):
                return f"{my_name} Heavy Strike blocked by {opp_name} consecutive Defense → {opp_name} takes 0"
            return f"{my_name} Heavy Strike → {opp_name} takes 1"

        if my_card == CardType.COUNTER:
            if opp_card == CardType.ATTACK:
                return f"{my_name} Counter vs {opp_name} Attack → {opp_name} takes 1, {my_name} takes 0"
            if opp_card == CardType.HEAVY_STRIKE:
                return f"{my_name} Counter vs {opp_name} Heavy Strike → both take 1"
            return f"{my_name} Counter vs {cn[opp_card]} → no effect"

        return "Unknown effect"
