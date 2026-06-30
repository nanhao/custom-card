"""Player state container for the custom card game.

Per architecture D2, Player is a data container (dataclass) with minimal
behavior. It holds hp, hand, and history. The GameManager mutates it.
Per architecture D9, this module is pure logic — no UI, no I/O.
"""

from dataclasses import dataclass, field

from src.backend.game.card import CardType


@dataclass
class Player:
    """Represents one player's state in the game.

    Attributes:
        hp: Current HP, initially 3, clamped to ≥0 (architecture D5).
        hand: Remaining cards. Modified by play_card().
        history: Cards played in previous rounds, chronologically.
    """

    hp: int = 3
    hand: list[CardType] = field(default_factory=list)
    history: list[CardType] = field(default_factory=list)

    def play_card(self, card: CardType) -> None:
        """Remove the first occurrence of `card` from hand.

        Equivalent to list.remove(card). Raises ValueError if the card
        is not present in the hand.

        Per architecture D6, this is called AFTER resolution and damage
        application, so resolution never depends on current hand state.

        Convenience wrapper around play_cards for single-card use.
        """
        self.play_cards([card])

    def play_cards(self, cards: list[CardType]) -> None:
        """Remove one occurrence of each card in `cards` from hand.

        Raises ValueError if any card is not present (no partial removal).
        Supports future multi-card gameplay (architecture D12).

        Per architecture D6, this is called AFTER resolution and damage
        application, so resolution never depends on current hand state.
        """
        if not cards:
            return
        # Validate all cards present before removing any (atomic)
        remaining = list(self.hand)
        for card in cards:
            try:
                remaining.remove(card)
            except ValueError:
                raise ValueError(
                    f"Card {card.name} is not in the player's hand."
                ) from None
        self.hand = remaining

    def take_damage(self, amount: int) -> None:
        """Reduce HP by `amount`, clamped to 0.

        Per architecture D5: HP never goes negative.
        """
        self.hp = max(0, self.hp - amount)

    def has_card(self, card: CardType) -> bool:
        """Return True if `card` is present in hand."""
        return card in self.hand

    def has_cards(self, cards: list[CardType]) -> bool:
        """Return True if all cards in `cards` are present in hand.

        Checks duplicates correctly (e.g., two ATK requires at least two ATK in hand).
        Supports future multi-card gameplay (architecture D12).
        """
        remaining = list(self.hand)
        for card in cards:
            if card not in remaining:
                return False
            remaining.remove(card)
        return True
