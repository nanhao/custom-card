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
        """
        try:
            self.hand.remove(card)
        except ValueError:
            raise ValueError(
                f"Card {card.name} is not in the player's hand."
            ) from None

    def take_damage(self, amount: int) -> None:
        """Reduce HP by `amount`, clamped to 0.

        Per architecture D5: HP never goes negative.
        """
        self.hp = max(0, self.hp - amount)

    def has_card(self, card: CardType) -> bool:
        """Return True if `card` is present in hand."""
        return card in self.hand
