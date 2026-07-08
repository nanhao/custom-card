"""Player state container for the custom card game v2 (design/11, design/14).

Per architecture D2, Player is a data container (dataclass) with minimal
behavior. It holds hp (default 20), energy (0-5), statuses, hand, draw_pile,
discard_pile, history, and build_history (design/14 D8).

Per architecture D9, this module is pure logic — no UI, no I/O.
"""

from dataclasses import dataclass, field

from src.backend.game.card import CardType

# Energy cap per design/11 intention §1
MAX_ENERGY = 5


@dataclass
class PlayerHistoryItem:
    """Single entry in a player's action history (design/14 D8).

    Records both the card played and the build target (if any)
    for a single round. Used by GameCore.get_state() to provide
    complete action history so Converter can derive policy indices.

    Attributes:
        card: The card played this round.
        build_target: The card built via BUILD, or None if no build occurred.
    """

    card: CardType
    build_target: CardType | None = None


@dataclass
class Player:
    """Represents one player's state in the v2 game.

    Attributes:
        hp: Current HP, default 20, clamped to >= 0.
        energy: Current energy, default 0, clamped to [0, 5].
        statuses: List of active status strings (e.g. "defense", "联防").
        hand: Cards currently in hand.
        draw_pile: Cards remaining in the draw pile.
        discard_pile: Cards in the discard pile.
        history: Cards played in previous rounds, chronologically.
                 Used for consecutive attack counting.
        build_history: Build targets for each round, parallel to history
                       (design/14 D8). None entries for non-build rounds.
    """

    hp: int = 20
    energy: int = 0
    statuses: list[str] = field(default_factory=list)
    hand: list[CardType] = field(default_factory=list)
    draw_pile: list[CardType] = field(default_factory=list)
    discard_pile: list[CardType] = field(default_factory=list)
    history: list[CardType] = field(default_factory=list)
    build_history: list[CardType | None] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Card manipulation
    # ------------------------------------------------------------------

    def play_card(self, card: CardType) -> None:
        """Remove one occurrence of `card` from hand.

        Raises ValueError if the card is not present in the hand.
        Convenience wrapper around play_cards for single-card use.
        """
        self.play_cards([card])

    def play_cards(self, cards: list[CardType]) -> None:
        """Remove one occurrence of each card in `cards` from hand.

        Atomic: validates all cards present before removing any.
        Raises ValueError if any card is not present.
        """
        if not cards:
            return
        remaining = list(self.hand)
        for card in cards:
            try:
                remaining.remove(card)
            except ValueError:
                raise ValueError(
                    f"Card {card.name} is not in the player's hand."
                ) from None
        self.hand = remaining

    def has_card(self, card: CardType) -> bool:
        """Return True if `card` is present in hand."""
        return card in self.hand

    def has_cards(self, cards: list[CardType]) -> bool:
        """Return True if all cards in `cards` are present in hand.

        Checks duplicates correctly (e.g., two ATK requires at least two ATK in hand).
        """
        remaining = list(self.hand)
        for card in cards:
            if card not in remaining:
                return False
            remaining.remove(card)
        return True

    # ------------------------------------------------------------------
    # HP
    # ------------------------------------------------------------------

    def take_damage(self, amount: int) -> None:
        """Reduce HP by `amount`, clamped to 0.

        Per design: HP never goes negative.
        """
        self.hp = max(0, self.hp - amount)

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def add_energy(self, amount: int) -> None:
        """Increase energy by `amount`, clamped to [0, MAX_ENERGY].

        Args:
            amount: Positive integer amount of energy to add.
        """
        self.energy = min(MAX_ENERGY, self.energy + amount)

    # ------------------------------------------------------------------
    # Statuses
    # ------------------------------------------------------------------

    def has_status(self, status: str) -> bool:
        """Return True if the player has the specified status."""
        return status in self.statuses
