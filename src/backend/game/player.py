"""Player state container for the custom card game v2 (design/11, design/16).

Per architecture D2, Player is a data container (dataclass) with minimal
behavior. It holds hp (default 20), energy (0-5), statuses, hand, draw_pile,
discard_pile, and history (list[PlayerHistoryItem] — design/16 D1).

Per architecture D9, this module is pure logic — no UI, no I/O.
"""

from dataclasses import dataclass, field

from src.backend.game.card_type import CardType
from src.backend.game.player_state import StatusType, PlayerStatus, has_active_status

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

    Attribute:
        hp: Current HP, default 20, clamped to >= 0.
        energy: Current energy, default 0, clamped to [0, 5].
        statuses: List of active PlayerStatus instances (design/15).
        hand: Cards currently in hand.
        draw_pile: Cards remaining in the draw pile.
        discard_pile: Cards in the discard pile.
        history: Action history as list[PlayerHistoryItem] (design/16).
                 Each entry records both the card played and build_target.
    """

    hp: int = 20
    energy: int = 0
    statuses: list[PlayerStatus] = field(default_factory=list)
    hand: list[CardType] = field(default_factory=list)
    draw_pile: list[CardType] = field(default_factory=list)
    discard_pile: list[CardType] = field(default_factory=list)
    history: list[PlayerHistoryItem] = field(default_factory=list)

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

    def grant_status(self, status_type: StatusType, lifetime: int = 1) -> None:
        """Grant a status with specified lifetime, appending to the statuses list.

        Per design/21: no deduplication — the caller is responsible for
        ensuring no duplicate statuses are granted.
        Per design/24: lifetime parameter added (default 1).

        Args:
            status_type: The status type to grant (e.g. StatusType.DEFENSE).
            lifetime: Number of rounds before expiry after ticks. Default 1.
        """
        self.statuses.append(PlayerStatus(status_type, lifetime=lifetime))

    def has_status(self, status_type: StatusType) -> bool:
        """Return True if the player has an active status of the given type.

        Only statuses with lifetime > 0 are considered active (design/15 D3).
        Delegates to has_active_status() in player_state.py.

        Args:
            status_type: The status type to check for (e.g. StatusType.DEFENSE).

        Returns:
            True if any PlayerStatus in statuses matches the type and has
            lifetime > 0.
        """
        return has_active_status(self.statuses, status_type)

    def get_consecutive_attack_count(self) -> int:
        """Count consecutive ATTACK cards from the end of history.

        Used by AttackDamageResolver to compute streak damage.

        Returns:
            Number of consecutive ATTACK cards at the tail of history.
        """
        count = 0
        for item in reversed(self.history):
            if item.card == CardType.ATTACK:
                count += 1
            else:
                break
        return count
