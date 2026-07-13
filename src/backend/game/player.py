"""Player state container for the custom card game v2 (design/11, design/16, design/38).

Per architecture D2, Player is a data container (dataclass) with minimal
behavior. It holds hp (default 30, design/38), energy (0-5, default 2, design/38),
statuses, hand, draw_pile, discard_pile, and history (list[PlayerHistoryItem] — design/16 D1).

Per architecture D9, this module is pure logic — no UI, no I/O.
"""

import random
from dataclasses import dataclass, field

from src.backend.game.card_type import CardType
from src.backend.game.constants import INITIAL_HP, INITIAL_ENERGY
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus, has_active_status

# Energy cap per design/11 intention §1
MAX_ENERGY = 5


@dataclass
class Player:
    """Represents one player's state in the v2 game.

    Attribute:
        hp: Current HP, default INITIAL_HP (design/38), clamped to >= 0.
        energy: Current energy, default INITIAL_ENERGY (design/38), clamped to [0, 5].
        statuses: List of active PlayerStatus instances (design/15).
        hand: Cards currently in hand.
        draw_pile: Cards remaining in the draw pile.
        discard_pile: Cards in the discard pile.
        history: Action history as list[PlayerHistoryItem] (design/16).
                 Each entry records both the card played and build_target.
    """

    hp: int = INITIAL_HP
    energy: int = INITIAL_ENERGY
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

    def _grant_status(self, status_type: StatusType, lifetime: int = 1) -> None:
        """Grant a status with specified lifetime (internal, no deduplication).

        External callers should use extend_status() to ensure proper
        status merging. This method always appends a new status without
        checking for duplicates.

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

    def extend_status(self, status_type: StatusType, extra: int) -> None:
        """Extend the lifetime of an existing active status.

        Finds the first active status (lifetime > 0) matching the given type
        and adds `extra` to its lifetime. If no matching active status exists,
        grants a new status with initial lifetime = extra.

        Per intention #36 D3: used by GrantFragileResolver for additive
        fragile (+2 rounds).

        Args:
            status_type: Status type to extend (e.g. StatusType.FRAGILE).
            extra: Number of rounds to add to the current lifetime.
        """
        for s in self.statuses:
            if s.status_type == status_type and s.lifetime > 0:
                s.lifetime += extra
                return
        # No active status found — grant a new one.
        self._grant_status(status_type, lifetime=extra)

    def remove_status(self, status_type: StatusType) -> None:
        """Remove all instances of the given status type.

        Per intention #36: used by RemoveDefenseResolver to strip opponent's
        DEFENSE when 破防重击 is played.

        Args:
            status_type: Status type to remove (e.g. StatusType.DEFENSE).
        """
        self.statuses = [
            s for s in self.statuses if s.status_type != status_type
        ]

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

    # ------------------------------------------------------------------
    # Deck manipulation
    # ------------------------------------------------------------------

    def remove_from_discard(self, card: CardType) -> None:
        """Remove one occurrence of `card` from discard_pile (design/37).

        Used by DismantleResolver to remove a card from the game when
        the玩家 selects the dismantle (拆解) half of 构建/拆解.

        Raises:
            ValueError: If the card is not found in discard_pile.
        """
        try:
            self.discard_pile.remove(card)
        except ValueError:
            raise ValueError(
                f"Card {card.name} is not in the player's discard pile."
            ) from None

    # ------------------------------------------------------------------
    # Deck drawing (design/35)
    # ------------------------------------------------------------------

    def draw_cards(self, count: int) -> int:
        """Draw up to `count` cards from draw_pile into hand.

        Implements classic DBG draw mechanics (design/35):
        1. Draw from draw_pile first.
        2. If draw_pile runs out and more cards are needed, shuffle
           discard_pile into draw_pile and continue drawing.
        3. If both piles are empty, stop early.

        Cards are appended to the end of hand; existing hand cards
        are unaffected.

        Args:
            count: Desired number of cards to draw. Must be >= 0.

        Returns:
            Number of cards actually drawn (0 <= return <= count).
        """
        if count <= 0:
            return 0

        drawn: int = 0
        needed: int = count

        while needed > 0:
            # If draw_pile is empty, try to recycle discard_pile
            if not self.draw_pile:
                if not self.discard_pile:
                    break  # Both piles empty, stop drawing
                # Shuffle discard_pile and move to draw_pile
                self.draw_pile = self.discard_pile
                self.discard_pile = []

            # Shuffle then draw front cards — each draw batch is random
            random.shuffle(self.draw_pile)
            take: int = min(needed, len(self.draw_pile))
            self.hand.extend(self.draw_pile[:take])
            self.draw_pile = self.draw_pile[take:]
            drawn += take
            needed -= take

        return drawn
