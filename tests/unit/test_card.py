"""Unit tests for src/backend/game/card.py — CardType enum and get_initial_hand()."""

import pytest
from src.backend.game.card import CardType, get_initial_hand


class TestCardType:
    """Tests for the CardType enum."""

    def test_card_type_values(self):
        """Verify all four card types have correct string values."""
        assert CardType.ATTACK.value == "attack"
        assert CardType.DEFENSE.value == "defense"
        assert CardType.HEAVY_STRIKE.value == "heavy"
        assert CardType.COUNTER.value == "counter"

    def test_card_type_count(self):
        """Verify exactly 4 card types exist."""
        assert len(CardType) == 4

    def test_card_type_identity(self):
        """Verify each card type is a unique enum member."""
        values = set(CardType)
        assert len(values) == 4


class TestGetInitialHand:
    """Tests for the get_initial_hand() factory function."""

    def test_correct_length(self):
        """Initial hand must contain exactly 7 cards."""
        hand = get_initial_hand()
        assert len(hand) == 7

    def test_correct_composition(self):
        """Initial hand must contain 3 ATTACK, 2 DEFENSE, 1 HEAVY_STRIKE, 1 COUNTER."""
        hand = get_initial_hand()
        assert hand.count(CardType.ATTACK) == 3
        assert hand.count(CardType.DEFENSE) == 2
        assert hand.count(CardType.HEAVY_STRIKE) == 1
        assert hand.count(CardType.COUNTER) == 1

    def test_correct_order(self):
        """Initial hand must match the standard order defined in interfaces.md."""
        hand = get_initial_hand()
        expected = [
            CardType.ATTACK, CardType.ATTACK, CardType.ATTACK,
            CardType.DEFENSE, CardType.DEFENSE,
            CardType.HEAVY_STRIKE,
            CardType.COUNTER,
        ]
        assert hand == expected

    def test_returns_new_list_each_call(self):
        """Each call must return a fresh list — no shared mutation bugs."""
        hand1 = get_initial_hand()
        hand2 = get_initial_hand()
        assert hand1 is not hand2  # Different objects
        assert hand1 == hand2       # Same content
        # Modifying one must not affect the other
        hand1[0] = CardType.COUNTER
        assert hand2[0] == CardType.ATTACK

    def test_all_elements_are_card_types(self):
        """Every element in the hand must be a CardType enum member."""
        hand = get_initial_hand()
        for card in hand:
            assert isinstance(card, CardType)
