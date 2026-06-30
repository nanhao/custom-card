"""Unit tests for src/backend/game/player.py — Player dataclass."""

import pytest
from src.backend.game.card import CardType, get_initial_hand
from src.backend.game.player import Player


class TestPlayerInit:
    """Tests for Player initialization."""

    def test_default_hp_is_3(self):
        """Player should start with 3 HP by default."""
        p = Player()
        assert p.hp == 3

    def test_custom_hp(self):
        """Player should accept a custom HP value."""
        p = Player(hp=1)
        assert p.hp == 1

    def test_default_hand_is_empty(self):
        """Player should start with an empty hand by default."""
        p = Player()
        assert p.hand == []

    def test_custom_hand(self):
        """Player should accept a custom hand."""
        hand = get_initial_hand()
        p = Player(hand=hand)
        assert p.hand == hand
        assert len(p.hand) == 7

    def test_default_history_is_empty(self):
        """Player should start with an empty history."""
        p = Player()
        assert p.history == []

    def test_custom_history(self):
        """Player should accept a custom history."""
        hist = [CardType.ATTACK, CardType.DEFENSE]
        p = Player(history=hist)
        assert p.history == hist

    def test_full_init(self):
        """Player should accept all fields together."""
        hand = get_initial_hand()
        hist = [CardType.ATTACK]
        p = Player(hp=2, hand=hand, history=hist)
        assert p.hp == 2
        assert p.hand == hand
        assert p.history == hist


class TestPlayCard:
    """Tests for Player.play_card()."""

    def test_removes_first_occurrence(self):
        """play_card should remove the first occurrence of the card."""
        hand = [CardType.ATTACK, CardType.DEFENSE, CardType.ATTACK, CardType.ATTACK]
        p = Player(hand=hand.copy())
        p.play_card(CardType.ATTACK)
        assert p.hand == [CardType.DEFENSE, CardType.ATTACK, CardType.ATTACK]

    def test_removes_one_instance_only(self):
        """play_card should remove only one card, not all matching."""
        hand = [CardType.ATTACK, CardType.ATTACK, CardType.ATTACK]
        p = Player(hand=hand.copy())
        p.play_card(CardType.ATTACK)
        assert len(p.hand) == 2
        assert p.hand.count(CardType.ATTACK) == 2

    def test_hand_length_decreases(self):
        """After play_card, hand length should decrease by 1."""
        p = Player(hand=get_initial_hand())
        assert len(p.hand) == 7
        p.play_card(CardType.ATTACK)
        assert len(p.hand) == 6

    def test_raises_value_error_if_card_not_in_hand(self):
        """play_card with a card not in hand should raise ValueError."""
        p = Player(hand=[CardType.DEFENSE])
        with pytest.raises(ValueError):
            p.play_card(CardType.ATTACK)

    def test_raises_value_error_if_hand_empty(self):
        """play_card on an empty hand should raise ValueError."""
        p = Player(hand=[])
        with pytest.raises(ValueError):
            p.play_card(CardType.ATTACK)

    def test_tc_x8_duplicate_removal(self):
        """TC-X8: play_card(ATTACK) with [ATK, DEF, ATK, ATK] → 2 ATTACK remain."""
        hand = [
            CardType.ATTACK, CardType.DEFENSE,
            CardType.ATTACK, CardType.ATTACK,
        ]
        p = Player(hand=hand.copy())
        p.play_card(CardType.ATTACK)
        assert p.hand.count(CardType.ATTACK) == 2
        assert len(p.hand) == 3


class TestPlayCards:
    """Tests for Player.play_cards() — multi-card support (architecture D12)."""

    def test_removes_multiple_cards(self):
        """play_cards should remove all specified cards."""
        p = Player(hand=[CardType.ATTACK, CardType.DEFENSE, CardType.ATTACK])
        p.play_cards([CardType.ATTACK, CardType.DEFENSE])
        assert p.hand == [CardType.ATTACK]

    def test_removes_duplicate_cards(self):
        """play_cards should remove one per specified card, even duplicates."""
        p = Player(hand=[CardType.ATTACK, CardType.ATTACK, CardType.ATTACK])
        p.play_cards([CardType.ATTACK, CardType.ATTACK])
        assert p.hand == [CardType.ATTACK]

    def test_empty_list_does_nothing(self):
        """play_cards with empty list should not modify hand."""
        p = Player(hand=get_initial_hand())
        original = list(p.hand)
        p.play_cards([])
        assert p.hand == original

    def test_raises_if_any_card_missing(self):
        """play_cards should raise ValueError if any card is not present."""
        p = Player(hand=[CardType.ATTACK])
        with pytest.raises(ValueError):
            p.play_cards([CardType.ATTACK, CardType.DEFENSE])
        # Hand should be unchanged (atomic)
        assert p.hand == [CardType.ATTACK]

    def test_raises_if_insufficient_duplicates(self):
        """play_cards should raise ValueError if not enough duplicates."""
        p = Player(hand=[CardType.ATTACK])
        with pytest.raises(ValueError):
            p.play_cards([CardType.ATTACK, CardType.ATTACK])
        assert p.hand == [CardType.ATTACK]

    def test_single_card_same_as_play_card(self):
        """play_cards with one card should work identically to play_card."""
        p = Player(hand=get_initial_hand())
        p.play_cards([CardType.ATTACK])
        assert len(p.hand) == 6
        assert p.hand.count(CardType.ATTACK) == 2


class TestTakeDamage:
    """Tests for Player.take_damage()."""

    def test_reduces_hp(self):
        """take_damage should decrease HP by the given amount."""
        p = Player(hp=3)
        p.take_damage(1)
        assert p.hp == 2

    def test_multiple_damage(self):
        """Multiple take_damage calls should accumulate."""
        p = Player(hp=3)
        p.take_damage(1)
        p.take_damage(2)
        assert p.hp == 0

    def test_clamps_to_zero(self):
        """HP should never go below 0 (architecture D5)."""
        p = Player(hp=1)
        p.take_damage(3)
        assert p.hp == 0

    def test_clamps_to_zero_from_zero(self):
        """HP at 0 should stay at 0 after more damage."""
        p = Player(hp=0)
        p.take_damage(5)
        assert p.hp == 0

    def test_zero_damage_does_nothing(self):
        """Zero damage should not change HP."""
        p = Player(hp=2)
        p.take_damage(0)
        assert p.hp == 2


class TestHasCard:
    """Tests for Player.has_card()."""

    def test_returns_true_if_card_present(self):
        """has_card should return True when the card exists."""
        p = Player(hand=[CardType.ATTACK])
        assert p.has_card(CardType.ATTACK) is True

    def test_returns_false_if_card_absent(self):
        """has_card should return False when the card doesn't exist."""
        p = Player(hand=[CardType.DEFENSE])
        assert p.has_card(CardType.ATTACK) is False

    def test_returns_false_if_hand_empty(self):
        """has_card should return False on an empty hand."""
        p = Player(hand=[])
        assert p.has_card(CardType.ATTACK) is False

    def test_returns_true_for_duplicate_cards(self):
        """has_card should return True even if multiple copies exist."""
        p = Player(hand=[CardType.ATTACK, CardType.ATTACK])
        assert p.has_card(CardType.ATTACK) is True


class TestHasCards:
    """Tests for Player.has_cards() — multi-card support (architecture D12)."""

    def test_returns_true_if_all_cards_present(self):
        """has_cards should return True when all cards are present."""
        p = Player(hand=[CardType.ATTACK, CardType.DEFENSE])
        assert p.has_cards([CardType.ATTACK, CardType.DEFENSE]) is True

    def test_returns_false_if_any_card_absent(self):
        """has_cards should return False if any card is missing."""
        p = Player(hand=[CardType.ATTACK])
        assert p.has_cards([CardType.ATTACK, CardType.DEFENSE]) is False

    def test_respects_duplicates(self):
        """has_cards should check duplicates correctly."""
        p = Player(hand=[CardType.ATTACK])
        assert p.has_cards([CardType.ATTACK, CardType.ATTACK]) is False
        p2 = Player(hand=[CardType.ATTACK, CardType.ATTACK])
        assert p2.has_cards([CardType.ATTACK, CardType.ATTACK]) is True

    def test_empty_list_returns_true(self):
        """has_cards with empty list should always return True."""
        p = Player(hand=[])
        assert p.has_cards([]) is True
        p2 = Player(hand=[CardType.ATTACK])
        assert p2.has_cards([]) is True

    def test_single_card_same_as_has_card(self):
        """has_cards with one card should work identically to has_card."""
        p = Player(hand=[CardType.ATTACK])
        assert p.has_cards([CardType.ATTACK]) == p.has_card(CardType.ATTACK)
        assert p.has_cards([CardType.DEFENSE]) == p.has_card(CardType.DEFENSE)
