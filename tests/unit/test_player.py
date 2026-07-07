"""Unit tests for src/backend/game/player.py — Player dataclass.

Test cases per design/11 test_cases.md §2 (TC-PL-01 ~ TC-PL-09).
"""

import pytest

from src.backend.game.card import CardType
from src.backend.game.player import Player


class TestPlayerDefaults:
    """TC-PL-01: default values."""

    def test_default_values(self):
        """Player() creates with hp=20, energy=0, all lists empty."""
        p = Player()
        assert p.hp == 20
        assert p.energy == 0
        assert p.statuses == []
        assert p.hand == []
        assert p.draw_pile == []
        assert p.discard_pile == []
        assert p.history == []


class TestTakeDamage:
    """TC-PL-02, TC-PL-03: take_damage."""

    def test_normal_damage(self):
        """TC-PL-02: take_damage(5) reduces hp from 20 to 15."""
        p = Player(hp=20)
        p.take_damage(5)
        assert p.hp == 15

    def test_damage_clamped_to_zero(self):
        """TC-PL-03: take_damage(10) with hp=3 results in hp=0."""
        p = Player(hp=3)
        p.take_damage(10)
        assert p.hp == 0


class TestAddEnergy:
    """TC-PL-04, TC-PL-05, TC-PL-06: add_energy."""

    def test_normal_increase(self):
        """TC-PL-04: add_energy(2) with energy=0 results in energy=2."""
        p = Player(energy=0)
        p.add_energy(2)
        assert p.energy == 2

    def test_clamped_to_max(self):
        """TC-PL-05: add_energy(3) with energy=4 results in energy=5."""
        p = Player(energy=4)
        p.add_energy(3)
        assert p.energy == 5

    def test_no_effect_at_max(self):
        """TC-PL-06: add_energy(1) with energy=5 results in energy=5."""
        p = Player(energy=5)
        p.add_energy(1)
        assert p.energy == 5


class TestHasStatus:
    """TC-PL-07: has_status."""

    def test_has_existing_status(self):
        """has_status('defense') returns True when statuses=['defense']."""
        p = Player(statuses=["defense"])
        assert p.has_status("defense") is True

    def test_has_missing_status(self):
        """has_status('联防') returns False when only 'defense' is present."""
        p = Player(statuses=["defense"])
        assert p.has_status("联防") is False


class TestPlayCard:
    """TC-PL-08, TC-PL-09: play_card."""

    def test_removes_card_from_hand(self):
        """TC-PL-08: play_card removes the card from hand."""
        p = Player(hand=[CardType.ATTACK, CardType.DEFENSE])
        p.play_card(CardType.ATTACK)
        assert p.hand == [CardType.DEFENSE]

    def test_raises_if_card_not_in_hand(self):
        """TC-PL-09: play_card with missing card raises ValueError."""
        p = Player(hand=[CardType.ATTACK])
        with pytest.raises(ValueError):
            p.play_card(CardType.DEFENSE)


class TestPlayCards:
    """Multi-card play_card support."""

    def test_play_cards_removes_multiple(self):
        """play_cards removes each specified card from hand atomically."""
        p = Player(hand=[CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE])
        p.play_cards([CardType.ATTACK, CardType.DEFENSE])
        assert p.hand == [CardType.ATTACK]

    def test_play_cards_empty_list(self):
        """play_cards with empty list is a no-op."""
        p = Player(hand=[CardType.ATTACK])
        hand_before = list(p.hand)
        p.play_cards([])
        assert p.hand == hand_before
