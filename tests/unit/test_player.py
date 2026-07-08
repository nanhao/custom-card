"""Unit tests for src/backend/game/player.py — Player dataclass.

Test cases per design/11 test_cases.md §2 (TC-PL-01 ~ TC-PL-09)
and design/14 test_cases.md §4 (TC-BH-01 ~ TC-BH-04).
"""

import pytest

from src.backend.game.card import CardType
from src.backend.game.player import Player, PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus


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
    """TC-HS-01 ~ TC-HS-05 (design/15): has_status with StatusType."""

    def test_has_existing_status(self):
        """TC-HS-01: has_status(DEFENSE) returns True when statuses has DEFENSE with lifetime>0."""
        p = Player(statuses=[PlayerStatus(StatusType.DEFENSE, 1)])
        assert p.has_status(StatusType.DEFENSE) is True

    def test_has_missing_status(self):
        """TC-HS-02: has_status(GUARD) returns False when only DEFENSE is present."""
        p = Player(statuses=[PlayerStatus(StatusType.DEFENSE, 1)])
        assert p.has_status(StatusType.GUARD) is False

    def test_empty_statuses_returns_false(self):
        """TC-HS-03: statuses=[] → has_status returns False."""
        p = Player(statuses=[])
        assert p.has_status(StatusType.DEFENSE) is False

    def test_lifetime_zero_not_considered_active(self):
        """TC-HS-04: lifetime=0 → has_status returns False (expired)."""
        p = Player(statuses=[PlayerStatus(StatusType.DEFENSE, 0)])
        assert p.has_status(StatusType.DEFENSE) is False

    def test_multiple_statuses_both_detected(self):
        """TC-HS-05: multiple active statuses → both detected."""
        p = Player(statuses=[
            PlayerStatus(StatusType.DEFENSE, 1),
            PlayerStatus(StatusType.GUARD, 1),
        ])
        assert p.has_status(StatusType.DEFENSE) is True
        assert p.has_status(StatusType.GUARD) is True


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


class TestPlayerHistoryNewType:
    """TC-PL-H01, TC-PL-H02, TC-PL-H03: Player.history type change (design/16)."""

    def test_default_history_empty(self):
        """TC-PL-H01: New Player has empty history."""
        p = Player()
        assert p.history == []

    def test_history_element_type(self):
        """TC-PL-H02: history elements are PlayerHistoryItem."""
        p = Player(history=[PlayerHistoryItem(CardType.ATTACK, None)])
        assert len(p.history) == 1
        assert isinstance(p.history[0], PlayerHistoryItem)
        assert p.history[0].card == CardType.ATTACK
        assert p.history[0].build_target is None

    def test_no_build_history_attribute(self):
        """TC-PL-H03: Player has no build_history attribute."""
        p = Player()
        with pytest.raises(AttributeError):
            _ = p.build_history


class TestPlayerHistoryItem:
    """PlayerHistoryItem dataclass."""

    def test_create_with_card_only(self):
        """PlayerHistoryItem with build_target=None."""
        item = PlayerHistoryItem(card=CardType.ATTACK, build_target=None)
        assert item.card == CardType.ATTACK
        assert item.build_target is None

    def test_create_with_build_target(self):
        """PlayerHistoryItem with build_target set."""
        item = PlayerHistoryItem(card=CardType.BUILD, build_target=CardType.ATTACK)
        assert item.card == CardType.BUILD
        assert item.build_target == CardType.ATTACK

    def test_equality(self):
        """Two PlayerHistoryItem with same values are equal."""
        a = PlayerHistoryItem(CardType.ATTACK, None)
        b = PlayerHistoryItem(CardType.ATTACK, None)
        assert a == b

    def test_inequality(self):
        """Different cards or build_targets produce inequality."""
        a = PlayerHistoryItem(CardType.ATTACK, None)
        b = PlayerHistoryItem(CardType.DEFENSE, None)
        assert a != b


class TestGrantStatus:
    """TC-PS-01, TC-PS-02: grant_status (design/21)."""

    def test_grant_status_on_empty(self):
        """TC-PS-01: grant_status(DEFENSE) adds PlayerStatus with lifetime=1."""
        p = Player(statuses=[])
        p.grant_status(StatusType.DEFENSE)
        assert len(p.statuses) == 1
        assert p.statuses[0].status_type == StatusType.DEFENSE
        assert p.statuses[0].lifetime == 1

    def test_grant_status_appends_not_overwrites(self):
        """TC-PS-02: grant_status appends, does not replace existing."""
        p = Player(statuses=[PlayerStatus(StatusType.DEFENSE, 1)])
        p.grant_status(StatusType.GUARD)
        assert len(p.statuses) == 2
        assert p.statuses[0].status_type == StatusType.DEFENSE
        assert p.statuses[1].status_type == StatusType.GUARD
        assert p.statuses[1].lifetime == 1
