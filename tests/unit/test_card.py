"""Unit tests for src/backend/game/card.py — CardType enum, CardDef, and get_initial_draw_pile().

Test cases per design/11 test_cases.md §1 (TC-CD-01 ~ TC-CD-05).
"""

import pytest
from dataclasses import FrozenInstanceError

from src.backend.game.card import CardType, CardDef, EffectType, get_card_def, get_initial_draw_pile


class TestCardType:
    """Tests for the CardType enum — 7 values."""

    def test_card_type_has_seven_members(self):
        """TC-CD-01: CardType 枚举包含7个值."""
        assert len(CardType) == 7
        assert CardType.ATTACK.value == "attack"
        assert CardType.DEFENSE.value == "defense"
        assert CardType.HEAVY.value == "heavy"
        assert CardType.COUNTER.value == "counter"
        assert CardType.ENERGY.value == "energy"
        assert CardType.BUILD.value == "build"
        assert CardType.FLEX.value == "flex"

    def test_card_type_identity(self):
        """Verify each card type is a unique enum member."""
        values = set(CardType)
        assert len(values) == 7


class TestGetCardDef:
    """TC-CD-02, TC-CD-03, TC-CD-05: get_card_def correctness and immutability."""

    def test_attack_priority(self):
        """TC-CD-02: ATTACK → priority=10, single damage effect."""
        cd = get_card_def(CardType.ATTACK)
        assert cd.priority == 10
        assert cd.effects == ((EffectType.DAMAGE, 10),)
        assert cd.build_cost == 1

    def test_defense_priority(self):
        """TC-CD-02: DEFENSE → priority=11."""
        cd = get_card_def(CardType.DEFENSE)
        assert cd.priority == 11
        assert cd.build_cost == 1

    def test_flex_multi_effects(self):
        """TC-CD-02: FLEX → priority=None, two effects (grant_defense=11, flex_damage=10)."""
        cd = get_card_def(CardType.FLEX)
        assert cd.priority is None
        assert cd.effects == ((EffectType.GRANT_DEFENSE, 11), (EffectType.FLEX_DAMAGE, 10))

    def test_build_cost_counter(self):
        """TC-CD-03: COUNTER → build_cost=2."""
        cd = get_card_def(CardType.COUNTER)
        assert cd.build_cost == 2

    def test_build_cost_flex(self):
        """TC-CD-03: FLEX → build_cost=2."""
        cd = get_card_def(CardType.FLEX)
        assert cd.build_cost == 2

    def test_build_cost_attack(self):
        """TC-CD-03: ATTACK → build_cost=1."""
        cd = get_card_def(CardType.ATTACK)
        assert cd.build_cost == 1

    def test_card_def_immutable(self):
        """TC-CD-05: CardDef is frozen — cannot modify fields."""
        cd = get_card_def(CardType.ATTACK)
        with pytest.raises(FrozenInstanceError):
            cd.priority = 5  # type: ignore[misc]

    def test_display_name_set(self):
        """Verify each CardDef has a non-empty display_name."""
        for ct in CardType:
            cd = get_card_def(ct)
            assert cd.display_name, f"Missing display_name for {ct.name}"

    def test_icon_set(self):
        """Verify each CardDef has a non-empty icon."""
        for ct in CardType:
            cd = get_card_def(ct)
            assert cd.icon, f"Missing icon for {ct.name}"


class TestGetInitialDrawPile:
    """TC-CD-04: get_initial_draw_pile correctness."""

    def test_correct_length(self):
        """TC-CD-04: 返回7张卡."""
        pile = get_initial_draw_pile()
        assert len(pile) == 7

    def test_correct_composition(self):
        """TC-CD-04: 2×ATTACK, 1 each of DEFENSE/HEAVY/COUNTER/ENERGY/BUILD, no FLEX."""
        pile = get_initial_draw_pile()
        assert pile.count(CardType.ATTACK) == 2
        assert pile.count(CardType.DEFENSE) == 1
        assert pile.count(CardType.HEAVY) == 1
        assert pile.count(CardType.COUNTER) == 1
        assert pile.count(CardType.ENERGY) == 1
        assert pile.count(CardType.BUILD) == 1
        assert pile.count(CardType.FLEX) == 0

    def test_returns_new_list_each_call(self):
        """Each call returns a fresh list — no shared-mutation bugs."""
        pile1 = get_initial_draw_pile()
        pile2 = get_initial_draw_pile()
        assert pile1 is not pile2
        assert pile1 == pile2
        pile1[0] = CardType.FLEX
        assert pile2[0] == CardType.ATTACK
