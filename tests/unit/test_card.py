"""Unit tests for src/backend/game/card_type.py and card_def.py — CardType enum, CardDef, and get_initial_draw_pile().

Per design/22: card.py split into card_type.py + card_def.py.
Per design/23: CardDef.icon field removed — display logic moved to frontend.
Test cases per design/22 and design/23 test_cases.md.
"""

import pytest
from dataclasses import FrozenInstanceError

from src.backend.game.card_type import CardType
from src.backend.game.card_def import CardDef, get_card_def, get_initial_draw_pile
from src.backend.game.resolver import (
    AttackDamageResolver,
    BonusDefenseResolver,
    BuildResolver,
    FlexDamageResolver,
    GainEnergyResolver,
    GrantDefenseResolver,
    GrantFragileResolver,
    GrantGuardResolver,
    HeavyDamageResolver,
    ReflectResolver,
    RemoveDefenseResolver,
)


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
    """Per design/21: CardDef simplified — no effects, no priority; build_cost retained."""

    def test_card_def_no_effects_field(self):
        """TC-CD-01: CardDef no longer has effects field."""
        cd = get_card_def(CardType.ATTACK)
        with pytest.raises(AttributeError):
            _ = cd.effects

    def test_card_def_no_priority_field(self):
        """TC-CD-02: CardDef no longer has priority field."""
        cd = get_card_def(CardType.ATTACK)
        with pytest.raises(AttributeError):
            _ = cd.priority

    def test_card_def_fields(self):
        """TC-CD-04 (design/23): CardDef has card_type, build_cost, display_name, resolvers. No icon."""
        cd = get_card_def(CardType.ATTACK)
        assert cd.card_type == CardType.ATTACK
        assert cd.build_cost == 1
        assert cd.display_name
        assert cd.resolvers
        # Per design/23: icon field removed.
        assert "icon" not in CardDef.__dataclass_fields__

    def test_card_def_has_description_field(self):
        """TC-01-01: CardDef has description field, non-empty for all cards."""
        assert "description" in CardDef.__dataclass_fields__
        for ct in CardType:
            cd = get_card_def(ct)
            assert isinstance(cd.description, str), f"{ct.name}: description is not str"
            assert cd.description, f"{ct.name}: description is empty"

    def test_card_def_has_tags_field(self):
        """TC-01-02: CardDef has tags field, type is tuple[str, ...]."""
        assert "tags" in CardDef.__dataclass_fields__
        for ct in CardType:
            cd = get_card_def(ct)
            assert isinstance(cd.tags, tuple), f"{ct.name}: tags is not tuple"

    def test_attack_tags(self):
        """TC-01-03: ATTACK tags = ('攻',)."""
        assert get_card_def(CardType.ATTACK).tags == ("攻",)

    def test_defense_tags(self):
        """TC-01-04: DEFENSE tags = ('防',)."""
        assert get_card_def(CardType.DEFENSE).tags == ("防",)

    def test_flex_tags(self):
        """TC-01-05: FLEX tags = ('攻', '防')."""
        assert get_card_def(CardType.FLEX).tags == ("攻", "防")

    def test_energy_tags_empty(self):
        """TC-01-06: ENERGY tags = ()."""
        assert get_card_def(CardType.ENERGY).tags == ()

    def test_build_tags_empty(self):
        """TC-01-07: BUILD tags = ()."""
        assert get_card_def(CardType.BUILD).tags == ()

    def test_heavy_tags(self):
        """HEAVY tags = ('攻',)."""
        assert get_card_def(CardType.HEAVY).tags == ("攻",)

    def test_counter_tags(self):
        """COUNTER tags = ('防',)."""
        assert get_card_def(CardType.COUNTER).tags == ("防",)

    def test_build_cost_counter(self):
        """TC-CD-04: COUNTER → build_cost=2."""
        cd = get_card_def(CardType.COUNTER)
        assert cd.build_cost == 2

    def test_build_cost_flex(self):
        """TC-CD-04: FLEX → build_cost=2."""
        cd = get_card_def(CardType.FLEX)
        assert cd.build_cost == 2

    def test_build_cost_attack(self):
        """TC-CD-04: ATTACK → build_cost=1."""
        cd = get_card_def(CardType.ATTACK)
        assert cd.build_cost == 1

    def test_card_def_immutable(self):
        """CardDef is frozen — cannot modify fields."""
        cd = get_card_def(CardType.ATTACK)
        with pytest.raises(FrozenInstanceError):
            cd.build_cost = 5  # type: ignore[misc]

    def test_display_name_set(self):
        """Verify each CardDef has a non-empty display_name."""
        for ct in CardType:
            cd = get_card_def(ct)
            assert cd.display_name, f"Missing display_name for {ct.name}"

    def test_counter_display_name(self):
        """TC-CD-03 (design/24): COUNTER display_name is '防守反击'."""
        cd = get_card_def(CardType.COUNTER)
        assert cd.display_name == "防守反击"

    def test_display_names_intention36(self):
        """TC-02: display_name changes per intention #36."""
        assert get_card_def(CardType.ATTACK).display_name == "连续击"
        assert get_card_def(CardType.DEFENSE).display_name == "格挡"
        assert get_card_def(CardType.HEAVY).display_name == "破防重击"
        assert get_card_def(CardType.COUNTER).display_name == "防守反击"
        assert get_card_def(CardType.ENERGY).display_name == "聚能"
        assert get_card_def(CardType.BUILD).display_name == "构建"
        assert get_card_def(CardType.FLEX).display_name == "攻守兼备"

    def test_icon_removed(self):
        """Design/23 TC-CD-02: accessing icon raises AttributeError."""
        for ct in CardType:
            cd = get_card_def(ct)
            with pytest.raises(AttributeError):
                _ = cd.icon


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


class TestCardDefResolvers:
    """Per design/22: CardDef.resolvers field tests (TC-CD-01 ~ TC-CD-10)."""

    def test_resolvers_field_present(self):
        """TC-CD-01: CardDef has resolvers field."""
        cd = get_card_def(CardType.ATTACK)
        assert cd.resolvers == (AttackDamageResolver,)

    def test_resolvers_count_per_card_type(self):
        """TC-CD-02: each CardType has correct number of resolvers (intention #36)."""
        expected = {
            CardType.ATTACK: 1, CardType.DEFENSE: 2, CardType.HEAVY: 3,
            CardType.COUNTER: 3, CardType.ENERGY: 1, CardType.BUILD: 1,
            CardType.FLEX: 3,
        }
        for ct, count in expected.items():
            cd = get_card_def(ct)
            assert len(cd.resolvers) == count, f"{ct.name}: expected {count}, got {len(cd.resolvers)}"

    def test_resolvers_types_per_card_type(self):
        """TC-CD-03: each CardType's resolvers have correct types (intention #36)."""
        expected = {
            CardType.ATTACK: {AttackDamageResolver},
            CardType.DEFENSE: {GrantDefenseResolver, GrantGuardResolver},
            CardType.HEAVY: {HeavyDamageResolver, RemoveDefenseResolver, GrantFragileResolver},
            CardType.COUNTER: {GrantDefenseResolver, ReflectResolver, GrantGuardResolver},
            CardType.ENERGY: {GainEnergyResolver},
            CardType.BUILD: {BuildResolver},
            CardType.FLEX: {GrantDefenseResolver, BonusDefenseResolver, FlexDamageResolver},
        }
        for ct, types_set in expected.items():
            cd = get_card_def(ct)
            actual = set(cd.resolvers)
            assert actual == types_set, f"{ct.name}: expected {types_set}, got {actual}"

    def test_resolvers_immutable(self):
        """TC-CD-08: resolvers is an immutable tuple."""
        cd = get_card_def(CardType.ATTACK)
        with pytest.raises(TypeError):
            cd.resolvers[0] = BuildResolver  # type: ignore[index]

    def test_resolver_class_instantiable(self):
        """TC-CD-09: resolver classes from CardDef can be instantiated."""
        resolver_cls = get_card_def(CardType.ATTACK).resolvers[0]
        instance = resolver_cls()
        assert isinstance(instance, AttackDamageResolver)
        assert instance.get_priority() == 10

    def test_build_cost_preserved(self):
        """TC-CD-10: build_cost unchanged after refactoring."""
        assert get_card_def(CardType.COUNTER).build_cost == 2
        assert get_card_def(CardType.FLEX).build_cost == 2
        assert get_card_def(CardType.ATTACK).build_cost == 1


class TestEffectTypeRemoved:
    """TC-CD-03: EffectType enum has been removed."""

    def test_effect_type_cannot_be_imported(self):
        """TC-CD-03: Importing EffectType raises ImportError."""
        with pytest.raises(ImportError):
            from src.backend.game.card_type import EffectType  # noqa: F811
