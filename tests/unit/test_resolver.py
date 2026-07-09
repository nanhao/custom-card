"""Unit tests for resolver.py — Resolver strategy pattern (design/21).

Test cases per design/21 test_cases.md.
"""

import pytest

from src.backend.game.card import CardType
from src.backend.game.game_core import GameCore
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus
from src.backend.game.resolver import (
    ResolveDetails,
    ResolveContext,
    ResolutionResult,
    Resolver,
    CardResolverRegistry,
    CardResolver,
    AttackDamageResolver,
    GrantDefenseResolver,
    GrantGuardResolver,
    HeavyDamageResolver,
    ReflectResolver,
    GainEnergyResolver,
    FlexDamageResolver,
    BuildResolver,
)


# ---------------------------------------------------------------------------
# Helper: create GameCore + players, then build ResolveContext
# ---------------------------------------------------------------------------


def _make_ctx(
    hp_1=20, hp_2=20,
    energy_1=0, energy_2=0,
    statuses_1=None, statuses_2=None,
    history_1=None,
    source="1", opponent_card=CardType.ATTACK,
    owner_card=CardType.ATTACK,
):
    """Create a GameCore with players, then build ResolveContext."""
    if statuses_1 is None:
        statuses_1 = []
    if statuses_2 is None:
        statuses_2 = []
    if history_1 is None:
        history_1 = []
    gm = GameCore(mode="pvp")
    gm.start_game()
    gm.player_1.hp = hp_1
    gm.player_2.hp = hp_2
    gm.player_1.energy = energy_1
    gm.player_2.energy = energy_2
    gm.player_1.statuses = list(statuses_1)
    gm.player_2.statuses = list(statuses_2)
    gm.player_1.history = list(history_1)
    if source == "1":
        return gm, ResolveContext(
            card_owner=gm.player_1, opponent=gm.player_2,
            card_played_by_owner=owner_card,
            card_played_by_opponent=opponent_card,
            game_core=gm,
        )
    else:
        return gm, ResolveContext(
            card_owner=gm.player_2, opponent=gm.player_1,
            card_played_by_owner=owner_card,
            card_played_by_opponent=opponent_card,
            game_core=gm,
        )


# =============================================================================
# 1. CardResolverRegistry — TC-RG-01 ~ TC-RG-04
# =============================================================================


class TestCardResolverRegistry:
    """TC-RG-01 ~ TC-RG-04: Registry correctness."""

    def test_attack_has_one_resolver(self):
        resolvers = CardResolverRegistry.get_resolvers(CardType.ATTACK)
        assert len(resolvers) == 1
        assert isinstance(resolvers[0], AttackDamageResolver)

    def test_defense_has_two_resolvers(self):
        resolvers = CardResolverRegistry.get_resolvers(CardType.DEFENSE)
        assert len(resolvers) == 2
        types = {type(r) for r in resolvers}
        assert types == {GrantDefenseResolver, GrantGuardResolver}

    def test_all_card_types_registered(self):
        expected = {CardType.ATTACK: 1, CardType.DEFENSE: 2, CardType.HEAVY: 1,
                    CardType.COUNTER: 2, CardType.ENERGY: 1, CardType.BUILD: 1,
                    CardType.FLEX: 2}
        for ct, count in expected.items():
            resolvers = CardResolverRegistry.get_resolvers(ct)
            assert len(resolvers) == count, f"{ct.name} expected {count}, got {len(resolvers)}"

    def test_register_appends(self):
        before = len(CardResolverRegistry.get_resolvers(CardType.ATTACK))
        dummy = BuildResolver()
        CardResolverRegistry.register(CardType.ATTACK, dummy)
        try:
            after = len(CardResolverRegistry.get_resolvers(CardType.ATTACK))
            assert after == before + 1
        finally:
            CardResolverRegistry._registry[CardType.ATTACK].pop()


# =============================================================================
# 2. Resolver base — TC-RS-BASE-01, TC-RS-BASE-02
# =============================================================================


class TestResolverBase:
    def test_resolver_is_abstract(self):
        with pytest.raises(TypeError):
            Resolver()

    def test_subclass_must_implement_methods(self):
        class _Incomplete(Resolver):
            pass
        with pytest.raises(TypeError):
            _Incomplete()


# =============================================================================
# 3. AttackDamageResolver — streak computed from card_owner.history
# =============================================================================


class TestAttackDamageResolver:
    """TC-ATK-01 ~ TC-ATK-05."""

    def test_basic_attack_streak_zero(self):
        """streak=0 (empty history) → damage=1."""
        gm, ctx = _make_ctx(hp_2=20, source="1")
        r = AttackDamageResolver()
        r.resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 19

    def test_streak_two_deals_three(self):
        """streak=2 → damage=3."""
        gm, ctx = _make_ctx(
            hp_2=10, source="1",
            history_1=[PlayerHistoryItem(CardType.ATTACK, None),
                        PlayerHistoryItem(CardType.ATTACK, None)],
        )
        AttackDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 7

    def test_streak_capped_at_three(self):
        """streak=5 → damage=3 (capped)."""
        gm, ctx = _make_ctx(
            hp_2=10, source="1",
            history_1=[PlayerHistoryItem(CardType.ATTACK, None)] * 5,
        )
        AttackDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 7

    def test_attack_blocked_by_defense(self):
        """DEFENSE status blocks attack."""
        gm, ctx = _make_ctx(
            hp_2=10, source="1",
            statuses_2=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        AttackDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_player_two_attacks(self):
        """P2 attacks P1."""
        gm, ctx = _make_ctx(hp_1=10, source="2")
        AttackDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_1.hp == 9

    def test_get_priority_returns_10(self):
        assert AttackDamageResolver().get_priority() == 10


# =============================================================================
# 4. GrantDefenseResolver
# =============================================================================


class TestGrantDefenseResolver:
    def test_grants_defense_to_player_one(self):
        gm, ctx = _make_ctx(source="1")
        GrantDefenseResolver().resolve(ctx, ResolveDetails())
        assert gm.player_1.statuses[0].status_type == StatusType.DEFENSE
        assert gm.player_1.statuses[0].lifetime == 1

    def test_grants_defense_to_player_two(self):
        gm, ctx = _make_ctx(source="2")
        GrantDefenseResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.statuses[0].status_type == StatusType.DEFENSE

    def test_get_priority_returns_11(self):
        assert GrantDefenseResolver().get_priority() == 11


# =============================================================================
# 5. GrantGuardResolver — priority 12, reads statuses directly
# =============================================================================


class TestGrantGuardResolver:
    """TC-GG-01 ~ TC-GG-03. Priority 12 ensures pre-round DEFENSE is visible."""

    def test_grants_guard_with_pre_existing_defense(self):
        """card_owner has DEFENSE → GUARD granted."""
        gm, ctx = _make_ctx(
            source="1",
            statuses_1=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        GrantGuardResolver().resolve(ctx, ResolveDetails())
        assert any(ps.status_type == StatusType.GUARD for ps in gm.player_1.statuses)

    def test_no_guard_without_pre_defense(self):
        """card_owner has no DEFENSE → no GUARD."""
        gm, ctx = _make_ctx(source="1")
        GrantGuardResolver().resolve(ctx, ResolveDetails())
        assert not any(ps.status_type == StatusType.GUARD for ps in gm.player_1.statuses)

    def test_no_duplicate_guard(self):
        """Already has GUARD → no duplicate."""
        gm, ctx = _make_ctx(
            source="1",
            statuses_1=[
                PlayerStatus(StatusType.DEFENSE, 1),
                PlayerStatus(StatusType.GUARD, 1),
            ],
        )
        GrantGuardResolver().resolve(ctx, ResolveDetails())
        guard_count = sum(1 for ps in gm.player_1.statuses
                          if ps.status_type == StatusType.GUARD)
        assert guard_count == 1

    def test_get_priority_returns_12(self):
        """GrantGuardResolver priority is now 12 (before GrantDefense at 11)."""
        assert GrantGuardResolver().get_priority() == 12


# =============================================================================
# 6. HeavyDamageResolver
# =============================================================================


class TestHeavyDamageResolver:
    def test_deals_damage_without_guard(self):
        gm, ctx = _make_ctx(hp_2=10, source="1")
        HeavyDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 9

    def test_blocked_by_guard(self):
        gm, ctx = _make_ctx(hp_2=10, source="1",
                            statuses_2=[PlayerStatus(StatusType.GUARD, 1)])
        HeavyDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_not_blocked_by_defense(self):
        gm, ctx = _make_ctx(hp_2=10, source="1",
                            statuses_2=[PlayerStatus(StatusType.DEFENSE, 1)])
        HeavyDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 9

    def test_get_priority_returns_10(self):
        assert HeavyDamageResolver().get_priority() == 10


# =============================================================================
# 7. ReflectResolver
# =============================================================================


class TestReflectResolver:
    def test_reflects_attack(self):
        gm, ctx = _make_ctx(hp_2=10, source="1", opponent_card=CardType.ATTACK)
        ReflectResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 9
        assert gm.player_1.hp == 20

    def test_reflects_heavy(self):
        gm, ctx = _make_ctx(hp_2=10, source="1", opponent_card=CardType.HEAVY)
        ReflectResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 9

    def test_no_reflect_against_defense(self):
        gm, ctx = _make_ctx(hp_2=10, source="1", opponent_card=CardType.DEFENSE)
        ReflectResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_no_reflect_against_energy(self):
        gm, ctx = _make_ctx(hp_2=10, source="1", opponent_card=CardType.ENERGY)
        ReflectResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_get_priority_returns_10(self):
        assert ReflectResolver().get_priority() == 10


# =============================================================================
# 8. GainEnergyResolver
# =============================================================================


class TestGainEnergyResolver:
    def test_gains_one_energy(self):
        gm, ctx = _make_ctx(energy_1=0, source="1")
        GainEnergyResolver().resolve(ctx, ResolveDetails())
        assert gm.player_1.energy == 1

    def test_energy_clamped_by_player(self):
        gm, ctx = _make_ctx(energy_1=5, source="1")
        GainEnergyResolver().resolve(ctx, ResolveDetails())
        assert gm.player_1.energy == 5

    def test_get_priority_returns_10(self):
        assert GainEnergyResolver().get_priority() == 10


# =============================================================================
# 9. FlexDamageResolver
# =============================================================================


class TestFlexDamageResolver:
    def test_damage_without_defense(self):
        gm, ctx = _make_ctx(hp_2=10, source="1")
        FlexDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 9

    def test_blocked_by_defense(self):
        gm, ctx = _make_ctx(hp_2=10, source="1",
                            statuses_2=[PlayerStatus(StatusType.DEFENSE, 1)])
        FlexDamageResolver().resolve(ctx, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_get_priority_returns_10(self):
        assert FlexDamageResolver().get_priority() == 10


# =============================================================================
# 10. BuildResolver
# =============================================================================


class TestBuildResolver:
    def test_no_state_change(self):
        gm, ctx = _make_ctx(hp_1=20, hp_2=20, energy_1=0, source="1")
        details = ResolveDetails()
        BuildResolver().resolve(ctx, details)
        assert gm.player_1.hp == 20
        assert gm.player_2.hp == 20
        assert gm.player_1.energy == 0
        assert gm.player_1.statuses == []
        assert gm.player_2.statuses == []
        assert len(details.details) == 1

    def test_get_priority_returns_10(self):
        assert BuildResolver().get_priority() == 10


# =============================================================================
# 11. Priority Ordering
# =============================================================================


class TestPriorityOrdering:
    """GrantGuardResolver (12) runs before GrantDefenseResolver (11)."""

    def test_guard_before_defense(self):
        """DEFENSE card: GUARD (12) checked before DEFENSE (11) granted."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.statuses = [PlayerStatus(StatusType.DEFENSE, 1)]
        result = CardResolver.resolve(gm, [CardType.DEFENSE], [CardType.ENERGY])
        types = {ps.status_type for ps in gm.player_1.statuses}
        assert StatusType.DEFENSE in types
        assert StatusType.GUARD in types

    def test_defense_blocks_attack(self):
        """DEFENSE (pri=11) resolves before ATTACK (pri=10)."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20
        CardResolver.resolve(gm, [CardType.DEFENSE], [CardType.ATTACK])
        assert gm.player_1.hp == 20


# =============================================================================
# 12. CardResolver Orchestration
# =============================================================================


class TestCardResolverOrchestration:
    def test_resolve_returns_result_with_details(self):
        gm = GameCore(mode="pvp")
        gm.start_game()
        result = CardResolver.resolve(gm, [CardType.ATTACK], [CardType.DEFENSE])
        assert isinstance(result, ResolutionResult)
        assert len(result.details) > 0

    def test_full_resolution_attack_vs_counter(self):
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20
        CardResolver.resolve(gm, [CardType.ATTACK], [CardType.COUNTER])
        assert gm.player_1.hp == 19
        assert gm.player_2.hp == 20
        assert gm.player_2.has_status(StatusType.DEFENSE)

    def test_get_consecutive_attack(self):
        history = [PlayerHistoryItem(CardType.ATTACK, None),
                    PlayerHistoryItem(CardType.ATTACK, None),
                    PlayerHistoryItem(CardType.DEFENSE, None),
                    PlayerHistoryItem(CardType.ATTACK, None)]
        assert CardResolver.get_consecutive_attack(history) == 1

    def test_get_consecutive_attack_empty(self):
        assert CardResolver.get_consecutive_attack([]) == 0

    def test_get_consecutive_attack_all_attacks(self):
        history = [PlayerHistoryItem(CardType.ATTACK, None)] * 3
        assert CardResolver.get_consecutive_attack(history) == 3

    def test_get_consecutive_attack_counter_breaks(self):
        history = [PlayerHistoryItem(CardType.ATTACK, None),
                    PlayerHistoryItem(CardType.COUNTER, None)]
        assert CardResolver.get_consecutive_attack(history) == 0
