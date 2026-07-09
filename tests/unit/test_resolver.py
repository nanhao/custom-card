"""Unit tests for resolver.py — Resolver strategy pattern (design/22).

Per design/22: CardResolverRegistry removed; card → CardType mapping
now lives in CardDef.resolvers.
Test cases per design/22 test_cases.md.
"""

import pytest

from src.backend.game.card_type import CardType
from src.backend.game.card_def import get_card_def
from src.backend.game.game_core import GameCore
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus
from src.backend.game.resolver import (
    ResolveDetails,
    ResolveContext,
    ResolutionResult,
    Resolver,
    CardResolver,
    AttackDamageResolver,
    GrantDefenseResolver,
    GrantGuardResolver,
    GrantFragileResolver,
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
        )
    else:
        return gm, ResolveContext(
            card_owner=gm.player_2, opponent=gm.player_1,
            card_played_by_owner=owner_card,
            card_played_by_opponent=opponent_card,
        )


def _build_tasks(player_1, player_2, cards_1, cards_2):
    """Build (Resolver, ResolveContext) tasks for CardResolver.resolve().

    Per design/22: GameCore.play_round() builds tasks from
    get_card_def(card_type).resolvers; tests mirror this pattern.
    """
    from src.backend.game.player import Player  # noqa: F811

    tasks = []
    for card_type in cards_1:
        ctx = ResolveContext(
            card_owner=player_1,
            opponent=player_2,
            card_played_by_owner=card_type,
            card_played_by_opponent=cards_2[0],
        )
        for resolver_cls in get_card_def(card_type).resolvers:
            tasks.append((resolver_cls(), ctx))
    for card_type in cards_2:
        ctx = ResolveContext(
            card_owner=player_2,
            opponent=player_1,
            card_played_by_owner=card_type,
            card_played_by_opponent=cards_1[0],
        )
        for resolver_cls in get_card_def(card_type).resolvers:
            tasks.append((resolver_cls(), ctx))
    return tasks


# =============================================================================
# 1. CardDef.resolvers — replaces CardResolverRegistry (TC-RG-01 ~ TC-RG-03)
# =============================================================================


class TestCardDefResolvers:
    """Per design/22: CardDef.resolvers replaces CardResolverRegistry."""

    def test_attack_has_one_resolver(self):
        """TC-RG-01: ATTACK has AttackDamageResolver."""
        cd = get_card_def(CardType.ATTACK)
        assert len(cd.resolvers) == 1
        assert cd.resolvers[0] is AttackDamageResolver

    def test_defense_has_two_resolvers(self):
        """TC-RG-02: DEFENSE has GrantDefenseResolver + GrantGuardResolver."""
        cd = get_card_def(CardType.DEFENSE)
        assert len(cd.resolvers) == 2
        assert set(cd.resolvers) == {GrantDefenseResolver, GrantGuardResolver}

    def test_all_card_types_have_resolvers(self):
        """TC-RG-03: every CardType has correct resolver counts."""
        expected = {CardType.ATTACK: 1, CardType.DEFENSE: 2, CardType.HEAVY: 2,
                    CardType.COUNTER: 2, CardType.ENERGY: 1, CardType.BUILD: 1,
                    CardType.FLEX: 2}
        for ct, count in expected.items():
            cd = get_card_def(ct)
            assert len(cd.resolvers) == count, f"{ct.name}: expected {count}, got {len(cd.resolvers)}"

    def test_registry_file_deleted(self):
        """TC-RG-03: registry.py no longer exists."""
        from pathlib import Path
        registry_path = Path(__file__).parent.parent.parent / \
            "src" / "backend" / "game" / "resolver" / "registry.py"
        assert not registry_path.exists(), "registry.py should be deleted"

    def test_card_resolver_registry_cannot_be_imported(self):
        """TC-RG-01, TC-RG-02: CardResolverRegistry is not importable."""
        with pytest.raises(ImportError):
            from src.backend.game.resolver import CardResolverRegistry  # noqa: F811
        with pytest.raises(ImportError):
            from src.backend.game.resolver.registry import CardResolverRegistry  # noqa: F811


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
# 10. ResolveContext — build_target field (design/25)
# =============================================================================


class TestResolveContextBuildTarget:
    """TC-CTX-01 ~ TC-CTX-02: ResolveContext build_target field."""

    def test_build_target_defaults_to_none(self):
        """TC-CTX-01: default value is None when not specified."""
        gm, ctx = _make_ctx(source="1")
        assert ctx.build_target is None

    def test_build_target_explicitly_set(self):
        """TC-CTX-02: can explicitly set build_target."""
        gm, ctx = _make_ctx(
            source="1",
            owner_card=CardType.BUILD,
        )
        ctx.build_target = CardType.HEAVY
        assert ctx.build_target == CardType.HEAVY


# =============================================================================
# 11. BuildResolver
# =============================================================================


class TestBuildResolver:
    """TC-BLD-01 ~ TC-BLD-06: BuildResolver settlement logic (design/25)."""

    def test_build_attack_cost_1(self):
        """TC-BLD-01: BUILD ATTACK (cost=1), energy 2 -> 1, card added to discard."""
        gm, ctx = _make_ctx(
            hp_1=20, energy_1=2, source="1",
            owner_card=CardType.BUILD,
        )
        ctx.build_target = CardType.ATTACK
        gm.player_1.discard_pile = []
        details = ResolveDetails()
        BuildResolver().resolve(ctx, details)
        assert gm.player_1.energy == 1
        assert CardType.ATTACK in gm.player_1.discard_pile
        assert len(gm.player_1.discard_pile) == 1
        assert any("BUILD attack (cost 1)" in d for d in details.details)

    def test_build_heavy_cost_2(self):
        """TC-BLD-02: BUILD HEAVY (cost=2), energy 2 -> 0, card added to discard."""
        gm, ctx = _make_ctx(
            hp_1=20, energy_1=2, source="1",
            owner_card=CardType.BUILD,
        )
        ctx.build_target = CardType.HEAVY
        gm.player_1.discard_pile = []
        details = ResolveDetails()
        BuildResolver().resolve(ctx, details)
        assert gm.player_1.energy == 0
        assert CardType.HEAVY in gm.player_1.discard_pile
        assert any("BUILD heavy (cost 2)" in d for d in details.details)

    def test_build_flex_with_existing_discard(self):
        """TC-BLD-03: BUILD FLEX (cost=2) with existing discard, energy 5 -> 3."""
        gm, ctx = _make_ctx(
            hp_1=20, energy_1=5, source="1",
            owner_card=CardType.BUILD,
        )
        ctx.build_target = CardType.FLEX
        gm.player_1.discard_pile = [CardType.ATTACK]
        details = ResolveDetails()
        BuildResolver().resolve(ctx, details)
        assert gm.player_1.energy == 3
        assert gm.player_1.discard_pile == [CardType.ATTACK, CardType.FLEX]

    def test_build_target_none_is_noop(self):
        """TC-BLD-04: build_target is None -> no state changes (defensive)."""
        gm, ctx = _make_ctx(
            hp_1=20, energy_1=2, source="1",
            owner_card=CardType.BUILD,
        )
        ctx.build_target = None
        gm.player_1.discard_pile = []
        details = ResolveDetails()
        BuildResolver().resolve(ctx, details)
        assert gm.player_1.energy == 2
        assert gm.player_1.discard_pile == []

    def test_build_energy_detail_recorded(self):
        """TC-BLD-05: detail message includes target card and cost."""
        gm, ctx = _make_ctx(
            hp_1=20, energy_1=3, source="1",
            owner_card=CardType.BUILD,
        )
        ctx.build_target = CardType.ENERGY
        details = ResolveDetails()
        BuildResolver().resolve(ctx, details)
        assert len(details.details) == 1
        assert "BUILD" in details.details[0]
        assert "cost 1" in details.details[0]

    def test_get_priority_returns_10(self):
        """TC-BLD-06: priority is 10."""
        assert BuildResolver().get_priority() == 10


# =============================================================================
# 12. Priority Ordering
# =============================================================================


class TestPriorityOrdering:
    """GrantGuardResolver (12) runs before GrantDefenseResolver (11)."""

    def test_guard_before_defense(self):
        """DEFENSE card: GUARD (12) checked before DEFENSE (11) granted."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.statuses = [PlayerStatus(StatusType.DEFENSE, 1)]
        tasks = _build_tasks(gm.player_1, gm.player_2, [CardType.DEFENSE], [CardType.ENERGY])
        result = CardResolver.resolve(tasks)
        types = {ps.status_type for ps in gm.player_1.statuses}
        assert StatusType.DEFENSE in types
        assert StatusType.GUARD in types

    def test_defense_blocks_attack(self):
        """DEFENSE (pri=11) resolves before ATTACK (pri=10)."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20
        tasks = _build_tasks(gm.player_1, gm.player_2, [CardType.DEFENSE], [CardType.ATTACK])
        CardResolver.resolve(tasks)
        assert gm.player_1.hp == 20


# =============================================================================
# 13. CardResolver Orchestration
# =============================================================================


class TestCardResolverOrchestration:
    def test_resolve_returns_result_with_details(self):
        gm = GameCore(mode="pvp")
        gm.start_game()
        tasks = _build_tasks(gm.player_1, gm.player_2, [CardType.ATTACK], [CardType.DEFENSE])
        result = CardResolver.resolve(tasks)
        assert isinstance(result, ResolutionResult)
        assert len(result.details) > 0

    def test_full_resolution_attack_vs_counter(self):
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20
        tasks = _build_tasks(gm.player_1, gm.player_2, [CardType.ATTACK], [CardType.COUNTER])
        CardResolver.resolve(tasks)
        assert gm.player_1.hp == 19
        assert gm.player_2.hp == 20
        assert gm.player_2.has_status(StatusType.DEFENSE)

    def test_get_consecutive_attack(self):
        """Player.get_consecutive_attack_count() — only tail ATTACK counted."""
        history = [PlayerHistoryItem(CardType.ATTACK, None),
                    PlayerHistoryItem(CardType.ATTACK, None),
                    PlayerHistoryItem(CardType.DEFENSE, None),
                    PlayerHistoryItem(CardType.ATTACK, None)]
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.history = history
        assert gm.player_1.get_consecutive_attack_count() == 1

    def test_get_consecutive_attack_empty(self):
        """Player.get_consecutive_attack_count() — empty history returns 0."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        assert gm.player_1.get_consecutive_attack_count() == 0

    def test_get_consecutive_attack_all_attacks(self):
        """Player.get_consecutive_attack_count() — all ATTACK returns count."""
        history = [PlayerHistoryItem(CardType.ATTACK, None)] * 3
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.history = history
        assert gm.player_1.get_consecutive_attack_count() == 3

    def test_get_consecutive_attack_counter_breaks(self):
        """Player.get_consecutive_attack_count() — non-ATTACK breaks streak."""
        history = [PlayerHistoryItem(CardType.ATTACK, None),
                    PlayerHistoryItem(CardType.COUNTER, None)]
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.history = history
        assert gm.player_1.get_consecutive_attack_count() == 0


# =============================================================================
# 14. GrantFragileResolver — design/24
# =============================================================================


class TestGrantFragileResolver:
    """TC-GF-01 ~ TC-GF-04: GrantFragileResolver."""

    def test_priority_is_10(self):
        """TC-GF-01: priority returns 10."""
        assert GrantFragileResolver().get_priority() == 10

    def test_grants_fragile_to_opponent(self):
        """TC-GF-02: grants FRAGILE (lifetime=2) when opponent has no GUARD."""
        gm, ctx = _make_ctx(source="1")
        details = ResolveDetails()
        GrantFragileResolver().resolve(ctx, details)
        assert gm.player_2.has_status(StatusType.FRAGILE)
        fragile = [ps for ps in gm.player_2.statuses if ps.status_type == StatusType.FRAGILE]
        assert len(fragile) == 1
        assert fragile[0].lifetime == 2
        assert any("gains FRAGILE" in d for d in details.details)

    def test_blocked_by_opponent_guard(self):
        """TC-GF-03: blocked when opponent has GUARD."""
        gm, ctx = _make_ctx(
            source="1",
            statuses_2=[PlayerStatus(StatusType.GUARD, 1)],
        )
        details = ResolveDetails()
        GrantFragileResolver().resolve(ctx, details)
        assert not gm.player_2.has_status(StatusType.FRAGILE)
        assert any("blocked by opponent GUARD" in d for d in details.details)

    def test_not_blocked_by_defense_alone(self):
        """FRAGILE is blocked by GUARD only, not DEFENSE."""
        gm, ctx = _make_ctx(
            source="1",
            statuses_2=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        details = ResolveDetails()
        GrantFragileResolver().resolve(ctx, details)
        assert gm.player_2.has_status(StatusType.FRAGILE)


# =============================================================================
# 15. FRAGILE blocks DEFENSE/ GUARD — design/24
# =============================================================================


class TestFragileBlocksDefense:
    """TC-GD-01 ~ TC-GD-02: FRAGILE blocks GrantDefenseResolver."""

    def test_defense_blocked_by_fragile(self):
        """TC-GD-01: with FRAGILE, DEFENSE is not granted."""
        gm, ctx = _make_ctx(
            source="1",
            statuses_1=[PlayerStatus(StatusType.FRAGILE, 2)],
        )
        details = ResolveDetails()
        GrantDefenseResolver().resolve(ctx, details)
        assert not gm.player_1.has_status(StatusType.DEFENSE)
        assert any("blocked by FRAGILE" in d for d in details.details)

    def test_defense_granted_without_fragile(self):
        """TC-GD-02: without FRAGILE, DEFENSE is granted normally."""
        gm, ctx = _make_ctx(source="1")
        details = ResolveDetails()
        GrantDefenseResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.DEFENSE)


class TestFragileBlocksGuard:
    """TC-GG-01 ~ TC-GG-02: FRAGILE blocks GrantGuardResolver."""

    def test_guard_blocked_by_fragile(self):
        """TC-GG-01: with FRAGILE and pre-round DEFENSE, GUARD is not granted."""
        gm, ctx = _make_ctx(
            source="1",
            statuses_1=[
                PlayerStatus(StatusType.DEFENSE, 1),
                PlayerStatus(StatusType.FRAGILE, 2),
            ],
        )
        details = ResolveDetails()
        GrantGuardResolver().resolve(ctx, details)
        assert not gm.player_1.has_status(StatusType.GUARD)
        assert any("blocked by FRAGILE" in d for d in details.details)

    def test_guard_granted_without_fragile(self):
        """TC-GG-02: with pre-round DEFENSE and no FRAGILE, GUARD is granted."""
        gm, ctx = _make_ctx(
            source="1",
            statuses_1=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        details = ResolveDetails()
        GrantGuardResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.GUARD)


# =============================================================================
# 16. Priority interaction — design/24 D1
# =============================================================================


class TestFragilePriorityInteraction:
    """TC-PR-01 ~ TC-PR-02: Priority ensures same-round defense before fragile."""

    def test_heavy_vs_defense_same_round(self):
        """TC-PR-01: DEFENSE (pri=11) granted before FRAGILE (pri=10) applied."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20
        tasks = _build_tasks(gm.player_1, gm.player_2, [CardType.DEFENSE], [CardType.HEAVY])
        CardResolver.resolve(tasks)
        # P1: DEFENSE granted (pri=11 before pri=10 fragile)
        assert gm.player_1.has_status(StatusType.DEFENSE)
        # P1: FRAGILE applied (HEAVY's GrantFragileResolver pri=10, no GUARD)
        assert gm.player_1.has_status(StatusType.FRAGILE)
        # P1: HEAVY damage (1) applied (no GUARD)
        assert gm.player_1.hp == 19

    def test_heavy_vs_no_defense(self):
        """TC-PR-02: HEAVY without opposing defense, both effects apply."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20
        tasks = _build_tasks(gm.player_1, gm.player_2, [CardType.ATTACK], [CardType.HEAVY])
        CardResolver.resolve(tasks)
        # P1 takes HEAVY damage
        assert gm.player_1.hp == 19
        # P1 gains FRAGILE
        assert gm.player_1.has_status(StatusType.FRAGILE)
        # P2 takes ATTACK damage
        assert gm.player_2.hp == 19

    def test_heavy_vs_guard_blocks_both(self):
        """GUARD blocks both damage AND fragile from HEAVY."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20
        gm.player_1.statuses = [PlayerStatus(StatusType.GUARD, 1)]
        tasks = _build_tasks(gm.player_1, gm.player_2, [CardType.ENERGY], [CardType.HEAVY])
        CardResolver.resolve(tasks)
        # P1 HP unchanged (GUARD blocks HEAVY damage)
        assert gm.player_1.hp == 20
        # P1 no FRAGILE (GUARD blocks GrantFragileResolver)
        assert not gm.player_1.has_status(StatusType.FRAGILE)
