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
        expected = {CardType.ATTACK: 1, CardType.DEFENSE: 2, CardType.HEAVY: 1,
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
# 12. CardResolver Orchestration
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
