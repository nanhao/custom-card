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
# Helper: set up GameCore resolution context for individual resolver tests
# ---------------------------------------------------------------------------


def _setup_game_core(
    hp_1=20, hp_2=20,
    energy_1=0, energy_2=0,
    statuses_1=None, statuses_2=None,
    source="1", opponent_card=CardType.ATTACK,
    attack_streak=0, had_defense_pre=False,
):
    """Create a GameCore with players and resolution context set."""
    if statuses_1 is None:
        statuses_1 = []
    if statuses_2 is None:
        statuses_2 = []
    gm = GameCore(mode="pvp")
    gm.start_game()
    gm.player_1.hp = hp_1
    gm.player_2.hp = hp_2
    gm.player_1.energy = energy_1
    gm.player_2.energy = energy_2
    gm.player_1.statuses = list(statuses_1)
    gm.player_2.statuses = list(statuses_2)
    gm._resolve_source = source
    gm._resolve_opponent_card = opponent_card
    gm._resolve_attack_streak = attack_streak
    gm._resolve_had_defense_pre = had_defense_pre
    return gm


# =============================================================================
# 1. CardResolverRegistry — TC-RG-01 ~ TC-RG-04
# =============================================================================


class TestCardResolverRegistry:
    """TC-RG-01 ~ TC-RG-04: Registry correctness."""

    def test_attack_has_one_resolver(self):
        """TC-RG-01: ATTACK returns 1 Resolver (AttackDamageResolver)."""
        resolvers = CardResolverRegistry.get_resolvers(CardType.ATTACK)
        assert len(resolvers) == 1
        assert isinstance(resolvers[0], AttackDamageResolver)

    def test_defense_has_two_resolvers(self):
        """TC-RG-02: DEFENSE returns 2 Resolvers."""
        resolvers = CardResolverRegistry.get_resolvers(CardType.DEFENSE)
        assert len(resolvers) == 2
        types = {type(r) for r in resolvers}
        assert types == {GrantDefenseResolver, GrantGuardResolver}

    def test_all_card_types_registered(self):
        """TC-RG-03: All 7 CardTypes return non-empty lists."""
        expected = {
            CardType.ATTACK: 1,
            CardType.DEFENSE: 2,
            CardType.HEAVY: 1,
            CardType.COUNTER: 2,
            CardType.ENERGY: 1,
            CardType.BUILD: 1,
            CardType.FLEX: 2,
        }
        for ct, count in expected.items():
            resolvers = CardResolverRegistry.get_resolvers(ct)
            assert len(resolvers) == count, f"CardType.{ct.name} expected {count}, got {len(resolvers)}"
            assert len(resolvers) > 0

    def test_register_appends(self):
        """TC-RG-04: register() appends, does not overwrite."""
        before = len(CardResolverRegistry.get_resolvers(CardType.ATTACK))
        dummy = BuildResolver()
        CardResolverRegistry.register(CardType.ATTACK, dummy)
        after = len(CardResolverRegistry.get_resolvers(CardType.ATTACK))
        assert after == before + 1


# =============================================================================
# 2. Resolver base — TC-RS-BASE-01, TC-RS-BASE-02
# =============================================================================


class TestResolverBase:
    """TC-RS-BASE-01, TC-RS-BASE-02: Resolver ABC."""

    def test_resolver_is_abstract(self):
        """TC-RS-BASE-01: Cannot instantiate Resolver directly."""
        with pytest.raises(TypeError):
            Resolver()  # type: ignore[abstract]

    def test_subclass_must_implement_methods(self):
        """TC-RS-BASE-02: Subclass missing abstract methods raises TypeError."""

        class _Incomplete(Resolver):
            pass  # Missing get_priority and resolve

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]


# =============================================================================
# 3. AttackDamageResolver — TC-ATK-01 ~ TC-ATK-05
# =============================================================================


class TestAttackDamageResolver:
    """TC-ATK-01 ~ TC-ATK-05."""

    def test_basic_attack_streak_zero(self):
        """TC-ATK-01: streak=0 → damage=1."""
        gm = _setup_game_core(hp_2=20, source="1", attack_streak=0)
        initial_hp = gm.player_2.hp
        details = ResolveDetails()
        r = AttackDamageResolver()
        r.resolve(gm, details)
        assert gm.player_2.hp == initial_hp - 1
        assert len(details.details) == 1

    def test_streak_two_deals_three(self):
        """TC-ATK-02: streak=2 → damage=min(2+1, 3)=3."""
        gm = _setup_game_core(hp_2=10, source="1", attack_streak=2)
        r = AttackDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 7

    def test_streak_capped_at_three(self):
        """TC-ATK-03: streak=5 → damage=min(5+1, 3)=3."""
        gm = _setup_game_core(hp_2=10, source="1", attack_streak=5)
        r = AttackDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 7

    def test_attack_blocked_by_defense(self):
        """TC-ATK-04: DEFENSE status blocks attack."""
        gm = _setup_game_core(
            hp_2=10, source="1", attack_streak=0,
            statuses_2=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        r = AttackDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 10  # No damage

    def test_player_two_attacks(self):
        """TC-ATK-05: P2 attacks P1."""
        gm = _setup_game_core(hp_1=10, source="2", attack_streak=0)
        r = AttackDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_1.hp == 9

    def test_get_priority_returns_10(self):
        """AttackDamageResolver priority is 10."""
        assert AttackDamageResolver().get_priority() == 10


# =============================================================================
# 4. GrantDefenseResolver — TC-GD-01 ~ TC-GD-04
# =============================================================================


class TestGrantDefenseResolver:
    """TC-GD-01 ~ TC-GD-04."""

    def test_grants_defense_to_player_one(self):
        """TC-GD-01: P1 gains DEFENSE with lifetime=1."""
        gm = _setup_game_core(source="1")
        r = GrantDefenseResolver()
        r.resolve(gm, ResolveDetails())
        assert len(gm.player_1.statuses) == 1
        assert gm.player_1.statuses[0].status_type == StatusType.DEFENSE
        assert gm.player_1.statuses[0].lifetime == 1

    def test_grants_defense_to_player_two(self):
        """TC-GD-02: P2 gains DEFENSE."""
        gm = _setup_game_core(source="2")
        r = GrantDefenseResolver()
        r.resolve(gm, ResolveDetails())
        assert len(gm.player_2.statuses) == 1
        assert gm.player_2.statuses[0].status_type == StatusType.DEFENSE

    def test_get_priority_returns_11(self):
        """GrantDefenseResolver priority is 11."""
        assert GrantDefenseResolver().get_priority() == 11


# =============================================================================
# 5. GrantGuardResolver — TC-GG-01 ~ TC-GG-03
# =============================================================================


class TestGrantGuardResolver:
    """TC-GG-01 ~ TC-GG-03."""

    def test_grants_guard_with_pre_existing_defense(self):
        """TC-GG-01: had_defense_pre=True → GUARD granted."""
        gm = _setup_game_core(source="1", had_defense_pre=True)
        r = GrantGuardResolver()
        r.resolve(gm, ResolveDetails())
        assert any(
            ps.status_type == StatusType.GUARD for ps in gm.player_1.statuses
        )

    def test_no_guard_without_pre_defense(self):
        """TC-GG-02: had_defense_pre=False → no GUARD."""
        gm = _setup_game_core(source="1", had_defense_pre=False)
        r = GrantGuardResolver()
        r.resolve(gm, ResolveDetails())
        assert not any(
            ps.status_type == StatusType.GUARD for ps in gm.player_1.statuses
        )

    def test_no_duplicate_guard(self):
        """TC-GG-03: Already has GUARD → no duplicate."""
        gm = _setup_game_core(
            source="1", had_defense_pre=True,
            statuses_1=[PlayerStatus(StatusType.GUARD, 1)],
        )
        r = GrantGuardResolver()
        r.resolve(gm, ResolveDetails())
        guard_count = sum(
            1 for ps in gm.player_1.statuses if ps.status_type == StatusType.GUARD
        )
        assert guard_count == 1

    def test_get_priority_returns_11(self):
        """GrantGuardResolver priority is 11."""
        assert GrantGuardResolver().get_priority() == 11


# =============================================================================
# 6. HeavyDamageResolver — TC-HV-01 ~ TC-HV-03
# =============================================================================


class TestHeavyDamageResolver:
    """TC-HV-01 ~ TC-HV-03."""

    def test_deals_damage_without_guard(self):
        """TC-HV-01: No GUARD → 1 damage."""
        gm = _setup_game_core(hp_2=10, source="1")
        r = HeavyDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 9

    def test_blocked_by_guard(self):
        """TC-HV-02: GUARD blocks HEAVY."""
        gm = _setup_game_core(
            hp_2=10, source="1",
            statuses_2=[PlayerStatus(StatusType.GUARD, 1)],
        )
        r = HeavyDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_not_blocked_by_defense(self):
        """TC-HV-03: DEFENSE alone does not block HEAVY."""
        gm = _setup_game_core(
            hp_2=10, source="1",
            statuses_2=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        r = HeavyDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 9

    def test_get_priority_returns_10(self):
        """HeavyDamageResolver priority is 10."""
        assert HeavyDamageResolver().get_priority() == 10


# =============================================================================
# 7. ReflectResolver — TC-RF-01 ~ TC-RF-04
# =============================================================================


class TestReflectResolver:
    """TC-RF-01 ~ TC-RF-04."""

    def test_reflects_attack(self):
        """TC-RF-01: Opponent ATTACK → reflect 1 damage."""
        gm = _setup_game_core(hp_2=10, source="1", opponent_card=CardType.ATTACK)
        r = ReflectResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 9
        assert gm.player_1.hp == 20  # Source player unharmed

    def test_reflects_heavy(self):
        """TC-RF-02: Opponent HEAVY → reflect 1 damage."""
        gm = _setup_game_core(hp_2=10, source="1", opponent_card=CardType.HEAVY)
        r = ReflectResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 9

    def test_no_reflect_against_defense(self):
        """TC-RF-03: Opponent DEFENSE → no reflect."""
        gm = _setup_game_core(hp_2=10, source="1", opponent_card=CardType.DEFENSE)
        r = ReflectResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_no_reflect_against_energy(self):
        """TC-RF-04: Opponent ENERGY → no reflect."""
        gm = _setup_game_core(hp_2=10, source="1", opponent_card=CardType.ENERGY)
        r = ReflectResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_get_priority_returns_10(self):
        """ReflectResolver priority is 10."""
        assert ReflectResolver().get_priority() == 10


# =============================================================================
# 8. GainEnergyResolver — TC-GE-01 ~ TC-GE-02
# =============================================================================


class TestGainEnergyResolver:
    """TC-GE-01 ~ TC-GE-02."""

    def test_gains_one_energy(self):
        """TC-GE-01: energy increases by 1."""
        gm = _setup_game_core(energy_1=0, source="1")
        r = GainEnergyResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_1.energy == 1

    def test_energy_clamped_by_player(self):
        """TC-GE-02: energy=5 → stays at 5 (Player clamping)."""
        gm = _setup_game_core(energy_1=5, source="1")
        r = GainEnergyResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_1.energy == 5

    def test_get_priority_returns_10(self):
        """GainEnergyResolver priority is 10."""
        assert GainEnergyResolver().get_priority() == 10


# =============================================================================
# 9. FlexDamageResolver — TC-FD-01 ~ TC-FD-02
# =============================================================================


class TestFlexDamageResolver:
    """TC-FD-01 ~ TC-FD-02."""

    def test_damage_without_defense(self):
        """TC-FD-01: Opponent no DEFENSE → 1 damage."""
        gm = _setup_game_core(hp_2=10, source="1")
        r = FlexDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 9

    def test_blocked_by_defense(self):
        """TC-FD-02: Opponent has DEFENSE → no damage."""
        gm = _setup_game_core(
            hp_2=10, source="1",
            statuses_2=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        r = FlexDamageResolver()
        r.resolve(gm, ResolveDetails())
        assert gm.player_2.hp == 10

    def test_get_priority_returns_10(self):
        """FlexDamageResolver priority is 10."""
        assert FlexDamageResolver().get_priority() == 10


# =============================================================================
# 10. BuildResolver — TC-BD-01
# =============================================================================


class TestBuildResolver:
    """TC-BD-01."""

    def test_no_state_change(self):
        """TC-BD-01: BUILD does not modify HP, energy, or statuses."""
        gm = _setup_game_core(hp_1=20, hp_2=20, energy_1=0, source="1")
        initial_hp_1 = gm.player_1.hp
        initial_hp_2 = gm.player_2.hp
        initial_energy = gm.player_1.energy
        initial_statuses_1 = list(gm.player_1.statuses)
        initial_statuses_2 = list(gm.player_2.statuses)

        r = BuildResolver()
        details = ResolveDetails()
        r.resolve(gm, details)

        assert gm.player_1.hp == initial_hp_1
        assert gm.player_2.hp == initial_hp_2
        assert gm.player_1.energy == initial_energy
        assert gm.player_1.statuses == initial_statuses_1
        assert gm.player_2.statuses == initial_statuses_2
        assert len(details.details) == 1  # Stores description only

    def test_get_priority_returns_10(self):
        """BuildResolver priority is 10."""
        assert BuildResolver().get_priority() == 10


# =============================================================================
# 11. Priority Ordering — TC-PO-01 ~ TC-PO-02
# =============================================================================


class TestPriorityOrdering:
    """TC-PO-01 ~ TC-PO-02."""

    def test_priority_11_before_10(self):
        """TC-PO-01: DEFENSE (pri=11) resolves before ATTACK (pri=10)."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.ATTACK]
        gm.player_1.hp = 20
        gm.player_2.hp = 20

        result = CardResolver.resolve(
            game_core=gm,
            cards_1=[CardType.DEFENSE],
            cards_2=[CardType.ATTACK],
            p1_streak=0, p2_streak=0,
            p1_had_defense_pre=False,
            p2_had_defense_pre=False,
        )
        # P1 DEFENSE should block P2 ATTACK: P1 hp unchanged
        assert gm.player_1.hp == 20
        assert len(result.details) > 0

    def test_same_priority_order_independent(self):
        """TC-PO-02: Two priority-11 resolvers are independent."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20

        result = CardResolver.resolve(
            game_core=gm,
            cards_1=[CardType.DEFENSE],
            cards_2=[CardType.ENERGY],
            p1_streak=0, p2_streak=0,
            p1_had_defense_pre=True,
            p2_had_defense_pre=False,
        )
        # Both GrantDefense and GrantGuard should have run
        types = {ps.status_type for ps in gm.player_1.statuses}
        assert StatusType.DEFENSE in types
        assert StatusType.GUARD in types


# =============================================================================
# 12. CardResolver Orchestration — TC-CR-01 ~ TC-CR-04
# =============================================================================


class TestCardResolverOrchestration:
    """TC-CR-01 ~ TC-CR-04."""

    def test_resolve_returns_result_with_details(self):
        """TC-CR-01: CardResolver.resolve() returns ResolutionResult."""
        gm = GameCore(mode="pvp")
        gm.start_game()

        result = CardResolver.resolve(
            game_core=gm,
            cards_1=[CardType.ATTACK],
            cards_2=[CardType.DEFENSE],
            p1_streak=0, p2_streak=0,
            p1_had_defense_pre=False,
            p2_had_defense_pre=False,
        )
        assert isinstance(result, ResolutionResult)
        assert len(result.details) > 0

    def test_full_resolution_attack_vs_counter(self):
        """TC-CR-02: ATTACK vs COUNTER full resolution."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 20
        gm.player_2.hp = 20

        CardResolver.resolve(
            game_core=gm,
            cards_1=[CardType.ATTACK],
            cards_2=[CardType.COUNTER],
            p1_streak=0, p2_streak=0,
            p1_had_defense_pre=False,
            p2_had_defense_pre=False,
        )
        # P2 COUNTER grants DEFENSE and reflects ATTACK → P1 takes 1
        # P1 ATTACK blocked by P2 DEFENSE → P2 takes 0
        assert gm.player_1.hp == 19  # took reflect damage
        assert gm.player_2.hp == 20  # attack blocked by defense
        # P2 has DEFENSE from COUNTER's GrantDefenseResolver
        assert gm.player_2.has_status(StatusType.DEFENSE)

    def test_get_consecutive_attack(self):
        """TC-CR-03: get_consecutive_attack unchanged."""
        history = [
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.DEFENSE, None),
            PlayerHistoryItem(CardType.ATTACK, None),
        ]
        assert CardResolver.get_consecutive_attack(history) == 1

    def test_get_consecutive_attack_empty(self):
        """TC-CR-04: Empty history returns 0."""
        assert CardResolver.get_consecutive_attack([]) == 0

    def test_get_consecutive_attack_all_attacks(self):
        """Three consecutive attacks → 3."""
        history = [
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.ATTACK, None),
        ]
        assert CardResolver.get_consecutive_attack(history) == 3

    def test_get_consecutive_attack_counter_breaks(self):
        """COUNTER breaks the streak."""
        history = [
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.COUNTER, None),
        ]
        assert CardResolver.get_consecutive_attack(history) == 0
