"""Unit tests for new/modified resolvers — intention #36 (TC-03 ~ TC-08).

Tests for: GainEnergyResolver (consecutive bonus), GrantFragileResolver (additive),
ReflectResolver (tag-based), RemoveDefenseResolver, BonusDefenseResolver.

Reuses existing resolver test helpers (_make_ctx, _build_tasks) where possible.
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
    AttackDamageResolver,
    BonusDefenseResolver,
    GainEnergyResolver,
    GrantDefenseResolver,
    GrantFragileResolver,
    GrantGuardResolver,
    HeavyDamageResolver,
    ReflectResolver,
    RemoveDefenseResolver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gm(**kwargs):
    """Create a fresh GameCore with two players."""
    gm = GameCore(mode="pvp")
    gm.start_game()
    for attr, value in kwargs.items():
        if attr.startswith("p1_"):
            setattr(gm.player_1, attr[3:], value)
        elif attr.startswith("p2_"):
            setattr(gm.player_2, attr[3:], value)
    return gm


def _ctx(gm, source, owner_card, opponent_card):
    """Build ResolveContext for a specific source player."""
    if source == "1":
        return ResolveContext(
            card_owner=gm.player_1,
            opponent=gm.player_2,
            card_played_by_owner=owner_card,
            card_played_by_opponent=opponent_card,
        )
    else:
        return ResolveContext(
            card_owner=gm.player_2,
            opponent=gm.player_1,
            card_played_by_owner=owner_card,
            card_played_by_opponent=opponent_card,
        )


# ===========================================================================
# TC-03: 连续击 (ATTACK) — unchanged formula, verify by resolver test
# ===========================================================================

class TestAttackDamageResolver36:
    """Verify AttackDamageResolver is unchanged by intention #36."""

    def test_first_attack_no_defense_deals_1(self):
        """TC-03-01: first consecutive attack deals 1 damage."""
        gm = _make_gm(p2_hp=20)
        ctx = _ctx(gm, "1", CardType.ATTACK, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        AttackDamageResolver().resolve(ctx, details)
        assert gm.player_2.hp == 19

    def test_second_consecutive_attack_deals_2(self):
        """TC-03-02: second consecutive attack deals 2 damage."""
        gm = _make_gm(
            p2_hp=20,
            p1_history=[PlayerHistoryItem(CardType.ATTACK)],
        )
        ctx = _ctx(gm, "1", CardType.ATTACK, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        AttackDamageResolver().resolve(ctx, details)
        assert gm.player_2.hp == 18

    def test_streak_capped_at_3(self):
        """TC-03-03: max damage is 3."""
        gm = _make_gm(
            p2_hp=20,
            p1_history=[
                PlayerHistoryItem(CardType.ATTACK),
                PlayerHistoryItem(CardType.ATTACK),
                PlayerHistoryItem(CardType.ATTACK),
            ],
        )
        ctx = _ctx(gm, "1", CardType.ATTACK, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        AttackDamageResolver().resolve(ctx, details)
        assert gm.player_2.hp == 17  # 20 - 3

    def test_blocked_by_defense(self):
        """TC-03-04: ATTACK blocked by opponent DEFENSE."""
        gm = _make_gm(p2_hp=20, p2_statuses=[PlayerStatus(StatusType.DEFENSE, 1)])
        ctx = _ctx(gm, "1", CardType.ATTACK, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        AttackDamageResolver().resolve(ctx, details)
        assert gm.player_2.hp == 20


# ===========================================================================
# TC-04: 格挡 (DEFENSE) — unchanged, verify GrantDefenseResolver + GrantGuardResolver
# ===========================================================================

class TestDefenseResolver36:
    """Verify DEFENSE resolvers unchanged by intention #36."""

    def test_grant_defense(self):
        """TC-04-01: DEFENSE grants +1 defense."""
        gm = _make_gm()
        ctx = _ctx(gm, "1", CardType.DEFENSE, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantDefenseResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.DEFENSE)

    def test_defense_blocked_by_fragile(self):
        """TC-04-02: FRAGILE blocks DEFENSE."""
        gm = _make_gm(p1_statuses=[PlayerStatus(StatusType.FRAGILE, 1)])
        ctx = _ctx(gm, "1", CardType.DEFENSE, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantDefenseResolver().resolve(ctx, details)
        assert not gm.player_1.has_status(StatusType.DEFENSE)

    def test_guard_triggered_after_defense(self):
        """TC-04-03: playing DEFENSE after defense-granting card → GUARD."""
        gm = _make_gm(
            p1_history=[PlayerHistoryItem(CardType.DEFENSE)],
        )
        ctx = _ctx(gm, "1", CardType.DEFENSE, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantGuardResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.GUARD)

    def test_no_guard_without_prior_defense(self):
        """TC-04-04: no GUARD if last card was not defense-granting."""
        gm = _make_gm(
            p1_history=[PlayerHistoryItem(CardType.ATTACK)],
        )
        ctx = _ctx(gm, "1", CardType.DEFENSE, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantGuardResolver().resolve(ctx, details)
        assert not gm.player_1.has_status(StatusType.GUARD)


# ===========================================================================
# TC-05: 破防重击 (HEAVY) — new RemoveDefenseResolver + additive fragile
# ===========================================================================

class TestHeavyResolver36:
    """Tests for HEAVY's new resolver chain."""

    def test_heavy_deals_1_damage_no_guard(self):
        """TC-05-01: HEAVY deals 1 damage if no GUARD."""
        gm = _make_gm(p2_hp=20)
        ctx = _ctx(gm, "1", CardType.HEAVY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        HeavyDamageResolver().resolve(ctx, details)
        assert gm.player_2.hp == 19

    def test_heavy_damage_blocked_by_guard(self):
        """TC-05-02: HEAVY damage blocked by GUARD."""
        gm = _make_gm(
            p2_hp=20,
            p2_statuses=[PlayerStatus(StatusType.GUARD, 1)],
        )
        ctx = _ctx(gm, "1", CardType.HEAVY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        HeavyDamageResolver().resolve(ctx, details)
        assert gm.player_2.hp == 20

    def test_remove_defense_resolver(self):
        """TC-05-03: RemoveDefenseResolver removes opponent DEFENSE."""
        gm = _make_gm(
            p2_statuses=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        ctx = _ctx(gm, "1", CardType.HEAVY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        RemoveDefenseResolver().resolve(ctx, details)
        assert not gm.player_2.has_status(StatusType.DEFENSE)

    def test_remove_defense_no_defense_no_error(self):
        """TC-05-04: RemoveDefenseResolver with no DEFENSE does not raise."""
        gm = _make_gm()
        ctx = _ctx(gm, "1", CardType.HEAVY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        RemoveDefenseResolver().resolve(ctx, details)

    def test_fragile_additive_with_existing(self):
        """TC-05-05: fragile +2 with existing FRAGILE(1) → lifetime=3."""
        gm = _make_gm(
            p2_statuses=[PlayerStatus(StatusType.FRAGILE, 1)],
        )
        ctx = _ctx(gm, "1", CardType.HEAVY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantFragileResolver().resolve(ctx, details)
        fragile = [s for s in gm.player_2.statuses if s.status_type == StatusType.FRAGILE]
        assert len(fragile) == 1
        assert fragile[0].lifetime == 3

    def test_fragile_from_scratch(self):
        """TC-05-06: fragile with no existing → lifetime=2."""
        gm = _make_gm()
        ctx = _ctx(gm, "1", CardType.HEAVY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantFragileResolver().resolve(ctx, details)
        assert gm.player_2.has_status(StatusType.FRAGILE)
        fragile = [s for s in gm.player_2.statuses if s.status_type == StatusType.FRAGILE]
        assert fragile[0].lifetime == 2

    def test_fragile_blocked_by_guard(self):
        """Fragile still blocked by GUARD."""
        gm = _make_gm(
            p2_statuses=[PlayerStatus(StatusType.GUARD, 1)],
        )
        ctx = _ctx(gm, "1", CardType.HEAVY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantFragileResolver().resolve(ctx, details)
        assert not gm.player_2.has_status(StatusType.FRAGILE)

    def test_guard_blocks_damage_and_fragile_but_not_defense_removal(self):
        """TC-05-07: GUARD blocks damage+fragile, but DEFENSE still removed."""
        gm = _make_gm(
            p2_hp=20,
            p2_statuses=[
                PlayerStatus(StatusType.GUARD, 1),
                PlayerStatus(StatusType.DEFENSE, 1),
            ],
        )
        ctx = _ctx(gm, "1", CardType.HEAVY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        # All three resolvers run
        HeavyDamageResolver().resolve(ctx, details)
        RemoveDefenseResolver().resolve(ctx, details)
        GrantFragileResolver().resolve(ctx, details)
        # Damage blocked
        assert gm.player_2.hp == 20
        # DEFENSE removed (GUARD doesn't block RemoveDefense)
        assert not gm.player_2.has_status(StatusType.DEFENSE)
        # Fragile blocked
        assert not gm.player_2.has_status(StatusType.FRAGILE)


# ===========================================================================
# TC-06: 防守反击 (COUNTER) — tag-based reflect + new GrantGuardResolver
# ===========================================================================

class TestCounterResolver36:
    """Tests for COUNTER's modified ReflectResolver and new GrantGuardResolver."""

    def test_grant_defense(self):
        """TC-06-01: COUNTER grants DEFENSE."""
        gm = _make_gm()
        ctx = _ctx(gm, "1", CardType.COUNTER, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantDefenseResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.DEFENSE)

    def test_reflect_against_attack_tag_no_defense(self):
        """TC-06-02: opponent plays ATTACK (攻 tag) w/o DEFENSE → reflect 1."""
        gm = _make_gm(p2_hp=20)
        ctx = _ctx(gm, "1", CardType.COUNTER, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        ReflectResolver().resolve(ctx, details)
        assert gm.player_2.hp == 19

    def test_reflect_blocked_by_opponent_defense(self):
        """TC-06-03: opponent has DEFENSE → no reflect."""
        gm = _make_gm(
            p2_hp=20,
            p2_statuses=[PlayerStatus(StatusType.DEFENSE, 1)],
        )
        ctx = _ctx(gm, "1", CardType.COUNTER, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        ReflectResolver().resolve(ctx, details)
        assert gm.player_2.hp == 20

    def test_no_reflect_non_attack_tag(self):
        """TC-06-04: opponent DEFENSE (no 攻 tag) → no reflect."""
        gm = _make_gm(p2_hp=20)
        ctx = _ctx(gm, "1", CardType.COUNTER, opponent_card=CardType.DEFENSE)
        details = ResolveDetails()
        ReflectResolver().resolve(ctx, details)
        assert gm.player_2.hp == 20

    def test_reflect_against_flex_attack_tag(self):
        """TC-06-05: opponent FLEX (has 攻 tag) w/o DEFENSE → reflect 1."""
        gm = _make_gm(p2_hp=20)
        ctx = _ctx(gm, "1", CardType.COUNTER, opponent_card=CardType.FLEX)
        details = ResolveDetails()
        ReflectResolver().resolve(ctx, details)
        assert gm.player_2.hp == 19

    def test_no_reflect_opponent_pass(self):
        """TC-06-06: opponent passes → no reflect."""
        gm = _make_gm(p2_hp=20)
        ctx = _ctx(gm, "1", CardType.COUNTER, opponent_card=None)
        details = ResolveDetails()
        ReflectResolver().resolve(ctx, details)
        assert gm.player_2.hp == 20

    def test_counter_guard_triggered(self):
        """TC-06-07: COUNTER after defense-granting card → GUARD."""
        gm = _make_gm(
            p1_history=[PlayerHistoryItem(CardType.DEFENSE)],
        )
        ctx = _ctx(gm, "1", CardType.COUNTER, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantGuardResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.GUARD)

    def test_counter_no_guard_without_prior_defense(self):
        """TC-06-08: COUNTER without prior defense → no GUARD."""
        gm = _make_gm(
            p1_history=[PlayerHistoryItem(CardType.ATTACK)],
        )
        ctx = _ctx(gm, "1", CardType.COUNTER, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantGuardResolver().resolve(ctx, details)
        assert not gm.player_1.has_status(StatusType.GUARD)


# ===========================================================================
# TC-07: 聚能 (ENERGY) — consecutive bonus
# ===========================================================================

class TestEnergyResolver36:
    """Tests for GainEnergyResolver (consecutive bonus)."""

    def test_first_energy_gains_1(self):
        """TC-07-01: first ENERGY in sequence → +1 energy."""
        gm = _make_gm(p1_energy=0)
        ctx = _ctx(gm, "1", CardType.ENERGY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GainEnergyResolver().resolve(ctx, details)
        assert gm.player_1.energy == 1

    def test_consecutive_energy_gains_2(self):
        """TC-07-02: second consecutive ENERGY → +2 energy."""
        gm = _make_gm(
            p1_energy=0,
            p1_history=[PlayerHistoryItem(CardType.ENERGY)],
        )
        ctx = _ctx(gm, "1", CardType.ENERGY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GainEnergyResolver().resolve(ctx, details)
        assert gm.player_1.energy == 2

    def test_consecutive_energy_capped_at_5(self):
        """TC-07-03: energy capped at 5 even with bonus."""
        gm = _make_gm(
            p1_energy=4,
            p1_history=[PlayerHistoryItem(CardType.ENERGY)],
        )
        ctx = _ctx(gm, "1", CardType.ENERGY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GainEnergyResolver().resolve(ctx, details)
        assert gm.player_1.energy == 5

    def test_non_consecutive_energy_after_other_card(self):
        """ENERGY after non-ENERGY card → +1 energy."""
        gm = _make_gm(
            p1_energy=0,
            p1_history=[PlayerHistoryItem(CardType.ATTACK)],
        )
        ctx = _ctx(gm, "1", CardType.ENERGY, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GainEnergyResolver().resolve(ctx, details)
        assert gm.player_1.energy == 1


# ===========================================================================
# TC-08: 攻守兼备 (FLEX) — bonus defense + unchanged damage
# ===========================================================================

class TestFlexResolver36:
    """Tests for FLEX's BonusDefenseResolver."""

    def test_flex_grant_defense(self):
        """TC-08-01: FLEX grants base DEFENSE."""
        gm = _make_gm()
        ctx = _ctx(gm, "1", CardType.FLEX, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        GrantDefenseResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.DEFENSE)

    def test_bonus_defense_after_defense_tag_card(self):
        """TC-08-02: last round played DEFENSE (防 tag) → bonus +1 defense."""
        gm = _make_gm(
            p1_history=[PlayerHistoryItem(CardType.DEFENSE)],
        )
        ctx = _ctx(gm, "1", CardType.FLEX, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        BonusDefenseResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.DEFENSE)

    def test_no_bonus_without_defense_tag_history(self):
        """TC-08-03: last round ATTACK (no 防 tag) → no bonus defense."""
        gm = _make_gm(
            p1_history=[PlayerHistoryItem(CardType.ATTACK)],
        )
        ctx = _ctx(gm, "1", CardType.FLEX, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        BonusDefenseResolver().resolve(ctx, details)
        assert not gm.player_1.has_status(StatusType.DEFENSE)

    def test_bonus_defense_after_counter(self):
        """Bonus defense triggers after COUNTER (防 tag)."""
        gm = _make_gm(
            p1_history=[PlayerHistoryItem(CardType.COUNTER)],
        )
        ctx = _ctx(gm, "1", CardType.FLEX, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        BonusDefenseResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.DEFENSE)

    def test_bonus_defense_after_flex(self):
        """Bonus defense triggers after FLEX (防 tag)."""
        gm = _make_gm(
            p1_history=[PlayerHistoryItem(CardType.FLEX)],
        )
        ctx = _ctx(gm, "1", CardType.FLEX, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        BonusDefenseResolver().resolve(ctx, details)
        assert gm.player_1.has_status(StatusType.DEFENSE)

    def test_bonus_defense_blocked_by_fragile(self):
        """Fragile blocks bonus defense."""
        gm = _make_gm(
            p1_statuses=[PlayerStatus(StatusType.FRAGILE, 1)],
            p1_history=[PlayerHistoryItem(CardType.DEFENSE)],
        )
        ctx = _ctx(gm, "1", CardType.FLEX, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        BonusDefenseResolver().resolve(ctx, details)
        assert not gm.player_1.has_status(StatusType.DEFENSE)

    def test_no_bonus_defense_first_round(self):
        """TC-08: first round has no history → no bonus defense."""
        gm = _make_gm()
        ctx = _ctx(gm, "1", CardType.FLEX, opponent_card=CardType.ATTACK)
        details = ResolveDetails()
        BonusDefenseResolver().resolve(ctx, details)
        assert not gm.player_1.has_status(StatusType.DEFENSE)
