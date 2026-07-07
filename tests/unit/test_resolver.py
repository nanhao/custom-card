"""Unit tests for src/backend/game/resolver.py — priority-layered card resolution.

Test cases per design/11 test_cases.md §3 (TC-RS-01 ~ TC-RS-11).
"""

from src.backend.game.card import CardType
from src.backend.game.resolver import CardResolver


class TestPriorityLayered:
    """TC-RS-01: Priority 11 resolves before priority 10."""

    def test_defense_blocks_attack(self):
        """P1 DEFENSE(11) vs P2 ATTACK(10); no pre-existing statuses."""
        result = CardResolver.resolve(
            cards_1=[CardType.DEFENSE],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_1 == 0
        assert result.damage_2 == 0
        assert "defense" in result.statuses_granted_1


class TestSamePriority:
    """TC-RS-02: Same-priority effects resolve simultaneously."""

    def test_attack_vs_attack_with_streak(self):
        """P1 ATTACK(10) streak=0, P2 ATTACK(10) streak=1."""
        result = CardResolver.resolve(
            cards_1=[CardType.ATTACK],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=1,
        )
        assert result.damage_1 == 2  # P2: streak 1 → 1+1=2
        assert result.damage_2 == 1  # P1: streak 0 → 0+1=1


class TestFlexMultiPriority:
    """TC-RS-03: FLEX dual-priority independent resolution."""

    def test_flex_vs_attack(self):
        """P1 FLEX vs P2 ATTACK."""
        result = CardResolver.resolve(
            cards_1=[CardType.FLEX],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_1 == 0
        assert result.damage_2 == 1
        assert "defense" in result.statuses_granted_1


class TestDefenseGrantGuard:
    """TC-RS-04: Defense grants guard when player already has defense."""

    def test_defense_with_pre_existing_defense(self):
        """P1 already has 'defense', P1 plays DEFENSE → gets '联防'."""
        result = CardResolver.resolve(
            cards_1=[CardType.DEFENSE],
            cards_2=[CardType.HEAVY],
            player_1_statuses=["defense"],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_1 == 0
        assert "联防" in result.statuses_granted_1


class TestCounterReflect:
    """TC-RS-05a~05d: Counter reflect mechanics."""

    def test_counter_vs_attack(self):
        """TC-RS-05a: COUNTER vs ATTACK."""
        result = CardResolver.resolve(
            cards_1=[CardType.COUNTER],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_1 == 0
        assert result.damage_2 == 1
        assert "defense" in result.statuses_granted_1

    def test_counter_vs_heavy(self):
        """TC-RS-05b: COUNTER vs HEAVY."""
        result = CardResolver.resolve(
            cards_1=[CardType.COUNTER],
            cards_2=[CardType.HEAVY],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_1 == 1  # HEAVY penetrates defense
        assert result.damage_2 == 1  # Reflect
        assert "defense" in result.statuses_granted_1

    def test_counter_vs_defense(self):
        """TC-RS-05c: COUNTER vs DEFENSE."""
        result = CardResolver.resolve(
            cards_1=[CardType.COUNTER],
            cards_2=[CardType.DEFENSE],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_1 == 0
        assert result.damage_2 == 0
        assert "defense" in result.statuses_granted_1
        assert "defense" in result.statuses_granted_2

    def test_counter_vs_energy(self):
        """TC-RS-05d: COUNTER vs ENERGY — no reflect."""
        result = CardResolver.resolve(
            cards_1=[CardType.COUNTER],
            cards_2=[CardType.ENERGY],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_1 == 0
        assert result.damage_2 == 0
        assert result.energy_delta_2 == 1
        assert "defense" in result.statuses_granted_1


class TestEnergyGain:
    """TC-RS-06, TC-RS-07: Energy gain mechanics."""

    def test_energy_gain(self):
        """TC-RS-06: ENERGY card gives +1 energy_delta."""
        result = CardResolver.resolve(
            cards_1=[CardType.ENERGY],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.energy_delta_1 == 1

    def test_energy_delta_unaffected_by_cap(self):
        """TC-RS-07: energy_delta always 1 (clamping done by Player)."""
        result = CardResolver.resolve(
            cards_1=[CardType.ENERGY],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.energy_delta_1 == 1


class TestBuildCard:
    """TC-RS-08: BUILD card produces no damage or energy in resolver."""

    def test_build_vs_attack(self):
        """P1 BUILD, P2 ATTACK."""
        result = CardResolver.resolve(
            cards_1=[CardType.BUILD],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_1 == 1  # P2 attack hits P1
        assert result.damage_2 == 0  # BUILD has no damage effect
        assert result.energy_delta_1 == 0
        assert result.energy_delta_2 == 0
        assert result.statuses_granted_1 == []
        assert result.statuses_granted_2 == []


class TestAttackStreak:
    """TC-RS-09a~09d: Consecutive attack counting."""

    def test_streak_normal(self):
        """TC-RS-09a: streak=2 → damage = min(2+1,3) = 3."""
        result = CardResolver.resolve(
            cards_1=[CardType.ATTACK],
            cards_2=[CardType.ENERGY],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=2,
            p2_attack_streak=0,
        )
        assert result.damage_2 == 3

    def test_streak_max_cap(self):
        """streak=5 → damage = min(5+1,3) = 3."""
        result = CardResolver.resolve(
            cards_1=[CardType.ATTACK],
            cards_2=[CardType.ENERGY],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=5,
            p2_attack_streak=0,
        )
        assert result.damage_2 == 3

    def test_streak_zero(self):
        """TC-RS-09c: streak=0 → damage=1."""
        result = CardResolver.resolve(
            cards_1=[CardType.ATTACK],
            cards_2=[CardType.ENERGY],
            player_1_statuses=[],
            player_2_statuses=[],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_2 == 1

    def test_attack_blocked_by_defense(self):
        """Attack blocked by defense status — 0 damage regardless of streak."""
        result = CardResolver.resolve(
            cards_1=[CardType.ATTACK],
            cards_2=[CardType.DEFENSE],
            player_1_statuses=[],
            player_2_statuses=["defense"],
            p1_attack_streak=3,
            p2_attack_streak=0,
        )
        assert result.damage_2 == 0


class TestHeavyStrike:
    """TC-RS-10, TC-RS-11: Heavy strike mechanics."""

    def test_heavy_blocked_by_guard(self):
        """TC-RS-10: HEAVY blocked by '联防'."""
        result = CardResolver.resolve(
            cards_1=[CardType.HEAVY],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=["联防"],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_2 == 0

    def test_heavy_not_blocked_by_defense(self):
        """TC-RS-11: HEAVY NOT blocked by plain 'defense'."""
        result = CardResolver.resolve(
            cards_1=[CardType.HEAVY],
            cards_2=[CardType.ATTACK],
            player_1_statuses=[],
            player_2_statuses=["defense"],
            p1_attack_streak=0,
            p2_attack_streak=0,
        )
        assert result.damage_2 == 1


class TestGetConsecutiveAttack:
    """Static get_consecutive_attack helper."""

    def test_continuous_attack_count(self):
        """Count consecutive ATTACK from end of history."""
        history = [CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE, CardType.ATTACK]
        assert CardResolver.get_consecutive_attack(history) == 1

    def test_counter_breaks_streak(self):
        """TC-RS-09d: COUNTER breaks attack streak."""
        history = [CardType.ATTACK, CardType.COUNTER]
        assert CardResolver.get_consecutive_attack(history) == 0

    def test_empty_history(self):
        """Empty history → streak = 0."""
        assert CardResolver.get_consecutive_attack([]) == 0

    def test_defense_breaks_streak(self):
        """DEFENSE breaks attack streak."""
        history = [CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE]
        assert CardResolver.get_consecutive_attack(history) == 0
