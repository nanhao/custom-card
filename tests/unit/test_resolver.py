"""Unit tests for src/backend/game/resolver.py — CardResolver and ResolutionResult.

Covers TC-GROUP-1, TC-GROUP-2, TC-GROUP-3, TC-GROUP-4 from test_cases.md.
"""

import pytest
from src.backend.game.card import CardType
from src.backend.game.resolver import CardResolver, ResolutionResult

# Short aliases for readability
A = CardType.ATTACK
D = CardType.DEFENSE
H = CardType.HEAVY_STRIKE
C = CardType.COUNTER


# ---------------------------------------------------------------------------
# ResolutionResult tests
# ---------------------------------------------------------------------------

class TestResolutionResult:
    """Tests for the ResolutionResult value object."""

    def test_creation_and_attributes(self):
        """ResolutionResult should hold damage and details."""
        result = ResolutionResult(damage_a=1, damage_b=2, details=["A takes 1", "B takes 2"])
        assert result.damage_a == 1
        assert result.damage_b == 2
        assert result.details == ["A takes 1", "B takes 2"]

    def test_zero_damage(self):
        """ResolutionResult should handle zero damage."""
        result = ResolutionResult(damage_a=0, damage_b=0, details=["no damage"])
        assert result.damage_a == 0
        assert result.damage_b == 0


# ---------------------------------------------------------------------------
# Helper constants for all table-driven tests
# ---------------------------------------------------------------------------
E = []  # Empty history


# ---------------------------------------------------------------------------
# TC-GROUP-1: PRD Examples (7 cases)
# ---------------------------------------------------------------------------

PRD_EXAMPLES = [
    # (id, card_a, card_b, hist_a, hist_b, dmg_a, dmg_b, desc)
    ("TC1",  A, A, E, E, 1, 1, "Both Attack (×1), both take 1"),
    ("TC2",  D, D, E, E, 0, 0, "Both Defense, no damage"),
    ("TC3",  A, D, E, E, 0, 0, "Attack blocked by Defense"),
    ("TC4",  A, C, E, E, 1, 0, "Counter vs Attack, A takes 1"),
    ("TC5",  H, C, E, E, 1, 1, "Both deal 1, both take 1"),
    ("TC6",  A, A, [A], E, 1, 2, "A Attack ×2, B Attack ×1"),
    ("TC7",  D, H, [D], E, 0, 0, "Consecutive Defense blocks Heavy Strike"),
]


# ---------------------------------------------------------------------------
# TC-GROUP-2: Full Resolution Matrix — Round 1 (16 cases)
# ---------------------------------------------------------------------------

MATRIX_R1 = [
    # (id, card_a, card_b, dmg_a, dmg_b, reason)
    ("TC-M1",  A, A, 1, 1, "Both ×1 Attack"),
    ("TC-M2",  A, D, 0, 0, "Attack blocked by Defense"),
    ("TC-M3",  A, H, 1, 1, "Attack ×1 vs HS ×1"),
    ("TC-M4",  A, C, 1, 0, "Counter blocks Attack, deals 1"),
    ("TC-M5",  D, A, 0, 0, "Defense blocks Attack"),
    ("TC-M6",  D, D, 0, 0, "Both Defense"),
    ("TC-M7",  D, H, 1, 0, "No consecutive Defense → A takes 1"),
    ("TC-M8",  D, C, 0, 0, "No effect"),
    ("TC-M9",  H, A, 1, 1, "HS vs Attack, both deal 1"),
    ("TC-M10", H, D, 0, 1, "No consecutive Defense → B takes 1"),  # dmg_a=0: HS A→B 1, DEF B no dmg
    ("TC-M11", H, H, 1, 1, "Both HS, both take 1"),
    ("TC-M12", H, C, 1, 1, "Both deal 1"),
    ("TC-M13", C, A, 0, 1, "Counter deals 1 vs Attack"),
    ("TC-M14", C, D, 0, 0, "No effect"),
    ("TC-M15", C, H, 1, 1, "Counter vs HS, both take 1"),
    ("TC-M16", C, C, 0, 0, "Neither triggers"),
]


# ---------------------------------------------------------------------------
# TC-GROUP-3: Consecutive Attack Tracking (5 cases)
# ---------------------------------------------------------------------------

CONSEC_ATTACK = [
    # (id, card_a, card_b, hist_a, hist_b, dmg_a, dmg_b, desc)
    ("TC-A1", A, A, [A], E, 1, 2, "A ×2 consecutive"),
    ("TC-A2", A, A, [A, A], E, 1, 3, "A ×3 consecutive"),
    ("TC-A3", A, A, [D], [A], 2, 1, "A ×1, B ×2 (interrupted by Defense)"),
    ("TC-A4", A, A, [A, D, A], E, 1, 2, "A streak broken by Defense, now ×2"),
    ("TC-A5", A, D, [A, A], E, 0, 0, "Attack ×3 blocked by Defense"),
]


# ---------------------------------------------------------------------------
# TC-GROUP-4: Consecutive Defense vs Heavy Strike (7 cases)
# ---------------------------------------------------------------------------

CONSEC_DEFENSE = [
    # (id, card_a, card_b, hist_a, hist_b, dmg_a, dmg_b, desc)
    ("TC-D1", D, H, [D], E, 0, 0, "Consecutive Defense blocks HS"),
    ("TC-D2", D, H, [A], E, 1, 0, "No consecutive Defense, A takes 1"),
    ("TC-D3", D, H, [D, A, D], E, 0, 0, "Last round was Defense, blocks"),
    ("TC-D4", H, D, E, [D], 0, 0, "B consecutive Defense blocks A HS"),
    ("TC-D5", H, D, E, E, 0, 1, "B no prev Defense, R1 → B takes 1"),
    ("TC-D6", D, H, E, E, 1, 0, "R1, no prev → A takes 1"),
    ("TC-D7", D, H, [A, D, A], E, 1, 0, "Last card ATK, not DEF → A takes 1"),
]


# ---------------------------------------------------------------------------
# Parameterized test runners
# ---------------------------------------------------------------------------

class TestResolverPRDExamples:
    """TC-GROUP-1: Verify all 7 PRD examples."""

    @pytest.mark.parametrize(
        "tid,card_a,card_b,hist_a,hist_b,dmg_a,dmg_b,desc",
        PRD_EXAMPLES,
        ids=[t[0] for t in PRD_EXAMPLES],
    )
    def test_prd_example(self, tid, card_a, card_b, hist_a, hist_b, dmg_a, dmg_b, desc):
        result = CardResolver.resolve(card_a, card_b, list(hist_a), list(hist_b))
        assert result.damage_a == dmg_a, f"{tid}: expected dmg_a={dmg_a}, got {result.damage_a}"
        assert result.damage_b == dmg_b, f"{tid}: expected dmg_b={dmg_b}, got {result.damage_b}"
        assert len(result.details) > 0, f"{tid}: expected non-empty details"


class TestResolverMatrix:
    """TC-GROUP-2: Full 4×4 resolution matrix (round 1, empty histories)."""

    @pytest.mark.parametrize(
        "tid,card_a,card_b,dmg_a,dmg_b,reason",
        MATRIX_R1,
        ids=[t[0] for t in MATRIX_R1],
    )
    def test_matrix_r1(self, tid, card_a, card_b, dmg_a, dmg_b, reason):
        result = CardResolver.resolve(card_a, card_b, [], [])
        assert result.damage_a == dmg_a, f"{tid}: expected dmg_a={dmg_a}, got {result.damage_a}"
        assert result.damage_b == dmg_b, f"{tid}: expected dmg_b={dmg_b}, got {result.damage_b}"


class TestResolverConsecutiveAttack:
    """TC-GROUP-3: Consecutive attack tracking."""

    @pytest.mark.parametrize(
        "tid,card_a,card_b,hist_a,hist_b,dmg_a,dmg_b,desc",
        CONSEC_ATTACK,
        ids=[t[0] for t in CONSEC_ATTACK],
    )
    def test_consecutive_attack(self, tid, card_a, card_b, hist_a, hist_b, dmg_a, dmg_b, desc):
        result = CardResolver.resolve(card_a, card_b, list(hist_a), list(hist_b))
        assert result.damage_a == dmg_a, f"{tid}: expected dmg_a={dmg_a}, got {result.damage_a}"
        assert result.damage_b == dmg_b, f"{tid}: expected dmg_b={dmg_b}, got {result.damage_b}"


class TestResolverConsecutiveDefense:
    """TC-GROUP-4: Consecutive Defense vs Heavy Strike."""

    @pytest.mark.parametrize(
        "tid,card_a,card_b,hist_a,hist_b,dmg_a,dmg_b,desc",
        CONSEC_DEFENSE,
        ids=[t[0] for t in CONSEC_DEFENSE],
    )
    def test_consecutive_defense(self, tid, card_a, card_b, hist_a, hist_b, dmg_a, dmg_b, desc):
        result = CardResolver.resolve(card_a, card_b, list(hist_a), list(hist_b))
        assert result.damage_a == dmg_a, f"{tid}: expected dmg_a={dmg_a}, got {result.damage_a}"
        assert result.damage_b == dmg_b, f"{tid}: expected dmg_b={dmg_b}, got {result.damage_b}"


# ---------------------------------------------------------------------------
# Edge-case unit tests for resolver
# ---------------------------------------------------------------------------

class TestResolverEdgeCases:
    """TC-GROUP-6 edge cases for the resolver (TC-X9)."""

    def test_tc_x9_no_double_count_hs_vs_def(self):
        """TC-X9: HS vs DEF with empty history → dmg(DEF player)=1, not 2."""
        result = CardResolver.resolve(H, D, [], [])
        assert result.damage_b == 1  # DEF player takes 1 from HS
        assert result.damage_a == 0  # HS player takes 0 from DEF

    def test_defense_never_deals_damage_d7(self):
        """Architecture D7: Defense is pure-block — never deals damage."""
        # DEF vs ATTACK
        result = CardResolver.resolve(D, A, [], [])
        assert result.damage_a == 0
        assert result.damage_b == 0
        # DEF vs HEAVY (with prev DEF — blocks)
        result = CardResolver.resolve(D, H, [D], [])
        assert result.damage_a == 0
        assert result.damage_b == 0
        # DEF vs HEAVY (no prev DEF — still Defense deals 0, HS deals the 1)
        result = CardResolver.resolve(D, H, [], [])
        assert result.damage_a == 1  # From HS, not from DEF
        assert result.damage_b == 0


class TestHelperFunctions:
    """Tests for CardResolver helper functions."""

    def test_get_consecutive_attack_empty(self):
        """Empty history → 0 consecutive attacks."""
        assert CardResolver.get_consecutive_attack([]) == 0

    def test_get_consecutive_attack_single(self):
        """Single attack in history → 1."""
        assert CardResolver.get_consecutive_attack([A]) == 1

    def test_get_consecutive_attack_multiple(self):
        """Multiple consecutive attacks → correct count."""
        assert CardResolver.get_consecutive_attack([A, A, A]) == 3

    def test_get_consecutive_attack_interrupted(self):
        """Attack streak broken by non-attack → count from last attack only."""
        assert CardResolver.get_consecutive_attack([A, D, A]) == 1

    def test_get_consecutive_attack_non_attack_last(self):
        """No trailing attacks → 0."""
        assert CardResolver.get_consecutive_attack([A, D]) == 0

    def test_played_defense_last_round_empty(self):
        """Empty history → False."""
        assert CardResolver.played_defense_last_round([]) is False

    def test_played_defense_last_round_true(self):
        """Last card is Defense → True."""
        assert CardResolver.played_defense_last_round([A, D]) is True

    def test_played_defense_last_round_false(self):
        """Last card is not Defense → False."""
        assert CardResolver.played_defense_last_round([D, A]) is False
