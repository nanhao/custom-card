"""Tests for B3: StrongDefenseResolver uses extend_status instead of
granting a new independent DEFENSE status.

Per design/37 test_cases.md T6.3: When a player already has DEFENSE(lifetime=1),
playing Iron Defense (IRON_DEFENSE) should create a NEW DEFENSE(lifetime=2)
alongside the existing one, NOT merge them into DEFENSE(lifetime=3).

Note: Status lifetime tick happens AFTER card resolution (Step 8), so
DEFENSE(1) will expire after the round. Tests check after_resolve state
(before tick) to verify the statuses are independent.
"""

from src.backend.game.card_type import CardType
from src.backend.game.game_core import GameCore
from src.backend.game.player_state import StatusType, PlayerStatus


class TestB3StrongDefenseGrantsNewStatus:
    """B3: Iron Defense grants a NEW DEFENSE(2), not extending existing."""

    def test_iron_defense_creates_separate_defense_before_tick(self) -> None:
        """Player has DEFENSE(1), plays IRON_DEFENSE → after_resolve has 2 DEFENSE.

        Check after_resolve statuses (captured before tick at Step 6):
        Should have [DEFENSE(1), DEFENSE(2)], NOT merged [DEFENSE(3)].
        After tick, DEFENSE(1) expires (lifetime 1→0) and DEFENSE(2)
        becomes DEFENSE(1), so final state is 1 DEFENSE — that's expected.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.statuses = [PlayerStatus(StatusType.DEFENSE, lifetime=1)]
        gm.player_1.hand = [CardType.IRON_DEFENSE]
        gm.player_2.hand = [CardType.ENERGY]

        result = gm.play_round([CardType.IRON_DEFENSE], [CardType.ENERGY])

        # Check after_resolve (before status tick): both DEFENSE should exist
        after_statuses = result.after_resolve.statuses_1
        defense_in_after = [
            s for s in after_statuses if s.status_type == StatusType.DEFENSE
        ]
        assert len(defense_in_after) == 2, (
            f"B3 FAIL: Expected 2 DEFENSE in after_resolve, got {len(defense_in_after)}: "
            f"{[(s.status_type.name, s.lifetime) for s in after_statuses]}"
        )
        lifetimes = sorted([s.lifetime for s in defense_in_after])
        assert lifetimes == [1, 2], (
            f"B3 FAIL: Expected lifetimes [1, 2] in after_resolve, got {lifetimes}"
        )

    def test_iron_defense_with_existing_lifetime2_preserves_both(self) -> None:
        """Player has DEFENSE(2), plays IRON_DEFENSE → after tick, 2 DEFENSE remain.

        DEFENSE(2) → tick → DEFENSE(1), new DEFENSE(2) → tick → DEFENSE(1).
        Final: [DEFENSE(1), DEFENSE(1)] — two independent statuses survive.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.statuses = [PlayerStatus(StatusType.DEFENSE, lifetime=2)]
        gm.player_1.hand = [CardType.IRON_DEFENSE]
        gm.player_2.hand = [CardType.ENERGY]

        result = gm.play_round([CardType.IRON_DEFENSE], [CardType.ENERGY])

        # After full round (tick applied): both DEFENSE should survive with lifetime=1
        defense_statuses = [
            s for s in gm.player_1.statuses
            if s.status_type == StatusType.DEFENSE
        ]
        assert len(defense_statuses) == 2, (
            f"B3 FAIL: Expected 2 DEFENSE after round, got {len(defense_statuses)}: "
            f"{[(s.status_type.name, s.lifetime) for s in gm.player_1.statuses]}"
        )
        lifetimes = sorted([s.lifetime for s in defense_statuses])
        assert lifetimes == [1, 1], (
            f"B3 FAIL: Expected lifetimes [1, 1], got {lifetimes}"
        )

    def test_iron_defense_without_existing_defense(self) -> None:
        """Player has no DEFENSE, plays IRON_DEFENSE → single DEFENSE.

        After tick: DEFENSE(2) → DEFENSE(1).
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.IRON_DEFENSE]
        gm.player_2.hand = [CardType.ENERGY]

        gm.play_round([CardType.IRON_DEFENSE], [CardType.ENERGY])

        defense_statuses = [
            s for s in gm.player_1.statuses
            if s.status_type == StatusType.DEFENSE
        ]
        assert len(defense_statuses) == 1
        assert defense_statuses[0].lifetime == 1  # 2 - 1 tick = 1

    def test_iron_defense_blocked_by_fragile(self) -> None:
        """Player has FRAGILE → IRON_DEFENSE is blocked, no DEFENSE granted."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.statuses = [PlayerStatus(StatusType.FRAGILE, lifetime=1)]
        gm.player_1.hand = [CardType.IRON_DEFENSE]
        gm.player_2.hand = [CardType.ENERGY]

        gm.play_round([CardType.IRON_DEFENSE], [CardType.ENERGY])

        defense_statuses = [
            s for s in gm.player_1.statuses
            if s.status_type == StatusType.DEFENSE
        ]
        assert len(defense_statuses) == 0, (
            "IRON_DEFENSE should be blocked by FRAGILE"
        )

    def test_iron_defense_no_merge_with_defense3(self) -> None:
        """Player has DEFENSE(3), plays IRON_DEFENSE → after tick: DEFENSE(2)+DEFENSE(1).

        Bug would merge: DEFENSE(3+2=5) → tick → DEFENSE(4).
        Correct: DEFENSE(3) + new DEFENSE(2) → tick → DEFENSE(2) + DEFENSE(1).
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.statuses = [PlayerStatus(StatusType.DEFENSE, lifetime=3)]
        gm.player_1.hand = [CardType.IRON_DEFENSE]
        gm.player_2.hand = [CardType.ENERGY]

        gm.play_round([CardType.IRON_DEFENSE], [CardType.ENERGY])

        defense_statuses = [
            s for s in gm.player_1.statuses
            if s.status_type == StatusType.DEFENSE
        ]
        assert len(defense_statuses) == 2, (
            f"B3 FAIL: Expected 2 DEFENSE, got {len(defense_statuses)}"
        )
        lifetimes = sorted([s.lifetime for s in defense_statuses])
        assert lifetimes == [1, 2], (
            f"B3 FAIL: Expected lifetimes [1, 2], got {lifetimes}"
        )
