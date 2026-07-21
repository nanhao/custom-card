"""B5: GrantGuardResolver does not falsely trigger after FRAGILE blocks DEFENSE.

Bug: GrantGuardResolver checks if last round's card has '防' tag as a proxy for
"did player gain DEFENSE last round". When FRAGILE blocks DEFENSE gain, the tag
check still returns True → GUARD is wrongly granted.

Fix: Use had_defense_after field (set by GameCore after resolution) instead of
tag check. This reflects whether DEFENSE was ACTUALLY granted.
"""

from src.backend.game.card_type import CardType
from src.backend.game.game_core import GameCore
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import PlayerStatus, StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails
from src.backend.game.resolver.grant_guard_resolver import GrantGuardResolver


class TestB5GuardFragileFalseTrigger:
    """B5: GUARD must NOT be granted when last round's DEFENSE was blocked by FRAGILE."""

    def test_fragile_blocks_defense_then_no_guard_integration(self):
        """Full integration: 3-round scenario proving the bug.

        R1: P1 plays HEAVY → P2 gets FRAGILE(2)
        R1 tick: FRAGILE(1) — still active
        R2: P2 plays FLEX (has '防' tag), GrantDefenseResolver blocked by FRAGILE
            → P2 does NOT gain DEFENSE. had_defense_after=False.
        R2 tick: FRAGILE expires
        R3: P2 plays DEFENSE → GrantGuardResolver should check had_defense_after
            from R2 (which is False), so GUARD should NOT be granted.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()

        # R1: P1 HEAVY, P2 pass
        gm.player_1.hand = [CardType.HEAVY]
        gm.player_2.hand = []
        result_r1 = gm.play_round([CardType.HEAVY], [])

        # After R1: P2 has FRAGILE(2) from HEAVY.
        p2_statuses_after_r1 = [s.status_type for s in result_r1.after_end_actions.statuses_2]
        assert StatusType.FRAGILE in p2_statuses_after_r1, (
            f"Expected P2 to have FRAGILE after R1, got {p2_statuses_after_r1}"
        )

        # R2: P1 pass, P2 plays FLEX (has '防' tag but FRAGILE blocks DEFENSE)
        gm.player_1.hand = []
        gm.player_2.hand = [CardType.FLEX]
        result_r2 = gm.play_round([], [CardType.FLEX])

        # After R2: P2 should NOT have DEFENSE (blocked by FRAGILE).
        r2_after_statuses = set(s.status_type for s in result_r2.after_resolve.statuses_2)
        assert StatusType.DEFENSE not in r2_after_statuses, (
            f"P2 should NOT have DEFENSE after R2 (blocked by FRAGILE), "
            f"got {r2_after_statuses}"
        )
        # P2's had_defense_after for R2 must be False.
        assert gm.player_2.history[-1].had_defense_after is False, (
            "had_defense_after must be False when DEFENSE was blocked by FRAGILE"
        )

        # R3: P2 plays DEFENSE → GrantGuardResolver should NOT grant GUARD.
        gm.player_1.hand = []
        gm.player_2.hand = [CardType.DEFENSE]
        result_r3 = gm.play_round([], [CardType.DEFENSE])

        # Check that GUARD was NOT granted to P2.
        r3_after_statuses = set(s.status_type for s in result_r3.after_resolve.statuses_2)
        assert StatusType.GUARD not in r3_after_statuses, (
            f"BUG: GUARD was granted despite P2 not having actual DEFENSE last round. "
            f"Statuses: {r3_after_statuses}"
        )

    def test_had_defense_after_false_prevents_guard_direct(self):
        """Direct resolver test: had_defense_after=False → no GUARD granted."""
        gm = GameCore(mode="pvp")
        gm.start_game()

        # Simulate: last round player played FLEX but FRAGILE blocked DEFENSE.
        gm.player_1.history = [
            PlayerHistoryItem(CardType.FLEX, None, had_defense_after=False)
        ]

        ctx = ResolveContext(
            card_owner=gm.player_1,
            opponent=gm.player_2,
            card_played_by_owner=CardType.DEFENSE,
            card_played_by_opponent=None,
        )
        details = ResolveDetails()
        GrantGuardResolver().resolve(ctx, details)

        # GUARD must NOT be granted because had_defense_after was False.
        assert not gm.player_1.has_status(StatusType.GUARD), (
            "BUG: GUARD granted when had_defense_after=False"
        )

    def test_had_defense_after_true_allows_guard_direct(self):
        """Direct resolver test: had_defense_after=True → GUARD granted (sanity check)."""
        gm = GameCore(mode="pvp")
        gm.start_game()

        gm.player_1.history = [
            PlayerHistoryItem(CardType.DEFENSE, None, had_defense_after=True)
        ]

        ctx = ResolveContext(
            card_owner=gm.player_1,
            opponent=gm.player_2,
            card_played_by_owner=CardType.DEFENSE,
            card_played_by_opponent=None,
        )
        details = ResolveDetails()
        GrantGuardResolver().resolve(ctx, details)

        assert gm.player_1.has_status(StatusType.GUARD), (
            "GUARD should be granted when had_defense_after=True"
        )
