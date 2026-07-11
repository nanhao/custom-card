"""Unit tests for Player.extend_status() and remove_status() — intention #36.

Per design/36 test_cases.md TC-09, TC-10.
"""

import pytest

from src.backend.game.player import Player
from src.backend.game.player_state import StatusType, PlayerStatus


class TestExtendStatus:
    """TC-09: Player.extend_status()."""

    def test_extend_existing_status(self):
        """TC-09-01: extend existing FRAGILE by 2 turns."""
        p = Player(statuses=[PlayerStatus(StatusType.FRAGILE, 1)])
        p.extend_status(StatusType.FRAGILE, 2)
        assert p.has_status(StatusType.FRAGILE) is True
        # find the fragile status and check lifetime
        fragile = [s for s in p.statuses if s.status_type == StatusType.FRAGILE]
        assert len(fragile) == 1
        assert fragile[0].lifetime == 3

    def test_extend_non_existing_status_grants_new(self):
        """TC-09-02: extend non-existing status grants it with lifetime=extra."""
        p = Player(statuses=[])
        p.extend_status(StatusType.FRAGILE, 2)
        assert p.has_status(StatusType.FRAGILE) is True
        fragile = [s for s in p.statuses if s.status_type == StatusType.FRAGILE]
        assert len(fragile) == 1
        assert fragile[0].lifetime == 2

    def test_extend_finds_first_active_only(self):
        """When multiple FRAGILE exist, extends the first active one."""
        p = Player(statuses=[
            PlayerStatus(StatusType.FRAGILE, 1),
            PlayerStatus(StatusType.FRAGILE, 3),
        ])
        p.extend_status(StatusType.FRAGILE, 2)
        assert p.statuses[0].lifetime == 3  # first one extended
        assert p.statuses[1].lifetime == 3  # second unchanged


class TestRemoveStatus:
    """TC-10: Player.remove_status()."""

    def test_remove_existing_defense(self):
        """TC-10-01: remove DEFENSE status."""
        p = Player(statuses=[PlayerStatus(StatusType.DEFENSE, 1)])
        p.remove_status(StatusType.DEFENSE)
        assert p.has_status(StatusType.DEFENSE) is False

    def test_remove_does_not_affect_other_statuses(self):
        """Removing DEFENSE leaves GUARD intact."""
        p = Player(statuses=[
            PlayerStatus(StatusType.DEFENSE, 1),
            PlayerStatus(StatusType.GUARD, 2),
        ])
        p.remove_status(StatusType.DEFENSE)
        assert p.has_status(StatusType.DEFENSE) is False
        assert p.has_status(StatusType.GUARD) is True

    def test_remove_non_existing_no_error(self):
        """TC-10-02: removing non-existing status does not raise."""
        p = Player(statuses=[])
        p.remove_status(StatusType.DEFENSE)
        assert p.statuses == []

    def test_remove_all_instances_of_type(self):
        """Removes all DEFENSE instances, not just the first."""
        p = Player(statuses=[
            PlayerStatus(StatusType.DEFENSE, 1),
            PlayerStatus(StatusType.DEFENSE, 2),
        ])
        p.remove_status(StatusType.DEFENSE)
        assert p.has_status(StatusType.DEFENSE) is False
