"""Unit tests for player_state module — StatusType + PlayerStatus (design/15, design/24).

Test cases per design/15 test_cases.md §1 (TC-ST-01~03) and §2 (TC-PS-01~04),
plus design/24 test_cases.md §1 (TC-ST-01~03) and §2 (TC-PS-01~02).
"""

import pytest

from src.backend.game.player_state import StatusType, PlayerStatus


class TestStatusType:
    """TC-ST-01 ~ TC-ST-03: StatusType enum."""

    def test_enum_members_exist(self):
        """TC-ST-01: DEFENSE, GUARD and FRAGILE exist with correct values."""
        assert len(StatusType) == 3
        assert StatusType.DEFENSE.value == "defense"
        assert StatusType.GUARD.value == "guard"
        assert StatusType.FRAGILE.value == "fragile"

    def test_construct_from_string(self):
        """TC-ST-02: StatusType("defense") → DEFENSE, StatusType("guard") → GUARD, StatusType("fragile") → FRAGILE."""
        assert StatusType("defense") == StatusType.DEFENSE
        assert StatusType("guard") == StatusType.GUARD
        assert StatusType("fragile") == StatusType.FRAGILE

    def test_invalid_string_raises(self):
        """TC-ST-03: StatusType("invalid") raises ValueError."""
        with pytest.raises(ValueError):
            StatusType("invalid")


class TestPlayerStatus:
    """TC-PS-01 ~ TC-PS-04: PlayerStatus dataclass."""

    def test_construct_and_access(self):
        """TC-PS-01 (design/15): construction with StatusType.DEFENSE, lifetime=1."""
        ps = PlayerStatus(status_type=StatusType.DEFENSE, lifetime=1)
        assert ps.status_type == StatusType.DEFENSE
        assert ps.lifetime == 1

    def test_fragile_construction(self):
        """TC-PS-01 (design/24): FRAGILE status with lifetime=2."""
        ps = PlayerStatus(status_type=StatusType.FRAGILE, lifetime=2)
        assert ps.status_type == StatusType.FRAGILE
        assert ps.lifetime == 2

    def test_fragile_to_dict(self):
        """TC-PS-02 (design/24): FRAGILE serializes to {"type":"fragile","lifetime":2}."""
        ps = PlayerStatus(StatusType.FRAGILE, lifetime=2)
        assert ps.to_dict() == {"type": "fragile", "lifetime": 2}

    def test_no_default_values(self):
        """TC-PS-02: PlayerStatus has no default_factory — fields must be explicit."""
        with pytest.raises(TypeError):
            PlayerStatus()  # type: ignore[call-arg]

    def test_equality(self):
        """TC-PS-03: same values → equal."""
        a = PlayerStatus(StatusType.DEFENSE, 1)
        b = PlayerStatus(StatusType.DEFENSE, 1)
        assert a == b

    def test_inequality(self):
        """Different status_type or lifetime → not equal."""
        a = PlayerStatus(StatusType.DEFENSE, 1)
        b = PlayerStatus(StatusType.GUARD, 1)
        c = PlayerStatus(StatusType.DEFENSE, 2)
        assert a != b
        assert a != c

    def test_lifetime_zero_boundary(self):
        """TC-PS-04: lifetime=0 is valid (expired, will be removed on next tick)."""
        ps = PlayerStatus(StatusType.GUARD, lifetime=0)
        assert ps.lifetime == 0
        assert ps.status_type == StatusType.GUARD
