"""Unit tests for ai/registry.py — AI registry discovery and loading.

Covers TC-REG1 through TC-REG4 from test_cases.md.
Tests list_ais, load_ai, error handling for unknown names,
and stability across calls.
"""

import pytest

from src.backend.ai.registry import list_ais, load_ai


# ---------------------------------------------------------------------------
# TC-REG1: list_ais returns known AIs
# ---------------------------------------------------------------------------


class TestListAis:
    """TC-REG1: list_ais() returns the declarative registry."""

    def test_returns_list_of_dicts(self):
        """list_ais() must return a list of dicts with at least 1 entry."""
        result = list_ais()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_each_entry_has_name_and_description(self):
        """Each entry must have 'name' and 'description' keys."""
        for entry in list_ais():
            assert "name" in entry, f"Entry {entry} missing 'name'"
            assert "description" in entry, f"Entry {entry} missing 'description'"

    def test_rnn_v1_1_exists(self):
        """Entry 'rnn_v1.1' must exist with a non-empty description."""
        ais = list_ais()
        names = [a["name"] for a in ais]
        assert "rnn_v1.1" in names
        for a in ais:
            if a["name"] == "rnn_v1.1":
                assert len(a["description"]) > 0
                break

    def test_entries_dont_expose_internal_data(self):
        """list_ais() must not expose weight_path or module fields."""
        for entry in list_ais():
            assert "weight_path" not in entry, (
                "weight_path is an internal detail"
            )
            assert "module" not in entry, (
                "module is an internal detail"
            )


# ---------------------------------------------------------------------------
# TC-REG2: load_ai returns callable module
# ---------------------------------------------------------------------------


class TestLoadAi:
    """TC-REG2: load_ai() returns a module with the required interface."""

    def test_load_ai_returns_object(self):
        """load_ai('rnn_v1.1') must return an object."""
        ai_module = load_ai("rnn_v1.1")
        assert ai_module is not None

    def test_load_ai_has_init_hidden_state(self):
        """The loaded module must have a callable init_hidden_state."""
        ai_module = load_ai("rnn_v1.1")
        assert callable(ai_module.init_hidden_state)

    def test_load_ai_has_select_action(self):
        """The loaded module must have a callable select_action."""
        ai_module = load_ai("rnn_v1.1")
        assert callable(ai_module.select_action)

    def test_init_hidden_state_returns_zeros(self):
        """Calling init_hidden_state through the registry must work."""
        ai_module = load_ai("rnn_v1.1")
        result = ai_module.init_hidden_state()
        assert result == [0.0] * 8

    def test_load_ai_is_cached(self):
        """Loading the same AI twice returns the same object (cached)."""
        ai1 = load_ai("rnn_v1.1")
        ai2 = load_ai("rnn_v1.1")
        assert ai1 is ai2


# ---------------------------------------------------------------------------
# TC-REG3: load_ai raises on unknown name
# ---------------------------------------------------------------------------


class TestLoadAiUnknown:
    """TC-REG3: load_ai() raises ValueError for unknown AI names."""

    def test_raises_value_error_for_unknown_name(self):
        """load_ai('nonexistent_ai') must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown AI"):
            load_ai("nonexistent_ai")

    def test_error_message_contains_name(self):
        """The error message must include the requested name."""
        with pytest.raises(ValueError, match="nonexistent_ai"):
            load_ai("nonexistent_ai")

    def test_error_message_suggests_available(self):
        """The error message should list available AIs."""
        with pytest.raises(ValueError, match="Available"):
            load_ai("nonexistent_ai")


# ---------------------------------------------------------------------------
# TC-REG4: list_ais is stable
# ---------------------------------------------------------------------------


class TestListAisStability:
    """TC-REG4: list_ais() returns identical results across calls."""

    def test_list_ais_stable_across_calls(self):
        """Multiple calls to list_ais() must return identical lists."""
        result1 = list_ais()
        result2 = list_ais()
        assert result1 == result2

    def test_same_order_across_calls(self):
        """Entries must appear in the same order every time."""
        result1 = list_ais()
        result2 = list_ais()
        assert [a["name"] for a in result1] == [a["name"] for a in result2]
