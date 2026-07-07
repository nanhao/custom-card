"""Unit tests for split-format AI registry.

Covers TC-REG1 through TC-REG3 from design/9 test_cases.md.
Tests split-format entries, unknown name handling, and list_ais output.
"""

import pytest

from src.backend.ai.registry import get_ai_info, list_ais


# ---------------------------------------------------------------------------
# TC-REG1: split-format entry resolves correctly
# ---------------------------------------------------------------------------


class TestSplitFormatEntry:
    """TC-REG1: split-format entry with policy/value sub-dicts."""

    def test_get_ai_info_returns_policy_and_value_keys(self):
        """get_ai_info('rnn_v2.1') has 'policy' and 'value' keys."""
        info = get_ai_info("rnn_v2.1")
        assert isinstance(info, dict)
        assert "policy" in info
        assert "value" in info
        assert "name" in info
        assert "description" in info

    def test_policy_subdict_has_required_keys(self):
        """'policy' sub-dict has weight_path, module, class."""
        info = get_ai_info("rnn_v2.1")
        policy = info["policy"]
        assert "weight_path" in policy
        assert "module" in policy
        assert "class" in policy

    def test_value_subdict_has_required_keys(self):
        """'value' sub-dict has weight_path, module, class."""
        info = get_ai_info("rnn_v2.1")
        value = info["value"]
        assert "weight_path" in value
        assert "module" in value
        assert "class" in value

    def test_policy_class_is_GRUPolicyModel(self):
        """Policy entry references GRUPolicyModel."""
        info = get_ai_info("rnn_v2.1")
        assert info["policy"]["class"] == "GRUPolicyModel"

    def test_value_class_is_GRUValueModel(self):
        """Value entry references GRUValueModel."""
        info = get_ai_info("rnn_v2.1")
        assert info["value"]["class"] == "GRUValueModel"

    def test_all_entries_have_split_format(self):
        """All registry entries have both 'policy' and 'value' sub-dicts."""
        all_ais = list_ais()
        for ai in all_ais:
            info = get_ai_info(ai["name"])
            assert "policy" in info, f"{ai['name']} missing 'policy'"
            assert "value" in info, f"{ai['name']} missing 'value'"


# ---------------------------------------------------------------------------
# TC-REG2: unknown name raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownName:
    """TC-REG2: unknown AI name raises ValueError."""

    def test_unknown_name_raises_valueerror(self):
        """get_ai_info('nonexistent') raises ValueError."""
        with pytest.raises(ValueError):
            get_ai_info("nonexistent")


# ---------------------------------------------------------------------------
# TC-REG3: list_ais returns all entries
# ---------------------------------------------------------------------------


class TestListAis:
    """TC-REG3: list_ais returns all entries with name + description."""

    def test_list_ais_returns_at_least_one_entry(self):
        """list_ais() returns non-empty list."""
        result = list_ais()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_list_ais_only_has_name_and_description(self):
        """Each entry only exposes 'name' and 'description'."""
        result = list_ais()
        for entry in result:
            assert set(entry.keys()) == {"name", "description"}

    def test_all_entries_are_split_format(self):
        """Every list_ais entry maps to a valid split-format registry entry."""
        for display_entry in list_ais():
            info = get_ai_info(display_entry["name"])
            # Verify the returned info matches display data
            assert info["name"] == display_entry["name"]
            assert info["description"] == display_entry["description"]
