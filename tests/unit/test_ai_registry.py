"""Unit tests for ai/registry.py — AI registry metadata.

Covers TC-REG1 through TC-REG6 from test_cases.md §2.
Tests list_ais, get_ai_info, error handling, stability,
and verifies that no model modules are loaded at import time.
"""

import pytest

from src.backend.ai.registry import list_ais, get_ai_info


# ---------------------------------------------------------------------------
# TC-REG1: list_ais returns known AIs [EXISTING]
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
        """list_ais() must not expose weight_path, module, or class fields."""
        for entry in list_ais():
            assert "weight_path" not in entry, (
                "weight_path is an internal detail"
            )
            assert "module" not in entry, (
                "module is an internal detail"
            )
            assert "class" not in entry, (
                "class is an internal detail"
            )


# ---------------------------------------------------------------------------
# TC-REG5: list_ais is stable [EXISTING]
# ---------------------------------------------------------------------------


class TestListAisStability:
    """TC-REG5: list_ais() returns identical results across calls."""

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


# ---------------------------------------------------------------------------
# TC-REG3: get_ai_info returns full entry
# ---------------------------------------------------------------------------


class TestGetAiInfo:
    """TC-REG3: get_ai_info() returns full registry entry with internal fields."""

    def test_returns_full_entry(self):
        """get_ai_info('rnn_v1.1') must return dict with all 5 keys."""
        info = get_ai_info("rnn_v1.1")
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "weight_path" in info
        assert "module" in info
        assert "class" in info

    def test_module_is_valid_import_path(self):
        """The 'module' field must be a valid module path string."""
        info = get_ai_info("rnn_v1.1")
        assert isinstance(info["module"], str)
        assert len(info["module"]) > 0
        # Must be a dotted module path
        assert "." in info["module"]

    def test_class_field_is_correct(self):
        """The 'class' field must be 'GRUCardModel'."""
        info = get_ai_info("rnn_v1.1")
        assert info["class"] == "GRUCardModel"

    def test_weight_path_is_string(self):
        """The 'weight_path' field must be a valid path string."""
        info = get_ai_info("rnn_v1.1")
        assert isinstance(info["weight_path"], str)
        assert len(info["weight_path"]) > 0


# ---------------------------------------------------------------------------
# TC-REG4: get_ai_info raises on unknown name
# ---------------------------------------------------------------------------


class TestGetAiInfoUnknown:
    """TC-REG4: get_ai_info() raises ValueError for unknown AI names."""

    def test_raises_value_error_for_unknown_name(self):
        """get_ai_info('nonexistent') must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown AI"):
            get_ai_info("nonexistent")

    def test_error_message_contains_name(self):
        """The error message must include the requested name."""
        with pytest.raises(ValueError, match="nonexistent"):
            get_ai_info("nonexistent")

    def test_error_message_suggests_available(self):
        """The error message should list available AI names."""
        with pytest.raises(ValueError, match="Available"):
            get_ai_info("nonexistent")


# ---------------------------------------------------------------------------
# TC-REG6: No module import at registry load time
# ---------------------------------------------------------------------------


class TestRegistryNoModelImport:
    """TC-REG6: Registry import must not trigger model loading."""

    def test_registry_import_does_not_import_model(self):
        """Importing ai.registry must not cause model module import.

        Verifies that the registry source contains no 'import' statements
        for model modules, torch, or importlib. This ensures the registry
        is pure metadata with no side-effect imports.
        """
        import src.backend.ai.registry as reg
        import inspect

        source = inspect.getsource(reg)
        # The registry must not contain any import statement at all
        # (after refactoring, it's pure declarative data + functions)
        # Check for import statements (excluding those in docstrings/comments)
        lines = source.splitlines()
        for line in lines:
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                # Only flag lines that are actual import statements (not in strings)
                assert False, (
                    f"registry must not contain import statements, but found: {stripped}"
                )
