"""Unit tests for AI registry — name.gen parsing and filesystem scanning (design/30).

Test cases per design/30 test_cases.md: TC-PN-01~08, TC-DP-01~03, TC-GI-01~03, TC-LA-01~07.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.backend.ai.registry import (
    _derive_weight_path,
    get_ai_info,
    list_ais,
    parse_ai_name,
)


# ===========================================================================
# 1. parse_ai_name() — Name Parsing (TC-PN-01 ~ TC-PN-08)
# ===========================================================================


class TestParseAiName:
    """TC-PN: parse_ai_name() parses 'name.gen' format."""

    def test_valid_name_dot_gen(self):
        """TC-PN-01: rnn_v1.3 → ('rnn_v1', 3)."""
        assert parse_ai_name("rnn_v1.3") == ("rnn_v1", 3)

    def test_single_digit_gen(self):
        """TC-PN-02: rnn_v1.1 → ('rnn_v1', 1)."""
        assert parse_ai_name("rnn_v1.1") == ("rnn_v1", 1)

    def test_multi_digit_gen(self):
        """TC-PN-03: rnn_v1.123 → ('rnn_v1', 123)."""
        assert parse_ai_name("rnn_v1.123") == ("rnn_v1", 123)

    def test_multiple_dots_rejected(self):
        """TC-PN-04: rnn_v2.0.1 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ai_name"):
            parse_ai_name("rnn_v2.0.1")

    def test_no_dot_rejected(self):
        """TC-PN-05: rnn_v1 raises ValueError."""
        with pytest.raises(ValueError):
            parse_ai_name("rnn_v1")

    def test_non_integer_gen_rejected(self):
        """TC-PN-06: rnn_v1.abc raises ValueError."""
        with pytest.raises(ValueError):
            parse_ai_name("rnn_v1.abc")

    def test_gen_zero_rejected(self):
        """TC-PN-07: rnn_v1.0 raises ValueError (gen must be >= 1)."""
        with pytest.raises(ValueError):
            parse_ai_name("rnn_v1.0")

    def test_negative_gen_rejected(self):
        """TC-PN-08: rnn_v1.-1 raises ValueError."""
        with pytest.raises(ValueError):
            parse_ai_name("rnn_v1.-1")


# ===========================================================================
# 2. _derive_weight_path() — Path Derivation (TC-DP-01 ~ TC-DP-03)
# ===========================================================================


class TestDeriveWeightPath:
    """TC-DP: _derive_weight_path() builds convention-based paths."""

    def test_policy_path(self):
        """TC-DP-01: ('rnn_v1', 3, 'policy') → .../gen3/policy.pth."""
        result = _derive_weight_path("rnn_v1", 3, "policy")
        expected = os.path.join("src", "backend", "ai", "rnn_v1", "gen3", "policy.pth")
        assert result == expected

    def test_value_path(self):
        """TC-DP-02: ('rnn_v1', 3, 'value') → .../gen3/value.pth."""
        result = _derive_weight_path("rnn_v1", 3, "value")
        expected = os.path.join("src", "backend", "ai", "rnn_v1", "gen3", "value.pth")
        assert result == expected

    def test_different_architecture(self):
        """TC-DP-03: ('rnn_v2', 1, 'policy') → .../gen1/policy.pth."""
        result = _derive_weight_path("rnn_v2", 1, "policy")
        expected = os.path.join("src", "backend", "ai", "rnn_v2", "gen1", "policy.pth")
        assert result == expected


# ===========================================================================
# 3. get_ai_info() — Registry Lookup (TC-GI-01 ~ TC-GI-03)
# ===========================================================================


# Minimal valid registry entry for testing
_MOCK_ENTRY = {
    "name": "rnn_v1",
    "description": "gru hidden-state=32, level=2",
    "policy": {
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUPolicyModel",
    },
    "value": {
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUValueModel",
    },
}


class TestGetAiInfo:
    """TC-GI: get_ai_info() returns full info with derived weight_path."""

    def test_valid_name_dot_gen_returns_derived_paths(self, monkeypatch):
        """TC-GI-01: get_ai_info('rnn_v1.3') returns derived weight_path.

        Note: get_ai_info does NOT check if the weight file exists on disk.
        """
        import src.backend.ai.registry as registry_module

        monkeypatch.setattr(registry_module, "_REGISTRY", [_MOCK_ENTRY])
        result = get_ai_info("rnn_v1.3")

        assert result["name"] == "rnn_v1"
        assert result["description"] == "gru hidden-state=32, level=2"
        assert result["policy"]["module"] == "src.backend.ai.rnn_v1.model"
        assert result["policy"]["class"] == "GRUPolicyModel"
        assert result["policy"]["weight_path"] == os.path.join(
            "src", "backend", "ai", "rnn_v1", "gen3", "policy.pth",
        )
        assert result["value"]["module"] == "src.backend.ai.rnn_v1.model"
        assert result["value"]["class"] == "GRUValueModel"
        assert result["value"]["weight_path"] == os.path.join(
            "src", "backend", "ai", "rnn_v1", "gen3", "value.pth",
        )

    def test_unknown_architecture_raises(self, monkeypatch):
        """TC-GI-02: unknown_model.1 raises ValueError."""
        import src.backend.ai.registry as registry_module

        monkeypatch.setattr(registry_module, "_REGISTRY", [_MOCK_ENTRY])
        with pytest.raises(ValueError, match="unknown|not found"):
            get_ai_info("unknown_model.1")

    def test_invalid_format_raises(self, monkeypatch):
        """TC-GI-03: 'invalid_format' (no dot) raises ValueError."""
        import src.backend.ai.registry as registry_module

        monkeypatch.setattr(registry_module, "_REGISTRY", [_MOCK_ENTRY])
        with pytest.raises(ValueError):
            get_ai_info("invalid_format")


# ===========================================================================
# 4. list_ais() — Filesystem Scanning (TC-LA-01 ~ TC-LA-07)
# ===========================================================================


def _make_gen_dirs(base: Path, name: str, gens: list[int]) -> None:
    """Create gen{i} directories under base/name/."""
    model_dir = base / name
    model_dir.mkdir(parents=True, exist_ok=True)
    for g in gens:
        (model_dir / f"gen{g}").mkdir()


class TestListAis:
    """TC-LA: list_ais() scans filesystem for gen directories."""

    def test_single_arch_single_gen(self, monkeypatch, tmp_path):
        """TC-LA-01: 1 architecture, gen1 → returns rnn_v1.1."""
        import src.backend.ai.registry as registry_module

        ai_dir = tmp_path / "ai"
        _make_gen_dirs(ai_dir, "rnn_v1", [1])

        monkeypatch.setattr(registry_module, "_REGISTRY", [_MOCK_ENTRY])
        monkeypatch.setattr(registry_module, "_AI_BASE_PATH", str(ai_dir))

        result = list_ais()
        assert len(result) == 1
        assert result[0]["name"] == "rnn_v1.1"
        assert "(gen1)" in result[0]["description"]

    def test_single_arch_multiple_gens(self, monkeypatch, tmp_path):
        """TC-LA-02: gen1, gen3, gen5 → sorted by gen ascending."""
        import src.backend.ai.registry as registry_module

        ai_dir = tmp_path / "ai"
        _make_gen_dirs(ai_dir, "rnn_v1", [1, 3, 5])

        monkeypatch.setattr(registry_module, "_REGISTRY", [_MOCK_ENTRY])
        monkeypatch.setattr(registry_module, "_AI_BASE_PATH", str(ai_dir))

        result = list_ais()
        names = [entry["name"] for entry in result]
        assert names == ["rnn_v1.1", "rnn_v1.3", "rnn_v1.5"]

    def test_multiple_architectures(self, monkeypatch, tmp_path):
        """TC-LA-03: rnn_v1 + rnn_v2, each with gens → sorted by name then gen."""
        import src.backend.ai.registry as registry_module

        ai_dir = tmp_path / "ai"
        _make_gen_dirs(ai_dir, "rnn_v1", [2, 1])
        _make_gen_dirs(ai_dir, "rnn_v2", [3, 1])

        entries = [
            _MOCK_ENTRY,
            {
                "name": "rnn_v2",
                "description": "gru hidden-state=64, level=3",
                "policy": {
                    "module": "src.backend.ai.rnn_v2.model",
                    "class": "GRUPolicyModelV2",
                },
                "value": {
                    "module": "src.backend.ai.rnn_v2.model",
                    "class": "GRUValueModelV2",
                },
            },
        ]
        monkeypatch.setattr(registry_module, "_REGISTRY", entries)
        monkeypatch.setattr(registry_module, "_AI_BASE_PATH", str(ai_dir))

        result = list_ais()
        names = [entry["name"] for entry in result]
        assert names == ["rnn_v1.1", "rnn_v1.2", "rnn_v2.1", "rnn_v2.3"]

    def test_arch_with_no_gen_dirs(self, monkeypatch, tmp_path):
        """TC-LA-04: architecture with no gen dirs is excluded."""
        import src.backend.ai.registry as registry_module

        ai_dir = tmp_path / "ai"
        # Create the AI base dir but no gen dirs
        (ai_dir / "rnn_v1").mkdir(parents=True)

        monkeypatch.setattr(registry_module, "_REGISTRY", [_MOCK_ENTRY])
        monkeypatch.setattr(registry_module, "_AI_BASE_PATH", str(ai_dir))

        result = list_ais()
        assert result == []

    def test_non_gen_items_ignored(self, monkeypatch, tmp_path):
        """TC-LA-05: __pycache__ and .txt files are ignored."""
        import src.backend.ai.registry as registry_module

        ai_dir = tmp_path / "ai"
        _make_gen_dirs(ai_dir, "rnn_v1", [1, 2])
        # Create noise
        (ai_dir / "rnn_v1" / "__pycache__").mkdir()
        (ai_dir / "rnn_v1" / "some_file.txt").write_text("noise")

        monkeypatch.setattr(registry_module, "_REGISTRY", [_MOCK_ENTRY])
        monkeypatch.setattr(registry_module, "_AI_BASE_PATH", str(ai_dir))

        result = list_ais()
        names = [entry["name"] for entry in result]
        assert names == ["rnn_v1.1", "rnn_v1.2"]

    def test_non_integer_gen_suffix_ignored(self, monkeypatch, tmp_path):
        """TC-LA-06: genX, gen/ are ignored; only gen1 is recognized."""
        import src.backend.ai.registry as registry_module

        ai_dir = tmp_path / "ai"
        _make_gen_dirs(ai_dir, "rnn_v1", [1])
        # Create noise gen dirs
        (ai_dir / "rnn_v1" / "genX").mkdir()
        (ai_dir / "rnn_v1" / "gen").mkdir()

        monkeypatch.setattr(registry_module, "_REGISTRY", [_MOCK_ENTRY])
        monkeypatch.setattr(registry_module, "_AI_BASE_PATH", str(ai_dir))

        result = list_ais()
        names = [entry["name"] for entry in result]
        assert names == ["rnn_v1.1"]

    def test_empty_registry_returns_empty(self, monkeypatch, tmp_path):
        """TC-LA-07: empty _REGISTRY returns []."""
        import src.backend.ai.registry as registry_module

        monkeypatch.setattr(registry_module, "_REGISTRY", [])
        result = list_ais()
        assert result == []
