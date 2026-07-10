"""AI registry — declarative list of available AI architectures.

Per design/30: registry declares network architecture only (module + class).
Weight paths are derived by naming convention: {name}.{gen} format,
where name maps to a registry entry and gen locates the weight directory.

Per design/9: split format with "policy" and "value" sub-dicts for each entry.

The registry exposes:
    parse_ai_name()         → parse "name.gen" into (name, gen)
    _derive_weight_path()   → convention-based weight file path
    list_ais()              → user-facing list (name.gen + description)
    get_ai_info()           → full entry with derived weight_paths
"""

import os
import re

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Project root: 4 levels up from this file (ai/registry.py)
_PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Base path for AI model directories and weight file scanning
_AI_BASE_PATH: str = os.path.join(_PROJECT_ROOT, "src", "backend", "ai")


# ---------------------------------------------------------------------------
# Declarative registry — architecture-only format (design/30)
# ---------------------------------------------------------------------------

_REGISTRY: list[dict] = [
    {
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
    },
]


# ---------------------------------------------------------------------------
# Public API — Name Parsing
# ---------------------------------------------------------------------------


def parse_ai_name(ai_name: str) -> tuple[str, int]:
    """Parse a "name.gen" string into (name, gen) tuple.

    Per design/30 D1: the '.' separates architecture name from gen number.
    Gen must be a positive integer (>= 1).

    Args:
        ai_name: AI identifier string, e.g. "rnn_v1.3".

    Returns:
        (name, gen) tuple where name is str and gen is int >= 1.

    Raises:
        ValueError: If format is invalid (no dot, multiple dots,
            non-integer gen, gen < 1).
    """
    parts = ai_name.split(".")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid ai_name {ai_name!r}: expected 'name.gen' format "
            f"(e.g. 'rnn_v1.3')"
        )
    name_part, gen_part = parts
    if not gen_part.lstrip("-").isdigit():
        raise ValueError(
            f"Invalid ai_name {ai_name!r}: gen must be an integer, "
            f"got {gen_part!r}"
        )
    gen = int(gen_part)
    if gen < 1:
        raise ValueError(
            f"Invalid ai_name {ai_name!r}: gen must be >= 1, got {gen}"
        )
    return name_part, gen


# ---------------------------------------------------------------------------
# Public API — Path Derivation
# ---------------------------------------------------------------------------


def _derive_weight_path(name: str, gen: int, suffix: str) -> str:
    """Derive weight file path by naming convention (design/30 D2).

    Convention: src/backend/ai/{name}/gen{gen}/{suffix}.pth
    Path is relative to project root.

    Args:
        name: Architecture name from registry (e.g. "rnn_v1").
        gen: Generation number (>= 1).
        suffix: "policy" or "value".

    Returns:
        Relative path string, e.g. "src/backend/ai/rnn_v1/gen3/policy.pth".
    """
    return os.path.join("src", "backend", "ai", name, f"gen{gen}", f"{suffix}.pth")


# ---------------------------------------------------------------------------
# Public API — AI Listing
# ---------------------------------------------------------------------------


def list_ais() -> list[dict]:
    """Return list of available AI opponents for display in the frontend.

    Per design/30 D3: scans the filesystem for gen directories under each
    registered architecture, returns name.gen + description for each.
    New gen directories are discovered automatically without registry changes.

    Returns:
        List of dicts, each with "name" (name.gen) and "description".
        Sorted by architecture name then gen ascending.
    """
    result: list[dict] = []
    for entry in _REGISTRY:
        arch_name = entry["name"]
        arch_desc = entry["description"]
        model_dir = os.path.join(_AI_BASE_PATH, arch_name)
        if not os.path.isdir(model_dir):
            continue
        # Scan for gen{i} directories
        gen_pattern = re.compile(r"^gen(\d+)$")
        gens: list[int] = []
        try:
            for item in os.listdir(model_dir):
                m = gen_pattern.match(item)
                if m and os.path.isdir(os.path.join(model_dir, item)):
                    gens.append(int(m.group(1)))
        except OSError:
            continue
        # Generate name.gen entries sorted by gen
        for gen in sorted(gens):
            result.append({
                "name": f"{arch_name}.{gen}",
                "description": f"{arch_desc} (gen{gen})",
            })
    return result


# ---------------------------------------------------------------------------
# Public API — AI Info Lookup
# ---------------------------------------------------------------------------


def get_ai_info(ai_name: str) -> dict:
    """Return full registry entry with derived weight paths.

    Per design/30: accepts name.gen format, parses the gen number,
    looks up the architecture in _REGISTRY, and derives weight_path
    by naming convention (D2).

    Used by AppManager and TrainingRunner to obtain module paths,
    class names, and weight paths for both policy and value networks.

    Args:
        ai_name: AI identifier in name.gen format (e.g. "rnn_v1.3").

    Returns:
        dict with keys: name, description, policy, value.
        Each of policy/value is a dict with weight_path, module, class.

    Raises:
        ValueError: If parse_ai_name fails or architecture not in registry.
    """
    arch_name, gen = parse_ai_name(ai_name)

    # Look up architecture in registry
    for entry in _REGISTRY:
        if entry["name"] == arch_name:
            policy_path = _derive_weight_path(arch_name, gen, "policy")
            value_path = _derive_weight_path(arch_name, gen, "value")
            return {
                "name": entry["name"],
                "description": entry["description"],
                "policy": {
                    **entry["policy"],
                    "weight_path": policy_path,
                },
                "value": {
                    **entry["value"],
                    "weight_path": value_path,
                },
            }

    known = [item["name"] for item in _REGISTRY]
    raise ValueError(
        f"Unknown AI: {ai_name!r} (architecture {arch_name!r} not found). "
        f"Available architectures: {', '.join(known)}"
    )
