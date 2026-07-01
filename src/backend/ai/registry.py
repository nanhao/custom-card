"""AI registry — declarative list of available AI opponents.

Per design/3 architecture D5, the registry is pure metadata.
No models are loaded or imported — model instantiation is handled
by AppManager. Adding a new AI requires only adding an entry here.

The registry exposes:
    list_ais()       → user-facing list (name + description only)
    get_ai_info()    → full entry with internal fields for AppManager
"""

# ---------------------------------------------------------------------------
# Declarative registry (per design/3 architecture)
# ---------------------------------------------------------------------------

_REGISTRY: list[dict] = [
    {
        "name": "rnn_v1.1",
        "description": "GRU-based model, random weights (gen1)",
        "weight_path": "src/backend/ai/rnn_v1/gen1/rnn_v1.pth",
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUCardModel",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_ais() -> list[dict]:
    """Return list of available AIs for display in the frontend.

    Each dict contains only user-facing keys: name and description.
    Internal fields (module, weight_path, class) are excluded.

    Returns:
        List of dicts, each with "name" and "description" keys.
    """
    return [
        {"name": entry["name"], "description": entry["description"]}
        for entry in _REGISTRY
    ]


def get_ai_info(name: str) -> dict:
    """Return the full registry entry for an AI, including internal fields.

    Used by AppManager to obtain the module path, class name, and weight
    path needed to instantiate and load an AI model.

    Args:
        name: AI name as listed in the registry (e.g. "rnn_v1.1").

    Returns:
        dict with keys: name, description, weight_path, module, class.

    Raises:
        ValueError: If the name is not found in the registry.
    """
    for entry in _REGISTRY:
        if entry["name"] == name:
            return dict(entry)  # Return a copy to prevent mutation

    known = [item["name"] for item in _REGISTRY]
    raise ValueError(
        f"Unknown AI: {name!r}. Available: {', '.join(known)}"
    )
