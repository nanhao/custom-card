"""AI registry — declarative list of available AI opponents.

Per ai_module.md §3, the registry is a flat declarative list.
Adding a new AI requires only adding an entry here — no code changes
to the controller or other modules.

The registry is a pure data module: list_ais() does not load any models.
load_ai() is called lazily only when the controller needs the AI for inference.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declarative registry (per ai_module.md §3.1)
# ---------------------------------------------------------------------------

_REGISTRY: list[dict] = [
    {
        "name": "rnn_v1.1",
        "description": "GRU-based model, random weights (gen1)",
        "weight_path": "src/backend/ai/rnn_v1/gen1/rnn_v1.pth",
        "module": "src.backend.ai.rnn_v1.model",
    },
]

# Cache for loaded AI modules to avoid repeated imports.
_loaded_ais: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_ais() -> list[dict]:
    """Return list of available AIs for display in the frontend.

    Each dict contains only user-facing keys: name and description.
    Weight paths and module names are excluded.

    Returns:
        List of dicts, each with "name" and "description" keys.
    """
    return [
        {"name": entry["name"], "description": entry["description"]}
        for entry in _REGISTRY
    ]


def load_ai(name: str) -> object:
    """Load and return the AI module for the given name.

    The returned object exposes two functions:
        init_hidden_state() → list[float]
        select_action(state: dict, hidden_state: list[float]) → tuple[str, list[float]]

    Modules are cached after first load so subsequent calls are cheap.

    Args:
        name: AI name as listed in the registry (e.g. "rnn_v1.1").

    Returns:
        The loaded AI module object.

    Raises:
        ValueError: If the name is not found in the registry.
        ImportError: If the registered module cannot be imported.
    """
    # Check cache first
    if name in _loaded_ais:
        return _loaded_ais[name]

    # Find the registry entry
    entry = None
    for item in _REGISTRY:
        if item["name"] == name:
            entry = item
            break

    if entry is None:
        known = [item["name"] for item in _REGISTRY]
        raise ValueError(
            f"Unknown AI: {name!r}. Available: {', '.join(known)}"
        )

    # Import the module
    module_name = entry["module"]
    try:
        ai_module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import AI module {module_name!r} for AI {name!r}: {exc}"
        ) from exc

    # Validate that the module exposes the required interface
    if not callable(getattr(ai_module, "init_hidden_state", None)):
        raise ImportError(
            f"AI module {module_name!r} missing init_hidden_state() function"
        )
    if not callable(getattr(ai_module, "select_action", None)):
        raise ImportError(
            f"AI module {module_name!r} missing select_action() function"
        )

    _loaded_ais[name] = ai_module
    logger.info("Loaded AI %r from %s", name, module_name)
    return ai_module
