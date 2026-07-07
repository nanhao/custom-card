"""AI registry — declarative list of available AI opponents.

Per design/9 architecture: split format with "policy" and "value"
sub-dicts for each entry, referencing GRUPolicyModel and GRUValueModel.

The registry exposes:
    list_ais()       → user-facing list (name + description only)
    get_ai_info()    → full entry with "policy" and "value" sub-dicts
"""

# ---------------------------------------------------------------------------
# Declarative registry — split format (design/9)
# ---------------------------------------------------------------------------

_REGISTRY: list[dict] = [
    {
        "name": "rnn_v1.1",
        "description": "Split policy/value, random weights (gen1)",
        "policy": {
            "weight_path": "src/backend/ai/rnn_v1/gen1/policy.pth",
            "module": "src.backend.ai.rnn_v1.model",
            "class": "GRUPolicyModel",
        },
        "value": {
            "weight_path": "src/backend/ai/rnn_v1/gen1/value.pth",
            "module": "src.backend.ai.rnn_v1.model",
            "class": "GRUValueModel",
        },
    },
    {
        "name": "rnn_v1.2",
        "description": "Split policy/value, random weights (gen1)",
        "policy": {
            "weight_path": "src/backend/ai/rnn_v1/gen2/policy.pth",
            "module": "src.backend.ai.rnn_v1.model",
            "class": "GRUPolicyModel",
        },
        "value": {
            "weight_path": "src/backend/ai/rnn_v1/gen2/value.pth",
            "module": "src.backend.ai.rnn_v1.model",
            "class": "GRUValueModel",
        },
    },
    {
        "name": "rnn_v1.3",
        "description": "Split policy/value, random weights (gen2)",
        "policy": {
            "weight_path": "src/backend/ai/rnn_v1/gen3/policy.pth",
            "module": "src.backend.ai.rnn_v1.model",
            "class": "GRUPolicyModel",
        },
        "value": {
            "weight_path": "src/backend/ai/rnn_v1/gen3/value.pth",
            "module": "src.backend.ai.rnn_v1.model",
            "class": "GRUValueModel",
        },
    },
    {
        "name": "rnn_v1.4",
        "description": "Split policy/value, random weights (gen3)",
        "policy": {
            "weight_path": "src/backend/ai/rnn_v1/gen4/policy.pth",
            "module": "src.backend.ai.rnn_v1.model",
            "class": "GRUPolicyModel",
        },
        "value": {
            "weight_path": "src/backend/ai/rnn_v1/gen4/value.pth",
            "module": "src.backend.ai.rnn_v1.model",
            "class": "GRUValueModel",
        },
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_ais() -> list[dict]:
    """Return list of available AIs for display in the frontend.

    Each dict contains only user-facing keys: name and description.
    Internal fields (policy, value sub-dicts) are excluded.

    Returns:
        List of dicts, each with "name" and "description" keys.
    """
    return [
        {"name": entry["name"], "description": entry["description"]}
        for entry in _REGISTRY
    ]


def get_ai_info(name: str) -> dict:
    """Return the full registry entry for an AI, including policy/value sub-dicts.

    Used by AppManager and TrainingRunner to obtain module paths, class names,
    and weight paths for both policy and value networks.

    Args:
        name: AI name as listed in the registry (e.g. "rnn_v2.1").

    Returns:
        dict with keys: name, description, policy, value.
        Each of policy/value is a dict with weight_path, module, class.

    Raises:
        ValueError: If the name is not found in the registry.
    """
    for entry in _REGISTRY:
        if entry["name"] == name:
            # Return a deep-ish copy to prevent external mutation
            return {
                "name": entry["name"],
                "description": entry["description"],
                "policy": dict(entry["policy"]),
                "value": dict(entry["value"]),
            }

    known = [item["name"] for item in _REGISTRY]
    raise ValueError(
        f"Unknown AI: {name!r}. Available: {', '.join(known)}"
    )
