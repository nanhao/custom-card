"""AI adapter registry — factory for AiInterface instances.

Resolves AI identifiers to pre-configured AiInterface instances
backed by RNN-based policy models. Reuses the existing ai.registry
module for architecture lookup and weight path derivation.
"""

from src.backend.ai.registry import get_ai_info
from src.backend.ai_adapter.ai_interface import AiInterface
from src.backend.ai_adapter.rnn_ai import RnnAiAdapter


class AiRegistry:
    """Factory that creates and caches AiInterface instances per AI name.

    Usage:
        adapter = AiRegistry.get_ai_interface("rnn_v1.3")
        card, build, dismantle, hidden = adapter.ai_select_action(state, h)
    """

    _adapters: dict[str, AiInterface] = {}

    @staticmethod
    def get_ai_interface(ai_name: str) -> AiInterface:
        """Return a cached AiInterface for the given AI name.

        On first call for a given ai_name, constructs an RnnAiAdapter
        with policy model metadata from the registry and caches it.
        Subsequent calls return the same cached instance.

        Args:
            ai_name: AI identifier in name.gen format (e.g. "rnn_v1.3").

        Returns:
            AiInterface instance ready for action selection.

        Raises:
            ValueError: If ai_name is unknown or format is invalid.
        """
        if ai_name not in AiRegistry._adapters:
            info = get_ai_info(ai_name)
            policy_info = info["policy"]
            value_info = info["value"]
            AiRegistry._adapters[ai_name] = RnnAiAdapter(
                policy_module=policy_info["module"],
                policy_class=policy_info["class"],
                policy_weight_path=policy_info["weight_path"],
                value_module=value_info["module"],
                value_class=value_info["class"],
                value_weight_path=value_info["weight_path"],
            )
        return AiRegistry._adapters[ai_name]
