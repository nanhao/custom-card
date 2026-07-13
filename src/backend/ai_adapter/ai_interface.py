"""Abstract AI interface for the custom card game.

Provides AiInterface — the abstract base class that app_manager
will depend on, decoupling it from concrete AI implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backend.game.card_type import CardType
    from src.backend.game.game_state_snapshot import GameStateSnapshot


class AiInterface(ABC):
    """Abstract interface for AI card selection."""

    @abstractmethod
    def ai_select_action(
        self, state: GameStateSnapshot, hidden_state: list[float],
    ) -> tuple[CardType | None, CardType | None, CardType | None, list[float]]:
        """Select the AI's card greedily using policy network."""
        ...

    @abstractmethod
    def ai_sample_action(
        self,
        state: GameStateSnapshot,
        hidden_state_policy: list[float],
        hidden_state_value: list[float],
        temperature: float,
        player_key: str = "2",
    ) -> tuple[
        CardType | None, CardType | None, CardType | None,
        list[float], float | None, float, list[float],
    ]:
        """Select AI card with temperature-based sampling."""
        ...
