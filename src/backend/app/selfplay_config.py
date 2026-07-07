"""Self-play configuration and result value objects.

Per design/5, these dataclasses are used by AppManager.run_selfplay()
and the SelfPlayRunner to configure and report self-play runs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfPlayConfig:
    """Immutable configuration for a self-play run.

    Attributes:
        ai1: AI name for player 1 (from registry).
        ai2: AI name for player 2 (from registry).
        game_count: Number of games to play (>=1).
        sample_output: Path for PPO sample JSON array file, or None.
        temperature: Sampling temperature (0.0 = greedy).
        temp_step: First N rounds use sampling, then greedy. None = no limit.
        temp_seed: Random seed for sampling reproducibility, or None.
        temp_side: Which side to sample — "player1", "player2", or "both".
    """

    ai1: str
    ai2: str
    game_count: int = 100
    sample_output: str | None = None
    temperature: float = 1.0
    temp_step: int | None = None
    temp_seed: int | None = None
    temp_side: str = "both"
    perspective: str = "both"

    def __post_init__(self):
        """Validate field values after initialization.

        Raises:
            ValueError: If temp_side, perspective is invalid or
                temperature is negative.
        """
        if self.temp_side not in ("player1", "player2", "both"):
            raise ValueError(
                f"Invalid temp_side: {self.temp_side!r}. "
                f"Must be 'player1', 'player2', or 'both'."
            )
        if self.perspective not in ("ai1", "ai2", "both"):
            raise ValueError(
                f"Invalid perspective: {self.perspective!r}. "
                f"Must be 'ai1', 'ai2', or 'both'."
            )
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature!r}"
            )


@dataclass
class SelfPlayResult:
    """Aggregate results of a self-play run.

    Attributes:
        total_games: Number of games played.
        ai1_wins: Games won by player 1 (ai1).
        ai2_wins: Games won by player 2 (ai2).
        draws: Games ending in a draw.
        avg_rounds: Average rounds per game.
    """

    total_games: int
    ai1_wins: int
    ai2_wins: int
    draws: int
    avg_rounds: float
