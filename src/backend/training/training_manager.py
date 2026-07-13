"""Training manager — orchestrates AI action sampling and self-play game loop.

Per design/14: policy output 13-d interleaved (build/play pairs),
mask_fn uses 13-d interleaved masking (hand cards + build energy constraints).

Per design/28: policy output 14-d — pass at index 0, even=play, odd=build.

Public interface:
    ai_sample_action(state, h_pol, h_val, ai_name, temp, player_key)
        -> (card_type, build_target, new_h_pol, log_prob, value, new_h_val)
    run_selfplay(config) -> SelfPlayResult
"""

from src.backend.ai_adapter.ai_registry import AiRegistry
from src.backend.game.card_type import CardType
from src.backend.game.game_state_snapshot import GameStateSnapshot
from src.backend.training.selfplay_config import SelfPlayConfig, SelfPlayResult
from src.backend.training.selfplay_runner import SelfPlayRunner


class TrainingManager:
    """Orchestrates AI temperature-based action sampling and self-play runs.

    Instances delegate AI inference to AiRegistry and self-play execution
    to SelfPlayRunner.
    """

    # ------------------------------------------------------------------
    # AI Temperature-Based Action Sampling (self-play mode)
    # ------------------------------------------------------------------

    def ai_sample_action(
        self,
        state: GameStateSnapshot,
        hidden_state_policy: list[float],
        hidden_state_value: list[float],
        ai_name: str,
        temperature: float,
        player_key: str = "2",
    ) -> tuple[CardType | None, CardType | None, CardType | None, list[float], float | None, float, list[float]]:
        """Select AI card with temperature-based sampling via AiInterface."""
        ai_interface = AiRegistry.get_ai_interface(ai_name)
        return ai_interface.ai_sample_action(
            state, hidden_state_policy, hidden_state_value,
            temperature, player_key,
        )

    # ------------------------------------------------------------------
    # Self-play game loop
    # ------------------------------------------------------------------

    def run_selfplay(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games and return aggregate results.

        Thin wrapper that delegates to SelfPlayRunner.
        """
        runner = SelfPlayRunner(self.ai_sample_action)
        return runner.run(config)
