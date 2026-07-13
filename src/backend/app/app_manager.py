"""App bridge layer — orchestrates game creation and AI card selection (design/14, design/28).

Per design/14: policy output 13-d interleaved (build/play pairs),
mask_fn uses 13-d interleaved masking (hand cards + build energy constraints).

Per design/28: policy output 14-d — pass at index 0, even=play, odd=build.

Public interface:
    create_game(mode, ai_name=None) -> (game_id, state)
    ai_select_action(state, hidden_state, ai_name) -> (card_type, build_target, new_hidden)
    ai_sample_action(state, h_pol, h_val, ai_name, temp, player_key)
        -> (card_type, build_target, new_h_pol, log_prob, value, new_h_val)
    run_selfplay(config) -> SelfPlayResult
"""

from math import gcd
import uuid

from src.backend.ai_adapter.ai_registry import AiRegistry
from src.backend.ai.constants import HIDDEN_SIZE
from src.backend.app.selfplay_config import SelfPlayConfig, SelfPlayResult
from src.backend.app.selfplay_runner import SelfPlayRunner
from src.backend.game.card_type import CardType
from src.backend.game.game_core import GameCore
from src.backend.game.game_state_snapshot import GameStateSnapshot
from src.backend.game.session_store import SessionStore


class AppManager:
    """Bridge layer orchestrating game creation and AI action selection.

    Delegates AI inference to AiRegistry — no direct model loading.
    """

    def __init__(self, session_store: SessionStore) -> None:
        """Create AppManager with a SessionStore for game persistence."""
        self._session_store = session_store

    # ------------------------------------------------------------------
    # Game Lifecycle
    # ------------------------------------------------------------------

    def create_game(
        self, mode: str, ai_name: str | None = None
    ) -> tuple[str, GameStateSnapshot]:
        """Create a new game session and return its ID and typed GameStateSnapshot.

        For PVE mode: validates ai_name, initializes hidden states to zeros (32-d).
        For PVP mode: creates a standard game with no AI metadata.

        Per design/19 D8: ai is not part of GameStateSnapshot — it is injected by
        the controller after to_dict() serialization.

        Args:
            mode: Game mode — "pvp" or "pve".
            ai_name: AI name (required if mode=="pve").

        Returns:
            (game_id, GameStateSnapshot) tuple.

        Raises:
            ValueError: If mode invalid or ai_name missing/unknown.
        """
        if mode not in ("pvp", "pve"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'pvp' or 'pve'.")

        if mode == "pve":
            if not ai_name:
                raise ValueError(
                    "Missing AI name: mode is 'pve' but no ai_name provided"
                )

        game_id = str(uuid.uuid4())
        game_core = GameCore(mode=mode)
        game_core.start_game()

        ai_state = None
        ai_value_state = None
        if mode == "pve":
            ai_state = [0.0] * HIDDEN_SIZE  # 32
            ai_value_state = [0.0] * HIDDEN_SIZE  # 32

        self._session_store.save(
            game_id, game_core,
            ai_state=ai_state,
            ai_value_state=ai_value_state,
            ai=ai_name,
        )

        gs = game_core.get_state()
        return game_id, gs

    # ------------------------------------------------------------------
    # AI Action Selection (greedy, PVE mode)
    # ------------------------------------------------------------------

    def ai_select_action(
        self, state: GameStateSnapshot, hidden_state: list[float], ai_name: str
    ) -> tuple[CardType | None, CardType | None, CardType | None, list[float]]:
        ai_interface = AiRegistry.get_ai_interface(ai_name)
        return ai_interface.ai_select_action(state, hidden_state)

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
