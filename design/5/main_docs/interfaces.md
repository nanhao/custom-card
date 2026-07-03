# Interface Design — AI Self-Play Script

Public interfaces only. Private methods omitted per design rule.

---

## 1. `app/selfplay_config.py` — SelfPlayConfig

Immutable configuration value object. Defined in `selfplay_config.py` (extracted from `app_manager.py` to respect 500-line file limit).

```
@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for a self-play run.

    Attributes:
        ai1: AI name for player 1 (from registry).
        ai2: AI name for player 2 (from registry).
        game_count: Number of games to play (>=1).
        sample_output: Path for PPO sample JSONL file, or None.
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
```

---

## 2. `app/selfplay_config.py` — SelfPlayResult

Aggregate statistics returned after all games complete.

```
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
```

---

## 3. `app/app_manager.py` — AppManager (Modified)

Existing `ai_select_action()` updated for 4-tuple return. New method `ai_sample_action()` added. `run_selfplay()` is a thin wrapper delegating to `SelfPlayRunner` (see §4).

```
class AppManager:
    """(existing __init__, create_game unchanged)"""

    # --- Modified: updated for 4-tuple return ---

    def ai_select_action(
        self, state: dict, hidden_state: list[float], ai_name: str
    ) -> tuple[str, list[float]]:
        """Select the AI's card greedily (argmax). Updated for 4-tuple return.

        Args:
            state: Game state dict. AI always plays as Player "2".
            hidden_state: Current GRU hidden state as list of 8 floats.
            ai_name: AI name from registry.

        Returns:
            (card_str, new_hidden_state) tuple.
        """

    # --- New: temperature-based sampling ---

    def ai_sample_action(
        self,
        state: dict,
        hidden_state: list[float],
        ai_name: str,
        temperature: float,
        player_key: str = "2",
    ) -> tuple[str, list[float], float | None]:
        """Select an AI action with temperature-based sampling.

        When temperature > 0:
            Samples from softmax(logits / temperature) distribution.
            Returns log_prob of the selected action.
        When temperature = 0:
            Greedy argmax (same as ai_select_action).
            Returns log_prob = None.

        Args:
            state: Game state dict.
            hidden_state: Current GRU hidden state as list of 8 floats.
            ai_name: AI name from registry.
            temperature: Sampling temperature. 0 = greedy.
            player_key: Which player the AI controls — "1" or "2".

        Returns:
            (card_str, new_hidden_state, log_prob) tuple.
            log_prob is float if temperature > 0, else None.
        """

    # --- New: self-play game loop (thin delegation wrapper) ---

    def run_selfplay(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games via SelfPlayRunner.

        Thin wrapper that delegates to SelfPlayRunner.run().
        The actual game loop logic lives in selfplay_runner.py.

        Args:
            config: SelfPlayConfig with all run parameters.

        Returns:
            SelfPlayResult with win/loss/draw counts and avg rounds.
        """
```

### Progress Output Format (handled by `SelfPlayRunner`)

Per game:
```
Game 5/100 | rnn_v1.1 vs rnn_v1.1 | Winner: P1 | Rounds: 6
```

At end:
```
=== Summary ===
Total: 100 games
rnn_v1.1 (P1): 48 wins (48.0%)
rnn_v1.1 (P2): 42 wins (42.0%)
Draws: 10 (10.0%)
Avg rounds: 5.3
Samples saved to: data/samples.jsonl
```

---

## 4. `app/selfplay_runner.py` — SelfPlayRunner

Encapsulates the self-play game loop (per design D1). Created and invoked
by `AppManager.run_selfplay()`. Receives a `sample_action_fn` callable via
dependency injection rather than holding a reference to AppManager.

```
class SelfPlayRunner:
    """Execute self-play games and collect results."""

    def __init__(self, sample_action_fn):
        """Create a runner with a sample-action callable.

        Args:
            sample_action_fn: Callable matching AppManager.ai_sample_action
                signature: (state, hidden, ai_name, temp, player_key)
                → (card_str, new_hidden, log_prob, value).
        """

    def run(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games and return aggregate results.

        1. Validate ai1/ai2 exist in registry
        2. For each game:
           a. Create GameCore and start_game()
           b. Loop rounds: query both AIs via AppManager.ai_sample_action()
           c. Apply temp_step/temp_side rules for sampling vs greedy
           d. Record PPO sample per step if config.sample_output is set
           e. At game end, assign reward to all recorded steps
           f. Write samples to file (if enabled)
           g. Print game summary line
        3. Print aggregate statistics
        4. Return SelfPlayResult

        Args:
            config: SelfPlayConfig with all run parameters.

        Returns:
            SelfPlayResult with win/loss/draw counts and avg rounds.
        """
```

---

## 5. `scripts/selfplay.py` — CLI Entry Point

Thin wrapper. No business logic — only argument parsing and delegation.

```
"""AI Self-Play CLI.

Usage:
    python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1
    python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 --game 200 --temperature 0.5 --temp_step 3 --temp_side both --sample_output data/samples.jsonl
"""

import argparse
from src.backend.app.app_manager import AppManager
from src.backend.app.selfplay_config import SelfPlayConfig
from src.backend.game.session_store import SessionStore

# CLI argument parsing, then:
#   config = SelfPlayConfig(ai1=..., ai2=..., ...)
#   app = AppManager(SessionStore())
#   result = app.run_selfplay(config)
```

---

## 6. `app/converter.py` — Converter (Modified)

Adds `player_key` parameter to `build_input_vector()`.

```
class Converter:
    """(existing constants and methods unchanged)"""

    @staticmethod
    def build_input_vector(state: dict, player_key: str = "2") -> list[float]:
        """Build the 14-d input vector from a game state dict.

        Args:
            state: Game state dict from GameCore.get_state().
            player_key: Which player's perspective — "1" or "2".
                        Default "2" preserves backward compatibility.

        Returns:
            list[float] of length 14:
                indices 0–3:  This player's card counts
                indices 4–5:  HP values (this player, opponent), clamped [0,3]
                indices 6–9:  One-hot of this player's last move
                indices 10–13: One-hot of opponent's last move
        """
```

---

## 7. `ai/rnn_v1/model.py` — GRUCardModel (Modified)

`forward()` adds `temperature` parameter and returns `log_probs` as 4th element.

```
class GRUCardModel(nn.Module):
    """(existing constants and __init__ unchanged)"""

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one GRU step with optional temperature scaling.

        When temperature = 0: greedy — probs are one-hot at argmax position.
        When temperature > 0: softmax(logits / temperature).

        Args:
            x: Input tensor of shape (1, 1, 14).
            hidden: Hidden state tensor of shape (1, 1, 8).
            mask_fn: Optional callable that takes logits (1,4) and
                     returns masked logits (1,4).
            temperature: Temperature for softmax scaling. 0 = greedy.

        Returns:
            (probs, log_probs, value, new_hidden) tuple:
                probs: (1, 4) probability distribution over card types.
                log_probs: (1, 4) log-probabilities.
                value: (1, 1) win-rate estimate in [-1, 1].
                new_hidden: (1, 1, 8) updated GRU hidden state.
        """
```

---

## 8. `game/` — Unchanged

No changes to `game/` package.
