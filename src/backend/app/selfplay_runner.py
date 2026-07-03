"""Self-play game loop — runs N AI-vs-AI games with temperature sampling.

Per design/5 D1, the game loop is business logic and lives in the app/
layer. SelfPlayRunner encapsulates the loop: creating GameCore instances,
querying both AIs via AppManager.ai_sample_action(), recording PPO
samples, and producing aggregate statistics.

This module is separate from app_manager.py to respect the 500-line
file limit (code_rule.md).
"""

import json
from pathlib import Path

import torch

from src.backend.ai.registry import get_ai_info
from src.backend.app.converter import Converter
from src.backend.app.selfplay_config import SelfPlayConfig, SelfPlayResult
from src.backend.game.card import CardType
from src.backend.game.game_core import GameCore


class SelfPlayRunner:
    """Execute self-play games and collect results.

    Receives a sample_action callable (dependency injection) rather
    than depending on AppManager directly. This breaks the bidirectional
    dependency between AppManager and SelfPlayRunner.

    Creates GameCore instances for game simulation. Handles PPO
    sample recording and progress logging.

    Attributes:
        _sample_action: Callable matching AppManager.ai_sample_action
            signature: (state, hidden, ai_name, temp, player_key)
            → (card_str, new_hidden, log_prob, value).
    """

    def __init__(self, sample_action_fn):
        """Create a SelfPlayRunner with a sample-action callable.

        Args:
            sample_action_fn: Callable with the same signature as
                AppManager.ai_sample_action().
        """
        self._sample_action = sample_action_fn

    def run(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games and return aggregate results.

        The loop (per design/5 D1):
        1. Validates that ai1 and ai2 exist in the registry.
        2. Sets torch random seed if temp_seed is provided.
        3. For each game (1..game_count):
           a. Creates a GameCore and calls start_game().
           b. Initializes hidden states for both AIs.
           c. Loops rounds: gets AI actions via AppManager.ai_sample_action(),
              plays the round, records PPO samples if configured.
           d. At game end, assigns rewards to all recorded steps.
           e. Writes samples to file and prints game summary.
        4. Prints aggregate statistics.
        5. Returns SelfPlayResult.

        Per design/5 D6: temp_step controls how many rounds use sampling;
        after that, temperature=0 (greedy) is used. temp_side controls
        which player(s) are sampled.

        Args:
            config: SelfPlayConfig with all run parameters.

        Returns:
            SelfPlayResult with win/loss/draw counts and avg rounds.

        Raises:
            ValueError: If ai1 or ai2 is not found in the registry.
        """
        # 1. Validate that both AIs exist in the registry
        get_ai_info(config.ai1)  # Raises ValueError if unknown
        get_ai_info(config.ai2)

        # 2. Set random seed for reproducibility (if provided)
        if config.temp_seed is not None:
            torch.manual_seed(config.temp_seed)

        # 3. Initialize counters
        ai1_wins = 0
        ai2_wins = 0
        draws = 0
        total_rounds = 0

        # Open sample output file handle if configured
        sample_fh = None
        if config.sample_output:
            sample_path = Path(config.sample_output)
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_fh = sample_path.open("w", encoding="utf-8")

        try:
            for game_idx in range(1, config.game_count + 1):
                rounds_played, winner, p1_reward, p2_reward = (
                    self._play_one_game(config, game_idx, sample_fh)
                )
                total_rounds += rounds_played

                if winner == "1":
                    ai1_wins += 1
                elif winner == "2":
                    ai2_wins += 1
                else:
                    draws += 1

        finally:
            if sample_fh is not None:
                sample_fh.close()

        # 4. Compute and return aggregate result
        total_games = config.game_count
        if total_games > 0:
            avg_rounds = total_rounds / total_games
        else:
            avg_rounds = 0.0

        result = SelfPlayResult(
            total_games=total_games,
            ai1_wins=ai1_wins,
            ai2_wins=ai2_wins,
            draws=draws,
            avg_rounds=avg_rounds,
        )

        # 5. Print summary
        self._print_summary(config, result)

        return result

    # ------------------------------------------------------------------
    # Internal: Single game execution
    # ------------------------------------------------------------------

    def _play_one_game(
        self,
        config: SelfPlayConfig,
        game_idx: int,
        sample_fh,
    ) -> tuple[int, str | None, float, float]:
        """Play one complete self-play game.

        Creates a GameCore, runs the round loop with AI actions,
        records PPO samples, and returns game outcome.

        Args:
            config: Self-play configuration.
            game_idx: 1-based game index (for logging).
            sample_fh: Open file handle for PPO sample JSONL, or None.

        Returns:
            (rounds_played, winner, p1_reward, p2_reward) tuple.
        """
        # a. Create game
        gm = GameCore(mode="pvp")
        gm.start_game()

        # b. Initialize hidden states
        h1 = [0.0] * 8
        h2 = [0.0] * 8

        # Per-step sample buffer for this game
        game_samples: list[dict] = []

        # c. Round loop
        while not gm.is_over:
            pre_state = gm.get_state()
            round_num = pre_state["round"]

            # Determine if this round should use temperature sampling
            should_sample = (
                config.temp_step is None
                or round_num <= config.temp_step
            )

            # --- Player 1 action ---
            temp_p1 = (
                config.temperature
                if (should_sample
                    and config.temp_side in ("player1", "both"))
                else 0.0
            )
            card1, h1, lp1, val1 = self._sample_action(
                pre_state, h1, config.ai1,
                temperature=temp_p1, player_key="1",
            )

            # --- Player 2 action ---
            temp_p2 = (
                config.temperature
                if (should_sample
                    and config.temp_side in ("player2", "both"))
                else 0.0
            )
            card2, h2, lp2, val2 = self._sample_action(
                pre_state, h2, config.ai2,
                temperature=temp_p2, player_key="2",
            )

            # Play the round
            try:
                gm.play_round(
                    [CardType(card1)], [CardType(card2)]
                )
            except ValueError as exc:
                print(
                    f"  [WARN] Game {game_idx}: "
                    f"play_round() failed — {exc}"
                )
                break

            # Record PPO samples (if sample output is configured)
            if sample_fh is not None:
                state_vec_p1 = Converter.build_input_vector(
                    pre_state, player_key="1"
                )
                state_vec_p2 = Converter.build_input_vector(
                    pre_state, player_key="2"
                )
                game_samples.append({
                    "player": "1",
                    "state": state_vec_p1,
                    "action": card1,
                    "value": val1,
                    "log_prob": lp1,
                })
                game_samples.append({
                    "player": "2",
                    "state": state_vec_p2,
                    "action": card2,
                    "value": val2,
                    "log_prob": lp2,
                })

        # d. Determine winner and rewards
        winner = gm.winner
        if winner == "1":
            p1_reward, p2_reward = 1.0, -1.0
        elif winner == "2":
            p1_reward, p2_reward = -1.0, 1.0
        else:  # draw
            p1_reward, p2_reward = 0.0, 0.0

        rounds_played = gm.round_log[-1].round if gm.round_log else 0

        # e. Write samples with rewards
        if sample_fh is not None and game_samples:
            self._write_samples(
                sample_fh, game_samples, p1_reward, p2_reward
            )

        # f. Print game summary
        winner_label = (
            "P1" if winner == "1"
            else "P2" if winner == "2"
            else "Draw"
        )
        print(
            f"Game {game_idx}/{config.game_count} | "
            f"{config.ai1} vs {config.ai2} | "
            f"Winner: {winner_label} | "
            f"Rounds: {rounds_played}"
        )

        return rounds_played, winner, p1_reward, p2_reward

    # ------------------------------------------------------------------
    # Internal: Sample writing
    # ------------------------------------------------------------------

    @staticmethod
    def _write_samples(
        sample_fh,
        game_samples: list[dict],
        p1_reward: float,
        p2_reward: float,
    ) -> None:
        """Write PPO sample JSONL lines for one game.

        Assigns the appropriate reward to each player's steps and writes
        one JSON object per line: {state, action, reward, value, log_prob}.

        Args:
            sample_fh: Open file handle in write mode.
            game_samples: List of per-step sample dicts with keys
                player, state, action, value, log_prob.
            p1_reward: Reward for all player 1 steps.
            p2_reward: Reward for all player 2 steps.
        """
        for entry in game_samples:
            reward = (
                p1_reward if entry["player"] == "1" else p2_reward
            )
            line = json.dumps({
                "state": entry["state"],
                "action": entry["action"],
                "reward": reward,
                "value": entry["value"],
                "log_prob": entry["log_prob"],
            }, ensure_ascii=False)
            sample_fh.write(line + "\n")

    # ------------------------------------------------------------------
    # Internal: Summary printing
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(
        config: SelfPlayConfig, result: SelfPlayResult
    ) -> None:
        """Print aggregate self-play statistics.

        Format matches design/5 interfaces.md §3 (Progress Output Format).

        Args:
            config: SelfPlayConfig used for the run.
            result: SelfPlayResult with win/loss/draw counts.
        """
        total = result.total_games
        print()
        print("=== Summary ===")
        print(f"Total: {total} games")
        if total > 0:
            pct1 = result.ai1_wins / total * 100
            pct2 = result.ai2_wins / total * 100
            pct_draw = result.draws / total * 100
            print(
                f"{config.ai1} (P1): {result.ai1_wins} wins "
                f"({pct1:.1f}%)"
            )
            print(
                f"{config.ai2} (P2): {result.ai2_wins} wins "
                f"({pct2:.1f}%)"
            )
            print(
                f"Draws: {result.draws} ({pct_draw:.1f}%)"
            )
            print(f"Avg rounds: {result.avg_rounds:.1f}")
        if config.sample_output:
            print(f"Samples saved to: {config.sample_output}")
