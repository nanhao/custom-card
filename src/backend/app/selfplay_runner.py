"""Self-play game loop — runs N AI-vs-AI games with temperature sampling.

Per design/5 D1, the game loop lives in the app/ layer.
SelfPlayRunner encapsulates the loop: creating GameCore instances,
querying both AIs via AppManager.ai_sample_action(), recording PPO
samples, and producing aggregate statistics.

Per design/6: emits structured log events via LoggerBuilder.get_logger()
at key lifecycle points.

Per design/7: sample output uses game → perspective → steps hierarchy
written as a single JSON array file.
"""

import json
from pathlib import Path

import torch

from src.backend.ai.registry import get_ai_info
from src.backend.app.converter import Converter
from src.backend.app.selfplay_config import SelfPlayConfig, SelfPlayResult
from src.backend.game.card import CardType
from src.backend.game.game_core import GameCore
from src.backend.log.structured_logger import LoggerBuilder


class SelfPlayRunner:
    """Execute self-play games and collect results.

    Receives a sample_action callable (dependency injection) to break
    the bidirectional dependency with AppManager.

    Creates GameCore instances, handles PPO sample recording via the
    game → perspective → steps hierarchy (design/7), and logs progress.

    Attributes:
        _sample_action: Callable with signature:
            (state, h_pol, h_val, ai_name, temp, player_key)
            → (card_str, new_h_pol, log_prob, value, new_h_val).
    """

    def __init__(self, sample_action_fn):
        """Create a SelfPlayRunner with a sample-action callable.

        Args:
            sample_action_fn: Callable with signature:
                (state, h_pol, h_val, ai_name, temp, player_key)
                → (card_str, new_h_pol, log_prob, value, new_h_val)
        """
        self._sample_action = sample_action_fn

    def run(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games with structured logging.

        Collects game sample dicts in memory and writes a single
        JSON array file at the end (design/7 D3).

        Returns:
            SelfPlayResult with win/loss/draw counts and avg rounds.

        Raises:
            ValueError: If ai1 or ai2 is not found in the registry.
        """
        # Retrieve the global singleton logger (no-op if disabled)
        logger = LoggerBuilder.get_logger()

        # Log run_start with all CLI parameters
        logger.log(
            "INFO", "selfplay.run_start",
            ai1=config.ai1,
            ai2=config.ai2,
            game_count=config.game_count,
            temperature=config.temperature,
            temp_step=config.temp_step,
            temp_side=config.temp_side,
            temp_seed=config.temp_seed,
            sample_output=config.sample_output,
        )

        # 1. Validate that both AIs exist in the registry
        get_ai_info(config.ai1)  # Raises ValueError if unknown
        get_ai_info(config.ai2)

        # Log ai_validated after successful registry lookup
        logger.log(
            "INFO", "selfplay.ai_validated",
            ai1=config.ai1, ai2=config.ai2,
        )

        # 2. Set random seed for reproducibility (if provided)
        if config.temp_seed is not None:
            torch.manual_seed(config.temp_seed)

        # 3. Initialize counters
        ai1_wins = 0
        ai2_wins = 0
        draws = 0
        total_rounds = 0

        # Track which AI models have been loaded (for model_loaded event)
        loaded_models: set[str] = set()

        # Collect structured game sample dicts in memory.
        # Per design/7 D3, all games are buffered and written as a
        # single JSON array file at the end (replaces per-game JSONL).
        all_game_samples: list[dict] = []

        try:
            for game_idx in range(1, config.game_count + 1):
                rounds_played, winner, p1_reward, p2_reward, game_sample = (
                    self._play_one_game(
                        config, game_idx, loaded_models,
                    )
                )
                total_rounds += rounds_played

                if game_sample is not None:
                    all_game_samples.append(game_sample)

                if winner == "1":
                    ai1_wins += 1
                elif winner == "2":
                    ai2_wins += 1
                else:
                    draws += 1

        finally:
            # Write all game samples as a single JSON array file.
            # Per design/7 D3, file is only created if sample_output
            # is configured. For game_count=0, this writes [] (TC-O4).
            if config.sample_output:
                sample_path = Path(config.sample_output)
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                with sample_path.open("w", encoding="utf-8") as fh:
                    json.dump(
                        all_game_samples, fh,
                        ensure_ascii=False,
                    )
                # Log sample_write event with total count
                total_steps = sum(
                    sum(len(p["steps"]) for p in g["perspectives"])
                    for g in all_game_samples
                )
                logger.log(
                    "DEBUG", "selfplay.sample_write",
                    game_count=len(all_game_samples),
                    total_steps=total_steps,
                )

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

        # Log run_end with aggregate statistics
        logger.log(
            "INFO", "selfplay.run_end",
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
        loaded_models: set[str],
    ) -> tuple[int, str | None, float, float, dict | None]:
        """Play one complete self-play game.

        Creates a GameCore, runs the round loop with AI actions,
        records PPO samples, emits structured log events, and
        returns game outcome plus the structured game sample.

        Per design/7, the game sample follows the game → perspective
        → steps hierarchy.

        Args:
            config: Self-play configuration.
            game_idx: 1-based game index (for logging).
            loaded_models: Set of AI names whose models are already
                loaded. Updated in-place when a model is first used.

        Returns:
            (rounds_played, winner, p1_reward, p2_reward, game_sample)
            tuple. game_sample is None if sample_output is not configured.
        """
        logger = LoggerBuilder.get_logger()

        # a. Create game and log game_start
        gm = GameCore(mode="pvp")
        gm.start_game()
        logger.log(
            "INFO", "selfplay.game_start",
            game_idx=game_idx, game_count=config.game_count,
        )

        # b. Initialize dual hidden states per player (design/9 D3)
        h1_pol = [0.0] * 8
        h1_val = [0.0] * 8
        h2_pol = [0.0] * 8
        h2_val = [0.0] * 8

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
            card1, h1_pol, lp1, val1, h1_val = self._sample_action(
                pre_state, h1_pol, h1_val, config.ai1,
                temperature=temp_p1, player_key="1",
            )

            # --- Player 2 action ---
            temp_p2 = (
                config.temperature
                if (should_sample
                    and config.temp_side in ("player2", "both"))
                else 0.0
            )
            card2, h2_pol, lp2, val2, h2_val = self._sample_action(
                pre_state, h2_pol, h2_val, config.ai2,
                temperature=temp_p2, player_key="2",
            )

            # Log model_loaded on first successful action per AI
            if config.ai1 not in loaded_models:
                logger.log(
                    "INFO", "selfplay.model_loaded",
                    ai_name=config.ai1,
                )
                loaded_models.add(config.ai1)
            if config.ai2 not in loaded_models:
                logger.log(
                    "INFO", "selfplay.model_loaded",
                    ai_name=config.ai2,
                )
                loaded_models.add(config.ai2)

            # Log round_action after both AIs have selected cards
            logger.log(
                "DEBUG", "selfplay.round_action",
                round=round_num,
                card1=card1,
                card2=card2,
                temp1=temp_p1,
                temp2=temp_p2,
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
                # Log error with exception details
                logger.log(
                    "ERROR", "selfplay.error",
                    error_type="ValueError",
                    msg=str(exc),
                    game_idx=game_idx,
                    round=round_num,
                )
                break

            # Log round_result from the latest round-log entry
            if gm.round_log:
                last_rr = gm.round_log[-1]
                logger.log(
                    "DEBUG", "selfplay.round_result",
                    round=last_rr.round,
                    dmg1=last_rr.damage_1,
                    dmg2=last_rr.damage_2,
                    hp1=last_rr.hp_1_after,
                    hp2=last_rr.hp_2_after,
                )

            # Record PPO samples (if sample output is configured)
            if config.sample_output is not None:
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

        # Derive rounds_played from round log, or from sample count
        # if round 1 errored before completing (round_log empty).
        if gm.round_log:
            rounds_played = gm.round_log[-1].round
        elif game_samples:
            rounds_played = len(game_samples) // 2
        else:
            rounds_played = 0

        # e. Build structured game sample.
        # Per design/7 §2, rounds must be ≥1 — skip if none completed.
        game_sample = None
        if config.sample_output is not None and game_samples and rounds_played > 0:
            game_sample = self._build_game_sample(
                game_samples=game_samples,
                game_id=game_idx,
                winner=winner,
                rounds_played=rounds_played,
                p1_reward=p1_reward,
                p2_reward=p2_reward,
                ai1=config.ai1,
                ai2=config.ai2,
            )
            logger.log(
                "DEBUG", "selfplay.sample_game_built",
                game_idx=game_idx, count=len(game_samples),
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

        # Log game_end
        logger.log(
            "INFO", "selfplay.game_end",
            game_idx=game_idx,
            winner=winner_label,
            rounds=rounds_played,
        )

        return rounds_played, winner, p1_reward, p2_reward, game_sample

    # ------------------------------------------------------------------
    # Internal: Sample building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_game_sample(
        game_samples: list[dict],
        game_id: int,
        winner: str | None,
        rounds_played: int,
        p1_reward: float,
        p2_reward: float,
        ai1: str,
        ai2: str,
    ) -> dict:
        """Build a structured game sample dict (game → perspective → steps).

        Splits flat game_samples by 'player' field, assigns rewards,
        and returns the hierarchy defined in design/7 interfaces.md §2.
        """
        # Find the last step index for each player.
        # Only the terminal step receives the game-outcome reward;
        # all intermediate steps get 0.0 (sparse reward for GAE).
        last_p1_idx = max(
            (i for i, e in enumerate(game_samples) if e["player"] == "1"),
            default=-1,
        )
        last_p2_idx = max(
            (i for i, e in enumerate(game_samples) if e["player"] == "2"),
            default=-1,
        )

        p1_steps: list[dict] = []
        p2_steps: list[dict] = []
        for i, entry in enumerate(game_samples):
            if entry["player"] == "1":
                reward = p1_reward if i == last_p1_idx else 0.0
            else:
                reward = p2_reward if i == last_p2_idx else 0.0
            step = {
                "state": entry["state"],
                "action": entry["action"],
                "reward": reward,
                "value": entry["value"],
                "log_prob": entry["log_prob"],
            }
            if entry["player"] == "1":
                p1_steps.append(step)
            else:
                p2_steps.append(step)

        # Build the structured game sample dict
        return {
            "game_id": game_id,
            "winner": winner,
            "rounds": rounds_played,
            "perspectives": [
                {"player": "1", "ai": ai1, "steps": p1_steps},
                {"player": "2", "ai": ai2, "steps": p2_steps},
            ],
        }

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
