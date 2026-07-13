"""Self-play game loop — runs N AI-vs-AI games with temperature sampling.

Per design/5 D1, the game loop lives in the training/ layer.
SelfPlayRunner encapsulates the loop: creating GameCore instances,
querying both AIs via TrainingManager.ai_sample_action(), recording PPO
samples, and producing aggregate statistics.

Per design/6: emits structured log events via LoggerBuilder.get_logger()
at key lifecycle points.

Per design/7: sample output uses game → perspective → steps hierarchy
written as a single JSON array file.
"""

import torch

from src.backend.ai.registry import get_ai_info
from src.backend.training.rnn_v2.sample_collector import SampleCollector
from src.backend.training.selfplay_config import SelfPlayConfig, SelfPlayResult
from src.backend.game.game_core import GameCore
from src.backend.log.structured_logger import LoggerBuilder


class SelfPlayRunner:
    """Execute self-play games and collect results.

    Receives a sample_action callable (dependency injection) to break
    the bidirectional dependency with TrainingManager.

    Creates GameCore instances, handles PPO sample recording via the
    game → perspective → steps hierarchy (design/7), and logs progress.

    Attributes:
        _sample_action: Callable with signature:
            (state, h_pol, h_val, ai_name, temp, player_key)
            → (card_type, build_target, dismantle_target,
               new_h_pol, log_prob, value, new_h_val).
    """

    def __init__(self, sample_action_fn):
        """Create a SelfPlayRunner with a sample-action callable.

        Args:
            sample_action_fn: Callable with signature:
                (state, h_pol, h_val, ai_name, temp, player_key)
                → (card_type, build_target, dismantle_target,
                   new_h_pol, log_prob, value, new_h_val)
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
            perspective=config.perspective,
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

        # Sample collector for PPO training data (design/39).
        # Handles per-round recording, per-game building, and file writing.
        collector = SampleCollector(
            output_path=config.sample_output,
            perspective=config.perspective,
            ai1=config.ai1,
            ai2=config.ai2,
        )

        try:
            for game_idx in range(1, config.game_count + 1):
                rounds_played, winner = (
                    self._play_one_game(
                        config, game_idx, loaded_models, collector,
                    )
                )
                total_rounds += rounds_played

                if winner == "1":
                    ai1_wins += 1
                elif winner == "2":
                    ai2_wins += 1
                else:
                    draws += 1

        finally:
            # Write all collected game samples as a single JSON array file.
            # Per design/7 D3: flush delegates to SampleCollector.
            collector.flush()

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
        collector: SampleCollector,
    ) -> tuple[int, str | None]:
        """Play one complete self-play game.

        Creates a GameCore, runs the round loop with AI actions,
        records PPO samples via SampleCollector, emits structured
        log events, and returns game outcome.

        Per design/7, samples are recorded through the collector
        using the game → perspective → steps hierarchy.

        Args:
            config: Self-play configuration.
            game_idx: 1-based game index (for logging).
            loaded_models: Set of AI names whose models are already
                loaded. Updated in-place when a model is first used.
            collector: SampleCollector for PPO sample recording.

        Returns:
            (rounds_played, winner) tuple.
        """
        logger = LoggerBuilder.get_logger()

        # a. Create game and log game_start
        gm = GameCore(mode="pvp")
        gm.start_game()
        init_state = gm.get_state()
        logger.log(
            "INFO", "selfplay.game_start",
            game_idx=game_idx, game_count=config.game_count,
            hand1=[c.value for c in init_state.players["1"].hand],
            hand2=[c.value for c in init_state.players["2"].hand],
        )

        # b. Initialize dual hidden states per player (32-d, design/11)
        h1_pol = [0.0] * 32
        h1_val = [0.0] * 32
        h2_pol = [0.0] * 32
        h2_val = [0.0] * 32

        # c. Round loop
        while not gm.is_over:
            # Per design/27 D4: state with perspective so draw/discard piles are populated.
            pre_state = gm.get_state()
            round_num = pre_state.round

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
            try:
                card1, bt1, dt1, h1_pol, lp1, val1, h1_val = self._sample_action(
                    pre_state, h1_pol, h1_val, config.ai1,
                    temperature=temp_p1, player_key="1",
                )
            except Exception as exc:
                logger.log(
                    "ERROR", "selfplay.action_error",
                    game_idx=game_idx, round=round_num, player="1",
                    hand=[c.value for c in pre_state.players["1"].hand],
                    energy=pre_state.players["1"].energy,
                    temperature=temp_p1,
                    error_type=type(exc).__name__,
                    msg=str(exc),
                )
                raise

            # --- Player 2 action ---
            temp_p2 = (
                config.temperature
                if (should_sample
                    and config.temp_side in ("player2", "both"))
                else 0.0
            )
            try:
                card2, bt2, dt2, h2_pol, lp2, val2, h2_val = self._sample_action(
                    pre_state, h2_pol, h2_val, config.ai2,
                    temperature=temp_p2, player_key="2",
                )
            except Exception as exc:
                logger.log(
                    "ERROR", "selfplay.action_error",
                    game_idx=game_idx, round=round_num, player="2",
                    hand=[c.value for c in pre_state.players["2"].hand],
                    energy=pre_state.players["2"].energy,
                    temperature=temp_p2,
                    error_type=type(exc).__name__,
                    msg=str(exc),
                )
                raise

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

            # Log round_action after both AIs have selected cards.
            # Per design/27: include hand and energy for debugging.
            logger.log(
                "DEBUG", "selfplay.round_action",
                round=round_num,
                card1=card1.value if card1 else None,
                card2=card2.value if card2 else None,
                bt1=bt1.value if bt1 else None,
                bt2=bt2.value if bt2 else None,
                dt1=dt1.value if dt1 else None,
                dt2=dt2.value if dt2 else None,
                hand1=[c.value for c in pre_state.players["1"].hand],
                hand2=[c.value for c in pre_state.players["2"].hand],
                energy1=pre_state.players["1"].energy,
                energy2=pre_state.players["2"].energy,
                temp1=temp_p1,
                temp2=temp_p2,
            )

            # Play the round.
            # Per design/20: card1/card2 are already CardType from ai_sample_action.
            try:
                gm.play_round(
                    [card1] if card1 else [], [card2] if card2 else [],
                    build_target_1=bt1,
                    build_target_2=bt2,
                    dismantle_target_1=dt1,
                    dismantle_target_2=dt2,
                )
            except ValueError as exc:
                print(
                    f"  [WARN] Game {game_idx}: "
                    f"play_round() failed — {exc}"
                )
                logger.log(
                    "ERROR", "selfplay.error",
                    game_idx=game_idx,
                    round=round_num,
                    card1=card1.value if card1 else None,
                    card2=card2.value if card2 else None,
                    bt1=bt1.value if bt1 else None,
                    bt2=bt2.value if bt2 else None,
                    energy1=pre_state.players["1"].energy,
                    energy2=pre_state.players["2"].energy,
                    error_type="ValueError",
                    msg=str(exc),
                )
                break

            # Log round_result from the latest round-log entry
            if gm.round_log:
                last_rr = gm.round_log[-1]
                logger.log(
                    "DEBUG", "selfplay.round_result",
                    round=last_rr.round,
                    hp1_before=last_rr.before.hp_1,
                    hp2_before=last_rr.before.hp_2,
                    energy1_before=last_rr.before.energy_1,
                    energy2_before=last_rr.before.energy_2,
                    hp1_after=last_rr.after_resolve.hp_1,
                    hp2_after=last_rr.after_resolve.hp_2,
                    energy1_after=last_rr.after_resolve.energy_1,
                    energy2_after=last_rr.after_resolve.energy_2,
                )

            # Record PPO samples via collector (design/39).
            # Per design/10 D2: filtering is handled inside collector.
            if collector.is_active:
                collector.record_step(
                    pre_state, pre_state,
                    card1, bt1, dt1, val1, lp1,
                    card2, bt2, dt2, val2, lp2,
                )

        # d. Determine winner and rewards
        winner = gm.winner
        if winner == "1":
            p1_reward, p2_reward = 1.0, -1.0
        elif winner == "2":
            p1_reward, p2_reward = -1.0, 1.0
        else:  # draw
            p1_reward, p2_reward = 0.0, 0.0

        # Derive rounds_played from round log, or from collector step count
        # if round 1 errored before completing (round_log empty).
        # Per design/10 D2: account for perspective filtering when
        # computing rounds from sample count.
        if gm.round_log:
            rounds_played = gm.round_log[-1].round
        elif collector.is_active:
            rounds_played = collector.rounds_from_steps()
        else:
            rounds_played = 0

        # e. Build structured game sample via collector.
        # Per design/7 §2, rounds must be ≥1 — skip if none completed.
        collector.finish_game(
            game_id=game_idx,
            winner=winner,
            p1_reward=p1_reward,
            p2_reward=p2_reward,
            rounds_played=rounds_played,
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

        return rounds_played, winner

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
