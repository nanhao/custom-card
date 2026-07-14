"""Sample collector — collects, builds, and writes PPO training samples (design/39).

Extracted from SelfPlayRunner to separate sample-building concerns from the
game loop. Handles the full pipeline:
  1. record_step()  — per-round: create step dicts from state + action data
  2. finish_game()  — per-game: build structured game sample with rewards
  3. flush()        — global: write all game samples to a single JSON file

Per design/7: sample output uses game → perspective → steps hierarchy.
Per design/10 D2: perspective filtering controls which AI's data is recorded.
"""

import json
from pathlib import Path

from src.backend.ai_adapter.converter import Converter
from src.backend.game.card_type import CardType
from src.backend.game.game_state_snapshot import GameStateSnapshot
from src.backend.log.structured_logger import LoggerBuilder


class SampleCollector:
    """Collects and persists PPO training samples from self-play games.

    Manages a per-game step buffer and a global collection of structured
    game samples. When flush() is called, writes all samples as a single
    JSON array file (design/7 D3).

    Attributes:
        _output_path: Filesystem path for the output JSON file, or None.
        _perspective: Which AI perspective(s) to record — "ai1", "ai2", "both".
        _ai1: Name of player 1's AI (metadata in sample output).
        _ai2: Name of player 2's AI (metadata in sample output).
        _all_games: Accumulated structured game sample dicts.
        _step_buffer: Per-round step dicts for the current game.
    """

    def __init__(
        self,
        output_path: str | None,
        perspective: str,
        ai1: str,
        ai2: str,
    ) -> None:
        """Create a SampleCollector.

        Args:
            output_path: Path for JSON output file. None disables collection.
            perspective: "ai1", "ai2", or "both" — which data to record.
            ai1: AI name for player 1 (written into sample metadata).
            ai2: AI name for player 2 (written into sample metadata).
        """
        self._output_path = output_path
        self._perspective = perspective
        self._ai1 = ai1
        self._ai2 = ai2
        self._all_games: list[dict] = []
        self._step_buffer: list[dict] = []
        self._use_atte_v1 = ai1.startswith("atte_v1") or ai2.startswith("atte_v1")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether sample collection is enabled (output_path is set)."""
        return self._output_path is not None

    # ------------------------------------------------------------------
    # Per-round: record step data
    # ------------------------------------------------------------------

    def record_step(
        self,
        pre_state_p1: GameStateSnapshot,
        pre_state_p2: GameStateSnapshot,
        card1: CardType | None,
        bt1: CardType | None,
        dt1: CardType | None,
        val1: float,
        lp1: float | None,
        card2: CardType | None,
        bt2: CardType | None,
        dt2: CardType | None,
        val2: float,
        lp2: float | None,
    ) -> None:
        """Record one round's step data for the configured perspective(s).

        Builds input vectors from state snapshots and computes policy
        indices from card + build/dismantle targets via Converter.
        Appends to the internal step buffer.

        Args:
            pre_state_p1: Game state from player 1's perspective.
            pre_state_p2: Game state from player 2's perspective.
            card1: Card chosen by player 1, or None for pass.
            bt1: Build target for player 1, or None.
            dt1: Dismantle target for player 1, or None.
            val1: Value network output for player 1's state.
            lp1: Log probability of player 1's action (None if greedy).
            card2: Card chosen by player 2, or None for pass.
            bt2: Build target for player 2, or None.
            dt2: Dismantle target for player 2, or None.
            val2: Value network output for player 2's state.
            lp2: Log probability of player 2's action (None if greedy).
        """
        # Per design/27 D4: use perspective-specific states for input vectors.
        if self._use_atte_v1:
            from src.backend.ai_adapter.atte_v1_converter import AtteV1Converter
            inp1 = AtteV1Converter.build_input(pre_state_p1, "1")
            inp2 = AtteV1Converter.build_input(pre_state_p2, "2")
            state_vec_p1 = _atte_v1_input_to_dict(inp1)
            state_vec_p2 = _atte_v1_input_to_dict(inp2)
        else:
            state_vec_p1 = Converter.build_input_vector(pre_state_p1, player_key="1")
            state_vec_p2 = Converter.build_input_vector(pre_state_p2, player_key="2")

        if self._perspective in ("ai1", "both"):
            p1_action_idx = Converter.action_to_policy_index(card1, bt1, dt1)
            self._step_buffer.append({
                "player": "1",
                "input": state_vec_p1,
                "action": p1_action_idx,
                "value": val1,
                "log_prob": lp1,
            })

        if self._perspective in ("ai2", "both"):
            p2_action_idx = Converter.action_to_policy_index(card2, bt2, dt2)
            self._step_buffer.append({
                "player": "2",
                "input": state_vec_p2,
                "action": p2_action_idx,
                "value": val2,
                "log_prob": lp2,
            })

    # ------------------------------------------------------------------
    # Per-game: step count & rounds derivation
    # ------------------------------------------------------------------

    def step_count(self) -> int:
        """Return the number of buffered steps for the current game."""
        return len(self._step_buffer)

    def rounds_from_steps(self) -> int:
        """Derive rounds_played from step buffer size and perspective.

        When perspective is "both", each round produces 2 entries.
        When filtered to "ai1" or "ai2", each round produces 1 entry.
        """
        entries_per_round = 1 if self._perspective in ("ai1", "ai2") else 2
        if entries_per_round == 0:
            return 0
        return len(self._step_buffer) // entries_per_round

    # ------------------------------------------------------------------
    # Per-game: build structured sample & clear buffer
    # ------------------------------------------------------------------

    def finish_game(
        self,
        game_id: int,
        winner: str | None,
        p1_reward: float,
        p2_reward: float,
        rounds_played: int,
    ) -> dict | None:
        """Build a structured game sample from the step buffer.

        Assigns sparse rewards (terminal step gets game outcome, others 0),
        builds the game → perspective → steps hierarchy, adds to the global
        collection, and clears the per-game buffer.

        Args:
            game_id: 1-based game index for sample metadata.
            winner: "1", "2", or None (draw / unfinished).
            p1_reward: Outcome reward for player 1.
            p2_reward: Outcome reward for player 2.
            rounds_played: Total rounds in the game.

        Returns:
            The built game sample dict, or None if the buffer is empty
            or rounds_played <= 0.
        """
        if not self._step_buffer or rounds_played <= 0:
            self._step_buffer.clear()
            return None

        game_sample = self._build_game_sample(
            game_samples=self._step_buffer,
            game_id=game_id,
            winner=winner,
            rounds_played=rounds_played,
            p1_reward=p1_reward,
            p2_reward=p2_reward,
            ai1=self._ai1,
            ai2=self._ai2,
            perspective=self._perspective,
        )
        self._all_games.append(game_sample)
        self._step_buffer.clear()

        logger = LoggerBuilder.get_logger()
        logger.log(
            "DEBUG", "selfplay.sample_game_built",
            game_idx=game_id, count=len(self._step_buffer),
        )
        return game_sample

    # ------------------------------------------------------------------
    # Global: write all samples to file
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all collected game samples as a single JSON array file.

        Per design/7 D3: file is only created if _output_path is configured.
        For zero games, this writes an empty JSON array [].
        """
        if not self._output_path:
            return

        sample_path = Path(self._output_path)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w", encoding="utf-8") as fh:
            json.dump(self._all_games, fh, ensure_ascii=False)

        # Log sample_write with total step count
        total_steps = sum(
            sum(len(p["steps"]) for p in g["perspectives"])
            for g in self._all_games
        )
        logger = LoggerBuilder.get_logger()
        logger.log(
            "DEBUG", "selfplay.sample_write",
            game_count=len(self._all_games),
            total_steps=total_steps,
        )

    # ------------------------------------------------------------------
    # Static: build structured game sample dict
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
        perspective: str = "both",
    ) -> dict:
        """Build a structured game sample dict (game → perspective → steps).

        Splits flat game_samples by 'player' field, assigns rewards,
        and returns the hierarchy defined in design/7 interfaces.md §2.

        Per design/10 D2: only perspectives matching the `perspective`
        parameter are included in the output.

        Args:
            game_samples: Flat list of per-step dicts with player/input/action/
                          value/log_prob fields.
            game_id: Game index for metadata.
            winner: "1", "2", or None.
            rounds_played: Total rounds in this game.
            p1_reward: Outcome reward for player 1.
            p2_reward: Outcome reward for player 2.
            ai1: Name of player 1's AI.
            ai2: Name of player 2's AI.
            perspective: Which perspective(s) to include — "ai1", "ai2", "both".

        Returns:
            Structured dict: {game_id, winner, rounds, perspectives: [...]}
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
                "input": entry["input"],
                "action": entry["action"],
                "reward": reward,
                "value": entry["value"],
                "log_prob": entry["log_prob"],
            }
            if entry["player"] == "1":
                p1_steps.append(step)
            else:
                p2_steps.append(step)

        # Build the structured game sample dict.
        # Per design/10 D2: only include perspectives matching config.perspective.
        perspectives: list[dict] = []
        if perspective in ("ai1", "both"):
            perspectives.append({"player": "1", "ai": ai1, "steps": p1_steps})
        if perspective in ("ai2", "both"):
            perspectives.append({"player": "2", "ai": ai2, "steps": p2_steps})

        return {
            "game_id": game_id,
            "winner": winner,
            "rounds": rounds_played,
            "perspectives": perspectives,
        }


# ---------------------------------------------------------------------------
# Helper: serialize AtteV1Input to JSON-compatible dict
# ---------------------------------------------------------------------------


def _atte_v1_input_to_dict(inp) -> dict:
    """Convert an AtteV1Input object to a JSON-serializable dict."""
    def _player_to_dict(p):
        return {
            "hand_ids": list(p.hand_ids),
            "discard_ids": list(p.discard_ids),
            "draw_ids": list(p.draw_ids),
            "last_action": p.last_action,
            "hp": p.hp,
            "energy": p.energy,
            "status_ids": list(p.status_ids),
            "status_durations": list(p.status_durations),
            "opp_hand_ids": list(p.opp_hand_ids),
            "opp_discard_ids": list(p.opp_discard_ids),
            "opp_draw_ids": list(p.opp_draw_ids),
        }
    return {
        "own": _player_to_dict(inp.own),
        "opp": _player_to_dict(inp.opp),
        "round_in_cycle": inp.round_in_cycle,
    }
