"""Sample loader — loads and validates PPO training sample JSON files (design/31).

Extracted from training_runner.py to keep file size under 500 lines.
Stateless — all methods are static.
"""

import json
import os

from src.backend.ai.rnn_v2.constants import INPUT_SIZE, POLICY_OUTPUT_SIZE


class SampleLoader:
    """Load and validate training sample perspectives from JSON files.

    Stateless namespace class. Accepts a file path or directory of .json files,
    validates each step for required keys, correct dimensions, and valid types,
    then returns a list of deep-copied perspective dicts.
    """

    @staticmethod
    def load(sample_path: str) -> list[dict]:
        """Load and validate perspectives from sample file(s).

        If sample_path is a file, loads it directly.
        If sample_path is a directory, loads all .json files inside.

        Returns a list of perspective dicts, each with a "steps" key
        containing validated, deep-copied step records.

        Raises:
            RuntimeError: If no valid samples found, JSON parse fails,
                or any step fails validation.
        """
        json_files = SampleLoader._collect_json_files(sample_path)

        all_perspectives: list[dict] = []
        for file_path in json_files:
            print(f"Loading {file_path}...")
            games = SampleLoader._parse_json(file_path)
            print(f"  Parsed {len(games)} games, validating steps...")
            perspectives = SampleLoader._extract_perspectives(games, file_path)
            all_perspectives.extend(perspectives)

        if not all_perspectives:
            raise RuntimeError(
                f"no valid training samples found in {sample_path!r}"
            )

        total_steps = sum(len(p["steps"]) for p in all_perspectives)
        print(f"  Loaded {len(all_perspectives)} perspectives ({total_steps} steps)")
        return all_perspectives

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_json_files(sample_path: str) -> list[str]:
        """Collect .json file paths from a file or directory."""
        if os.path.isfile(sample_path):
            return [sample_path]
        if os.path.isdir(sample_path):
            return [
                os.path.join(sample_path, entry)
                for entry in sorted(os.listdir(sample_path))
                if entry.lower().endswith(".json")
            ]
        return []

    @staticmethod
    def _parse_json(file_path: str) -> list[dict]:
        """Parse a JSON file into a list of game dicts."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse sample file {file_path!r}: {exc}"
            ) from exc

    @staticmethod
    def _extract_perspectives(
        games: list[dict], file_path: str,
    ) -> list[dict]:
        """Validate all perspectives from loaded games and return deep-copied list."""
        all_perspectives: list[dict] = []
        total_games = len(games)
        for game_idx, game in enumerate(games):
            # Print progress every 10% of games
            if game_idx % max(1, total_games // 10) == 0:
                pct = int(game_idx / total_games * 100)
                print(f"  Validating... {pct}% ({game_idx}/{total_games} games)")
            for p_idx, perspective in enumerate(game.get("perspectives", [])):
                steps = SampleLoader._validate_perspective(
                    perspective, p_idx, game, file_path,
                )
                all_perspectives.append({
                    "steps": steps,
                    "game_id": game.get("game_id"),
                })
        return all_perspectives

    @staticmethod
    def _validate_perspective(
        perspective: dict,
        p_idx: int,
        game: dict,
        file_path: str,
    ) -> list[dict]:
        """Validate all steps in a perspective, return deep-copied step list."""
        validated: list[dict] = []
        game_id = game.get("game_id", "?")
        for s_idx, step in enumerate(perspective.get("steps", [])):
            SampleLoader._validate_step(step, s_idx, p_idx, game_id, file_path)
            validated.append({
                "input": dict(step["input"]) if isinstance(step["input"], dict) else list(step["input"]),
                "action": step["action"],
                "reward": float(step["reward"]),
                "value": float(step["value"]),
                "log_prob": (
                    float(step["log_prob"])
                    if step.get("log_prob") is not None
                    else 0.0
                ),
                "build_target": step.get("build_target"),
            })
        return validated

    @staticmethod
    def _validate_step(
        step: dict,
        s_idx: int,
        p_idx: int,
        game_id: str,
        file_path: str,
    ) -> None:
        """Validate a single step's required keys, types, and input dimension."""
        # Required keys
        for key in ("input", "action", "reward", "value"):
            if key not in step:
                raise RuntimeError(
                    f"Step {s_idx} in perspective {p_idx} "
                    f"of game {game_id} "
                    f"in {file_path!r} is missing required key {key!r}"
                )
        # Numeric validation
        try:
            float(step["reward"])
            float(step["value"])
            log_prob_raw = step.get("log_prob")
            float(log_prob_raw) if log_prob_raw is not None else 0.0
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Step {s_idx} in perspective {p_idx} "
                f"of game {game_id} "
                f"in {file_path!r}: invalid numeric value — {exc}"
            ) from exc
        # Action validation
        action = step["action"]
        if not isinstance(action, int) or action < 0 or action >= POLICY_OUTPUT_SIZE:
            raise RuntimeError(
                f"Step {s_idx} in perspective {p_idx} "
                f"of game {game_id} "
                f"in {file_path!r}: action must be int in [0, {POLICY_OUTPUT_SIZE - 1}], "
                f"got {action!r}"
            )
        # Input dimension — skip for atte_v1 dict format
        if not isinstance(step["input"], dict):
            if len(step["input"]) != INPUT_SIZE:
                raise RuntimeError(
                    f"Step {s_idx} in perspective {p_idx} "
                    f"of game {game_id} "
                    f"in {file_path!r}: input has {len(step['input'])} elements, "
                    f"expected {INPUT_SIZE}"
                )
