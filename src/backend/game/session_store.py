"""Disk-based game state persistence for the custom card game.

Per architecture D10, this module contains NO business logic — only filesystem
I/O for save/load/delete operations. All game rules live in the other game/
modules.

Per persistence.md, SessionStore uses a config.json file to determine the
save directory, and stores each game as a single JSON file named by UUID.

Per architecture D3 (intention #2), save() and load() support AI metadata
(mode, ai, ai_state) for PVE games. Legacy files without mode default to "pvp".
"""

import json
import logging
import os
from pathlib import Path

from src.backend.game.card import CardType, get_initial_hand
from src.backend.game.game_core import GameCore, RoundResult
from src.backend.game.player import Player

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAVE_DIR = "./saves"

# Mapping from CardType enum to JSON string and back.
# Used in both serialization and deserialization.
_CARD_TYPE_BY_VALUE: dict[str, CardType] = {c.value: c for c in CardType}


# ---------------------------------------------------------------------------
# Serialization helpers (private to SessionStore per persistence.md §5.2)
# ---------------------------------------------------------------------------


def _card_to_str(card: CardType) -> str:
    """Convert a CardType enum member to its JSON string value."""
    return card.value


def _str_to_card(value: str) -> CardType:
    """Convert a JSON string back to a CardType enum member.

    Raises ValueError if the string does not match any CardType value.
    """
    if value not in _CARD_TYPE_BY_VALUE:
        raise ValueError(f"Unknown card type: {value!r}")
    return _CARD_TYPE_BY_VALUE[value]


def _serialize_player(player: Player) -> dict:
    """Serialize a Player to a plain dict for JSON encoding.

    Returns:
        dict with keys: hp (int), hand (list[str]), history (list[str]).
    """
    return {
        "hp": player.hp,
        "hand": [_card_to_str(c) for c in player.hand],
        "history": [_card_to_str(c) for c in player.history],
    }


def _deserialize_player(data: dict) -> Player:
    """Reconstruct a Player from a dict produced by _serialize_player()."""
    return Player(
        hp=data["hp"],
        hand=[_str_to_card(s) for s in data["hand"]],
        history=[_str_to_card(s) for s in data["history"]],
    )


def _serialize_round_result(rr: RoundResult) -> dict:
    """Serialize a RoundResult to a plain dict for JSON encoding.

    Returns:
        dict matching the round_log entry format in persistence.md §5.1.
    """
    return {
        "round": rr.round,
        "cards_1": [_card_to_str(c) for c in rr.cards_1],
        "cards_2": [_card_to_str(c) for c in rr.cards_2],
        "damage_1": rr.damage_1,
        "damage_2": rr.damage_2,
        "hp_1_after": rr.hp_1_after,
        "hp_2_after": rr.hp_2_after,
        "details": rr.details,
        "is_game_over": rr.is_game_over,
        "winner": rr.winner,
    }


def _deserialize_round_result(data: dict) -> RoundResult:
    """Reconstruct a RoundResult from a dict produced by _serialize_round_result()."""
    return RoundResult(
        round=data["round"],
        cards_1=[_str_to_card(s) for s in data["cards_1"]],
        cards_2=[_str_to_card(s) for s in data["cards_2"]],
        damage_1=data["damage_1"],
        damage_2=data["damage_2"],
        hp_1_after=data["hp_1_after"],
        hp_2_after=data["hp_2_after"],
        details=data["details"],
        is_game_over=data["is_game_over"],
        winner=data["winner"],
    )


def _serialize_game_core(gm: GameCore, ai: str | None = None, ai_state: list[float] | None = None) -> dict:
    """Serialize the full GameCore state to a plain dict.

    Args:
        gm: The GameCore to serialize.
        ai: AI name for PVE games (optional).
        ai_state: GRU hidden state for PVE games (optional).

    Returns:
        dict matching the game save JSON structure in persistence.md §5.1,
        extended with mode, ai (if PVE), and ai_state (if PVE).
    """
    data = {
        "round": gm.round_number,
        "player_1": _serialize_player(gm.player_1),
        "player_2": _serialize_player(gm.player_2),
        "is_over": gm.is_over,
        "winner": gm.winner,
        "round_log": [_serialize_round_result(rr) for rr in gm.round_log],
        "mode": gm.mode,
    }
    # Only include AI fields for PVE games to keep PVP saves clean.
    if gm.mode == "pve" and ai is not None:
        data["ai"] = ai
    if ai_state is not None:
        data["ai_state"] = ai_state
    return data


def _deserialize_game_core(data: dict) -> tuple[GameCore, dict]:
    """Reconstruct a GameCore and metadata from a save file dict.

    Handles legacy files (no mode key) by defaulting to "pvp".
    Validates ai_state on load (fallback to zeros if invalid).

    Returns:
        (GameCore, metadata) tuple where metadata has:
            mode: "pvp" | "pve"
            ai: AI name or None
            ai_state: GRU hidden state or None

    Raises:
        ValueError: If required game state keys are missing, or if
                    mode is "pve" but ai field is missing.
    """
    required_keys = ("round", "player_1", "player_2", "is_over", "winner", "round_log")
    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"Missing required game state key: {key!r}"
            )

    # ---- Extract mode (default to "pvp" for legacy files) ----
    mode = data.get("mode", "pvp")

    # ---- Extract AI metadata ----
    ai = data.get("ai", None)
    ai_state = data.get("ai_state", None)

    # ---- Validate PVE config ----
    if mode == "pve":
        if not ai:
            raise ValueError(
                "AI_CONFIG_ERROR: save file has mode='pve' but no 'ai' field"
            )
        # Validate ai_state: must be list[float] of length 8, or fallback
        ai_state = _validate_ai_state(ai_state)
    else:
        # PVP: silently ignore any stray ai/ai_state fields
        ai = None
        ai_state = None

    # ---- Build GameCore ----
    gm = GameCore(mode=mode)
    gm.round_number = data["round"]
    gm.player_1 = _deserialize_player(data["player_1"])
    gm.player_2 = _deserialize_player(data["player_2"])
    gm.is_over = data["is_over"]
    gm.winner = data["winner"]
    gm.round_log = [_deserialize_round_result(rr) for rr in data["round_log"]]

    metadata = {"mode": mode, "ai": ai, "ai_state": ai_state}
    return gm, metadata


def _validate_ai_state(ai_state) -> list[float]:
    """Validate ai_state from a save file.

    Returns the valid ai_state, or a fresh [0.0]*8 if invalid.
    Logs a warning when fallback occurs.
    """
    if ai_state is None:
        logger.warning("ai_state is missing; defaulting to zeros")
        return [0.0] * 8
    if not isinstance(ai_state, list):
        logger.warning("ai_state is not a list; defaulting to zeros")
        return [0.0] * 8
    if len(ai_state) != 8:
        logger.warning("ai_state length is %d (expected 8); defaulting to zeros", len(ai_state))
        return [0.0] * 8
    try:
        return [float(v) for v in ai_state]
    except (ValueError, TypeError):
        logger.warning("ai_state contains non-numeric values; defaulting to zeros")
        return [0.0] * 8


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """Disk-based persistence for game sessions.

    Reads save_dir from a config.json file at construction time.
    Each game is stored as a single JSON file: {save_dir}/{game_id}.json.

    Per persistence.md: pure I/O, no game rules, no in-memory cache.
    """

    def __init__(self, config_path: str = "config.json") -> None:
        """Read save_dir from config file. Default to './saves' if absent.

        Args:
            config_path: Path to the JSON config file. If the file does not
                         exist, the default save_dir ("./saves") is used.
        """
        self._config_path = Path(config_path)
        self._save_dir: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def save_dir(self) -> Path:
        """The resolved save directory path (lazy-loaded from config)."""
        if self._save_dir is None:
            self._save_dir = self._resolve_save_dir()
        return self._save_dir

    def save(self, game_id: str, game_core: GameCore,
             ai_state: list[float] | None = None,
             ai: str | None = None) -> None:
        """Serialize GameCore and write to {save_dir}/{game_id}.json.

        Creates the save directory if it does not exist.
        Overwrites the file if it already exists.

        Args:
            game_id: Unique game identifier (UUID string).
            game_core: The GameCore to serialize.
            ai_state: GRU hidden state for PVE games (optional).
            ai: AI name for PVE games (optional).
        """
        self._ensure_save_dir()
        file_path = self._file_path(game_id)
        data = _serialize_game_core(game_core, ai=ai, ai_state=ai_state)
        file_path.write_text(json.dumps(data, indent=4, ensure_ascii=False))

    def load(self, game_id: str) -> tuple[GameCore | None, dict | None]:
        """Read and deserialize from {save_dir}/{game_id}.json.

        Returns:
            A (GameCore, metadata) tuple, or (None, None) if the file
            does not exist. metadata is a dict with keys:
                mode: "pvp" | "pve"
                ai: AI name or None
                ai_state: GRU hidden state (list[float]) or None

        Raises:
            ValueError: If the JSON file exists but is malformed or missing
                        required keys, or if PVE config is inconsistent.
        """
        file_path = self._file_path(game_id)
        if not file_path.exists():
            return None, None
        try:
            data = json.loads(file_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse game file: {file_path}"
            ) from exc
        return _deserialize_game_core(data)

    def delete(self, game_id: str) -> bool:
        """Remove {save_dir}/{game_id}.json.

        Returns:
            True if the file existed and was removed, False otherwise.
        """
        file_path = self._file_path(game_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_save_dir(self) -> Path:
        """Resolve save_dir from config.json, falling back to DEFAULT_SAVE_DIR."""
        if not self._config_path.exists():
            return Path(DEFAULT_SAVE_DIR).resolve()
        try:
            config_data = json.loads(self._config_path.read_text())
        except json.JSONDecodeError:
            return Path(DEFAULT_SAVE_DIR).resolve()

        save_dir_str = config_data.get("save_dir", DEFAULT_SAVE_DIR)
        return Path(save_dir_str).resolve()

    def _ensure_save_dir(self) -> None:
        """Create the save directory if it does not exist."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, game_id: str) -> Path:
        """Return the full path for a game's JSON file."""
        return self.save_dir / f"{game_id}.json"
