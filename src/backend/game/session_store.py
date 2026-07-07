"""Disk-based game state persistence for the custom card game v2 (design/11).

Per architecture D10, this module contains NO business logic — only filesystem
I/O for save/load/delete operations.

Per design/11, serialization now includes new Player fields (energy, statuses,
draw_pile, discard_pile) and GameCore fields (cycle_number, round_in_cycle).
ai_state length changed from 8 to 32.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from src.backend.game.card import CardType, get_initial_draw_pile
from src.backend.game.game_core import GameCore, RoundResult
from src.backend.game.player import Player
from src.backend.ai.rnn_v1.model import HIDDEN_SIZE

logger = logging.getLogger(__name__)

DEFAULT_SAVE_DIR = "./saves"
_CARD_TYPE_BY_VALUE: dict[str, CardType] = {c.value: c for c in CardType}
_EXPECTED_AI_STATE_LEN = HIDDEN_SIZE  # 32


# ---------------------------------------------------------------------------
# Serialization helpers
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

    Returns dict with keys: hp, energy, statuses, hand, draw_pile,
    discard_pile, history.
    """
    return {
        "hp": player.hp,
        "energy": player.energy,
        "statuses": list(player.statuses),
        "hand": [_card_to_str(c) for c in player.hand],
        "draw_pile": [_card_to_str(c) for c in player.draw_pile],
        "discard_pile": [_card_to_str(c) for c in player.discard_pile],
        "history": [_card_to_str(c) for c in player.history],
    }


def _deserialize_player(data: dict) -> Player:
    """Reconstruct a Player from a dict produced by _serialize_player().

    Raises ValueError if required keys are missing.
    """
    required = ("hp", "energy", "statuses", "hand", "draw_pile", "discard_pile", "history")
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required player key: {key!r}")
    return Player(
        hp=data["hp"],
        energy=data["energy"],
        statuses=data["statuses"],
        hand=[_str_to_card(s) for s in data["hand"]],
        draw_pile=[_str_to_card(s) for s in data["draw_pile"]],
        discard_pile=[_str_to_card(s) for s in data["discard_pile"]],
        history=[_str_to_card(s) for s in data["history"]],
    )


def _serialize_round_result(rr: RoundResult) -> dict:
    """Serialize a RoundResult to a plain dict."""
    return {
        "round": rr.round,
        "cycle": rr.cycle,
        "round_in_cycle": rr.round_in_cycle,
        "cards_1": [_card_to_str(c) for c in rr.cards_1],
        "cards_2": [_card_to_str(c) for c in rr.cards_2],
        "damage_1": rr.damage_1,
        "damage_2": rr.damage_2,
        "hp_1_after": rr.hp_1_after,
        "hp_2_after": rr.hp_2_after,
        "energy_1_after": rr.energy_1_after,
        "energy_2_after": rr.energy_2_after,
        "statuses_1": rr.statuses_1,
        "statuses_2": rr.statuses_2,
        "details": rr.details,
        "is_game_over": rr.is_game_over,
        "winner": rr.winner,
        "cycle_ended": rr.cycle_ended,
        "bleed_damage_1": rr.bleed_damage_1,
        "bleed_damage_2": rr.bleed_damage_2,
        "after_cycle": rr.after_cycle,
        "after_round_in_cycle": rr.after_round_in_cycle,
        "after_hp_1": rr.after_hp_1,
        "after_hp_2": rr.after_hp_2,
        "after_statuses_1": rr.after_statuses_1,
        "after_statuses_2": rr.after_statuses_2,
    }


def _deserialize_round_result(data: dict) -> RoundResult:
    """Reconstruct a RoundResult from a dict."""
    # Handle old format (no cycle/round_in_cycle) for backward compat
    return RoundResult(
        round=data["round"],
        cycle=data.get("cycle", 1),
        round_in_cycle=data.get("round_in_cycle", 1),
        cards_1=[_str_to_card(s) for s in data["cards_1"]],
        cards_2=[_str_to_card(s) for s in data["cards_2"]],
        damage_1=data["damage_1"],
        damage_2=data["damage_2"],
        hp_1_after=data["hp_1_after"],
        hp_2_after=data["hp_2_after"],
        energy_1_after=data.get("energy_1_after", 0),
        energy_2_after=data.get("energy_2_after", 0),
        statuses_1=data.get("statuses_1", []),
        statuses_2=data.get("statuses_2", []),
        details=data["details"],
        is_game_over=data["is_game_over"],
        winner=data["winner"],
        cycle_ended=data.get("cycle_ended", False),
        bleed_damage_1=data.get("bleed_damage_1", 0),
        bleed_damage_2=data.get("bleed_damage_2", 0),
        after_cycle=data.get("after_cycle", data.get("cycle", 1)),
        after_round_in_cycle=data.get("after_round_in_cycle", data.get("round_in_cycle", 1)),
        after_hp_1=data.get("after_hp_1", data.get("hp_1_after", 20)),
        after_hp_2=data.get("after_hp_2", data.get("hp_2_after", 20)),
        after_statuses_1=data.get("after_statuses_1", []),
        after_statuses_2=data.get("after_statuses_2", []),
    )


def _serialize_game_core(
    gm: GameCore,
    ai: str | None = None,
    ai_state: list[float] | None = None,
    ai_value_state: list[float] | None = None,
) -> dict:
    """Serialize the full GameCore state to a plain dict."""
    data = {
        "round": gm.round_number,
        "cycle": gm.cycle_number,
        "round_in_cycle": gm.round_in_cycle,
        "player_1": _serialize_player(gm.player_1),
        "player_2": _serialize_player(gm.player_2),
        "is_over": gm.is_over,
        "winner": gm.winner,
        "round_log": [_serialize_round_result(rr) for rr in gm.round_log],
        "mode": gm.mode,
        "status_lifetimes_1": gm._status_lifetimes_1,
        "status_lifetimes_2": gm._status_lifetimes_2,
    }
    if gm.mode == "pve" and ai is not None:
        data["ai"] = ai
    if ai_state is not None:
        data["ai_state"] = ai_state
    if ai_value_state is not None:
        data["ai_value_state"] = ai_value_state
    return data


def _deserialize_game_core(data: dict) -> tuple[GameCore, dict]:
    """Reconstruct a GameCore and metadata from a save file dict.

    Raises ValueError if required keys are missing.
    """
    required_keys = ("round", "player_1", "player_2", "is_over", "winner", "round_log")
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required game state key: {key!r}")

    mode = data.get("mode", "pvp")
    ai = data.get("ai", None)
    ai_state = data.get("ai_state", None)
    ai_value_state = data.get("ai_value_state", None)

    if mode == "pve":
        if not ai:
            raise ValueError(
                "AI_CONFIG_ERROR: save file has mode='pve' but no 'ai' field"
            )
        ai_state = _validate_ai_state(ai_state)
        ai_value_state = _validate_ai_state(ai_value_state)
    else:
        ai = None
        ai_state = None
        ai_value_state = None

    gm = GameCore(mode=mode)
    gm.round_number = data["round"]
    gm.cycle_number = data.get("cycle", 1)
    gm.round_in_cycle = data.get("round_in_cycle", 1)
    gm.player_1 = _deserialize_player(data["player_1"])
    gm.player_2 = _deserialize_player(data["player_2"])
    gm.is_over = data["is_over"]
    gm.winner = data["winner"]
    gm.round_log = [_deserialize_round_result(rr) for rr in data["round_log"]]
    gm._status_lifetimes_1 = data.get("status_lifetimes_1", [])
    gm._status_lifetimes_2 = data.get("status_lifetimes_2", [])

    metadata = {
        "mode": mode, "ai": ai,
        "ai_state": ai_state, "ai_value_state": ai_value_state,
    }
    return gm, metadata


def _validate_ai_state(ai_state) -> list[float]:
    """Validate ai_state from a save file.

    Returns the valid ai_state, or a fresh zero list if invalid.
    Expected length is _EXPECTED_AI_STATE_LEN (32).
    """
    if ai_state is None:
        logger.warning("ai_state is missing; defaulting to zeros")
        return [0.0] * _EXPECTED_AI_STATE_LEN
    if not isinstance(ai_state, list):
        logger.warning("ai_state is not a list; defaulting to zeros")
        return [0.0] * _EXPECTED_AI_STATE_LEN
    if len(ai_state) != _EXPECTED_AI_STATE_LEN:
        logger.warning(
            "ai_state length is %d (expected %d); defaulting to zeros",
            len(ai_state), _EXPECTED_AI_STATE_LEN,
        )
        return [0.0] * _EXPECTED_AI_STATE_LEN
    try:
        return [float(v) for v in ai_state]
    except (ValueError, TypeError):
        logger.warning("ai_state contains non-numeric values; defaulting to zeros")
        return [0.0] * _EXPECTED_AI_STATE_LEN


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """Disk-based persistence for game sessions.

    Reads save_dir from a config.json file at construction time.
    Each game is stored as a single JSON file: {save_dir}/{game_id}.json.
    """

    def __init__(self, config_path: str = "config.json") -> None:
        """Read save_dir from config file. Default to './saves' if absent."""
        self._config_path = Path(config_path)
        self._save_dir: Path | None = None

    @property
    def save_dir(self) -> Path:
        """The resolved save directory path (lazy-loaded from config)."""
        if self._save_dir is None:
            self._save_dir = self._resolve_save_dir()
        return self._save_dir

    def save(self, game_id: str, game_core: GameCore,
             ai_state: list[float] | None = None,
             ai_value_state: list[float] | None = None,
             ai: str | None = None) -> None:
        """Serialize GameCore and write to {save_dir}/{game_id}.json."""
        self._ensure_save_dir()
        file_path = self._file_path(game_id)
        data = _serialize_game_core(
            game_core, ai=ai, ai_state=ai_state, ai_value_state=ai_value_state,
        )
        # Write to temp file then atomically replace to avoid partial writes
        fd, tmp_path = tempfile.mkstemp(dir=str(self.save_dir), suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, file_path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def load(self, game_id: str) -> tuple[GameCore | None, dict | None]:
        """Read and deserialize from {save_dir}/{game_id}.json.

        Returns (GameCore, metadata) or (None, None) if not found.
        """
        file_path = self._file_path(game_id)
        if not file_path.exists():
            return None, None
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse game file: {file_path}"
            ) from exc
        return _deserialize_game_core(data)

    def delete(self, game_id: str) -> bool:
        """Remove {save_dir}/{game_id}.json. Returns True if existed."""
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
