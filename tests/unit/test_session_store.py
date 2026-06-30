"""Unit tests for SessionStore — disk-based game state persistence.

Tests cover save, load, delete, round-trip consistency, config handling,
and corruption recovery per persistence.md §9.

All file I/O uses temporary directories (test rule: never touch real data).
"""

import json
import os
from pathlib import Path

import pytest

from src.backend.game.card import CardType
from src.backend.game.game_manager import GameManager, RoundResult

# Import is deferred until SessionStore is implemented
# from src.backend.game.session_store import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

A = CardType.ATTACK
D = CardType.DEFENSE
H = CardType.HEAVY_STRIKE
C = CardType.COUNTER


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory. Cleaned up by pytest automatically."""
    return tmp_path


@pytest.fixture
def populated_gm():
    """Create a GameManager with 2 completed rounds and return it.

    State after setup:
        round_number = 3 (after 2 completed rounds, game not over)
        player_1: hp=2, hand=[A,A,D,D,H,C], history=[A,H]
        player_2: hp=1, hand=[A,A,D,D,H,C],  history=[A,D]
    """
    gm = GameManager()
    gm.start_game()
    # R1: ATK vs ATK → 1-1, HP: 2-2
    gm.play_round([A], [A])
    # R2: HS vs DEF (B no prev DEF) → B takes 1, HP: 2-1
    gm.play_round([H], [D])
    return gm


# ---------------------------------------------------------------------------
# Test config handling
# ---------------------------------------------------------------------------


class TestSessionStoreConfig:
    """Tests for config.json reading and save_dir resolution."""

    def test_default_save_dir_when_config_missing(self, temp_dir):
        """When config.json is absent, use './saves' relative to cwd."""
        from src.backend.game.session_store import SessionStore

        config_path = temp_dir / "nonexistent_config.json"
        store = SessionStore(config_path=str(config_path))
        # save_dir resolves to project_root + saves
        assert store.save_dir.name == "saves"

    def test_custom_save_dir_from_config(self, temp_dir):
        """Config file 'save_dir' key is respected."""
        from src.backend.game.session_store import SessionStore

        custom_dir = temp_dir / "my_saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(custom_dir)}))

        store = SessionStore(config_path=str(config_path))
        assert store.save_dir == custom_dir


# ---------------------------------------------------------------------------
# Test save
# ---------------------------------------------------------------------------


class TestSessionStoreSave:
    """Tests for SessionStore.save()."""

    def test_save_creates_file(self, temp_dir, populated_gm):
        """save() creates a JSON file at the expected path."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        store.save("test-game-1", populated_gm)

        expected_path = saves_dir / "test-game-1.json"
        assert expected_path.exists()

    def test_save_creates_save_dir_if_missing(self, temp_dir, populated_gm):
        """save() creates the save_dir if it doesn't exist."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "nested" / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        store.save("game-1", populated_gm)

        assert saves_dir.exists()
        assert (saves_dir / "game-1.json").exists()

    def test_save_produces_valid_json(self, temp_dir, populated_gm):
        """Saved file contains valid JSON with all expected keys."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        store.save("game-1", populated_gm)

        data = json.loads((saves_dir / "game-1.json").read_text())
        assert data["round"] == 3
        assert data["player_1"]["hp"] == 2
        assert data["player_2"]["hp"] == 1
        assert data["is_over"] is False
        assert data["winner"] is None
        assert len(data["round_log"]) == 2

    def test_save_overwrite_updates_file(self, temp_dir):
        """Saving twice overwrites the file with updated state."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        saves_dir.mkdir()
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))

        # Save first state
        gm1 = GameManager()
        gm1.start_game()
        store.save("game-1", gm1)

        # Play a round, save again
        gm1.play_round([A], [A])
        store.save("game-1", gm1)

        data = json.loads((saves_dir / "game-1.json").read_text())
        assert data["round"] == 2
        assert data["player_1"]["hp"] == 2  # 3 - 1 from ATK vs ATK

    def test_save_game_over_state(self, temp_dir):
        """Saving a finished game preserves game-end state correctly."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [A])  # 3→2
        gm.play_round([A], [A])  # 2→0 → draw

        store = SessionStore(config_path=str(config_path))
        store.save("finished-game", gm)

        data = json.loads((saves_dir / "finished-game.json").read_text())
        assert data["is_over"] is True
        assert data["winner"] == "draw"
        assert data["player_1"]["hp"] == 0
        assert data["player_2"]["hp"] == 0


# ---------------------------------------------------------------------------
# Test load
# ---------------------------------------------------------------------------


class TestSessionStoreLoad:
    """Tests for SessionStore.load()."""

    def test_load_returns_none_for_missing_file(self, temp_dir):
        """load() returns None when the game file doesn't exist."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        saves_dir.mkdir()
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        result = store.load("nonexistent")
        assert result is None

    def test_load_restores_game_manager(self, temp_dir, populated_gm):
        """load() returns a GameManager with equivalent state to what was saved."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        store.save("game-1", populated_gm)

        restored = store.load("game-1")
        assert restored is not None
        assert isinstance(restored, GameManager)
        assert restored.round_number == populated_gm.round_number
        assert restored.player_1.hp == populated_gm.player_1.hp
        assert restored.player_2.hp == populated_gm.player_2.hp
        assert restored.is_over == populated_gm.is_over
        assert restored.winner == populated_gm.winner

    def test_load_restores_hand_correctly(self, temp_dir, populated_gm):
        """Player hands are restored with correct card types and order."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        store.save("game-1", populated_gm)

        restored = store.load("game-1")
        # Player 1 played A and H → 5 remaining cards
        assert len(restored.player_1.hand) == 5
        # Player 2 played D and D → 5 remaining cards
        assert len(restored.player_2.hand) == 5
        # All cards should be CardType instances
        for card in restored.player_1.hand:
            assert isinstance(card, CardType)
        for card in restored.player_2.hand:
            assert isinstance(card, CardType)

    def test_load_restores_history(self, temp_dir, populated_gm):
        """Player histories are restored correctly."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        store.save("game-1", populated_gm)

        restored = store.load("game-1")
        assert restored.player_1.history == [A, H]
        assert restored.player_2.history == [A, D]

    def test_load_restores_round_log(self, temp_dir, populated_gm):
        """The round_log list is restored with all RoundResult data."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        store.save("game-1", populated_gm)

        restored = store.load("game-1")
        assert len(restored.round_log) == 2
        r1 = restored.round_log[0]
        assert r1.round == 1
        assert r1.cards_1 == [A]
        assert r1.cards_2 == [A]
        assert r1.damage_1 == 1
        assert r1.damage_2 == 1
        assert len(r1.details) == 2


# ---------------------------------------------------------------------------
# Test round-trip consistency
# ---------------------------------------------------------------------------


class TestSessionStoreRoundTrip:
    """End-to-end save→load consistency tests."""

    def test_round_trip_game_continuable(self, temp_dir):
        """A saved game can be loaded and continued (play another round)."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))

        # Start and play one round
        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [D])
        store.save("game-1", gm)

        # Load and continue
        restored = store.load("game-1")
        result = restored.play_round([A], [A])
        assert result is not None
        assert restored.round_number == 3

    def test_round_trip_preserves_get_state(self, temp_dir, populated_gm):
        """get_state() returns identical data before and after round-trip."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        original_state = populated_gm.get_state()

        store = SessionStore(config_path=str(config_path))
        store.save("game-1", populated_gm)
        restored = store.load("game-1")

        restored_state = restored.get_state()
        assert restored_state == original_state


# ---------------------------------------------------------------------------
# Test delete
# ---------------------------------------------------------------------------


class TestSessionStoreDelete:
    """Tests for SessionStore.delete()."""

    def test_delete_returns_true_when_file_existed(self, temp_dir, populated_gm):
        """delete() returns True when the file was present and removed."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        store.save("game-1", populated_gm)

        result = store.delete("game-1")
        assert result is True
        assert not (saves_dir / "game-1.json").exists()

    def test_delete_returns_false_when_file_absent(self, temp_dir):
        """delete() returns False when no file existed."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        saves_dir.mkdir()
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        store = SessionStore(config_path=str(config_path))
        result = store.delete("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# Test corruption handling
# ---------------------------------------------------------------------------


class TestSessionStoreCorruption:
    """Tests for corrupted JSON files."""

    def test_load_corrupted_json_raises_value_error(self, temp_dir):
        """Loading a file with malformed JSON raises ValueError."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        saves_dir.mkdir()
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        corrupt_file = saves_dir / "corrupt.json"
        corrupt_file.write_text("this is not valid { json")

        store = SessionStore(config_path=str(config_path))
        with pytest.raises(ValueError, match="Failed to parse game file"):
            store.load("corrupt")

    def test_load_valid_json_missing_keys_raises_value_error(self, temp_dir):
        """Loading valid JSON that lacks expected game keys raises ValueError."""
        from src.backend.game.session_store import SessionStore

        saves_dir = temp_dir / "saves"
        saves_dir.mkdir()
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

        incomplete = saves_dir / "incomplete.json"
        incomplete.write_text(json.dumps({"round": 1}))  # missing player data

        store = SessionStore(config_path=str(config_path))
        with pytest.raises(ValueError, match="Missing required game state key"):
            store.load("incomplete")
