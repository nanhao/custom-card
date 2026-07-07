"""Unit tests for app/app_manager.py — AppManager class.

Covers TC-AM1 through TC-AM12 from design/3 test_cases.md §4.
Tests game creation (PVP/PVE), AI action selection, error handling,
model caching, and hidden state evolution.

Self-play tests (SelfPlayConfig validation, ai_sample_action) are in
test_app_manager_selfplay.py.
"""

import json
import uuid

import pytest
import torch

from src.backend.app.app_manager import AppManager
from src.backend.game.session_store import SessionStore
from tests.helpers import make_game_state


def _init_hidden():
    """Return initial hidden state (8 zeros)."""
    return [0.0] * 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_store(tmp_path):
    """Create a SessionStore that writes to a temporary saves/ directory."""
    saves_dir = tmp_path / "saves"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))
    return SessionStore(config_path=str(config_path))


@pytest.fixture
def app_manager(session_store):
    """Create an AppManager with a temporary SessionStore."""
    return AppManager(session_store)


# ---------------------------------------------------------------------------
# TC-AM1: create_game PVP
# ---------------------------------------------------------------------------


class TestCreateGamePVP:
    """TC-AM1: create_game in PVP mode."""

    def test_returns_game_id_and_state(self, app_manager):
        """create_game(mode='pvp') returns (game_id, state) tuple."""
        game_id, state = app_manager.create_game(mode="pvp")
        assert isinstance(game_id, str)
        # game_id should be a valid UUID
        uuid.UUID(game_id)
        assert isinstance(state, dict)

    def test_state_mode_is_pvp(self, app_manager):
        """State must have mode='pvp'."""
        _, state = app_manager.create_game(mode="pvp")
        assert state["mode"] == "pvp"

    def test_state_ai_is_none(self, app_manager):
        """State must have ai=None for PVP."""
        _, state = app_manager.create_game(mode="pvp")
        assert state["ai"] is None

    def test_state_has_correct_initial_values(self, app_manager):
        """State must have round=1, both HP=3, hands=7 each."""
        _, state = app_manager.create_game(mode="pvp")
        assert state["round"] == 1
        assert state["players"]["1"]["hp"] == 3
        assert state["players"]["2"]["hp"] == 3
        assert len(state["players"]["1"]["hand"]) == 7
        assert len(state["players"]["2"]["hand"]) == 7

    def test_game_is_saved_to_store(self, app_manager, session_store):
        """After creation, game must be loadable from SessionStore."""
        game_id, _ = app_manager.create_game(mode="pvp")
        gm, metadata = session_store.load(game_id)
        assert gm is not None
        assert metadata is not None


# ---------------------------------------------------------------------------
# TC-AM2: create_game PVE
# ---------------------------------------------------------------------------


class TestCreateGamePVE:
    """TC-AM2: create_game in PVE mode."""

    def test_returns_game_id_and_state(self, app_manager):
        """create_game(mode='pve', ai_name='rnn_v2.1') returns (game_id, state)."""
        game_id, state = app_manager.create_game(mode="pve", ai_name="rnn_v2.1")
        assert isinstance(game_id, str)
        uuid.UUID(game_id)
        assert isinstance(state, dict)

    def test_state_mode_is_pve(self, app_manager):
        """State must have mode='pve'."""
        _, state = app_manager.create_game(mode="pve", ai_name="rnn_v2.1")
        assert state["mode"] == "pve"

    def test_state_ai_is_correct(self, app_manager):
        """State must have ai='rnn_v2.1'."""
        _, state = app_manager.create_game(mode="pve", ai_name="rnn_v2.1")
        assert state["ai"] == "rnn_v2.1"

    def test_save_file_contains_ai_metadata(self, app_manager, session_store):
        """Save file must contain ai_state and ai fields."""
        game_id, _ = app_manager.create_game(mode="pve", ai_name="rnn_v2.1")
        gm, metadata = session_store.load(game_id)
        assert metadata["ai"] == "rnn_v2.1"
        assert metadata["ai_state"] == [0.0] * 8

    def test_player2_hand_is_standard_7_cards(self, app_manager):
        """Player 2's hand must be a standard 7-card starting hand."""
        _, state = app_manager.create_game(mode="pve", ai_name="rnn_v2.1")
        assert len(state["players"]["2"]["hand"]) == 7


# ---------------------------------------------------------------------------
# TC-AM3, TC-AM4: create_game PVE error cases
# ---------------------------------------------------------------------------


class TestCreateGamePVEErrors:
    """TC-AM3, TC-AM4: Error handling for PVE game creation."""

    def test_invalid_mode_raises_value_error(self, app_manager):
        """create_game with invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            app_manager.create_game(mode="pva")

    def test_unknown_ai_raises_value_error(self, app_manager):
        """TC-AM3: create_game with unknown AI raises ValueError."""
        with pytest.raises(ValueError):
            app_manager.create_game(mode="pve", ai_name="nonexistent")

    def test_unknown_ai_no_save_file(self, app_manager, session_store):
        """TC-AM3: No save file created when AI is unknown."""
        try:
            app_manager.create_game(mode="pve", ai_name="nonexistent")
        except ValueError:
            pass
        # The saves directory should be empty (no game was created)
        saves_dir = session_store.save_dir
        json_files = list(saves_dir.glob("*.json")) if saves_dir.exists() else []
        assert len(json_files) == 0

    def test_missing_ai_name_raises_value_error(self, app_manager):
        """TC-AM4: create_game(mode='pve') without ai_name raises ValueError."""
        with pytest.raises(ValueError, match="ai"):
            app_manager.create_game(mode="pve")

    def test_missing_ai_error_message_mentions_missing_ai(self, app_manager):
        """TC-AM4: Error message mentions missing AI."""
        with pytest.raises(ValueError, match="[Mm]issing"):
            app_manager.create_game(mode="pve")


# ---------------------------------------------------------------------------
# TC-AM5: create_game produces unique IDs
# ---------------------------------------------------------------------------


class TestCreateGameUniqueIds:
    """TC-AM5: Each create_game call produces unique IDs."""

    def test_different_ids(self, app_manager):
        """Two consecutive create_game calls return different IDs."""
        id1, _ = app_manager.create_game(mode="pvp")
        id2, _ = app_manager.create_game(mode="pvp")
        assert id1 != id2

    def test_both_games_independently_loadable(self, app_manager, session_store):
        """Both games can be loaded independently."""
        id1, _ = app_manager.create_game(mode="pvp")
        id2, _ = app_manager.create_game(mode="pvp")
        gm1, _ = session_store.load(id1)
        gm2, _ = session_store.load(id2)
        assert gm1 is not None
        assert gm2 is not None


# ---------------------------------------------------------------------------
# TC-AM6: ai_select_action returns legal card
# ---------------------------------------------------------------------------


class TestAiSelectAction:
    """TC-AM6–TC-AM9, TC-AM12: AI action selection."""

    def test_returns_tuple_of_str_and_list(self, app_manager):
        """TC-AM6: Returns (card_str, new_hidden) with full hand."""
        state = make_game_state()
        h0 = _init_hidden()
        card_str, new_hidden = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        assert isinstance(card_str, str)
        assert isinstance(new_hidden, list)
        assert len(new_hidden) == 8

    def test_card_str_is_valid_card(self, app_manager):
        """TC-AM6: Returned card must be a valid card type."""
        state = make_game_state()
        h0 = _init_hidden()
        card_str, _ = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        assert card_str in ("attack", "defense", "heavy", "counter")

    def test_hidden_state_evolves(self, app_manager):
        """TC-AM6: Hidden state must change after inference."""
        state = make_game_state()
        h0 = _init_hidden()
        _, new_hidden = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        assert new_hidden != [0.0] * 8, "Hidden state should evolve"

    def test_limited_hand_returns_only_legal_card(self, app_manager):
        """TC-AM7: With only one card, must return that card."""
        state = make_game_state(hand_2=["counter"])
        h0 = _init_hidden()
        card_str, _ = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        assert card_str == "counter"

    def test_hidden_state_evolves_across_calls(self, app_manager):
        """TC-AM8: Hidden state evolves across sequential calls."""
        state = make_game_state()
        h0 = _init_hidden()
        _, h1 = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        _, h2 = app_manager.ai_select_action(state, h1, "rnn_v2.1")
        assert h1 != h0
        assert h2 != h1

    def test_deterministic_with_same_input(self, app_manager):
        """TC-AM9: Same input → same output (deterministic)."""
        state = make_game_state()
        h0 = _init_hidden()
        card1, h1 = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        card2, h2 = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        assert card1 == card2
        assert h1 == h2

    def test_single_defense_hand_returns_defense(self, app_manager):
        """TC-AM12: AI with only 'defense' must return 'defense'."""
        state = make_game_state(hand_2=["defense"])
        h0 = _init_hidden()
        card_str, _ = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        assert card_str == "defense"


# ---------------------------------------------------------------------------
# TC-AM10: ai_select_action unknown AI
# ---------------------------------------------------------------------------


class TestAiSelectActionErrors:
    """TC-AM10: Error handling for ai_select_action."""

    def test_unknown_ai_raises_value_error(self, app_manager):
        """ai_select_action with unknown AI raises ValueError."""
        state = make_game_state()
        h0 = [0.0] * 8
        with pytest.raises(ValueError):
            app_manager.ai_select_action(state, h0, "nonexistent")


# ---------------------------------------------------------------------------
# TC-AM11: Model caching
# ---------------------------------------------------------------------------


class TestModelCaching:
    """TC-AM11: Model is cached and not reloaded."""

    def test_model_cached_between_calls(self, app_manager):
        """Two ai_select_action calls for same AI use cached model.

        Verifies caching by mocking importlib.import_module — the
        second call should not trigger a re-import.
        """
        from unittest.mock import patch

        state = make_game_state()
        h0 = [0.0] * 8

        # First call loads the model (importlib is called)
        card1, h1 = app_manager.ai_select_action(state, h0, "rnn_v2.1")
        assert card1 in ("attack", "defense", "heavy", "counter")

        # Second call should use cache — verify importlib is NOT called again
        with patch("importlib.import_module") as mock_import:
            card2, h2 = app_manager.ai_select_action(state, h1, "rnn_v2.1")
            mock_import.assert_not_called()
            assert card2 in ("attack", "defense", "heavy", "counter")
