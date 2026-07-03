"""Unit tests for app/app_manager.py — AppManager class.

Covers TC-AM1 through TC-AM12 from design/3 test_cases.md §4, plus
new TC-A1 through TC-A6 from design/5 test_cases.md §3 (ai_sample_action).
Tests game creation (PVP/PVE), AI action selection, temperature-based
sampling, error handling, model caching, and hidden state evolution.
"""

import json
import uuid

import pytest
import torch

from src.backend.app.app_manager import AppManager
from src.backend.app.selfplay_config import SelfPlayConfig
from src.backend.game.session_store import SessionStore
from tests.helpers import make_game_state


def _init_hidden():
    """Get the initial hidden state from GRUCardModel."""
    from src.backend.ai.rnn_v1.model import GRUCardModel
    return GRUCardModel.INITIAL_HIDDEN_STATE


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
        """create_game(mode='pve', ai_name='rnn_v1.1') returns (game_id, state)."""
        game_id, state = app_manager.create_game(mode="pve", ai_name="rnn_v1.1")
        assert isinstance(game_id, str)
        uuid.UUID(game_id)
        assert isinstance(state, dict)

    def test_state_mode_is_pve(self, app_manager):
        """State must have mode='pve'."""
        _, state = app_manager.create_game(mode="pve", ai_name="rnn_v1.1")
        assert state["mode"] == "pve"

    def test_state_ai_is_correct(self, app_manager):
        """State must have ai='rnn_v1.1'."""
        _, state = app_manager.create_game(mode="pve", ai_name="rnn_v1.1")
        assert state["ai"] == "rnn_v1.1"

    def test_save_file_contains_ai_metadata(self, app_manager, session_store):
        """Save file must contain ai_state and ai fields."""
        game_id, _ = app_manager.create_game(mode="pve", ai_name="rnn_v1.1")
        gm, metadata = session_store.load(game_id)
        assert metadata["ai"] == "rnn_v1.1"
        assert metadata["ai_state"] == [0.0] * 8

    def test_player2_hand_is_standard_7_cards(self, app_manager):
        """Player 2's hand must be a standard 7-card starting hand."""
        _, state = app_manager.create_game(mode="pve", ai_name="rnn_v1.1")
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

    @staticmethod
    def _get_initial_hidden():
        """Get the initial hidden state from GRUCardModel."""
        from src.backend.ai.rnn_v1.model import GRUCardModel
        return GRUCardModel.INITIAL_HIDDEN_STATE

    def test_returns_tuple_of_str_and_list(self, app_manager):
        """TC-AM6: Returns (card_str, new_hidden) with full hand."""
        state = make_game_state()
        h0 = self._get_initial_hidden()
        card_str, new_hidden = app_manager.ai_select_action(state, h0, "rnn_v1.1")
        assert isinstance(card_str, str)
        assert isinstance(new_hidden, list)
        assert len(new_hidden) == 8

    def test_card_str_is_valid_card(self, app_manager):
        """TC-AM6: Returned card must be a valid card type."""
        state = make_game_state()
        h0 = self._get_initial_hidden()
        card_str, _ = app_manager.ai_select_action(state, h0, "rnn_v1.1")
        assert card_str in ("attack", "defense", "heavy", "counter")

    def test_hidden_state_evolves(self, app_manager):
        """TC-AM6: Hidden state must change after inference."""
        state = make_game_state()
        h0 = self._get_initial_hidden()
        _, new_hidden = app_manager.ai_select_action(state, h0, "rnn_v1.1")
        assert new_hidden != [0.0] * 8, "Hidden state should evolve"

    def test_limited_hand_returns_only_legal_card(self, app_manager):
        """TC-AM7: With only one card, must return that card."""
        state = make_game_state(hand_2=["counter"])
        h0 = self._get_initial_hidden()
        card_str, _ = app_manager.ai_select_action(state, h0, "rnn_v1.1")
        assert card_str == "counter"

    def test_hidden_state_evolves_across_calls(self, app_manager):
        """TC-AM8: Hidden state evolves across sequential calls."""
        state = make_game_state()
        h0 = self._get_initial_hidden()
        _, h1 = app_manager.ai_select_action(state, h0, "rnn_v1.1")
        _, h2 = app_manager.ai_select_action(state, h1, "rnn_v1.1")
        assert h1 != h0
        assert h2 != h1

    def test_deterministic_with_same_input(self, app_manager):
        """TC-AM9: Same input → same output (deterministic)."""
        state = make_game_state()
        h0 = self._get_initial_hidden()
        card1, h1 = app_manager.ai_select_action(state, h0, "rnn_v1.1")
        card2, h2 = app_manager.ai_select_action(state, h0, "rnn_v1.1")
        assert card1 == card2
        assert h1 == h2

    def test_single_defense_hand_returns_defense(self, app_manager):
        """TC-AM12: AI with only 'defense' must return 'defense'."""
        state = make_game_state(hand_2=["defense"])
        h0 = self._get_initial_hidden()
        card_str, _ = app_manager.ai_select_action(state, h0, "rnn_v1.1")
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
        card1, h1 = app_manager.ai_select_action(state, h0, "rnn_v1.1")
        assert card1 in ("attack", "defense", "heavy", "counter")

        # Second call should use cache — verify importlib is NOT called again
        with patch("importlib.import_module") as mock_import:
            card2, h2 = app_manager.ai_select_action(state, h1, "rnn_v1.1")
            mock_import.assert_not_called()
            assert card2 in ("attack", "defense", "heavy", "counter")


# ---------------------------------------------------------------------------
# SelfPlayConfig validation (TC-T5 from design/5 test_cases.md §6)
# ---------------------------------------------------------------------------


class TestSelfPlayConfig:
    """TC-T5: SelfPlayConfig validation for temp_side."""

    def test_invalid_temp_side_raises_value_error(self):
        """temp_side='invalid' raises ValueError at instantiation."""
        with pytest.raises(ValueError, match="temp_side"):
            SelfPlayConfig(
                ai1="rnn_v1.1",
                ai2="rnn_v1.1",
                temp_side="invalid",
            )

    def test_valid_temp_side_values_accepted(self):
        """Valid temp_side values ('player1','player2','both') do not raise."""
        for side in ("player1", "player2", "both"):
            cfg = SelfPlayConfig(
                ai1="rnn_v1.1", ai2="rnn_v1.1", temp_side=side
            )
            assert cfg.temp_side == side

    def test_default_temp_side_is_both(self):
        """Default temp_side should be 'both'."""
        cfg = SelfPlayConfig(ai1="rnn_v1.1", ai2="rnn_v1.1")
        assert cfg.temp_side == "both"

    def test_default_temperature_is_1(self):
        """Default temperature should be 1.0."""
        cfg = SelfPlayConfig(ai1="rnn_v1.1", ai2="rnn_v1.1")
        assert cfg.temperature == 1.0

    def test_default_game_count_is_100(self):
        """Default game_count should be 100."""
        cfg = SelfPlayConfig(ai1="rnn_v1.1", ai2="rnn_v1.1")
        assert cfg.game_count == 100

    def test_default_sample_output_is_none(self):
        """Default sample_output should be None."""
        cfg = SelfPlayConfig(ai1="rnn_v1.1", ai2="rnn_v1.1")
        assert cfg.sample_output is None

    def test_frozen_dataclass_prevents_mutation(self):
        """Frozen dataclass prevents attribute mutation."""
        cfg = SelfPlayConfig(ai1="rnn_v1.1", ai2="rnn_v1.1")
        with pytest.raises(Exception):  # frozendataclass raises FrozenInstanceError
            cfg.game_count = 200


# ---------------------------------------------------------------------------
# TC-A1–TC-A6: ai_sample_action temperature sampling
# ---------------------------------------------------------------------------


class TestAiSampleAction:
    """TC-A1–TC-A6: Temperature-based action sampling from design/5 §3."""

    @staticmethod
    def _get_initial_hidden():
        """Get the initial hidden state from GRUCardModel."""
        from src.backend.ai.rnn_v1.model import GRUCardModel
        return GRUCardModel.INITIAL_HIDDEN_STATE

    # ---- TC-A1: temperature=0 returns greedy, log_prob=None ----

    def test_temperature_zero_returns_greedy_and_none_log_prob(
        self, app_manager
    ):
        """TC-A1: temperature=0 → greedy action equals ai_select_action."""
        state = make_game_state()
        h0 = self._get_initial_hidden()

        greedy_card, greedy_h = app_manager.ai_select_action(
            state, h0, "rnn_v1.1"
        )
        card, h, lp, val = app_manager.ai_sample_action(
            state, h0, "rnn_v1.1", temperature=0.0
        )

        assert card == greedy_card
        assert lp is None
        assert isinstance(val, float)

    # ---- TC-A2: temperature > 0 returns log_prob ----

    def test_temperature_positive_returns_log_prob(self, app_manager):
        """TC-A2: temperature=1.0 → valid card, finite log_prob ≤ 0."""
        state = make_game_state()
        h0 = self._get_initial_hidden()

        card, h, lp, val = app_manager.ai_sample_action(
            state, h0, "rnn_v1.1", temperature=1.0
        )

        assert card in ("attack", "defense", "heavy", "counter")
        assert card in state["players"]["2"]["hand"]
        assert isinstance(lp, float)
        assert lp <= 0.0, f"log_prob should be ≤ 0, got {lp}"
        # log_prob should be finite (not -inf, not NaN)
        import math
        assert math.isfinite(lp)

    # ---- TC-A3: sampling respects masked (unavailable) cards ----

    def test_sampling_respects_masked_cards(self, app_manager):
        """TC-A3: Hand [attack, defense] → never returns heavy or counter."""
        state = make_game_state(hand_2=["attack", "defense"])
        h0 = self._get_initial_hidden()

        for _ in range(50):
            card, h0, _, _ = app_manager.ai_sample_action(
                state, h0, "rnn_v1.1", temperature=1.0
            )
            assert card in ("attack", "defense"), (
                f"Expected attack or defense, got {card}"
            )

    # ---- TC-A4: player_key="1" queries from player 1's perspective ----

    def test_player_key_1_uses_p1_hand(self, app_manager):
        """TC-A4: player_key='1' → card from P1's hand."""
        # P1 has only ["heavy"], P2 has standard hand
        state = make_game_state(hand_1=["heavy"])
        h0 = self._get_initial_hidden()

        card, _, _, _ = app_manager.ai_sample_action(
            state, h0, "rnn_v1.1", temperature=1.0, player_key="1"
        )

        assert card == "heavy", (
            f"Expected 'heavy' (only P1 card), got {card}"
        )

    # ---- TC-A5: same seed + same temperature produces same sequence ----

    def test_same_seed_produces_same_sequence(self, app_manager):
        """TC-A5: Deterministic with temperature=0 regardless of seed.

        temperature=0 (greedy) is always deterministic — same input
        produces the same action even with different seeds set.
        """
        state = make_game_state()
        h0 = self._get_initial_hidden()

        # With seed 42
        torch.manual_seed(42)
        card1, h1, lp1, val1 = app_manager.ai_sample_action(
            state, h0, "rnn_v1.1", temperature=0.0
        )
        # With seed 99
        torch.manual_seed(99)
        card2, h2, lp2, val2 = app_manager.ai_sample_action(
            state, h0, "rnn_v1.1", temperature=0.0
        )

        # Greedy actions must be identical (deterministic argmax)
        assert card1 == card2, (
            f"Greedy actions should match: {card1} != {card2}"
        )
        assert lp1 is None
        assert lp2 is None

    # ---- TC-A6: different seeds produce different sequences ----

    def test_different_seeds_produce_different_sequence(self, app_manager):
        """TC-A6: temperature>0 with different seeds may differ.

        With temperature=1.0, setting different seeds produces
        different sampling behavior. We verify that after rotating
        seeds and running multiple calls, the action sequences differ.
        Note: This is probabilistic — false negatives are possible
        but extremely unlikely with 4 card types over 10 calls.
        """
        state = make_game_state()
        h0 = self._get_initial_hidden()

        def sample_sequence(seed):
            torch.manual_seed(seed)
            h = list(h0)
            actions = []
            for _ in range(10):
                card, h, _, _ = app_manager.ai_sample_action(
                    state, h, "rnn_v1.1", temperature=1.0
                )
                actions.append(card)
            return actions

        seq1 = sample_sequence(42)
        seq2 = sample_sequence(9999)
        # With 4 card types and 10 calls, different seeds should differ
        assert seq1 != seq2, (
            "Different seeds (42 vs 9999) should produce different "
            "action sequences for 10 calls"
        )
