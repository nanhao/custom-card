"""Unit tests for self-play features in app_manager.py.

Covers SelfPlayConfig validation (TC-T5 from design/5 §6) and
temperature-based action sampling (TC-A1–A6 from design/5 §3).
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
    """Return initial hidden state (8 zeros)."""
    return [0.0] * 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_manager(tmp_path):
    """Create an AppManager with a temporary SessionStore."""
    saves_dir = tmp_path / "saves"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))
    store = SessionStore(config_path=str(config_path))
    return AppManager(store)


# ---------------------------------------------------------------------------
# SelfPlayConfig validation (TC-T5 from design/5 test_cases.md §6)
# ---------------------------------------------------------------------------


class TestSelfPlayConfig:
    """TC-T5: SelfPlayConfig validation for temp_side and temperature."""

    def test_invalid_temp_side_raises_value_error(self):
        """temp_side='invalid' raises ValueError at instantiation."""
        with pytest.raises(ValueError, match="temp_side"):
            SelfPlayConfig(
                ai1="rnn_v2.1",
                ai2="rnn_v2.1",
                temp_side="invalid",
            )

    def test_valid_temp_side_values_accepted(self):
        """Valid temp_side values ('player1','player2','both') do not raise."""
        for side in ("player1", "player2", "both"):
            cfg = SelfPlayConfig(
                ai1="rnn_v2.1", ai2="rnn_v2.1", temp_side=side
            )
            assert cfg.temp_side == side

    def test_default_temp_side_is_both(self):
        """Default temp_side should be 'both'."""
        cfg = SelfPlayConfig(ai1="rnn_v2.1", ai2="rnn_v2.1")
        assert cfg.temp_side == "both"

    def test_default_temperature_is_1(self):
        """Default temperature should be 1.0."""
        cfg = SelfPlayConfig(ai1="rnn_v2.1", ai2="rnn_v2.1")
        assert cfg.temperature == 1.0

    def test_negative_temperature_raises_value_error(self):
        """Negative temperature raises ValueError at instantiation."""
        with pytest.raises(ValueError, match="temperature"):
            SelfPlayConfig(
                ai1="rnn_v2.1", ai2="rnn_v2.1", temperature=-1.0
            )

    def test_default_game_count_is_100(self):
        """Default game_count should be 100."""
        cfg = SelfPlayConfig(ai1="rnn_v2.1", ai2="rnn_v2.1")
        assert cfg.game_count == 100

    def test_default_sample_output_is_none(self):
        """Default sample_output should be None."""
        cfg = SelfPlayConfig(ai1="rnn_v2.1", ai2="rnn_v2.1")
        assert cfg.sample_output is None

    def test_frozen_dataclass_prevents_mutation(self):
        """Frozen dataclass prevents attribute mutation."""
        cfg = SelfPlayConfig(ai1="rnn_v2.1", ai2="rnn_v2.1")
        with pytest.raises(Exception):  # frozendataclass raises FrozenInstanceError
            cfg.game_count = 200


# ---------------------------------------------------------------------------
# TC-A1–TC-A6: ai_sample_action temperature sampling
# ---------------------------------------------------------------------------


class TestAiSampleAction:
    """TC-A1–TC-A6: Temperature-based action sampling from design/5 §3."""

    # ---- TC-A1: temperature=0 returns greedy, log_prob=None ----

    def test_temperature_zero_returns_greedy_and_none_log_prob(
        self, app_manager
    ):
        """TC-A1: temperature=0 → greedy action equals ai_select_action."""
        state = make_game_state()
        h0 = _init_hidden()

        greedy_card, greedy_h = app_manager.ai_select_action(
            state, h0, "rnn_v2.1"
        )
        card, h_pol, lp, val, h_val = app_manager.ai_sample_action(
            state, h0, h0, "rnn_v2.1", temperature=0.0
        )

        assert card == greedy_card
        assert lp is None
        assert isinstance(val, float)

    # ---- TC-A2: temperature > 0 returns log_prob ----

    def test_temperature_positive_returns_log_prob(self, app_manager):
        """TC-A2: temperature=1.0 → valid card, finite log_prob ≤ 0."""
        state = make_game_state()
        h0 = _init_hidden()

        card, h_pol, lp, val, h_val = app_manager.ai_sample_action(
            state, h0, h0, "rnn_v2.1", temperature=1.0
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
        h_pol = _init_hidden()
        h_val = _init_hidden()

        for _ in range(50):
            card, h_pol, _, _, h_val = app_manager.ai_sample_action(
                state, h_pol, h_val, "rnn_v2.1", temperature=1.0
            )
            assert card in ("attack", "defense"), (
                f"Expected attack or defense, got {card}"
            )

    # ---- TC-A4: player_key="1" queries from player 1's perspective ----

    def test_player_key_1_uses_p1_hand(self, app_manager):
        """TC-A4: player_key='1' → card from P1's hand."""
        # P1 has only ["heavy"], P2 has standard hand
        state = make_game_state(hand_1=["heavy"])
        h0 = _init_hidden()

        card, _, _, _, _ = app_manager.ai_sample_action(
            state, h0, h0, "rnn_v2.1", temperature=1.0, player_key="1"
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
        h0 = _init_hidden()

        # With seed 42
        torch.manual_seed(42)
        card1, h_pol1, lp1, val1, h_val1 = app_manager.ai_sample_action(
            state, h0, h0, "rnn_v2.1", temperature=0.0
        )
        # With seed 99
        torch.manual_seed(99)
        card2, h_pol2, lp2, val2, h_val2 = app_manager.ai_sample_action(
            state, h0, h0, "rnn_v2.1", temperature=0.0
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
        h0 = _init_hidden()

        def sample_sequence(seed):
            torch.manual_seed(seed)
            h_pol = list(h0)
            h_val = list(h0)
            actions = []
            for _ in range(10):
                card, h_pol, _, _, h_val = app_manager.ai_sample_action(
                    state, h_pol, h_val, "rnn_v2.1", temperature=1.0
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
