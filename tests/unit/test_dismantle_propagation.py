"""Tests for dismantle_target propagation through AI and API layers (design/37).

Covers the full dismantle flow from policy_index_to_action → ai_select_action
→ api.py → GameCore.play_round(). These tests were missing in the original
design/37 implementation, which led to runtime errors when AI selected
dismantle actions.
"""

import pytest
import torch

from src.backend.ai.constants import HIDDEN_SIZE, POLICY_OUTPUT_SIZE
from src.backend.app.converter import Converter
from src.backend.game.card_type import CardType
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus
from tests.helpers import make_game_state


# ============================================================================
# T1: ai_select_action returns dismantle_target
# ============================================================================


class TestAiSelectActionDismantle:
    """Verify ai_select_action returns dismantle_target when AI selects dismantle mode."""

    def test_ai_select_pass_returns_all_none(self, app_manager):
        """pass (index 0): card=None, build_target=None, dismantle_target=None."""
        state = make_game_state(
            hand_2=[CardType.ATTACK],
            hp_1=20, hp_2=20,
        )
        # index 0 = pass → all None
        _force_policy_index(app_manager, 0)
        card, bt, dt, _h = app_manager.ai_select_action(state, [0.0] * 32, "rnn_v1.1")
        assert card is None
        assert bt is None
        assert dt is None

    def test_ai_select_play_returns_correct_form(self, app_manager):
        """Pure play: build_target=None, dismantle_target=None."""
        state = make_game_state(
            hand_2=[CardType.ATTACK],
            hp_1=20, hp_2=20,
        )
        # Force model to output index 3 (play ATTACK)
        _force_policy_index(app_manager, 3)
        card, bt, dt, _h = app_manager.ai_select_action(state, [0.0] * 32, "rnn_v1.1")
        assert card == CardType.ATTACK
        assert bt is None
        assert dt is None

    def test_ai_select_build_returns_build_target(self, app_manager):
        """Build mode: dismantle_target=None."""
        state = make_game_state(
            hand_2=[CardType.BUILD],
            hp_1=20, hp_2=20, energy_2=2,
        )
        # index 4 = build ATTACK
        _force_policy_index(app_manager, 4)
        card, bt, dt, _h = app_manager.ai_select_action(state, [0.0] * 32, "rnn_v1.1")
        assert card == CardType.BUILD
        assert bt == CardType.ATTACK
        assert dt is None

    def test_ai_select_dismantle_returns_dismantle_target(self, app_manager):
        """Dismantle mode: build_target=None, dismantle_target is set."""
        state = make_game_state(
            hand_2=[CardType.BUILD],
            hp_1=20, hp_2=20, energy_2=0,
            discard_pile_2=[CardType.ATTACK, CardType.DEFENSE],
        )
        # index 5 = dismantle ATTACK
        _force_policy_index(app_manager, 5)
        card, bt, dt, _h = app_manager.ai_select_action(state, [0.0] * 32, "rnn_v1.1")
        assert card == CardType.BUILD
        assert bt is None
        assert dt == CardType.ATTACK

    def test_ai_select_dismantle_defense(self, app_manager):
        """Dismantle DEFENSE when discard has DEFENSE."""
        state = make_game_state(
            hand_2=[CardType.BUILD],
            hp_1=20, hp_2=20, energy_2=0,
            discard_pile_2=[CardType.DEFENSE],
        )
        # index 8 = dismantle DEFENSE
        _force_policy_index(app_manager, 8)
        card, bt, dt, _h = app_manager.ai_select_action(state, [0.0] * 32, "rnn_v1.1")
        assert card == CardType.BUILD
        assert bt is None
        assert dt == CardType.DEFENSE


# ============================================================================
# T2: ai_sample_action returns dismantle_target
# ============================================================================


class TestAiSampleActionDismantle:
    """Verify ai_sample_action returns dismantle_target."""

    def test_sample_dismantle_returns_dismantle_target(self, app_manager):
        """ai_sample_action in dismantle mode returns dt in the 7-tuple."""
        state = make_game_state(
            hand_2=[CardType.BUILD],
            hp_1=20, hp_2=20, energy_2=0,
            discard_pile_2=[CardType.ATTACK],
        )
        _force_policy_index(app_manager, 5)  # dismantle ATTACK
        card, bt, dt, h_pol, lp, val, h_val = app_manager.ai_sample_action(
            state, [0.0] * 32, [0.0] * 32, "rnn_v1.1",
            temperature=0.0, player_key="2",
        )
        assert card == CardType.BUILD
        assert bt is None
        assert dt == CardType.ATTACK
        assert isinstance(h_pol, list) and len(h_pol) == HIDDEN_SIZE
        assert isinstance(h_val, list) and len(h_val) == HIDDEN_SIZE
        assert lp is None  # temperature=0 → log_prob=None


# ============================================================================
# T3: API integration — dismantle_target_1 from request
# ============================================================================


class TestApiDismantleTarget:
    """Verify the API passes dismantle_target_1 to GameCore."""

    def test_pve_play_round_passes_dismantle_to_gamecore(self, api_client):
        """PVE dismantle: API forwards AI's dismantle_target to play_round.

        Creates a PVE game, patches ai_select_action to return BUILD + dismantle,
        sets up P2's hand/discard, and verifies the round result contains
        DISMANTLE in details.
        """
        from unittest.mock import patch

        # Create PVE game
        resp = api_client.post("/api/games", json={"mode": "pve", "ai": "rnn_v1.1"})
        assert resp.status_code == 201
        gid = resp.get_json()["game_id"]

        # Set up P2: BUILD in hand, ATTACK in discard, energy=0 (dismantle needs no energy)
        from src.backend.game.session_store import SessionStore
        app_instance = api_client.application
        store: SessionStore = app_instance.config["SESSION_STORE"]
        gm, metadata = store.load(gid)
        gm.player_2.hand = [CardType.BUILD]
        gm.player_2.energy = 0
        gm.player_2.discard_pile = [CardType.ATTACK]
        gm.player_1.hand = [CardType.DEFENSE]
        store.save(gid, gm, ai=metadata.get("ai"), ai_state=metadata.get("ai_state"),
                   ai_value_state=metadata.get("ai_value_state"))

        # Patch ai_select_action to pick dismantle ATTACK (P2 is AI in PVE)
        app_manager = app_instance.config["APP_MANAGER"]
        with patch.object(
            app_manager, "ai_select_action",
            return_value=(CardType.BUILD, None, CardType.ATTACK, [0.0] * 32),
        ):
            resp2 = api_client.post(f"/api/games/{gid}/play", json={
                "cards_1": ["defense"],
            })

        assert resp2.status_code == 200, (
            f"Expected 200, got {resp2.status_code}: {resp2.get_json()}"
        )
        rr = resp2.get_json()["round_result"]
        # Verify P2's dismantle happened
        details = rr["after_resolve"]["details"]
        assert any("DISMANTLE attack" in d or "dismantle" in d.lower() for d in details), \
            f"DISMANTLE not found in details: {details}"

    def test_pvp_dismantle_request_accepted(self, api_client):
        """PVP: dismantle_target_1 in request body is accepted and passed through."""
        from src.backend.game.session_store import SessionStore

        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]

        # Set hands manually (avoid random-deal flakiness from shuffling).
        app_instance = api_client.application
        store: SessionStore = app_instance.config["SESSION_STORE"]
        gm, _meta = store.load(gid)
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]
        store.save(gid, gm)

        # Round 1: P1 plays ATTACK (goes to discard), P2 plays DEFENSE
        resp = api_client.post(f"/api/games/{gid}/play", json={
            "cards_1": ["attack"],
            "cards_2": ["defense"],
        })
        assert resp.status_code == 200

        # Now manually inject BUILD into P1's hand and ATTACK to discard
        gm, _meta = store.load(gid)
        gm.player_1.hand = [CardType.BUILD, CardType.ATTACK]
        gm.player_1.discard_pile = [CardType.ATTACK, CardType.DEFENSE]
        gm.player_2.hand = [CardType.DEFENSE, CardType.ATTACK]
        store.save(gid, gm)

        # Round 2: P1 plays BUILD with dismantle ATTACK
        resp3 = api_client.post(f"/api/games/{gid}/play", json={
            "cards_1": ["build"],
            "cards_2": ["defense"],
            "dismantle_target_1": "attack",
        })
        assert resp3.status_code == 200, f"Expected 200, got {resp3.status_code}: {resp3.get_json()}"
        rr = resp3.get_json()["round_result"]
        # Verify dismantle happened: ATTACK should be removed from P1's discard
        # (We can check the details string)
        assert any("DISMANTLE attack" in d for d in rr["after_resolve"]["details"]), \
            f"DISMANTLE not found in details: {rr['after_resolve']['details']}"

    def test_dismantle_missing_from_discard_returns_400(self, api_client):
        """Dismantle target not in discard returns 400."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]

        # Set up: P1 has BUILD, discard is empty
        from src.backend.game.session_store import SessionStore
        app_instance = api_client.application
        store: SessionStore = app_instance.config["SESSION_STORE"]
        gm, metadata = store.load(gid)
        gm.player_1.hand = [CardType.BUILD]
        gm.player_1.discard_pile = []  # empty!
        gm.player_2.hand = [CardType.DEFENSE]
        store.save(gid, gm)

        resp2 = api_client.post(f"/api/games/{gid}/play", json={
            "cards_1": ["build"],
            "cards_2": ["defense"],
            "dismantle_target_1": "attack",
        })
        assert resp2.status_code == 400
        assert "not found in discard" in resp2.get_json()["error"].lower()

    def test_build_and_dismantle_mutually_exclusive_400(self, api_client):
        """Both build_target_1 and dismantle_target_1 → 400."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]

        from src.backend.game.session_store import SessionStore
        app_instance = api_client.application
        store: SessionStore = app_instance.config["SESSION_STORE"]
        gm, metadata = store.load(gid)
        gm.player_1.hand = [CardType.BUILD]
        gm.player_1.energy = 3
        gm.player_1.discard_pile = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]
        store.save(gid, gm)

        resp2 = api_client.post(f"/api/games/{gid}/play", json={
            "cards_1": ["build"],
            "cards_2": ["defense"],
            "build_target_1": "attack",
            "dismantle_target_1": "attack",
        })
        assert resp2.status_code == 400
        assert "mutually exclusive" in resp2.get_json()["error"].lower()

    def test_dismantle_no_energy_required(self, api_client):
        """Dismantle succeeds even when energy=0 (no energy cost)."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]

        from src.backend.game.session_store import SessionStore
        app_instance = api_client.application
        store: SessionStore = app_instance.config["SESSION_STORE"]
        gm, metadata = store.load(gid)
        gm.player_1.hand = [CardType.BUILD]
        gm.player_1.energy = 0  # energy=0
        gm.player_1.discard_pile = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]
        store.save(gid, gm)

        resp2 = api_client.post(f"/api/games/{gid}/play", json={
            "cards_1": ["build"],
            "cards_2": ["defense"],
            "dismantle_target_1": "attack",
        })
        assert resp2.status_code == 200, (
            f"Dismantle should work with energy=0, got {resp2.status_code}: "
            f"{resp2.get_json()}"
        )

    def test_dismantle_survives_save_load_roundtrip(self, api_client):
        """After save+load, dismantle_target is preserved in history."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]

        from src.backend.game.session_store import SessionStore
        app_instance = api_client.application
        store: SessionStore = app_instance.config["SESSION_STORE"]
        gm, metadata = store.load(gid)
        gm.player_1.hand = [CardType.BUILD]
        gm.player_1.energy = 0
        gm.player_1.discard_pile = [CardType.HEAVY]
        gm.player_2.hand = [CardType.DEFENSE]
        store.save(gid, gm)

        # Round 1: P1 dismantles HEAVY
        api_client.post(f"/api/games/{gid}/play", json={
            "cards_1": ["build"],
            "cards_2": ["defense"],
            "dismantle_target_1": "heavy",
        })

        # Save and reload — simulates page refresh
        gm2, meta2 = store.load(gid)
        store.save(gid, gm2)
        gm3, meta3 = store.load(gid)

        # Verify history has dismantle_target preserved after roundtrip
        hist = gm3.player_1.history
        assert len(hist) == 1
        assert hist[0].card == CardType.BUILD
        assert hist[0].dismantle_target == CardType.HEAVY
        assert hist[0].build_target is None

        # Play another round — should NOT crash on history processing
        gm3.player_1.hand = [CardType.ATTACK]
        gm3.player_2.hand = [CardType.DEFENSE]
        store.save(gid, gm3)
        resp2 = api_client.post(f"/api/games/{gid}/play", json={
            "cards_1": ["attack"],
            "cards_2": ["defense"],
        })
        assert resp2.status_code == 200, (
            f"Round after dismantle save/load should work, "
            f"got {resp2.status_code}: {resp2.get_json()}"
        )


class _FakeValueModel(torch.nn.Module):
    """A value model that always returns 0.0 and zero hidden state."""

    def forward(self, x, hidden):
        value = torch.zeros(1, 1)
        new_hidden = torch.zeros(1, 1, HIDDEN_SIZE)
        return value, new_hidden


# ============================================================================
# Helper: force policy model to output a specific index
# ============================================================================


class _FakePolicyModel(torch.nn.Module):
    """A policy model that always outputs a fixed action index via argmax."""

    def __init__(self, forced_index: int):
        super().__init__()
        self._forced = forced_index

    def forward(self, x, hidden, mask_fn=None, temperature=1.0):
        """Return probs with forced argmax, plus dummy log_probs and hidden."""
        probs = torch.full((1, POLICY_OUTPUT_SIZE), -100.0)
        # mask_fn is applied AFTER logit output but BEFORE softmax in the real model
        # Here we ignore mask_fn since we're forcing the output
        if mask_fn is not None:
            # Let mask_fn modify our probs (it may -inf some positions)
            masked = mask_fn(probs)
            # But we still force the desired index
            masked[0, self._forced] = 100.0
            probs = masked
        else:
            probs[0, self._forced] = 100.0

        log_probs = torch.full((1, POLICY_OUTPUT_SIZE), -100.0)
        log_probs[0, self._forced] = 0.0
        new_hidden = torch.zeros(1, 1, HIDDEN_SIZE)
        return probs, log_probs, new_hidden


def _force_policy_index(app_manager, index: int):
    """Inject a fake adapter into AiRegistry cache for 'rnn_v1.1'."""
    from src.backend.ai_adapter.ai_registry import AiRegistry
    from src.backend.ai_adapter.rnn_ai import RnnAiAdapter

    AiRegistry._adapters.clear()
    adapter = RnnAiAdapter(
        policy_module="src.backend.ai.rnn_v1.model",
        policy_class="GRUPolicyModel",
        policy_weight_path="dummy.pth",
        value_module="src.backend.ai.rnn_v1.model",
        value_class="GRUValueModel",
        value_weight_path="dummy.pth",
    )
    adapter._policy_model = _FakePolicyModel(index)
    adapter._value_model_cache = _FakeValueModel()
    AiRegistry._adapters["rnn_v1.1"] = adapter
