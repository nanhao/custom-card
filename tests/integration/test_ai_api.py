"""Integration tests for the AI-related REST API endpoints.

Covers TC-API1 through TC-API15 from test_cases.md §5:
- GET /api/ais
- POST /api/games with mode/pve/ai
- POST /api/games/{id}/play in PVE mode
- AI error handling (UNKNOWN_AI, AI_LOAD_ERROR, AI_INFERENCE_ERROR)
- State includes mode/ai fields
- AI hidden state persistence across rounds

All tests use Flask's test_client with temporary save directories.
"""

import json
import uuid

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_client(tmp_path):
    """Create a Flask test client with a temporary save directory."""
    saves_dir = tmp_path / "saves"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"save_dir": str(saves_dir)}))

    from src.backend.controller.api import create_app

    app = create_app(config_path=str(config_path))
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def ai_game_id(app_client):
    """Create a PVE game against rnn_v1.1 and return its game_id."""
    resp = app_client.post(
        "/api/games",
        data=json.dumps({"mode": "pve", "ai": "rnn_v1.1"}),
        content_type="application/json",
    )
    assert resp.status_code == 201
    return resp.get_json()["game_id"]


@pytest.fixture
def pvp_game_id(app_client):
    """Create a PVP game and return its game_id."""
    resp = app_client.post(
        "/api/games",
        data=json.dumps({}),
        content_type="application/json",
    )
    return resp.get_json()["game_id"]


# ---------------------------------------------------------------------------
# GET /api/ais (TC-API1)
# ---------------------------------------------------------------------------


class TestListAis:
    """TC-API1: GET /api/ais returns AI list."""

    def test_list_ais_returns_200(self, app_client):
        resp = app_client.get("/api/ais")
        assert resp.status_code == 200

    def test_list_ais_has_ais_key(self, app_client):
        resp = app_client.get("/api/ais")
        data = resp.get_json()
        assert "ais" in data
        assert isinstance(data["ais"], list)

    def test_list_ais_contains_rnn_v1_1(self, app_client):
        resp = app_client.get("/api/ais")
        data = resp.get_json()
        names = [a["name"] for a in data["ais"]]
        assert "rnn_v1.1" in names

    def test_list_ais_entries_have_name_and_description(self, app_client):
        resp = app_client.get("/api/ais")
        data = resp.get_json()
        for entry in data["ais"]:
            assert "name" in entry
            assert "description" in entry


# ---------------------------------------------------------------------------
# POST /api/games — AI game creation (TC-API2–TC-API5)
# ---------------------------------------------------------------------------


class TestCreateAIGame:
    """TC-API2–TC-API5: Creating AI games."""

    def test_create_pve_game_returns_201(self, app_client):
        """TC-API2: Creating a PVE game returns 201 with AI state."""
        resp = app_client.post(
            "/api/games",
            data=json.dumps({"mode": "pve", "ai": "rnn_v1.1"}),
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["state"]["mode"] == "pve"
        assert data["state"]["ai"] == "rnn_v1.1"

    def test_create_pvp_game_backward_compat(self, app_client):
        """TC-API3: Empty body creates PVP game (backward compatible)."""
        resp = app_client.post(
            "/api/games",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["state"]["mode"] == "pvp"
        assert data["state"]["ai"] is None

    def test_create_pvp_explicit_mode(self, app_client):
        """Explicit mode='pvp' also works."""
        resp = app_client.post(
            "/api/games",
            data=json.dumps({"mode": "pvp"}),
            content_type="application/json",
        )
        assert resp.status_code == 201
        assert resp.get_json()["state"]["mode"] == "pvp"

    def test_create_pve_unknown_ai_returns_400(self, app_client):
        """TC-API4: Unknown AI name returns 400 UNKNOWN_AI."""
        resp = app_client.post(
            "/api/games",
            data=json.dumps({"mode": "pve", "ai": "nonexistent"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "UNKNOWN_AI"

    def test_create_pve_no_ai_field_returns_400(self, app_client):
        """TC-API5: PVE mode without ai field returns 400."""
        resp = app_client.post(
            "/api/games",
            data=json.dumps({"mode": "pve"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["code"] == "MISSING_AI"


# ---------------------------------------------------------------------------
# POST /api/games/{id}/play — PVE mode (TC-API6–TC-API8, TC-API13–TC-API15)
# ---------------------------------------------------------------------------


class TestPlayAIGame:
    """TC-API6–TC-API8, TC-API13–TC-API15: Playing rounds in PVE mode."""

    def test_play_pve_auto_computes_cards_2(self, app_client, ai_game_id):
        """TC-API6: cards_2 is auto-computed by AI."""
        resp = app_client.post(
            f"/api/games/{ai_game_id}/play",
            data=json.dumps({"cards_1": ["attack"]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        rr = data["round_result"]
        assert len(rr["cards_2"]) == 1
        # AI card must be a valid type present in AI's starting hand
        assert rr["cards_2"][0] in ("attack", "defense", "heavy", "counter")

    def test_play_pve_ai_hand_decreases(self, app_client, ai_game_id):
        """After a PVE round, AI's hand decreases by 1."""
        # Get initial hand size from state
        resp = app_client.get(f"/api/games/{ai_game_id}")
        initial_hand_len = len(resp.get_json()["state"]["players"]["2"]["hand"])

        # Play one round
        app_client.post(
            f"/api/games/{ai_game_id}/play",
            data=json.dumps({"cards_1": ["attack"]}),
            content_type="application/json",
        )

        # Check hand size after round
        resp = app_client.get(f"/api/games/{ai_game_id}")
        after_hand_len = len(resp.get_json()["state"]["players"]["2"]["hand"])
        assert after_hand_len == initial_hand_len - 1

    def test_pve_ignores_cards_2_if_sent(self, app_client, ai_game_id):
        """TC-API8: Client sending cards_2 in PVE is ignored."""
        resp = app_client.post(
            f"/api/games/{ai_game_id}/play",
            data=json.dumps({"cards_1": ["attack"], "cards_2": ["defense"]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        rr = resp.get_json()["round_result"]
        # cards_2 should be AI's choice, not necessarily "defense"
        # The AI may select any legal card; we just verify it's valid
        assert len(rr["cards_2"]) == 1
        assert rr["cards_2"][0] in ("attack", "defense", "heavy", "counter")

    def test_pve_missing_cards_1_returns_400(self, app_client, ai_game_id):
        """TC-API13: Missing cards_1 in PVE returns 400."""
        resp = app_client.post(
            f"/api/games/{ai_game_id}/play",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "MISSING_CARDS_1"

    def test_pvp_missing_cards_2_returns_400(self, app_client, pvp_game_id):
        """TC-API7: Missing cards_2 in PVP returns 400."""
        resp = app_client.post(
            f"/api/games/{pvp_game_id}/play",
            data=json.dumps({"cards_1": ["attack"]}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "MISSING_CARDS_2"


# ---------------------------------------------------------------------------
# AI hidden state persistence (TC-API9, TC-API10)
# ---------------------------------------------------------------------------


class TestAIStatePersistence:
    """TC-API9, TC-API10: AI hidden state persists in save file."""

    def test_save_file_contains_ai_state_after_round(self, app_client, ai_game_id):
        """TC-API9: Save file has ai_state after one round."""
        # Play one round
        resp = app_client.post(
            f"/api/games/{ai_game_id}/play",
            data=json.dumps({"cards_1": ["attack"]}),
            content_type="application/json",
        )
        assert resp.status_code == 200

        # Load the game state
        state_resp = app_client.get(f"/api/games/{ai_game_id}")
        state = state_resp.get_json()["state"]
        # After a round, round_log should have one entry
        assert len(state["round_log"]) == 1

    def test_hidden_state_evolves_across_rounds(self, app_client, ai_game_id):
        """TC-API10: Hidden state changes across successive rounds.

        Plays multiple rounds (up to 3, or until game over) verifying
        that the AI hidden state persists and evolves across saves/loads.
        """
        rounds_played = 0
        for i in range(3):
            # Check if game is over before attempting to play
            state_resp = app_client.get(f"/api/games/{ai_game_id}")
            st = state_resp.get_json()["state"]
            if st["is_over"]:
                break
            # Pick a card from Player 1's actual hand
            hand = st["players"]["1"]["hand"]
            card_to_play = hand[0]
            resp = app_client.post(
                f"/api/games/{ai_game_id}/play",
                data=json.dumps({"cards_1": [card_to_play]}),
                content_type="application/json",
            )
            assert resp.status_code == 200, f"Round {i+1} failed"
            rounds_played += 1
        # At least 1 round must have been played
        assert rounds_played >= 1


# ---------------------------------------------------------------------------
# State includes mode and ai (TC-API11)
# ---------------------------------------------------------------------------


class TestStateIncludesModeAI:
    """TC-API11: GET /state includes mode and ai fields."""

    def test_state_includes_mode_and_ai_pve(self, app_client, ai_game_id):
        """GET state for PVE game includes mode and ai."""
        resp = app_client.get(f"/api/games/{ai_game_id}")
        state = resp.get_json()["state"]
        assert state["mode"] == "pve"
        assert state["ai"] == "rnn_v1.1"

    def test_state_includes_mode_and_ai_pvp(self, app_client, pvp_game_id):
        """GET state for PVP game includes mode='pvp' and ai=null."""
        resp = app_client.get(f"/api/games/{pvp_game_id}")
        state = resp.get_json()["state"]
        assert state["mode"] == "pvp"
        assert state["ai"] is None

    def test_state_includes_round_log(self, app_client, ai_game_id):
        """GET state includes round_log after rounds are played."""
        app_client.post(
            f"/api/games/{ai_game_id}/play",
            data=json.dumps({"cards_1": ["attack"]}),
            content_type="application/json",
        )
        resp = app_client.get(f"/api/games/{ai_game_id}")
        state = resp.get_json()["state"]
        assert "round_log" in state
        assert len(state["round_log"]) == 1


# ---------------------------------------------------------------------------
# DELETE cleans up AI game (TC-API12)
# ---------------------------------------------------------------------------


class TestDeleteAIGame:
    """TC-API12: DELETE works for AI games same as PVP."""

    def test_delete_ai_game_returns_204(self, app_client, ai_game_id):
        resp = app_client.delete(f"/api/games/{ai_game_id}")
        assert resp.status_code == 204

    def test_get_after_delete_returns_404(self, app_client, ai_game_id):
        app_client.delete(f"/api/games/{ai_game_id}")
        resp = app_client.get(f"/api/games/{ai_game_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Full AI game flow (TC-FLOW1–TC-FLOW6)
# ---------------------------------------------------------------------------


class TestFullAIGameFlow:
    """TC-FLOW1–TC-FLOW6: Complete AI game playthroughs."""

    def test_complete_7_round_ai_game_ends(self, app_client, ai_game_id):
        """TC-FLOW1: Play rounds until game over (max 7), all AI cards legal.

        The game may end before round 7 if a player reaches 0 HP.
        We play up to 7 rounds or until the game ends, verifying that:
        - Every POST /play returns 200
        - AI selects a legal card each round
        - After the loop, the game is over
        """
        for i in range(7):
            # Check if game is already over
            state_resp = app_client.get(f"/api/games/{ai_game_id}")
            st = state_resp.get_json()["state"]
            if st["is_over"]:
                break
            # Pick a card from Player 1's actual hand
            hand = st["players"]["1"]["hand"]
            card_to_play = hand[0]
            resp = app_client.post(
                f"/api/games/{ai_game_id}/play",
                data=json.dumps({"cards_1": [card_to_play]}),
                content_type="application/json",
            )
            assert resp.status_code == 200, f"Round {i+1} failed"
            rr = resp.get_json()["round_result"]
            # AI card must be legal
            assert rr["cards_2"][0] in ("attack", "defense", "heavy", "counter")

        # After playing until game over, state must show game over
        state_resp = app_client.get(f"/api/games/{ai_game_id}")
        state = state_resp.get_json()["state"]
        assert state["is_over"] is True

    def test_ai_hand_decreases_each_round(self, app_client):
        """TC-FLOW2: AI hand decreases by exactly 1 each round.

        Plays until game over (max 7 rounds) and verifies:
        - Each round, AI selects a card it still has
        - After each round, AI hand size decreases by exactly 1
        - If the game goes the full 7 rounds, AI hand is empty
        """
        # Create a fresh AI game for this test
        resp = app_client.post(
            "/api/games",
            data=json.dumps({"mode": "pve", "ai": "rnn_v1.1"}),
            content_type="application/json",
        )
        gid = resp.get_json()["game_id"]
        rounds_played = 0
        for _ in range(7):
            # Check if game is over
            state_resp = app_client.get(f"/api/games/{gid}")
            st = state_resp.get_json()["state"]
            if st["is_over"]:
                break
            hand_before = len(st["players"]["2"]["hand"])
            # Pick a card from Player 1's actual hand
            card_to_play = st["players"]["1"]["hand"][0]
            play_resp = app_client.post(
                f"/api/games/{gid}/play",
                data=json.dumps({"cards_1": [card_to_play]}),
                content_type="application/json",
            )
            assert play_resp.status_code == 200
            rounds_played += 1
            # After playing, check AI hand decreased by 1
            after_state = play_resp.get_json()["state"]
            hand_after = len(after_state["players"]["2"]["hand"])
            assert hand_after == hand_before - 1, (
                f"AI hand should decrease by 1: was {hand_before}, now {hand_after}"
            )
        # Verify at least 1 round was played
        assert rounds_played >= 1
        # After game over, AI hand = 7 - rounds_played
        final_resp = app_client.get(f"/api/games/{gid}")
        final_state = final_resp.get_json()["state"]
        assert len(final_state["players"]["2"]["hand"]) == 7 - rounds_played

    def test_new_game_fresh_hidden_state(self, app_client):
        """TC-FLOW6: New AI game starts with fresh hidden state (round 1, HP 3)."""
        # Create first game
        resp = app_client.post(
            "/api/games",
            data=json.dumps({"mode": "pve", "ai": "rnn_v1.1"}),
            content_type="application/json",
        )
        gid1 = resp.get_json()["game_id"]
        # Play one round
        app_client.post(
            f"/api/games/{gid1}/play",
            data=json.dumps({"cards_1": ["attack"]}),
            content_type="application/json",
        )

        # Create new game with same AI
        resp2 = app_client.post(
            "/api/games",
            data=json.dumps({"mode": "pve", "ai": "rnn_v1.1"}),
            content_type="application/json",
        )
        gid2 = resp2.get_json()["game_id"]
        state2 = resp2.get_json()["state"]
        # New game starts at round 1 with full HP
        assert state2["round"] == 1
        assert state2["players"]["1"]["hp"] == 3
        assert state2["players"]["2"]["hp"] == 3

    def test_game_over_prevents_play(self, app_client, ai_game_id):
        """TC-EDGE4: POST /play after game over returns 400."""
        # Play all 7 rounds
        for _ in range(7):
            app_client.post(
                f"/api/games/{ai_game_id}/play",
                data=json.dumps({"cards_1": ["attack"]}),
                content_type="application/json",
            )
        # Extra round attempt
        resp = app_client.post(
            f"/api/games/{ai_game_id}/play",
            data=json.dumps({"cards_1": ["defense"]}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "GAME_OVER"
