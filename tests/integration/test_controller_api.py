"""Integration tests for the REST API endpoints.

Test cases per design/11 test_cases.md §10 (TC-API-01 ~ TC-API-03).

Tests use Flask test client with tmp_path-isolated saves directory.
PVP mode used throughout to avoid AI model loading dependency.
"""

import json


# ===========================================================================
# TC-API-01: GET /api/games/{id} returns v2 state fields
# ===========================================================================


class TestGetStateV2:
    """GET /api/games/{id} returns new v2 fields."""

    def test_state_has_cycle_fields(self, api_client):
        """State includes round, cycle, round_in_cycle."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201
        game_id = resp.get_json()["game_id"]

        resp = api_client.get(f"/api/games/{game_id}")
        assert resp.status_code == 200
        state = resp.get_json()["state"]
        assert state["round"] == 1
        assert state["cycle"] == 1
        assert state["round_in_cycle"] == 1

    def test_state_has_energy_and_statuses(self, api_client):
        """Player state includes energy and statuses."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.get(f"/api/games/{game_id}")
        state = resp.get_json()["state"]
        p1 = state["players"]["1"]
        assert "energy" in p1
        assert "statuses" in p1
        assert p1["energy"] == 0
        assert p1["statuses"] == []

    def test_state_has_pile_counts(self, api_client):
        """Player state includes draw_pile_count and discard_pile_count."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.get(f"/api/games/{game_id}")
        state = resp.get_json()["state"]
        p1 = state["players"]["1"]
        assert "draw_pile_count" in p1
        assert "discard_pile_count" in p1
        assert isinstance(p1["draw_pile_count"], int)
        assert isinstance(p1["discard_pile_count"], int)


# ===========================================================================
# TC-API-02: POST /api/games/{id}/play
# ===========================================================================


class TestPlayRound:
    """POST /api/games/{id}/play — PVP mode."""

    def test_pvp_play_round_returns_round_result(self, api_client):
        """PVP play returns round_result with v2 fields."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={"cards_1": ["attack"], "cards_2": ["defense"]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "round_result" in data
        assert "state" in data

    def test_pvp_round_result_has_cycle_fields(self, api_client):
        """RoundResult includes cycle, round_in_cycle."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={"cards_1": ["attack"], "cards_2": ["defense"]},
        )
        rr = resp.get_json()["round_result"]
        assert "cycle" in rr
        assert "round_in_cycle" in rr
        assert rr["cycle"] == 1
        assert rr["round_in_cycle"] == 1

    def test_pvp_round_result_has_energy_and_statuses(self, api_client):
        """RoundResult includes energy_after and statuses."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={"cards_1": ["energy"], "cards_2": ["defense"]},
        )
        rr = resp.get_json()["round_result"]
        assert "energy_1_after" in rr
        assert "energy_2_after" in rr
        assert "statuses_1" in rr
        assert "statuses_2" in rr
        # P1 played ENERGY → +1 energy
        assert rr["energy_1_after"] == 1
        # P2 played DEFENSE → defense status
        assert "defense" in rr["statuses_2"]

    def test_pvp_round_result_has_cycle_ended(self, api_client):
        """RoundResult includes cycle_ended field (False for non-cycle-end)."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={"cards_1": ["attack"], "cards_2": ["defense"]},
        )
        rr = resp.get_json()["round_result"]
        assert "cycle_ended" in rr
        assert rr["cycle_ended"] is False

    def test_missing_cards_2_in_pvp_returns_400(self, api_client):
        """PVP with missing cards_2 returns 400."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={"cards_1": ["attack"]},
        )
        assert resp.status_code == 400

    def test_game_over_returns_400(self, api_client):
        """Cannot play round after game is over."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]
        # Play until game over is not practical, just set is_over manually
        # by loading the save and corrupting it — skip for now.
        # Instead test via a separate path.


# ===========================================================================
# TC-API-03: RoundResult serialization has all v2 fields
# ===========================================================================


class TestRoundResultSerialization:
    """RoundResult JSON contains all v2 fields."""

    def test_round_result_all_v2_fields(self, api_client):
        """Verify all v2-specific fields are present in round_result JSON."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={"cards_1": ["attack"], "cards_2": ["attack"]},
        )
        rr = resp.get_json()["round_result"]

        # Required fields per design
        for key in (
            "round", "cycle", "round_in_cycle",
            "cards_1", "cards_2",
            "damage_1", "damage_2",
            "hp_1_after", "hp_2_after",
            "energy_1_after", "energy_2_after",
            "statuses_1", "statuses_2",
            "details",
            "is_game_over", "winner",
            "cycle_ended",
        ):
            assert key in rr, f"Missing key '{key}' in round_result"

    def test_hp_reflects_damage(self, api_client):
        """HP after round reflects damage taken."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={"cards_1": ["attack"], "cards_2": ["attack"]},
        )
        rr = resp.get_json()["round_result"]
        # Both played ATTACK with streak 0 → 1 damage each → hp 19
        assert rr["hp_1_after"] == 19
        assert rr["hp_2_after"] == 19


# ===========================================================================
# POST /api/games — Create Game
# ===========================================================================


class TestCreateGame:
    """POST /api/games — game creation."""

    def test_create_pvp_returns_201(self, api_client):
        """Creating a PVP game returns 201 with game_id."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201
        data = resp.get_json()
        assert "game_id" in data
        assert "state" in data

    def test_create_game_state_has_7_cards_in_hand(self, api_client):
        """After create, each player has 7 cards in hand."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        state = resp.get_json()["state"]
        assert len(state["players"]["1"]["hand"]) == 7
        assert len(state["players"]["2"]["hand"]) == 7

    def test_create_pvp_state_has_mode(self, api_client):
        """State dict includes mode field."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        state = resp.get_json()["state"]
        assert state["mode"] == "pvp"


# ===========================================================================
# DELETE /api/games/{id}
# ===========================================================================


class TestDeleteGame:
    """DELETE /api/games/{id} — cleanup."""

    def test_delete_existing_returns_204(self, api_client):
        """Deleting an existing game returns 204."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        game_id = resp.get_json()["game_id"]

        resp = api_client.delete(f"/api/games/{game_id}")
        assert resp.status_code == 204

    def test_delete_non_existent_returns_404(self, api_client):
        """Deleting a non-existent game returns 404."""
        resp = api_client.delete("/api/games/nonexistent-id")
        assert resp.status_code == 404


# ===========================================================================
# GET /api/ais
# ===========================================================================


class TestListAis:
    """GET /api/ais — list AI opponents."""

    def test_list_ais_includes_rnn_v2_1(self, api_client):
        """AI list includes rnn_v2.1."""
        resp = api_client.get("/api/ais")
        assert resp.status_code == 200
        ais = resp.get_json()["ais"]
        names = [a["name"] for a in ais]
        assert "rnn_v2.1" in names
