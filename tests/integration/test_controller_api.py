"""Integration tests for REST API endpoints — v2.

Test cases per design/11 test_cases.md §10 (TC-API-01 ~ TC-API-03).
Uses Flask test client with tmp_path-isolated saves. PVP mode to avoid AI loading.
"""


class TestGetStateV2:
    """TC-API-01: GET /api/games/{id} returns v2 fields."""

    def test_state_has_cycle_fields(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        state = resp.get_json()["state"]
        assert state["round"] == 1
        assert state["cycle"] == 1
        assert state["round_in_cycle"] == 1

    def test_state_has_energy_and_statuses(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        p1 = resp.get_json()["state"]["players"]["1"]
        assert "energy" in p1
        assert "statuses" in p1
        assert p1["energy"] == 0

    def test_state_has_pile_counts(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        p1 = resp.get_json()["state"]["players"]["1"]
        assert isinstance(p1["draw_pile_count"], int)
        assert isinstance(p1["discard_pile_count"], int)


class TestPlayRound:
    """TC-API-02: POST /api/games/{id}/play."""

    def test_round_result_has_cycle_fields(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": ["attack"], "cards_2": ["defense"]})
        rr = resp.get_json()["round_result"]
        assert rr["cycle"] == 1
        assert rr["round_in_cycle"] == 1

    def test_round_result_has_energy_and_statuses(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": ["energy"], "cards_2": ["defense"]})
        rr = resp.get_json()["round_result"]
        assert rr["energy_1_after"] == 1
        assert "defense" in rr["statuses_2"]

    def test_round_result_has_cycle_ended(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": ["attack"], "cards_2": ["defense"]})
        assert resp.get_json()["round_result"]["cycle_ended"] is False

    def test_missing_cards_2_returns_400(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": ["attack"]})
        assert resp.status_code == 400


class TestRoundResultSerialization:
    """TC-API-03: RoundResult has all v2 fields."""

    def test_all_required_fields_present(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": ["attack"], "cards_2": ["attack"]})
        rr = resp.get_json()["round_result"]
        for key in ("round", "cycle", "round_in_cycle",
                    "damage_1", "damage_2",
                    "hp_1_after", "hp_2_after",
                    "energy_1_after", "energy_2_after",
                    "statuses_1", "statuses_2",
                    "is_game_over", "winner", "cycle_ended"):
            assert key in rr, f"Missing {key}"

    def test_hp_reflects_damage(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": ["attack"], "cards_2": ["attack"]})
        rr = resp.get_json()["round_result"]
        assert rr["hp_1_after"] == 19
        assert rr["hp_2_after"] == 19


class TestCreateAndDelete:
    def test_create_pvp_201(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201

    def test_create_has_7_cards(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        state = resp.get_json()["state"]
        assert len(state["players"]["1"]["hand"]) == 7

    def test_delete_204(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        assert api_client.delete(f"/api/games/{gid}").status_code == 204
