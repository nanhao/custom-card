"""Integration tests for REST API endpoints — v2.

Test cases per design/11 test_cases.md §10 (TC-API-01 ~ TC-API-03).
Uses Flask test client with tmp_path-isolated saves. PVP mode to avoid AI loading.
"""

import pytest

from src.backend.game.card_type import CardType


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
        assert p1["energy"] == 2  # design/38: initial energy=2

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
        state = resp.get_json()["state"]
        hand_1 = [c for c in state["players"]["1"]["hand"] if c != "build"]
        hand_2 = [c for c in state["players"]["2"]["hand"] if c != "build"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": [hand_1[0]], "cards_2": [hand_2[0]]})
        rr = resp.get_json()["round_result"]
        assert rr["cycle"] == 1
        assert rr["round_in_cycle"] == 1

    def test_round_result_has_energy_and_statuses(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        # Directly inject required cards into hands to avoid random draw flakiness
        from src.backend.game.card_type import CardType
        app_instance = api_client.application
        store = app_instance.config["SESSION_STORE"]
        gm, metadata = store.load(gid)
        gm.player_1.hand = [CardType.ENERGY, CardType.ATTACK, CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE, CardType.ATTACK, CardType.ATTACK]
        store.save(gid, gm)
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": ["energy"], "cards_2": ["defense"]})
        rr = resp.get_json()["round_result"]
        assert rr["after_resolve"]["energy_1"] == 3  # design/38: initial energy=2, ENERGY +1
        # Statuses now in design/15 dict format: [{"type": "defense", "lifetime": 1}]
        assert any(s["type"] == "defense" for s in rr["after_resolve"]["statuses_2"])

    def test_round_result_has_cycle_ended(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        state = resp.get_json()["state"]
        hand_1 = [c for c in state["players"]["1"]["hand"] if c != "build"]
        hand_2 = [c for c in state["players"]["2"]["hand"] if c != "build"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": [hand_1[0]], "cards_2": [hand_2[0]]})
        assert resp.get_json()["round_result"]["end_actions"]["cycle_ended"] is False

    def test_missing_cards_2_means_pass(self, api_client):
        """Per design/28: missing cards_2 = pass, should return 200 (not 400)."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        state = resp.get_json()["state"]
        hand_1 = state["players"]["1"]["hand"]
        # Pick a non-BUILD card to avoid BUILD dual-mode validation
        non_build = [c for c in hand_1 if c != "build"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": [non_build[0]]})
        assert resp.status_code == 200
        assert resp.get_json()["round_result"]["cards_2"] == []


class TestRoundResultSerialization:
    """TC-API-03: RoundResult has all v2 fields."""

    def test_all_required_fields_present(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        state = resp.get_json()["state"]
        hand_1 = [c for c in state["players"]["1"]["hand"] if c != "build"]
        hand_2 = [c for c in state["players"]["2"]["hand"] if c != "build"]
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": [hand_1[0]], "cards_2": [hand_2[0]]})
        rr = resp.get_json()["round_result"]
        for key in ("round", "cycle", "round_in_cycle",
                    "cards_1", "cards_2",
                    "before", "after_resolve", "end_actions", "after_end_actions"):
            assert key in rr, f"Missing {key}"

    def test_hp_reflects_damage(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        # Set hands manually to guarantee ATTACK for damage test
        from src.backend.game.session_store import SessionStore
        app_instance = api_client.application
        store: SessionStore = app_instance.config["SESSION_STORE"]
        gm, _meta = store.load(gid)
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.ATTACK]
        store.save(gid, gm)
        # Use attack for both to test HP damage
        resp = api_client.post(f"/api/games/{gid}/play",
                               json={"cards_1": ["attack"], "cards_2": ["attack"]})
        rr = resp.get_json()["round_result"]
        assert rr["after_resolve"]["hp_1"] == 29  # design/38: initial HP=30, -1
        assert rr["after_resolve"]["hp_2"] == 29  # design/38: initial HP=30, -1


class TestCreateAndDelete:
    def test_create_pvp_201(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201

    def test_create_has_5_cards(self, api_client):
        """TC-INIT-01: start_game returns hand with 5 cards (design/26 D2)."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        state = resp.get_json()["state"]
        assert len(state["players"]["1"]["hand"]) == 5

    def test_delete_204(self, api_client):
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        assert api_client.delete(f"/api/games/{gid}").status_code == 204


class TestGameStateSerialization:
    """TC-API-01~04: GameState JSON serialization format (design/19)."""

    def test_state_hand_is_string_array(self, api_client):
        """TC-API-01: state.players.1.hand is a string array."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        state = resp.get_json()["state"]
        hand = state["players"]["1"]["hand"]
        assert isinstance(hand, list)
        assert len(hand) == 5
        for c in hand:
            assert isinstance(c, str), f"{c!r} should be str"

    def test_state_has_draw_pile_always_visible(self, api_client):
        """design/34: draw_pile is always present in state (mutual visibility)."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        p1 = resp.get_json()["state"]["players"]["1"]
        # Per design/34: both players' piles are always visible
        assert "draw_pile" in p1
        assert isinstance(p1["draw_pile"], list)
        # draw_pile_count is also always present
        assert isinstance(p1["draw_pile_count"], int)

    def test_perspective_both_players_see_piles(self, api_client):
        """design/34: perspective param is deprecated — both players see all piles."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        resp = api_client.get(f"/api/games/{gid}?perspective=1")
        p1 = resp.get_json()["state"]["players"]["1"]
        # P1 sees pile contents
        assert "draw_pile" in p1
        assert isinstance(p1["draw_pile"], list)
        assert "discard_pile" in p1
        assert isinstance(p1["discard_pile"], list)
        # P2 also sees pile contents (design/34: mutual visibility)
        p2 = resp.get_json()["state"]["players"]["2"]
        assert "draw_pile" in p2
        assert isinstance(p2["draw_pile"], list)

    def test_create_returns_state_dict(self, api_client):
        """TC-API-03: POST /api/games returns state dict with expected structure."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        data = resp.get_json()
        assert "game_id" in data
        assert isinstance(data["game_id"], str)
        state = data["state"]
        assert isinstance(state, dict)
        assert "round" in state
        assert "players" in state
        assert "mode" in state

    def test_play_round_returns_state_dict(self, api_client):
        """TC-API-04: POST /play returns round_result and state dicts."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        gid = resp.get_json()["game_id"]
        state = resp.get_json()["state"]
        hand_1 = [c for c in state["players"]["1"]["hand"] if c != "build"]
        hand_2 = [c for c in state["players"]["2"]["hand"] if c != "build"]
        resp = api_client.post(
            f"/api/games/{gid}/play",
            json={"cards_1": [hand_1[0]], "cards_2": [hand_2[0]]},
        )
        data = resp.get_json()
        # round_result is a dict
        assert isinstance(data["round_result"], dict)
        # state is a dict
        assert isinstance(data["state"], dict)
        # state has expected fields
        assert data["state"]["round"] == 2


class TestListCards:
    """TC-11: GET /api/cards returns description and tags (intention #36)."""

    def test_cards_have_description(self, api_client):
        """TC-11-01: every card has non-empty description."""
        resp = api_client.get("/api/cards")
        data = resp.get_json()
        cards = data["cards"]
        assert len(cards) == 8
        for card in cards:
            assert "description" in card, f"missing description in {card['type']}"
            assert isinstance(card["description"], str)
            assert card["description"], f"empty description for {card['type']}"

    def test_cards_have_tags(self, api_client):
        """TC-11-02: every card has tags as list."""
        resp = api_client.get("/api/cards")
        cards = resp.get_json()["cards"]
        for card in cards:
            assert "tags" in card, f"missing tags in {card['type']}"
            assert isinstance(card["tags"], list)

    def test_energy_tags_empty(self, api_client):
        """TC-11-03: ENERGY tags is empty list."""
        resp = api_client.get("/api/cards")
        cards = resp.get_json()["cards"]
        energy = next(c for c in cards if c["type"] == "energy")
        assert energy["tags"] == []

    def test_new_display_names(self, api_client):
        """TC-11-04: display_names match intention #36."""
        resp = api_client.get("/api/cards")
        cards = {c["type"]: c for c in resp.get_json()["cards"]}
        assert cards["attack"]["display_name"] == "连续击"
        assert cards["defense"]["display_name"] == "格挡"
        assert cards["heavy"]["display_name"] == "破防重击"
        assert cards["counter"]["display_name"] == "防守反击"
        assert cards["energy"]["display_name"] == "聚能"
        assert cards["build"]["display_name"] == "构建/拆解"
        assert cards["flex"]["display_name"] == "攻守兼备"

    def test_tags_values(self, api_client):
        """Verify tag values for all cards."""
        resp = api_client.get("/api/cards")
        cards = {c["type"]: c for c in resp.get_json()["cards"]}
        assert cards["attack"]["tags"] == ["攻"]
        assert cards["defense"]["tags"] == ["防"]
        assert cards["heavy"]["tags"] == ["攻"]
        assert cards["counter"]["tags"] == ["防"]
        assert cards["flex"]["tags"] == ["攻", "防"]
        assert cards["build"]["tags"] == []
        assert cards["energy"]["tags"] == []
