"""Tests for B4: API doesn't handle build_target/dismantle_target deserialization errors.

When invalid build_target_1 or dismantle_target_1 strings are sent to the
play_round endpoint, the API should return a 400 with "INVALID_CARD" error code,
NOT a 500 Internal Server Error.

Per serializers.error_response(), the response body is:
    {"error": "<message>", "code": "<error_code>"}
"""


class TestB4BuildTargetDeserializationErrors:
    """B4: Invalid build_target/dismantle_target returns 400, not 500."""

    def test_invalid_build_target_1_returns_400(self, api_client) -> None:
        """Sending invalid build_target_1 string returns 400 with code INVALID_CARD."""
        # Create a game first
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201
        game_id = resp.get_json()["game_id"]

        # Send a play request with invalid build_target_1
        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={
                "cards_1": ["build"],
                "cards_2": ["defense"],
                "build_target_1": "invalid_card_name_xyz",
            },
        )

        # Should be 400, not 500
        assert resp.status_code == 400, (
            f"B4 FAIL: Expected 400, got {resp.status_code}. "
            f"Body: {resp.get_data(as_text=True)}"
        )
        data = resp.get_json()
        # error_response() puts the error code in "code" field, message in "error"
        assert data["code"] == "INVALID_CARD", (
            f"B4 FAIL: Expected error code INVALID_CARD, "
            f"got {data.get('code')}"
        )

    def test_invalid_dismantle_target_1_returns_400(self, api_client) -> None:
        """Sending invalid dismantle_target_1 string returns 400 with code INVALID_CARD."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={
                "cards_1": ["build"],
                "cards_2": ["defense"],
                "dismantle_target_1": "not_a_real_card",
            },
        )

        assert resp.status_code == 400
        data = resp.get_json()
        assert data["code"] == "INVALID_CARD"

    def test_valid_build_target_still_works(self, api_client) -> None:
        """Valid build_target should not cause a deserialization error."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201
        game_id = resp.get_json()["game_id"]

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={
                "cards_1": ["build"],
                "cards_2": ["defense"],
                "build_target_1": "attack",
            },
        )

        # Should not be a 500 deserialization error
        # (may be 200 success or 400 game logic error, but not 500)
        assert resp.status_code != 500, (
            f"B4 FAIL: Valid build_target should not cause 500. "
            f"Status: {resp.status_code}, Body: {resp.get_data(as_text=True)}"
        )
