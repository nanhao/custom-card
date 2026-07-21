"""B9: Corrupted save file causes unhandled 500 instead of JSON error.

Bug: store.load() calls are not wrapped in try/except, so corrupted
save files cause unhandled 500 errors (ValueError/KeyError → 500 HTML).

Fix: Wrap store.load() in try/except and return 500 JSON with "LOAD_ERROR".
"""

import json
import os


class TestB9CorruptedSaveFile:
    """B9: Corrupted save file returns 500 JSON with LOAD_ERROR, not generic 500."""

    def test_corrupted_save_get_state_returns_load_error(self, api_client):
        """GET /api/games/{id} with corrupted save → 500 JSON with LOAD_ERROR."""
        # Create a game normally
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201
        game_id = resp.get_json()["game_id"]

        # Get the session store from the app to find the save path
        # The api_client fixture creates a Flask app with a temp config
        from flask import current_app

        # Use the app's session store to find the correct save file path
        with api_client.application.app_context():
            store = current_app.config["SESSION_STORE"]
            save_path = os.path.join(str(store.save_dir), f"{game_id}.json")

        # Corrupt the save file by writing invalid JSON
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("this is not valid json {{{{{")

        # Now GET the game state — should return 500 with JSON body
        resp = api_client.get(f"/api/games/{game_id}")

        # B9 CHECK: Should be a server error with JSON body, not HTML
        assert resp.status_code == 500, (
            f"B9 FAIL: Expected 500 for corrupted save, got {resp.status_code}. "
            f"Body: {resp.get_data(as_text=True)[:200]}"
        )

        # Should be JSON, not HTML
        data = resp.get_json()
        assert data is not None, (
            f"B9 FAIL: Response is not valid JSON: {resp.get_data(as_text=True)[:200]}"
        )
        assert data.get("code") == "LOAD_ERROR", (
            f"B9 FAIL: Expected code='LOAD_ERROR', got {data}"
        )

    def test_corrupted_save_play_round_returns_load_error(self, api_client):
        """POST /api/games/{id}/play with corrupted save → 500 JSON with LOAD_ERROR."""
        resp = api_client.post("/api/games", json={"mode": "pvp"})
        assert resp.status_code == 201
        game_id = resp.get_json()["game_id"]

        from flask import current_app
        with api_client.application.app_context():
            store = current_app.config["SESSION_STORE"]
            save_path = os.path.join(str(store.save_dir), f"{game_id}.json")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("corrupted data {{{")

        resp = api_client.post(
            f"/api/games/{game_id}/play",
            json={"cards_1": ["attack"], "cards_2": ["defense"]},
        )

        assert resp.status_code == 500
        data = resp.get_json()
        assert data is not None
        assert data.get("code") == "LOAD_ERROR", (
            f"B9 FAIL: Expected code='LOAD_ERROR', got {data}"
        )
