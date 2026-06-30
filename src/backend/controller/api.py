"""Flask REST API adapter for the custom card game.

Per architecture D10, this module contains NO business logic. It only:
- Parses HTTP requests (deserialize JSON, extract path params)
- Delegates to src/backend/game/ classes
- Formats and returns HTTP responses (serialize to JSON, set status codes)

Per architecture D11, the face-down protocol is client-side. The API
receives both cards together via POST /play.

Routes:
    POST   /api/games              — Create a new game
    GET    /api/games/{game_id}    — Get current game state
    POST   /api/games/{game_id}/play  — Play a round
    DELETE /api/games/{game_id}    — Delete a game session
"""

import uuid

from flask import Flask, jsonify, request

from src.backend.controller.serializers import (
    card_to_str,
    error_response,
    round_result_to_dict,
    str_to_card,
)
from src.backend.game.game_manager import GameManager
from src.backend.game.session_store import SessionStore

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config_path: str = "config.json") -> Flask:
    """Create and configure the Flask application.

    Args:
        config_path: Path to the JSON config file for SessionStore.

    Returns:
        A configured Flask app instance with all routes registered.
    """
    app = Flask(__name__)

    # Store a SessionStore instance on the app for all routes to use.
    # Each request reads/writes through the filesystem per DC3/DC4.
    app.config["SESSION_STORE"] = SessionStore(config_path)

    _register_routes(app)
    return app


def _register_routes(app: Flask) -> None:
    """Register all API routes on the given Flask app."""

    # ------------------------------------------------------------------
    # POST /api/games — Create Game
    # ------------------------------------------------------------------

    @app.route("/api/games", methods=["POST"])
    def create_game():
        """Create a new game session and return its initial state.

        Request body: {} (empty)
        Success: 201 Created with {"game_id": "<uuid>", "state": {...}}
        """
        store: SessionStore = app.config["SESSION_STORE"]
        game_id = str(uuid.uuid4())

        gm = GameManager()
        gm.start_game()
        store.save(game_id, gm)

        return jsonify({
            "game_id": game_id,
            "state": gm.get_state(),
        }), 201

    # ------------------------------------------------------------------
    # GET /api/games/{game_id} — Get State
    # ------------------------------------------------------------------

    @app.route("/api/games/<game_id>", methods=["GET"])
    def get_state(game_id: str):
        """Return the current game snapshot.

        Success: 200 OK with {"game_id": "...", "state": {...}}
        Errors:  404 if game not found
        """
        store: SessionStore = app.config["SESSION_STORE"]
        gm = store.load(game_id)
        if gm is None:
            return jsonify(error_response(
                f"Game not found: {game_id}", "GAME_NOT_FOUND", 404
            )), 404

        return jsonify({
            "game_id": game_id,
            "state": gm.get_state(),
        }), 200

    # ------------------------------------------------------------------
    # POST /api/games/{game_id}/play — Play Round
    # ------------------------------------------------------------------

    @app.route("/api/games/<game_id>/play", methods=["POST"])
    def play_round(game_id: str):
        """Submit both players' cards and resolve the round.

        Request body: {"cards_1": ["<card>", ...], "cards_2": ["<card>", ...]}
        Success: 200 OK with {"round_result": {...}, "state": {...}}
        Errors:
            400 — invalid card string (INVALID_CARD)
            400 — card not in hand (CARD_NOT_IN_HAND)
            400 — game already over (GAME_OVER)
            400 — game not started (GAME_NOT_STARTED)
            404 — game not found
        """
        store: SessionStore = app.config["SESSION_STORE"]
        gm = store.load(game_id)
        if gm is None:
            return jsonify(error_response(
                f"Game not found: {game_id}", "GAME_NOT_FOUND", 404
            )), 404

        # Parse and validate the request body
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(error_response(
                "Request body must be valid JSON", "INVALID_JSON", 400
            )), 400

        # ---- Deserialize cards (with INVALID_CARD check) ----
        cards_1_raw = data.get("cards_1", [])
        cards_2_raw = data.get("cards_2", [])

        try:
            cards_1 = [str_to_card(s) for s in cards_1_raw]
            cards_2 = [str_to_card(s) for s in cards_2_raw]
        except ValueError as exc:
            return jsonify(error_response(
                str(exc), "INVALID_CARD", 400
            )), 400

        # ---- Delegate to GameManager (catches game-rule violations) ----
        try:
            result = gm.play_round(cards_1, cards_2)
        except ValueError as exc:
            error_msg = str(exc)
            # Map known error messages to appropriate error codes
            if "game is over" in error_msg.lower():
                code = "GAME_OVER"
            elif "game not started" in error_msg.lower():
                code = "GAME_NOT_STARTED"
            else:
                code = "CARD_NOT_IN_HAND"
            return jsonify(error_response(error_msg, code, 400)), 400

        # Persist updated state
        store.save(game_id, gm)

        return jsonify({
            "round_result": round_result_to_dict(result),
            "state": gm.get_state(),
        }), 200

    # ------------------------------------------------------------------
    # DELETE /api/games/{game_id} — Clean Up
    # ------------------------------------------------------------------

    @app.route("/api/games/<game_id>", methods=["DELETE"])
    def delete_game(game_id: str):
        """Remove a game session.

        Success: 204 No Content
        Errors:  404 if game not found
        """
        store: SessionStore = app.config["SESSION_STORE"]
        existed = store.delete(game_id)
        if not existed:
            return jsonify(error_response(
                f"Game not found: {game_id}", "GAME_NOT_FOUND", 404
            )), 404

        return "", 204
