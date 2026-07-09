"""Flask REST API adapter for the custom card game.

Per architecture D10, this module contains NO business logic. It only:
- Parses HTTP requests (deserialize JSON, extract path params)
- Delegates to src/backend/game/ classes
- Formats and returns HTTP responses (serialize to JSON, set status codes)

Per architecture D11, the face-down protocol is client-side. The API
receives both cards together via POST /play in PVP mode. In PVE mode
the client sends only cards_1 and the server auto-computes cards_2 via AI.

Routes:
    GET    /                       — Serve the React SPA (production) or dev info
    POST   /api/games              — Create a new game (accepts mode + ai)
    GET    /api/games/{game_id}    — Get current game state
    POST   /api/games/{game_id}/play  — Play a round (PVP: both cards; PVE: cards_1 only)
    DELETE /api/games/{game_id}    — Delete a game session
    GET    /api/ais                — List available AI opponents

In development, run the Vite dev server (port 5173) which proxies /api to Flask
(port 5000). In production, run `npm run build` then Flask serves dist/ at /.
"""

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Paths relative to this module.
# api.py → src/backend/controller/api.py
# frontend source → src/frontend/
# frontend build  → src/frontend/dist/
_FRONTEND_SRC = Path(__file__).resolve().parent.parent.parent / "frontend"
_FRONTEND_DIST = _FRONTEND_SRC / "dist"

from src.backend.ai.registry import list_ais
from src.backend.app.app_manager import AppManager
from src.backend.app.converter import Converter
from src.backend.controller.serializers import (
    card_to_str,
    error_response,
    game_state_to_dict,
    round_result_to_dict,
    str_to_card,
)
from src.backend.game.card_type import CardType
from src.backend.game.card_def import get_card_def
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

    # Enable CORS for the Vite dev server (different port during development).
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Store a SessionStore and AppManager on the app for all routes to use.
    # Each request reads/writes through the filesystem per DC3/DC4.
    # AppManager is created once at startup (per design D8).
    session_store = SessionStore(config_path)
    app.config["SESSION_STORE"] = session_store
    app.config["APP_MANAGER"] = AppManager(session_store)

    _register_routes(app)
    return app


def _register_routes(app: Flask) -> None:
    """Register all API routes on the given Flask app."""

    # ------------------------------------------------------------------
    # GET / — Serve the React SPA (production) or dev info
    # ------------------------------------------------------------------

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path: str):
        """Serve the built React app with SPA fallback.

        In production (dist/ present), serves static assets from dist/
        and index.html for all non-/api routes (client-side routing).

        In development (no dist/), returns a message pointing to the
        Vite dev server on port 5173.
        """
        # Skip API routes — let them be handled by their own route decorators.
        if path.startswith("api"):
            return jsonify(error_response("Not Found", "NOT_FOUND", 404)), 404

        if _FRONTEND_DIST.is_dir():
            # Try serving a specific file; fall back to index.html for SPA routing.
            file_path = _FRONTEND_DIST / path
            if path and file_path.is_file():
                return send_from_directory(str(_FRONTEND_DIST), path)
            return send_from_directory(str(_FRONTEND_DIST), "index.html")

        # Development mode: no build available.
        return (
            "<h2>Frontend not built</h2>"
            "<p>Run <code>npm run dev</code> in <code>src/frontend/</code> "
            "and open <a href='http://localhost:5173'>http://localhost:5173</a>.</p>"
        )

    # ------------------------------------------------------------------
    # POST /api/games — Create Game
    # ------------------------------------------------------------------

    @app.route("/api/games", methods=["POST"])
    def create_game():
        """Create a new game session and return its initial state.

        Request body (PVP, backward compatible):
            {} (empty) or {"mode": "pvp"}

        Request body (PVE):
            {"mode": "pve", "ai": "rnn_v1.1"}

        Success: 201 Created with {"game_id": "<uuid>", "state": {...}}
        Errors:
            400 — unknown AI name (UNKNOWN_AI)
            400 — mode is "pve" but no AI specified
        """
        app_manager: AppManager = app.config["APP_MANAGER"]

        # Parse request body (backward compatible: empty body = PVP)
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "pvp")
        ai_name = data.get("ai", None)

        # Delegate to AppManager for game creation
        # AppManager handles validation (unknown AI, missing AI for PVE)
        try:
            game_id, gs = app_manager.create_game(mode=mode, ai_name=ai_name)
        except ValueError as exc:
            error_msg = str(exc)
            if "unknown" in error_msg.lower():
                return jsonify(error_response(error_msg, "UNKNOWN_AI", 400)), 400
            if "missing" in error_msg.lower():
                return jsonify(error_response(error_msg, "MISSING_AI", 400)), 400
            return jsonify(error_response(error_msg, "BAD_REQUEST", 400)), 400

        state_dict = game_state_to_dict(gs)
        state_dict["ai"] = ai_name

        return jsonify({
            "game_id": game_id,
            "state": state_dict,
        }), 201

    # ------------------------------------------------------------------
    # GET /api/ais — List AI Opponents
    # ------------------------------------------------------------------

    @app.route("/api/ais", methods=["GET"])
    def list_ai_opponents():
        """Return the list of registered AI opponents.

        Success: 200 OK with {"ais": [{"name": "...", "description": "..."}]}
        """
        ais = list_ais()
        return jsonify({"ais": ais}), 200

    # ------------------------------------------------------------------
    # GET /api/cards — Card Metadata
    # ------------------------------------------------------------------

    @app.route("/api/cards", methods=["GET"])
    def list_cards():
        """Return metadata for all card types.

        Success: 200 OK with {"cards": [{"type": "attack", "display_name": "攻击", ...}]}
        """
        cards = []
        for ct in CardType:
            cd = get_card_def(ct)
            cards.append({
                "type": ct.value,
                "display_name": cd.display_name,
                "build_cost": cd.build_cost,
            })
        return jsonify({"cards": cards}), 200

    # ------------------------------------------------------------------
    # GET /api/games/{game_id} — Get State
    # ------------------------------------------------------------------

    @app.route("/api/games/<game_id>", methods=["GET"])
    def get_state(game_id: str):
        """Return the current game snapshot.

        Query params:
            perspective: "1" or "2" — if set, exposes that player's draw_pile
                         and discard_pile contents (opponent sees counts only).
                         If absent, both players see counts only.

        Success: 200 OK with {"game_id": "...", "state": {...}}
        Errors:  404 if game not found
        """
        store: SessionStore = app.config["SESSION_STORE"]
        gm, metadata = store.load(game_id)
        if gm is None:
            return jsonify(error_response(
                f"Game not found: {game_id}", "GAME_NOT_FOUND", 404
            )), 404

        # Optional player perspective for pile inspection
        perspective = request.args.get("perspective")

        gs = gm.get_state(player_perspective=perspective)
        state_dict = game_state_to_dict(gs)
        state_dict["ai"] = metadata["ai"] if metadata else None

        return jsonify({
            "game_id": game_id,
            "state": state_dict,
        }), 200

    # ------------------------------------------------------------------
    # POST /api/games/{game_id}/play — Play Round
    # ------------------------------------------------------------------

    @app.route("/api/games/<game_id>/play", methods=["POST"])
    def play_round(game_id: str):
        """Submit cards and resolve the round.

        PVP mode (unchanged):
            Request body: {"cards_1": ["<card>"], "cards_2": ["<card>"]}

        PVE mode (new):
            Request body: {"cards_1": ["<card>"]}
            Server auto-computes cards_2 via AI. cards_2 is ignored if sent.

        Success: 200 OK with {"round_result": {...}, "state": {...}}
        Errors:
            400 — invalid card (INVALID_CARD), card not in hand (CARD_NOT_IN_HAND),
                  game over (GAME_OVER), game not started (GAME_NOT_STARTED),
                  missing cards_1 in PVE (MISSING_CARDS_1)
            404 — game not found
            500 — AI load failure (AI_LOAD_ERROR), AI inference failure (AI_INFERENCE_ERROR)
        """
        store: SessionStore = app.config["SESSION_STORE"]
        gm, metadata = store.load(game_id)
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

        # ---- Deserialize Player 1's cards ----
        cards_1_raw = data.get("cards_1", [])
        if not cards_1_raw:
            return jsonify(error_response(
                "Missing required field: cards_1", "MISSING_CARDS_1", 400
            )), 400

        try:
            cards_1 = [str_to_card(s) for s in cards_1_raw]
        except ValueError as exc:
            return jsonify(error_response(
                str(exc), "INVALID_CARD", 400
            )), 400

        # ---- Determine Player 2's cards ----
        if gm.mode == "pve":
            # PVE: auto-compute via AppManager
            app_manager: AppManager = app.config["APP_MANAGER"]
            try:
                card_type, build_target_type, new_hidden_state = app_manager.ai_select_action(
                    gm.get_state(), metadata["ai_state"], metadata["ai"]
                )
            except ValueError as exc:
                return jsonify(error_response(
                    f"Failed to load AI {metadata.get('ai', 'unknown')!r}: {exc}",
                    "AI_LOAD_ERROR", 500
                )), 500
            except RuntimeError as exc:
                return jsonify(error_response(
                    f"AI load error: {exc}", "AI_LOAD_ERROR", 500
                )), 500
            except Exception as exc:
                return jsonify(error_response(
                    f"AI inference failed: {exc}", "AI_INFERENCE_ERROR", 500
                )), 500

            # Validate AI output (defense in depth).
            # Per design/20: ai_select_action returns CardType.
            if card_type not in Converter.ALL_CARD_TYPES:
                return jsonify(error_response(
                    f"AI returned invalid card: {card_type!r}",
                    "AI_INFERENCE_ERROR", 500
                )), 500

            cards_2 = [card_type]
            build_target_2 = build_target_type
        else:
            # PVP: deserialize cards_2 from request body
            cards_2_raw = data.get("cards_2", [])
            if not cards_2_raw:
                return jsonify(error_response(
                    "Missing required field: cards_2", "MISSING_CARDS_2", 400
                )), 400

            try:
                cards_2 = [str_to_card(s) for s in cards_2_raw]
            except ValueError as exc:
                return jsonify(error_response(
                    str(exc), "INVALID_CARD", 400
                )), 400
            build_target_2 = None

        # Parse build_target_1 from request (optional, only for BUILD)
        build_target_1_raw = data.get("build_target_1")
        build_target_1 = str_to_card(build_target_1_raw) if build_target_1_raw else None

        # ---- Delegate to GameCore ----
        try:
            result = gm.play_round(
                cards_1, cards_2,
                build_target_1=build_target_1,
                build_target_2=build_target_2,
            )
        except ValueError as exc:
            error_msg = str(exc)
            if "game is over" in error_msg.lower():
                code = "GAME_OVER"
            elif "game not started" in error_msg.lower():
                code = "GAME_NOT_STARTED"
            else:
                code = "CARD_NOT_IN_HAND"
            return jsonify(error_response(error_msg, code, 400)), 400

        # Persist updated state
        if gm.mode == "pve":
            store.save(
                game_id, gm,
                ai_state=new_hidden_state,
                ai_value_state=metadata.get("ai_value_state"),
                ai=metadata["ai"],
            )
        else:
            store.save(game_id, gm)

        # Build response state with AI metadata
        gs = gm.get_state()
        state_dict = game_state_to_dict(gs)
        state_dict["ai"] = metadata["ai"] if metadata else None

        return jsonify({
            "round_result": round_result_to_dict(result),
            "state": state_dict,
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
