# REST API Design — `src/backend/controller/`

## 1. Constraints (from architecture.md)

Per **D10**: This module contains NO business logic. It only:
- Parses HTTP requests (deserialize JSON, extract path params)
- Delegates to `src/backend/game/` classes
- Formats and returns HTTP responses (serialize to JSON, set status codes)

Per **D11**: The face-down protocol is handled client-side (hot-seat, single browser). The web UI buffers Player 1's choice → shows pass screen → Player 2 chooses → client sends both cards together via `POST /play`. The server never sees individual selections — it only receives the pair.

## 2. Technology Choices

| Decision | Choice | Reason |
|----------|--------|--------|
| Framework | Flask | Lightweight, minimal boilerplate, standard for thin adapters |
| Session store | `SessionStore` (disk-based) | Game state persisted to JSON files; survives server restart. See [persistence.md](persistence.md). |
| Serialization | Manual `jsonify()` helpers | CardType enum → string, RoundResult → dict — simple mapping, no ORM |
| CORS | flask-cors on `/api/*` | Web UI served from different port during development |

## 3. Endpoint Definitions

Base path: `/api`

### 3.1 `POST /api/games` — Create Game

Creates a new game session, initializes both players with standard hands.

| | |
|---|---|
| **Request body** | `{}` (empty) |
| **Success** | `201 Created` |
| **Response body** | `{"game_id": "<uuid>", "state": {...}}` (see §4.1) |

**Interaction flow:**

> See [create_game.puml](diagrams/endpoints/create_game.puml).

1. Generate a UUID for the game session.
2. Instantiate `GameManager`, call `start_game()`.
3. Save via `SessionStore` (write to disk).
4. Return `game_id` + serialized `get_state()`.

---

### 3.2 `GET /api/games/{game_id}` — Get State

Returns the current game snapshot. Used by the web UI to render the board (hands, HP, round info).

| | |
|---|---|
| **Path param** | `game_id: str` — session UUID |
| **Success** | `200 OK` |
| **Response body** | `{"game_id": "...", "state": {...}}` (see §4.1) |
| **Errors** | `404` — game not found |

**Interaction flow:**

> See [get_state.puml](diagrams/endpoints/get_state.puml).

1. Load game via `SessionStore.load(game_id)` → 404 if missing.
2. Call `manager.get_state()`.
3. Serialize and return.

> **Design note:** The response includes both players' full hand cards. The web UI is responsible for hiding the opponent's hand (rendering card backs) during the selection phase, matching the prototype's pass-screen approach.

---

### 3.3 `POST /api/games/{game_id}/play` — Play Round

Submit both players' cards in a single request. The server resolves immediately.

| | |
|---|---|
| **Path param** | `game_id: str` |
| **Request body** | `{"cards_1": ["<card>", ...], "cards_2": ["<card>", ...]}` |
| **Success** | `200 OK` |
| **Response body** | `{"round_result": {...}, "state": {...}}` (see §4.2, §4.1) |
| **Errors** | `400` — invalid card string, card not in hand, game already over |
| | `404` — game not found |

**Interaction flow:**

> See [play_round.puml](diagrams/endpoints/play_round.puml).

1. Load game via `SessionStore.load(game_id)` → 404 if missing.
2. Deserialize `cards_1` / `cards_2` from string arrays to `list[CardType]` → 400 if any string invalid.
3. Call `manager.play_round(cards_1, cards_2)`.
4. Catch `ValueError` → extract message, return `400 {"error": message}`.
5. Save updated state via `SessionStore.save(game_id, manager)`.
6. Serialize `RoundResult` + updated `get_state()` and return.

---

### 3.4 `DELETE /api/games/{game_id}` — Clean Up

Removes a game session from memory. Called by the client when navigating away or starting a new game.

| | |
|---|---|
| **Path param** | `game_id: str` |
| **Success** | `204 No Content` |
| **Errors** | `404` — game not found |

**Interaction flow:**

> See [delete_game.puml](diagrams/endpoints/delete_game.puml).

1. Delete via `SessionStore.delete(game_id)` → 404 if not present.
2. Return 204.

---

## 4. Response Schemas

### 4.1 Game State (`get_state()` serialization)

```json
{
  "round": 1,
  "players": {
    "1": {
      "hp": 3,
      "hand": ["attack", "attack", "attack", "defense", "defense", "heavy", "counter"],
      "history": []
    },
    "2": {
      "hp": 3,
      "hand": ["attack", "attack", "attack", "defense", "defense", "heavy", "counter"],
      "history": []
    }
  },
  "is_over": false,
  "winner": null,
  "last_round": null
}
```

- `hand`: Serialized via `card.value` (the string form). Order matches standard hand order.
- `last_round`: `null` before the first round; after each round, contains the most recent `RoundResult` (see §4.2). Enables the UI to display round history without a separate request.

### 4.2 Round Result

```json
{
  "round": 1,
  "cards_1": ["attack"],
  "cards_2": ["defense"],
  "damage_1": 0,
  "damage_2": 0,
  "hp_1_after": 3,
  "hp_2_after": 3,
  "details": [
    "Player 1 Attack blocked by Player 2's Defense → Player 2 takes 0",
    "Player 2 Defense vs Attack → Player 2 takes 0"
  ],
  "is_game_over": false,
  "winner": null
}
```

### 4.3 Error Response

All errors use a consistent format:

```json
{
  "error": "Human-readable description",
  "code": "ERROR_CODE"
}
```

| HTTP Status | `code` | When |
|-------------|--------|------|
| `400` | `INVALID_CARD` | Card string doesn't match any `CardType` value |
| `400` | `CARD_NOT_IN_HAND` | Card is valid but not in player's remaining hand |
| `400` | `GAME_OVER` | `play_round()` on a finished game |
| `400` | `GAME_NOT_STARTED` | `play_round()` before `start_game()` |
| `404` | `GAME_NOT_FOUND` | `game_id` not in session store |

---

## 5. Session Persistence

Game state is persisted to disk via `SessionStore`. See [persistence.md](persistence.md) for the full design.

Session lifecycle:
- **Created**: POST /games writes initial state to `{save_dir}/{game_id}.json`.
- **Active**: POST /play reloads state from disk, resolves round, re-saves to disk.
- **Terminal**: Game ended — GET /state still available (data on disk). POST /play returns 400 GAME_OVER.
- **Cleaned up**: DELETE /games removes the JSON file from disk.

No in-memory cache is maintained. Every request reads/writes the filesystem directly.

---

## 6. Internal Module Structure

```
src/backend/controller/
├── api.py              # Flask app, route definitions
└── serializers.py      # CardType ↔ str, RoundResult → dict, error dicts
```

| File | Responsibility |
|------|---------------|
| `api.py` | Route handlers. Each handler: parse request → load from `SessionStore` (imported from `game/`) → call `game/` → save to `SessionStore` → serialize → return response. Zero branching on game rules. |
| `serializers.py` | Pure transformation functions. `card_to_str(c)`, `str_to_card(s)`, `round_result_to_dict(rr)`, `error_response(msg, code, status)`. |

**Dependency direction (strict):**

```
api.py  ──imports──▶  serializers.py
   │                      │
   │                      ▼
   │               src/backend/game/*
   │
   └── imports game/* for GameManager, SessionStore
```

`api.py` imports `game/` classes directly for `GameManager` instantiation. `serializers.py` handles all type conversion and formatting — routes never do ad-hoc JSON construction.

---

## 7. Error Handling Strategy

The adapter's error handling is limited to three categories:

1. **Input validation** (400): Malformed JSON, invalid card strings, missing fields. Handled in the route before calling `game/`.
2. **Game rule violations** (400): `ValueError` from `GameManager.play_round()`. Caught and translated to error response.
3. **Session errors** (404): Unknown `game_id`. Handled by `SessionStore.load()` returning `None`.

All unexpected exceptions (500) propagate to Flask's default error handler — the adapter does NOT swallow unknown errors.

---

## 8. Design Decisions

**DC1 — Client-side face-down protocol.** The server never buffers individual card selections. The web UI handles the pass-screen flow (Player 1 selects → hide → Player 2 selects), then sends both cards atomically via `POST /play`. This keeps the API stateless between rounds and eliminates pending-selection edge cases (re-submission, timeout, spectator reveal).

**DC2 — Full hand exposure.** `GET /games/{id}` returns both players' complete hands. The web UI is responsible for hiding the opponent's cards (rendering card backs). This keeps the API simple and matches the prototype's approach.

**DC3 — Disk persistence via JSON files.** Game sessions are saved to `{save_dir}/{game_id}.json` after creation and after each round. Games survive server restarts. The storage layer (`SessionStore`) is a thin wrapper around filesystem I/O — swapping to Redis or SQLite requires only replacing that single class.

**DC4 — Stateless routes.** The Flask app does not use server-side sessions or cookies. All state is persisted to disk via `SessionStore`, keyed by UUID. This avoids sticky-session issues and allows server restarts without data loss.

---

## 9. Testing Strategy

Tests go in `tests/integration/` (Flask test client against live routes):

| Category | What to test |
|----------|-------------|
| Create game | 201 + valid state, game_id uniqueness |
| Get state | 200 with correct structure, 404 for unknown id |
| Play round | 200 with valid result, 400 for invalid cards, 400 when game over |
| Delete | 204 on success, 404 on unknown id, subsequent GET returns 404 |
| Full flow | Create → play 7 rounds → game over → state reflects winner |

All tests use Flask's `test_client()` — no real HTTP server needed.
