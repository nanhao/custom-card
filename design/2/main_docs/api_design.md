# API Design — Human vs AI Mode

## 1. Changes Overview

| Change | Endpoint | Type |
|--------|----------|------|
| New | `GET /api/ais` | List available AIs |
| Modified | `POST /api/games` | Now accepts `mode` + `ai` fields |
| Modified | `POST /api/games/{id}/play` | PVE: `cards_2` auto-computed by server |
| Unchanged | `GET /api/games/{id}` | Same response; mode added to state |
| Unchanged | `DELETE /api/games/{id}` | Same behavior |

---

## 2. SessionStore Interface Update

`SessionStore` signature changes to support PVE metadata:

```python
class SessionStore:
    def save(self, game_id: str, game_manager: GameManager, ai_state: list[float] | None = None) -> None
    def load(self, game_id: str) -> tuple[GameManager | None, dict | None]
        # Returns (gm, metadata) where metadata = {"ai": str|None, "ai_state": list[float]|None}
        # Returns (None, None) if game not found
    def delete(self, game_id: str) -> bool
```

- `save()`: If `ai_state` is provided, serialized to `"ai_state"` in the save JSON.
- `load()`: Always returns a tuple. The metadata dict has the following schema:

  | Key | Type | Description |
  |-----|------|-------------|
  | `mode` | `"pvp"` \| `"pve"` | Game mode; defaults to `"pvp"` for legacy files |
  | `ai` | `str` \| `None` | AI name for PVE; `None` for PVP or legacy files |
  | `ai_state` | `list[float]` \| `None` | GRU hidden state for PVE; `None` for PVP. Validated on load (fallback to zeros if invalid) |

- `delete()`: Unchanged.

---

## 3. New Endpoint: `GET /api/ais`

List all registered AI opponents.

| | |
|---|---|
| **Method** | `GET` |
| **Response** | `200 OK` |
| **Body** | `{"ais": [{"name": "rnn_v1.1", "description": "GRU-based model, random weights (gen1)"}]}` |

Controller calls `ai.registry.list_ais()` and returns the result.

---

## 4. Modified Endpoint: `POST /api/games`

Now accepts optional `mode` and `ai` fields to create AI games.

| | |
|---|---|
| **Request body** | `{"mode": "pvp", "ai": null}` (default, backward compatible) |
| | `{"mode": "pve", "ai": "rnn_v1.1"}` (AI game) |
| **Success** | `201 Created` |
| **Errors** | `400` — unknown AI name; `400` — mode is "pve" but no AI specified |
| **Response body** | `{"game_id": "<uuid>", "state": {...}}` |

**AI game creation flow:**

> See [create_ai_game.puml](diagrams/endpoints/create_ai_game.puml).

1. Validate request: if `mode == "pve"`, require `ai` field and check name exists in `ai.registry.list_ais()` (no model loading)
2. Generate UUID, create `GameManager(mode=mode)`, call `start_game()`
3. Initialize `ai_state = [0.0]*8` (hardcoded: `init_hidden_state()` always returns zeros, and we avoid loading the AI module at this stage)
4. Save AI name, `mode`, and `ai_state` to save file via `SessionStore`
5. Return `game_id` + serialized state

---

## 5. Modified Endpoint: `POST /api/games/{game_id}/play`

Behavior differs by game mode.

### 4.1 PVP Mode (unchanged)

Same as existing: client sends both `cards_1` and `cards_2`.

### 4.2 PVE Mode (new)

Client sends only `cards_1`. Server auto-computes AI's card.

| | |
|---|---|
| **Request body** | `{"cards_1": ["attack"]}` (PVE: `cards_2` ignored if present) |
| **Success** | `200 OK` |
| **Response body** | `{"round_result": {...}, "state": {...}}` |

> See [play_ai_round.puml](diagrams/endpoints/play_ai_round.puml).

1. Load game: `gm, metadata = store.load(game_id)` → 404 if `gm is None`
2. If `gm.mode == "pve"`:
   a. Load AI model via `ai.registry.load_ai(metadata["ai"])` (model cached after first load, see ai_module.md)
   b. Pass raw state dict to AI: `ai.select_action(gm.get_state(), metadata["ai_state"])`
   c. AI internally builds 14-d input vector, runs forward pass, returns `(card_str, new_hidden_state)`
   d. Validate `card_str` is one of `"attack"`, `"defense"`, `"heavy"`, `"counter"`; if not, return `500 AI_INFERENCE_ERROR`
   e. Convert `card_str` → `CardType` via `str_to_card()`, set `cards_2 = [card_type]`
3. If `gm.mode == "pvp"`: deserialize `cards_2` from request body (unchanged)
4. Call `gm.play_round(cards_1, cards_2)`
5. If PVE: call `store.save(game_id, gm, new_hidden_state)`
6. If PVP: call `store.save(game_id, gm)`
7. Return result

**Error handling (PVE-specific):**

| Scenario | HTTP | Code |
|----------|------|------|
| AI name not found (corrupted save) | 500 | AI_LOAD_ERROR |
| AI fails to select (model error) | 500 | AI_INFERENCE_ERROR |

All existing errors (INVALID_CARD, CARD_NOT_IN_HAND, GAME_OVER, etc.) still apply.

---

## 6. Updated Game State Schema

### 6.1 `get_state()` — added `mode` and `ai`

```json
{
  "round": 1,
  "players": { "1": {...}, "2": {...} },
  "is_over": false,
  "winner": null,
  "mode": "pve",
  "ai": "rnn_v1.1"
}
```

| Field | Type | When present |
|-------|------|-------------|
| `mode` | `"pvp"` \| `"pve"` | Always |
| `ai` | `str` \| `null` | PVE: AI name; PVP: `null` |

The frontend uses `mode` to determine game flow (pass screen vs auto-reveal) and `ai` to display the AI name alongside Player 2's area. The `ai` value is read from the save file metadata, not stored in GameManager.

**New Game flow (AC15):** When the game ends, the frontend reads `state.ai` from the final GET state response. The client then calls `POST /api/games` with `{"mode": "pve", "ai": "<state.ai>"}` to start a new game against the same AI. No dedicated restart endpoint is needed.

**Round log (AC18):** The `round_log` array (present in `GameManager`) is included in the `get_state()` response so the frontend can display accumulated round history. The response format is:
```json
{
  "round": 2,
  "players": {...},
  "is_over": false,
  "winner": null,
  "mode": "pve",
  "ai": "rnn_v1.1",
  "round_log": [
    {"round": 1, "cards_1": ["attack"], "cards_2": ["defense"], "damage_1": 0, "damage_2": 0, "hp_1_after": 3, "hp_2_after": 3, "details": [...], "is_game_over": false, "winner": null}
  ]
}
```

### 6.2 Save File — added `mode`, `ai`, `ai_state`

```json
{
  "round": 2,
  "player_1": {...},
  "player_2": {...},
  "is_over": false,
  "winner": null,
  "round_log": [...],
  "mode": "pve",
  "ai": "rnn_v1.1",
  "ai_state": [0.12, -0.34, 0.0, 0.56, -0.01, 0.78, -0.22, 0.03]
}
```

| Field | Type | When present |
|-------|------|-------------|
| `mode` | `str` | Always; defaults to `"pvp"` for legacy files |
| `ai` | `str` \| `null` | PVE: AI name; PVP: absent or `null` (silently ignored if present in PVP save) |
| `ai_state` | `list[float]` \| `null` | PVE: GRU hidden state, length 8; validated on load (fallback to zeros if invalid) |

**Validation rules on load:**
- If `mode == "pve"` but `ai` is missing/invalid → treat as corrupted, return `500 AI_CONFIG_ERROR`
- If `ai_state` is missing, wrong length, or contains non-numeric values → fall back to `[0.0]*8` with logged warning
- If `mode == "pvp"` and `ai_state`/`ai` are present → silently ignore

---

## 7. Serializer Changes (`serializers.py`)

No changes. The existing `card_to_str` / `str_to_card` / `round_result_to_dict` are sufficient. AI card selection returns `str` directly (e.g. `"attack"`), which maps directly to `CardType` via `str_to_card`.

---

## 8. Module: `game/game_manager.py`

### New constructor parameter

`__init__(self, mode: str = "pvp")` — `mode` is set at construction and never changed. This enforces immutability after creation (no post-construction `gm.mode = "pve"`).

### `get_state()` — updated return

Adds `"mode": self.mode` to the returned dict. The controller injects `"ai"` from the save file metadata when building the API response.
