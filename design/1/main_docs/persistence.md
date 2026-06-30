# Persistence Design — `src/backend/game/`

## 1. Constraints (from architecture.md)

Per **D10**: This module contains NO business logic. It only reads/writes `GameManager` state to disk. All game rules live in `src/backend/game/`.

Per **D11**: The face-down protocol is client-side. The persistence layer has no knowledge of selection phases or multi-device buffering.

## 2. Config File

### Location

`{project_root}/config.json`

### Format

```json
{
    "save_dir": "./saves"
}
```

- `save_dir`: Path relative to project root. Default: `"./saves"` (resolves to `{project_root}/saves/`).
- `SessionStore` reads this file at initialization.
- If `config.json` does not exist, `SessionStore` uses the default value.
- If the `save_dir` does not exist on first write, `SessionStore` creates it automatically.

## 3. File Structure

```
{project_root}/
├── config.json
├── saves/                    # {save_dir} default
│   ├── <uuid1>.json
│   ├── <uuid2>.json
│   └── ...
```

Each game session is stored as a single JSON file named by its UUID.

## 4. SessionStore Interface

`src/backend/game/session_store.py`

```python
class SessionStore:
    def __init__(self, config_path: str = "config.json"):
        """Read save_dir from config file. Default to './saves' if absent."""

    def save(self, game_id: str, game_manager: GameManager) -> None:
        """Serialize GameManager and write to {save_dir}/{game_id}.json.
        Creates the file if it does not exist; overwrites if it does."""

    def load(self, game_id: str) -> GameManager | None:
        """Read and deserialize from {save_dir}/{game_id}.json.
        Returns None if the file does not exist."""

    def delete(self, game_id: str) -> bool:
        """Remove {save_dir}/{game_id}.json. Returns True if it existed."""
```

| Method | Called by | Trigger |
|--------|----------|---------|
| `save()` | `POST /api/games`, `POST /api/games/{id}/play` | Game creation, after each round resolution |
| `load()` | `GET /api/games/{id}`, `POST /api/games/{id}/play` | State retrieval, round pre-validation |
| `delete()` | `DELETE /api/games/{id}` | Explicit cleanup |

## 5. Serialization Format

### 5.1 Game Save JSON

Each `{game_id}.json` contains the full `GameManager` state:

```json
{
    "round": 2,
    "player_1": {
        "hp": 3,
        "hand": ["attack", "defense", "heavy"],
        "history": ["counter"]
    },
    "player_2": {
        "hp": 2,
        "hand": ["attack", "attack", "defense", "counter"],
        "history": ["attack"]
    },
    "is_over": false,
    "winner": null,
    "round_log": [
        {
            "round": 1,
            "cards_1": ["attack"],
            "cards_2": ["counter"],
            "damage_1": 0,
            "damage_2": 0,
            "hp_1_after": 3,
            "hp_2_after": 3,
            "details": ["Player 1 Attack vs Player 2 Counter: 0 damage each"],
            "is_game_over": false,
            "winner": null
        }
    ]
}
```

### 5.2 Serialization

Serialization (`GameManager` ↔ `dict` ↔ JSON) is internal to `SessionStore`. The controller never touches raw dicts — it only calls `store.save(gm)` and `store.load(id)`.

## 6. Lifecycle Integration

```
POST /api/games
  → api.py generates UUID, creates GameManager, calls store.save(game_id, gm)
  → 201 Created

GET /api/games/{id}
  → store.load(game_id)
  → 404 if None
  → gm.get_state() → 200 OK

POST /api/games/{id}/play
  → store.load(game_id)
  → 404 if None
  → gm.play_round(cards_1, cards_2)
  → store.save(game_id, gm)           ← persist updated state
  → 200 OK {round_result, state}

DELETE /api/games/{id}
  → store.delete(game_id)
  → 404 if False (file did not exist)
  → 204 No Content
```

All reads and writes go through the filesystem. There is no in-memory cache. This keeps the server stateless and allows games to survive process restarts.

## 7. Module Structure

`session_store.py` lives in `src/backend/game/`, alongside the other game-logic modules:

```
src/backend/game/
├── card.py
├── player.py
├── resolver.py
├── game_manager.py
└── session_store.py    # SessionStore: disk-based game state persistence
```

The controller (`src/backend/controller/api.py`) imports `SessionStore` from `game/` the same way it imports `GameManager`.

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| `config.json` missing | Use default `"./saves"` |
| `save_dir` does not exist | Create it on first `save()` call |
| JSON file missing on `load()` | Return `None` → controller returns 404 |
| JSON file corrupted (malformed JSON) | Raise `ValueError` → controller returns 500 |
| Disk write fails (permissions, disk full) | Raise `OSError` → controller returns 500 |

The `SessionStore` does not catch exceptions — it lets them propagate to the Flask route handlers, which translate them to appropriate HTTP responses.

## 9. Testing Strategy

Tests go in `tests/unit/` (no HTTP dependency, filesystem only):

| Category | What to test |
|----------|-------------|
| Save | File created with correct content, overwrite updates content |
| Load | Returns correct GameManager, returns None for missing file |
| Delete | Returns True on success, returns False when file absent |
| Round-trip | save → load produces equivalent GameManager (same state) |
| Config | Default dir when config missing; custom dir from config |
| Corruption | Corrupted JSON raises ValueError on load |
