# Interface Definitions

## 1. `CardType` (Enum) — `src/backend/game/card.py`

```python
class CardType(Enum):
    ATTACK = "attack"
    DEFENSE = "defense"
    HEAVY_STRIKE = "heavy"
    COUNTER = "counter"
```

**Public function** — `get_initial_hand() -> list[CardType]`
> Returns a **new** copy of the standard 7-card starting hand: `[ATTACK, ATTACK, ATTACK, DEFENSE, DEFENSE, HEAVY_STRIKE, COUNTER]`. Returns a fresh list on each call to prevent shared-mutation bugs.

---

## 2. `Player` — `src/backend/game/player.py`

A data container representing one player's state. No methods beyond property access.

| Attribute | Type | Description |
|-----------|------|-------------|
| `hp` | `int` | Current HP, initially 3, clamped to ≥0 |
| `hand` | `list[CardType]` | Remaining cards, standard order: [ATTACK×3, DEFENSE×2, HEAVY_STRIKE×1, COUNTER×1] |
| `history` | `list[CardType]` | Cards played in previous rounds, chronologically. Each round appends the card(s) played that round. Currently 1 card/round; future multi-card will append all cards from the round. |

**Public methods:**

- **`play_card(card: CardType) -> None`**
  Removes the first occurrence of `card` from `hand` (equivalent to `list.remove(card)`). Raises `ValueError` if card not in hand. Convenience wrapper around `play_cards`.

- **`play_cards(cards: list[CardType]) -> None`**
  Removes one occurrence of each card in `cards` from `hand`. Raises `ValueError` if any card is not present (no partial removal). Supports future multi-card gameplay.

- **`take_damage(amount: int) -> None`**
  Reduces `hp` by `amount`, clamped to 0.

- **`has_card(card: CardType) -> bool`**
  Returns `True` if `card` is present in hand.

- **`has_cards(cards: list[CardType]) -> bool`**
  Returns `True` if all cards in `cards` are present in hand (with sufficient duplicates). Supports future multi-card gameplay.

---

## 3. `ResolutionResult` — `src/backend/game/resolver.py`

Value object returned by the resolver.

| Attribute | Type | Description |
|-----------|------|-------------|
| `damage_1` | `int` | Damage dealt to Player 1 this round |
| `damage_2` | `int` | Damage dealt to Player 2 this round |
| `details` | `list[str]` | Human-readable resolution steps (for logging) |

---

## 4. `CardResolver` — `src/backend/game/resolver.py`

**Stateless.** All resolution logic is pure functions. No class instantiation needed.

- **`resolve(cards_1: list[CardType], cards_2: list[CardType], history_1: list[CardType], history_2: list[CardType]) -> ResolutionResult`**
  Computes damage for both players given their played cards and history. Accepts lists for future multi-card support; current version expects exactly 1 card per list. Handles all 4×4 card combinations per PRD §3.

- **`get_consecutive_attack(history: list[CardType]) -> int`**
  Counts consecutive Attack cards from the end of history. Used internally by `resolve`.

- **`played_defense_last_round(history: list[CardType]) -> bool`**
  Checks if the last card in history was Defense. Used internally by `resolve`.

---

## 5. `GameManager` — `src/backend/game/game_manager.py`

Manages the full game lifecycle. This is the only stateful component.

| Attribute | Type | Description |
|-----------|------|-------------|
| `player_1` | `Player` | Player 1 state |
| `player_2` | `Player` | Player 2 state |
| `round_number` | `int` | Current round (1–7) |
| `is_over` | `bool` | Whether game has ended |
| `winner` | `str \| None` | `"1"`, `"2"`, `"draw"`, or `None` |
| `round_log` | `list[RoundResult]` | History of all completed rounds |

**Public methods:**

- **`start_game() -> None`**
  Initializes both players with 3 HP and standard 7-card hand. Sets round to 1. Can be called at any time to reinitialize a fresh game state (doubles as reset).

- **`play_round(cards_1: list[CardType], cards_2: list[CardType]) -> RoundResult`**
  Validates all cards are in respective hands and game is not over, calls `CardResolver.resolve()`, applies damage, discards cards from hands, appends to history, checks game-end conditions, and increments round if game continues.
  Accepts lists for future multi-card support; current version expects exactly 1 card per list.
  Raises `ValueError` if game is already over, any card not in hand, or hand is empty.

- **`get_state() -> dict`**
  Returns a read-only snapshot for UI/logging:
  `{ "round": int, "players": { "1": {"hp": int, "hand": list[CardType], "history": list[CardType]}, "2": {...} }, "is_over": bool, "winner": str|None }`.

---

## 6. `RoundResult` — `src/backend/game/game_manager.py`

Value object returned by `GameManager.play_round()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `round` | `int` | Round number |
| `cards_1` | `list[CardType]` | Cards played by Player 1 |
| `cards_2` | `list[CardType]` | Cards played by Player 2 |
| `damage_1` | `int` | Damage Player 1 took |
| `damage_2` | `int` | Damage Player 2 took |
| `hp_1_after` | `int` | Player 1's HP after resolution |
| `hp_2_after` | `int` | Player 2's HP after resolution |
| `details` | `list[str]` | Resolution steps |
| `is_game_over` | `bool` | Whether game ended this round |
| `winner` | `str \| None` | Winner if game ended |

---

## 7. CLI Adapter — `src/backend/script/cli.py`

**Constraint:** Contains NO business logic. Only parses arguments, calls `game/` classes, and prints output.

- **`main() -> None`**
  Entry point. Parses CLI args, instantiates `GameManager`, runs the interactive game loop. The face-down protocol is implemented as a simple two-step prompt: ask Player 1 for a card, clear screen, ask Player 2 for a card, then call `play_round()`.

- **`display_round_result(result: RoundResult) -> None`**
  Prints round outcome to stdout, including played cards, damage, HP after resolution.

- **`display_game_over(manager: GameManager) -> None`**
  Prints final winner and game summary.

---

## 8. REST API Adapter — `src/backend/controller/api.py`

**Constraint:** Contains NO business logic. Only handles HTTP request/response, delegates to `game/`.

- **`POST /game/start`** → `{"status": "started", "state": ...}`
  Creates a new `GameManager` instance (server-side session).

- **`POST /game/play`** → `RoundResult` as JSON
  Body: `{"cards_1": ["attack"], "cards_2": ["defense"]}`. Calls `play_round()`, returns result. Arrays support future multi-card play; current version expects exactly 1 element each.

- **`GET /game/state`** → `get_state()` as JSON
  Returns current game snapshot for UI rendering.

**Session management** (implementation detail, not part of the `game/` interface): The API adapter maintains a session-keyed map of `GameManager` instances. The face-down protocol is enforced client-side (web UI buffers Player 1's choice, then sends both together).
