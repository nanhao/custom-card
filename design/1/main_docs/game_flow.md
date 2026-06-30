# Game Flow

## 1. Game Lifecycle

The game is a finite state machine with four states:

```
UNINIT → PLAYING → (ROUND_RESULT) → GAME_OVER
                  ↑        │
                  └────────┘ (if game not over)
```

Refer to [game_lifecycle.puml](diagrams/game_lifecycle.puml) for the full state diagram.

| State | Description | Transition Trigger |
|-------|-------------|-------------------|
| `UNINIT` | Controller constructed, no players | `start_game()` → `PLAYING` |
| `PLAYING` | Waiting for card selection | `play_round(1, 2)` → check game-end |
| `ROUND_RESULT` | Round resolved, damage applied | If game over → `GAME_OVER`; else → `PLAYING` |
| `GAME_OVER` | Winner determined or draw | Terminal state |

---

## 2. Round Execution Flow

Each call to `GameManager.play_round(cards_1, cards_2)` (in `src/backend/game/game_manager.py`) executes the following sequence. Accepts lists for future multi-card support; current version expects exactly 1 card per list.

Refer to [round_flow.puml](diagrams/round_flow.puml) for a sequence diagram of this flow.

### Step-by-step:

1. **Validate** — raise `ValueError` if `is_over` is `True` or any card in `cards_1`/`cards_2` is not in the player's hand.
2. **Resolve** — call `CardResolver.resolve(cards_1, cards_2, player_1.history, player_2.history)` → `ResolutionResult`.
3. **Apply damage** — `player_1.take_damage(result.damage_1)`, `player_2.take_damage(result.damage_2)`.
4. **Discard cards** — call `player_1.play_cards(cards_1)`, `player_2.play_cards(cards_2)`.
5. **Record history** — append each card in `cards_1` to `player_1.history`, each card in `cards_2` to `player_2.history`.
6. **Check game-end** (in priority order):
   - Both HP = 0 → `is_over = True`, `winner = "draw"`
   - Exactly one HP = 0 → `is_over = True`, `winner = "1"` or `"2"`
   - Round ≥ 7 → `is_over = True`, winner = higher HP or `"draw"`
   - Otherwise → `round_number += 1`, continue
7. **Return** `RoundResult` with all round data.

**Design invariant:** Since `max_rounds == initial_hand_size` (both 7), each player always has at least one card remaining at the start of every round. No underflow check needed.

---

## 3. Win Condition Priority

Evaluated after each round's damage is applied:

| Priority | Condition | Winner |
|----------|-----------|--------|
| 1 | Both HP = 0 | `"draw"` |
| 2 | Player 1 HP = 0, Player 2 HP > 0 | `"2"` |
| 3 | Player 2 HP = 0, Player 1 HP > 0 | `"1"` |
| 4 | Round 7 reached, Player 1 HP > Player 2 HP | `"1"` |
| 5 | Round 7 reached, Player 2 HP > Player 1 HP | `"2"` |
| 6 | Round 7 reached, Player 1 HP = Player 2 HP | `"draw"` |
| — | None of above | Game continues |
