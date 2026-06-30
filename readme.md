# custom-card

A 2-player turn-based card game with simultaneous card resolution.

## Game Rules

- 2 players, 3 HP each
- Each player holds 7 cards: 3× Attack, 2× Defense, 1× Heavy Strike, 1× Counter
- Max 7 rounds. The player with higher HP wins at the end; draw if equal.

### Card Effects

| Card | Effect |
|------|--------|
| **Attack** | Deals damage equal to consecutive Attack count (×1, ×2, ×3...) |
| **Defense** | Blocks all damage except Heavy Strike without consecutive Defense |
| **Heavy Strike** | Deals 1 damage unless opponent played Defense in current AND previous rounds |
| **Counter** | vs Attack: opponent takes 1. vs Heavy Strike: both take 1. Otherwise: no damage |

## Project Structure

```
src/backend/game/        Pure game logic (no UI, no I/O)
  card.py                CardType enum + initial hand factory
  player.py              Player state container
  resolver.py            Stateless card resolution engine
  game_manager.py        Game state machine + round orchestration

tests/
  unit/                  Unit tests (per-module)
  integration/           Integration tests (full game flows)
  e2e/                   End-to-end tests (future)
```

## Running Tests

```bash
# All tests
python -m pytest tests/

# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/
```
