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
  session_store.py       Disk-based game persistence (save/load/delete)

src/backend/controller/  REST API adapter (thin, no business logic)
  api.py                 Flask routes + static serving
  serializers.py         CardType ⇄ string, RoundResult → dict, error formatting

src/backend/scripts/     Server entry point
  run.py                 Start Flask server (python src/backend/scripts/run.py)

src/frontend/            React SPA (Vite + React 19)
  src/api.js             REST API client
  src/App.jsx            Main game UI component
  src/App.css            All styles

tests/
  unit/                  Unit tests (per-module)
  integration/           Integration tests (API routes, full game flows)
  e2e/                   End-to-end tests (future)

src/frontend/tests/      Frontend tests (vitest + React Testing Library)
  api.test.js            API client unit tests
  App.test.jsx            UI component tests
```

## Quick Start

### Development (hot-reload)

```bash
# Terminal 1 — start the Flask API server
python src/backend/scripts/run.py --debug

# Terminal 2 — start the Vite dev server
cd src/frontend
npm install    # first time only
npm run dev
```

Open **http://localhost:5173** in your browser. The Vite dev server proxies
`/api` requests to Flask automatically.

### Production

```bash
# Build the React frontend
cd src/frontend && npm run build

# Start the server (serves both API and static frontend)
cd ../.. && python src/backend/scripts/run.py
```

Open **http://127.0.0.1:5000** in your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/games` | Create a new game |
| `GET` | `/api/games/{id}` | Get current game state |
| `POST` | `/api/games/{id}/play` | Submit cards and resolve a round |
| `DELETE` | `/api/games/{id}` | Delete a game session |

## Running Tests

### Backend (Python)

```bash
# All backend tests (188 tests)
python -m pytest tests/

# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/
```

### Frontend (JavaScript)

```bash
cd src/frontend

# Run frontend tests once (27 tests)
npm test

# Watch mode (re-run on file changes)
npm run test:watch
```
