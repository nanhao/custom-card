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
  card_type.py           CardType enum (7 values)
  card_def.py            CardDef dataclass + per-card Resolver mapping
  player.py              Player state container
  resolver/              Card resolution engine (strategy pattern, per-effect Resolver subclasses)
  game_manager.py        Game state machine + round orchestration
  session_store.py       Disk-based game persistence (save/load/delete)

src/backend/config/       Backend-wide configuration
  backend.json           Shared config (logging, future: database, API)

src/backend/ai/           AI opponents (no game logic)
  registry.py            AI registry: list, get metadata
  training_config.py     TrainingConfig dataclass (PPO hyperparameters)
  training_runner.py     PPO training loop (GAE, loss, gradient accumulation)
  training_utils.py      Training artifact generation (CSV + PNG charts)
  rnn_v1/                GRU neural network model (PyTorch) — GRUPolicyModel + GRUValueModel

src/backend/app/          Bridge layer (orchestrates game + AI)
  app_manager.py         Game creation, AI action selection, model lifecycle
  converter.py           Card ↔ tensor conversion, input vector construction
  selfplay_config.py     SelfPlayConfig and SelfPlayResult dataclasses
  selfplay_runner.py     Self-play game loop (AI vs AI, PPO sample generation)

src/backend/log/          Structured logging facility (generic)
  structured_logger.py   StructuredLogger + LoggerBuilder singleton

src/backend/controller/  REST API adapter (thin, no business logic)
  api.py                 Flask routes + static serving
  serializers.py         CardType ⇄ string, RoundResult → dict, error formatting

src/backend/scripts/     Entry points
  run.py                 Start Flask server (python src/backend/scripts/run.py)
  selfplay.py            AI self-play (python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1)
  train.py               PPO training (python src/backend/scripts/train.py --sample-path ... --output-dir ... --model rnn_v1.1 --epoch 10)

src/frontend/            React SPA (Vite + React 19)
  src/api.js             REST API client
  src/App.jsx            App shell (menu, navigation, game lifecycle)
  src/GameBoard.jsx      In-game UI (hand area, status, result, log)
  src/App.css            All styles

tests/
  unit/                  Unit tests (per-module)
  integration/           Integration tests (API routes, full game flows)
  e2e/                   End-to-end tests (future)

src/frontend/tests/      Frontend tests (vitest + React Testing Library)
  api.test.js            API client unit tests
  App.test.jsx           UI component tests
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
# Build frontend then start server
npm run build --prefix src/frontend ; python src/backend/scripts/run.py
```

Open **http://127.0.0.1:5000** in your browser.

### AI Self-Play

```bash
# Run 100 games between two AI instances
python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1
python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 --game 100 --temperature 2 --temp_step 1 

# With temperature sampling, perspective filtering, and PPO sample output
python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 \
    --game 200 --temperature 0.5 --temp_step 3 --temp_side both \
    --perspective ai1 --sample_output data/samples.json --temp_seed 42

python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 --game 1000 --temperature 2 --temp_step 2000 --temp_side both --perspective both --sample_output samples/v1/rnn_v1_self_play.json --perspective ai1
```

**Structured Logging:** Set `enabled` to `true` in `src/backend/config/backend.json`
to enable machine-parseable event logging to `logs/{yyyy}-{mm}-{dd}.log`.
When enabled, the self-play lifecycle events (run_start, game_start, round_action,
etc.) are written with a unique traceId for post-hoc analysis. See
design/6/main_docs/ for the full specification.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ai1` | `str` | *(required)* | AI name for player 1 (e.g. `rnn_v2.1`) |
| `--ai2` | `str` | *(required)* | AI name for player 2 (e.g. `rnn_v2.1`) |
| `--game` | `int` | `100` | Number of games to play |
| `--temperature` | `float` | `1.0` | Sampling temperature (`0` = greedy, `>0` = stochastic). Higher values flatten the distribution (more exploration) |
| `--temp_step` | `int` | *(none)* | First N rounds use sampling, then switch to greedy. Omit for no limit |
| `--temp_side` | `str` | `both` | Which side to apply temperature sampling: `player1`, `player2`, or `both` |
| `--temp_seed` | `int` | *(none)* | Random seed for reproducible sampling |
| `--sample_output` | `str` | *(none)* | File path for PPO sample JSON array output (game -> perspective -> steps hierarchy). Omit to skip sample recording |
| `--perspective` | `str` | `both` | Perspectives to include in sample file: `ai1`, `ai2`, or `both` |

### AI Training

Train the split policy/value networks using three-phase PPO on self-play samples.
Training hyperparameters are configured in `src/backend/config/backend.json` under
the `training` section.

```bash
# Train with split networks (three-phase PPO)
python src/backend/scripts/train.py ^
    --sample-path samples/rnn_v2_self_play.json ^
    --output-dir src/backend/ai/rnn_v1/gen2 ^
    --model rnn_v1.1 ^
    --epoch 10

# Train on all JSON files in a directory
python src/backend/scripts/train.py ^
    --sample-path data/samples/ ^
    --output-dir src/backend/ai/rnn_v1/gen2 ^
    --model rnn_v1.1 ^
    --epoch 10

python src/backend/scripts/train.py --sample-path samples/v1 --output-dir src/backend/ai/rnn_v1/gen2 --model rnn_v1.1 --epoch 100
```

**Training Outputs:** After training, two weight files are saved
(`{output_dir}/policy.pth` + `{output_dir}/value.pth`) plus loss/grad charts in `loss/`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sample-path` | `str` | *(required)* | File path or directory of JSON sample files |
| `--output-dir` | `str` | *(required)* | Directory to save `policy.pth` and `value.pth` |
| `--model` | `str` | *(required)* | AI registry name to train (e.g. `rnn_v1.1`) |
| `--epoch` | `int` | *(required)* | Number of training epochs for both value and policy phases |

**Hyperparameters** (in `config/backend.json` -> `training`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `value_lr` | `float` | `0.001` | Learning rate for value optimizer (AdamW) |
| `policy_lr` | `float` | `0.001` | Learning rate for policy optimizer (AdamW) |
| `weight_decay` | `float` | `0.01` | Weight decay for AdamW |
| `gamma` | `float` | `0.99` | Discount factor for GAE |
| `gae_lambda` | `float` | `0.95` | GAE lambda parameter |
| `clip_eps` | `float` | `0.2` | PPO clipping epsilon |
| `max_grad_norm` | `float` | `0.5` | Maximum gradient norm for clipping |
| `value_coef` | `float` | `0.5` | Value loss coefficient (Phase 1 only) |
| `entropy_coef` | `float` | `0.01` | Entropy bonus coefficient (Phase 3 only) |
| `batch_size` | `int` | `64` | Perspectives per gradient accumulation group |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/ais` | List available AI opponents |
| `POST` | `/api/games` | Create a new game (supports `mode` and `ai` params for PVE) |
| `GET` | `/api/games/{id}` | Get current game state |
| `POST` | `/api/games/{id}/play` | Submit cards and resolve a round (PVE: auto-computes AI card) |
| `DELETE` | `/api/games/{id}` | Delete a game session |

## Running Tests

### Backend (Python)

```bash
# All backend tests (506 tests)
python -m pytest tests/

# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/
```

### Frontend (JavaScript)

```bash
cd src/frontend

# Run frontend tests once (28 tests)
npm test

# Watch mode (re-run on file changes)
npm run test:watch
```
