# custom-card

A 2-player turn-based card game with simultaneous card resolution.

## Game Rules (v2)

- 2 players, each starting with 30 HP
- 8 card types: Attack (连续击), Defense (格挡), Heavy Strike (破防重击), Counter (防守反击), Energy (聚能), Build (构建), Flex (攻守兼备), Iron Defense (严防死守)
- **Energy System**: Each player has an energy bar (0–5), starting at 2. Playing cards costs no energy; only building new cards via Build card consumes energy.
- **Deck System**: Each player has a draw pile and discard pile. Initial draw pile: 3× Attack + 2× Defense + 1× Energy + 1× Build (7 cards).
- **Cycle (轮) System**: 3 rounds per cycle. At each cycle start, all hands are discarded, then 5 cards are drawn via smart draw (discard pile shuffled into draw pile when draw pile is empty). At cycle end, each player loses 1 HP (bleed).
- **Priority Resolution**: Card effects resolve by priority (10, 11, or 12) — higher priority resolves first.
- **Status System**: Defense (防御), Guard (联防), Fragile (脆弱) — each with managed lifetimes.
- **Mutual Visibility**: Both players can see each other's hand, discard pile, and draw pile.
- **Win Condition**: Opponent HP reaches 0; no maximum round limit.

## Project Structure

```
src/backend/game/        Pure game logic (no UI, no I/O)
  card_type.py           CardType enum (8 values)
  card_def.py            CardDef dataclass + per-card Resolver mapping
  player.py              Player state container (HP, energy, statuses, piles, hand, draw_cards)
  player_history_item.py PlayerHistoryItem dataclass (card, build_target, dismantle_target)
  player_state.py        StatusType enum + PlayerStatus dataclass with lifetime
  player_state_snapshot.py  PlayerStateSnapshot read-only dataclass
  game_state_snapshot.py GameStateSnapshot read-only dataclass
  game_core.py           GameCore — state machine, play_round, cycle management, end detection
  round_result.py        RoundResult four-phase dataclass
  constants.py           Game-wide constants (HP, energy, deck composition)
  session_store.py       Disk-based JSON persistence (save/load/delete)
  resolver/              Card resolution engine (13 Resolver subclasses, strategy pattern)

src/backend/ai_adapter/  Bridge: game ↔ AI tensor conversion
  ai_interface.py        Abstract AI interface
  ai_registry.py         AiRegistry — AI registration, model loading/caching
  converter.py           Card↔index bidirectional mapping, 62-d input vector construction
  atte_v1_converter.py   Atte v1-specific converter
  rnn_ai.py              RNN-based AI implementation (rnn_v1/rnn_v2)
  atte_v1_ai.py          Attention-based AI implementation

src/backend/ai/           AI models (pure ML, no game logic)
  registry.py            Declarative AI registry (model classes, modules, weight paths)
  rnn_v1/model.py        GRU models: GRUPolicyModel + GRUValueModel
  rnn_v2/model.py        GRU models with skip connection (design/34)
  rnn_v2/constants.py    INPUT_SIZE=62, POLICY_OUTPUT_SIZE=24, HIDDEN_SIZE=32
  atte_v1/model.py       Attention-based transformer model
  atte_v1/constants.py   Atte v1 constants

src/backend/training/    PPO training pipeline
  training_runner.py     BaseTrainingRunner — three-phase PPO (value→advantage→policy)
  training_config.py     TrainingConfig dataclass (PPO hyperparameters)
  training_utils.py      CSV loss/grad logging + PNG charts
  training_factory.py    Training runner factory
  selfplay_config.py     SelfPlayConfig + SelfPlayResult dataclasses
  rnn_v2/selfplay_runner.py     SelfPlayRunner — AI vs AI game loop, PPO sample generation
  rnn_v2/v2_training_runner.py  V2TrainingRunner subclass
  rnn_v2/sample_loader.py       SampleLoader — JSON sample loading
  rnn_v2/sample_collector.py    PPO sample collection
  rnn_v2/training_manager.py    Training orchestration
  atte_v1/atte_v1_training_runner.py  Atte v1 training runner

src/backend/app/          Thin orchestration layer
  app_manager.py         AppManager — game creation, AI action selection

src/backend/config/       Backend-wide configuration
  backend.json           Shared config (logging, training hyperparameters)

src/backend/controller/  REST API adapter (thin, no business logic)
  api.py                 Flask routes + static serving
  serializers.py         CardType ⇄ string, RoundResult → dict, error formatting

src/backend/log/          Structured logging facility
  structured_logger.py   StructuredLogger + LoggerBuilder singleton

src/backend/scripts/     Entry points
  run.py                 Start Flask server
  selfplay.py            AI self-play
  train.py               PPO training (split policy/value networks)
  iteration_train.py     Iterative PPO pipeline — self-play → train → validate → repeat

src/frontend/            React SPA (Vite + React 19)
  src/api.js             REST API client
  src/App.jsx            App shell (menu, navigation, game lifecycle)
  src/GameBoard.jsx      In-game UI (hand area, status, result, log)
  src/main.jsx           React entry point
  src/App.css            All styles

tests/
  unit/                  Unit tests (26 test files, 592 tests)
  integration/           Integration tests (API routes, full game flows)
  e2e/                   End-to-end tests (future)

src/frontend/tests/      Frontend tests (vitest + React Testing Library, 13 tests)
  api.test.js            API client unit tests
  App.test.jsx           UI component tests
  B10.test.jsx           PVP pass phase highlight tests
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

python src/backend/scripts/selfplay.py --ai1 atte_v1.1 --ai2 atte_v1.1 --game 500 --temperature 1 --temp_step 2000 --temp_side both --perspective both --sample_output samples/atte/gen1/atte_v1_self_play.json --perspective ai1
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
    --policy-epoch 10

# Train with different epoch counts for policy and value networks
python src/backend/scripts/train.py ^
    --sample-path samples/rnn_v2_self_play.json ^
    --output-dir src/backend/ai/rnn_v1/gen2 ^
    --model rnn_v1.1 ^
    --policy-epoch 10 ^
    --value-epoch 5

# Train on all JSON files in a directory
python src/backend/scripts/train.py ^
    --sample-path data/samples/ ^
    --output-dir src/backend/ai/rnn_v1/gen2 ^
    --model rnn_v1.1 ^
    --policy-epoch 10

# Train with custom validation split (15% validation set)
python src/backend/scripts/train.py ^
    --sample-path samples/rnn_v2_self_play.json ^
    --output-dir src/backend/ai/rnn_v1/gen2 ^
    --model rnn_v1.1 ^
    --policy-epoch 10 ^
    --validation-split 0.15

python src/backend/scripts/train.py --sample-path samples/v1 --output-dir src/backend/ai/rnn_v1/gen2 --model rnn_v1.1 --policy-epoch 100
```

### Iterative PPO Training

Automated pipeline that iteratively trains AI generations: each iteration runs
self-play/cross-play to generate samples, trains a new generation, then validates
against historical generations. Continues advancing until validation fails.

```bash
# Start iterative training from gen1 with atte_v1
python src/backend/scripts/iteration_train.py --ai atte_v1 --start-gen 1 --sample-dir samples/ --game 2000 --policy-epoch 5 --value-epoch 10

# With different epoch counts
python src/backend/scripts/iteration_train.py --ai atte_v1 --start-gen 1 --sample-dir samples/atte_v1 --game 1000 --policy-epoch 10 --value-epoch 10
```

**Pipeline Workflow (per iteration):**

| Step | Description |
|------|-------------|
| ① Self-Play | gen vs gen, `--game` games, perspective=both → `{sample_dir}/{ai}/{gen}/self_play.json` |
| ② Cross-Play | gen vs gen-1 (`game/2`), gen vs gen-2 (`game/4`), gen vs gen-3 (`game/8`), gen vs gen-4 (`game/16`), perspective=ai1 → `{sample_dir}/{ai}/{gen}/{k}_play.json` |
| ③ Training | Three-phase PPO on all samples in `{sample_dir}/{ai}/{gen}/` → gen+1 weights at `src/backend/ai/{name}/gen{gen+1}/` |
| ④ Validation | gen+1 vs gen/1/2/3/4, 500 games each. Condition: (ai1_wins - ai2_wins) / 500 > 5%. All must pass. |

- Historical generations that don't exist are skipped (e.g. gen-1 at start_gen=1).
- For `atte_v1`, weight files are automatically split into e1/e2 + main files after training.
- Training hyperparameters (LR, gamma, clip_eps, etc.) are loaded from `config/backend.json` (same as `train.py`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ai` | `str` | *(required)* | AI name to train (e.g. `atte_v1`, `rnn_v2`) |
| `--start-gen` | `int` | *(required)* | Starting generation number (weights must exist at `src/backend/ai/{name}/gen{start_gen}/`) |
| `--sample-dir` | `str` | *(required)* | Root directory for sample output (e.g. `samples/`) |
| `--game` | `int` | *(required)* | Base game count for self-play per generation |
| `--policy-epoch` | `int` | `5` | Policy network training epochs per generation |
| `--value-epoch` | `int` | `10` | Value network training epochs per generation |

**Training Outputs:** After each generation, weights are saved to
`src/backend/ai/{name}/gen{gen+1}/`. For `atte_v1` this includes 4 files
(`policy_e1_e2.pth`, `policy.pth`, `value_e1_e2.pth`, `value.pth`);
for `rnn_v2` it includes 2 files (`policy.pth`, `value.pth`).
Loss/grad artifacts are generated in `loss/`.

**Training Outputs:** After training, two weight files are saved
(`{output_dir}/policy.pth` + `{output_dir}/value.pth`) plus loss/grad charts in `loss/`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sample-path` | `str` | *(required)* | File path or directory of JSON sample files |
| `--output-dir` | `str` | *(required)* | Directory to save `policy.pth` and `value.pth` |
| `--model` | `str` | *(required)* | AI registry name to train (e.g. `rnn_v1.1`) |
| `--policy-epoch` | `int` | *(required)* | Number of training epochs for policy network (Phase 3) |
| `--value-epoch` | `int` | *(optional)* | Number of training epochs for value network (Phase 1). Defaults to `--policy-epoch`. |
| `--validation-split` | `float` | *(optional)* | Fraction of samples reserved for validation set (0 <= split < 1). Defaults to `training.validation_split` in `backend.json` (0.1). |

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
| `validation_split` | `float` | `0.1` | Fraction of samples reserved for validation set (0 = disabled) |

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
# All backend tests (592 tests)
python -m pytest tests/

# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/
```

### Frontend (JavaScript)

```bash
cd src/frontend

# Run frontend tests once (13 tests)
npm test

# Watch mode (re-run on file changes)
npm run test:watch
```
