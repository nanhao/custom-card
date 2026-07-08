# custom-card — Project Overview

A 2-player turn-based card game with simultaneous card resolution, AI opponents (GRU-based neural networks), and a PPO reinforcement learning training pipeline.

---

## 1. Game Mechanics (v2)

The current game (design/11+) introduces a deep strategic system on top of the original v1 rules:

- **Players**: 2, each starting with 20 HP.
- **Card Types**: 7 — Attack, Defense, Heavy Strike, Counter, Energy, Build, Flex (攻守兼备).
- **Energy System**: Each player has an energy bar (0–5). Cards have a `build_cost` (1 or 2). Playing cards costs _no_ energy; only _building_ new cards via the Build card consumes energy.
- **Deck System**: Each player has a draw pile and discard pile. No initial hand — cards are drawn at cycle start.
- **Cycle (轮) System**: A cycle consists of 5 rounds. At each cycle start, all hands are discarded, discard piles shuffled into draw piles, and 7 cards drawn. At cycle end, each player loses 1 HP (bleed) to prevent infinite loops.
- **Priority Resolution**: Each card effect has a priority (10 or 11). Priority 11 effects (mostly defensive buffs) resolve before priority 10 effects (damage).
- **Status System**: Players can gain statuses like "Defense" (防御) and "Joint Defense" (联防) with managed lifetimes.
- **Win Condition**: A player wins when the opponent's HP reaches 0; no maximum round limit.

---

## 2. Architecture Overview

The project follows a **layered architecture** with strict separation of concerns:

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│        React 19 SPA (Vite + vitest)             │
│   App.jsx → GameBoard.jsx → api.js              │
└─────────────────┬───────────────────────────────┘
                  │  REST (JSON) over HTTP
┌─────────────────▼───────────────────────────────┐
│               Controller Layer                   │
│          Flask API (thin adapter)                │
│   api.py → serializers.py                       │
└────────┬────────────────────────────┬───────────┘
         │                            │
┌────────▼────────┐          ┌────────▼───────────┐
│   App Layer     │          │   Game Layer       │
│  (Bridge)       │          │  (Pure Logic)      │
│                 │          │                    │
│ AppManager      │◄────────►│ GameCore           │
│ Converter       │          │ CardResolver       │
│ SelfPlayRunner  │          │ Player / CardDef   │
│ SelfPlayConfig  │          │ SessionStore       │
└────────┬────────┘          └────────────────────┘
         │
┌────────▼────────────────────────────────────────┐
│               AI Layer (pure ML)                 │
│  registry.py  │  training_runner.py             │
│  training_config.py  │  training_utils.py       │
│  rnn_v1/model.py (GRUPolicyModel + GRUValueModel)│
└─────────────────────────────────────────────────┘
         │
┌────────▼────────┐   ┌───────────────────────────┐
│    Log Layer    │   │     Config                 │
│ StructuredLogger│   │  backend.json              │
└─────────────────┘   └───────────────────────────┘
```

**Key principles**:
- **Game layer** is pure logic — no UI, no I/O, no ML dependencies.
- **AI layer** is pure ML — no game logic, no knowledge of rules.
- **App layer** bridges the two, handling card↔tensor conversion, action selection, and masking.
- **Controller layer** is a thin HTTP adapter with no business logic.
- **Frontend** is a standalone React SPA communicating via REST.

---

## 3. Backend Code Structure

### 3.1 Game Layer (`src/backend/game/`)

Pure game logic. The only module that knows game rules.

| File | Responsibility |
|------|---------------|
| `card.py` | `CardType` enum (7 values) + `CardDef` dataclass (priority, effects, build_cost, display) |
| `player.py` | `Player` class: HP, energy, statuses, draw/discard piles, hand, history |
| `resolver.py` | `CardResolver` — stateless priority-layered effect resolution engine |
| `game_core.py` | `GameCore` — game state machine (create, play_round, cycle management, end detection) + `RoundResult` dataclass |
| `session_store.py` | `SessionStore` — pure disk-based JSON persistence (save/load/delete games) |

### 3.2 AI Layer (`src/backend/ai/`)

Pure ML — no knowledge of game rules.

| File | Responsibility |
|------|---------------|
| `registry.py` | Declarative AI registry — maps AI names to model classes and weight paths |
| `training_config.py` | `TrainingConfig` dataclass — PPO hyperparameters |
| `training_runner.py` | PPO training loop — GAE computation, policy/value loss, gradient accumulation (three-phase) |
| `training_utils.py` | Training artifact generation — CSV loss/grad logging + PNG charts |
| `rnn_v1/model.py` | Split GRU models: `GRUPolicyModel` (15→32→13) + `GRUValueModel` (15→32→1) |

**Model Architecture (design/14)**:
- Input: 15-d vector (HP×2, energy×2, round_in_cycle, cycle, card_counts×7, last_action×2)
- Policy: GRU(32) → Linear(32→13) → softmax (13-d interleaved: play/build pairs)
- Value: GRU(32) → Linear(32→1) → scalar win-rate estimate
- Mask function (injected via forward): blocks illegal actions (cards not in hand, insufficient energy)

### 3.3 App Layer (`src/backend/app/`)

Bridge between game logic and AI — the only layer that understands both.

| File | Responsibility |
|------|---------------|
| `app_manager.py` | `AppManager` — game creation, AI action selection (greedy + sampling), model lifecycle (loading/caching policy + value models) |
| `converter.py` | `Converter` — card↔index bidirectional mapping, input vector construction (15-d), illegal action masking |
| `selfplay_config.py` | `SelfPlayConfig` + `SelfPlayResult` dataclasses |
| `selfplay_runner.py` | `SelfPlayRunner` — AI vs AI game loop with PPO sample generation (game→perspective→steps hierarchy) |

### 3.4 Controller Layer (`src/backend/controller/`)

Thin HTTP adapter — no business logic.

| File | Responsibility |
|------|---------------|
| `api.py` | Flask app factory + routes: `POST /api/games`, `GET /api/games/{id}`, `POST /api/games/{id}/play`, `DELETE /api/games/{id}`, `GET /api/ais` |
| `serializers.py` | CardType↔string conversion, RoundResult→dict formatting, error response helpers |

### 3.5 Other Backend Modules

| Module | Responsibility |
|--------|---------------|
| `src/backend/log/` | `StructuredLogger` — machine-parseable structured logging with traceId, threadId, processId, millisecond timestamps |
| `src/backend/config/backend.json` | Shared config: logging switch/destination, training hyperparameters |
| `src/backend/scripts/run.py` | Entry point — start Flask server |
| `src/backend/scripts/selfplay.py` | CLI — AI self-play with configurable temperature, perspective filtering, sample output |
| `src/backend/scripts/train.py` | CLI — PPO training with split policy/value networks |

---

## 4. Frontend Code Structure (`src/frontend/`)

React 19 SPA built with **Vite** (dev server + build), tested with **vitest** + React Testing Library.

| File | Responsibility |
|------|---------------|
| `src/App.jsx` | App shell — main menu (Hot-seat PVP / PVE mode selection), game lifecycle management |
| `src/GameBoard.jsx` | In-game UI — hand area, card selection, status display, opponent info, result/log display, deck viewer |
| `src/api.js` | REST API client — wraps fetch calls to all backend endpoints |
| `src/App.css` | All styles (single CSS file) |
| `tests/api.test.js` | API client unit tests |
| `tests/App.test.jsx` | UI component tests |

**Game Modes**:
- **Hot-seat PVP**: Two human players on the same device (face-down card selection protocol)
- **PVE (Human vs AI)**: Player selects cards; server auto-computes AI response

---

## 5. AI Training Pipeline

The project includes a full PPO reinforcement learning loop:

```
Self-play (AI vs AI)
    ↓  generate samples (game → perspective → steps)
Train policy network (PPO clip loss)
Train value network (MSE on returns)
    ↓  update weights in gen{N}/ directory
Register new AI version in registry.py
```

**Sample format** (design/7, 8, 10): Hierarchical JSON — each game contains two player perspectives, each perspective contains an array of (input, action, log_prob, reward, value) steps.

**Training** (design/9): Split policy/value networks trained separately with **AdamW** optimizer. Three phases per epoch: value training, policy training, combined fine-tuning with entropy bonus.

**Artifacts**:
- Weights: `gen{N}/policy.pth` + `gen{N}/value.pth`
- Charts: `loss/loss.png`, `loss/grad.png`
- Logs: `logs/{yyyy}-{mm}-{dd}.log` (structured, machine-parseable)

---

## 6. Testing

| Layer | Framework | Location | Count |
|-------|-----------|----------|-------|
| Unit tests (backend) | pytest | `tests/unit/` | ~500+ tests across game, converter, app_manager, card, resolver |
| Integration tests (backend) | pytest | `tests/integration/` | API routes + full game flows |
| Frontend tests | vitest | `src/frontend/tests/` | ~28 tests (API + UI components) |
| E2E tests | (future) | `tests/e2e/` | — |

**TDD workflow** is the project standard — tests are written before implementation.

---

## 7. Key Design Decisions

1. **Pure game logic** — `game/` has zero dependencies on AI, UI, or I/O.
2. **Pure AI models** — `ai/` has zero knowledge of game rules; all card↔tensor conversion lives in `app/converter.py`.
3. **Split policy/value networks** (design/9) — avoids interference between the two heads.
4. **Mask function as closure injection** — illegal action masking is defined in `AppManager` and passed into `model.forward()`.
5. **Disk-based persistence** — `SessionStore` uses JSON files, no database.
6. **Structured logging** — `StructuredLogger` emits machine-parseable log lines with traceId/threadId/processId for post-hoc analysis.
7. **Breaking changes are acceptable** — new game versions drop backward compatibility for old saves, weights, and samples.

---

## 8. Quick Start

```bash
# Backend
python src/backend/scripts/run.py --debug

# Frontend (separate terminal)
cd src/frontend ; npm run dev

# Open http://localhost:5173

# Tests
python -m pytest tests/
cd src/frontend ; npm test
```
