# custom-card — Project Overview

A 2-player turn-based card game with simultaneous card resolution, AI opponents (GRU-based neural networks), and a PPO reinforcement learning training pipeline. Players can see each other's hands (design/27).

---

## 1. Game Mechanics (v2)

The current game (design/11+) introduces a deep strategic system on top of the original v1 rules:

- **Players**: 2, each starting with 30 HP (design/38).
- **Card Types**: 8 — Attack (连续击), Defense (格挡), Heavy Strike (破防重击), Counter (防守反击), Energy (聚能), Build/Dismantle (构建/拆解), Flex (攻守兼备), Iron Defense (严防死守).
- **Energy System**: Each player has an energy bar (0–5), starting at 2 (design/38). Cards have a `build_cost` (1 or 2). Playing cards costs _no_ energy; only _building_ new cards via the Build card consumes energy.
- **Deck System**: Each player has a draw pile and discard pile. No initial hand — cards are drawn at cycle start via `Player.draw_cards()`. Initial draw pile (design/38): 3× Attack + 2× Defense + 1× Energy + 1× Build (7 cards, no Heavy/Counter/Flex/Iron Defense).
- **Cycle (轮) System**: A cycle consists of 3 rounds (design/26 D1). At each cycle start, all hands are discarded to discard pile, then 5 cards are drawn from draw pile via smart draw (design/35): discard pile is only shuffled into draw pile when draw pile is empty during drawing. The initial draw pile contains 7 cards (design/26 D3), so 2 cards remain after the first deal. At cycle end, each player loses 1 HP (bleed) to prevent infinite loops.
- **Priority Resolution**: Each card effect has a priority (10, 11, or 12). Higher priority effects resolve first.
- **Status System**: Players can gain statuses with managed lifetimes:
  - **Defense (防御)** — blocks attack damage.
  - **Guard (联防)** — blocks heavy strike damage and its fragile effect.
  - **Fragile (脆弱)** — prevents gaining defense/guard (design/24).
  - Status remaining turns are displayed in the frontend UI (design/38).
- **Mutual Hand & Pile Visibility**: Both players can see each other's current hand (design/27), discard pile, and draw pile (design/34).
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
│ Converter       │          │ CardResolver (9)   │
│ SelfPlayRunner  │          │ Player / CardDef   │
│ SelfPlayConfig  │          │ GameState          │
│                 │          │ RoundResult        │
│                 │          │ PlayerStatus       │
│                 │          │ SessionStore       │
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
| `card_type.py` | `CardType` enum (7 values) |
| `card_def.py` | `CardDef` dataclass (build_cost, display_name, resolvers mapping per design/22) |
| `player.py` | `Player` + `PlayerHistoryItem`: HP, energy, statuses, piles, hand, history, `draw_cards()` (design/16, design/35) |
| `player_state.py` | `StatusType` enum + `PlayerStatus` dataclass with lifetime (design/15) |
| `game_state.py` | `GameState` + `PlayerState` typed dataclass snapshots (design/19) |
| `round_result.py` | `RoundResult` four-phase dataclass: `BeforeResolve`, `AfterResolve`, `EndActions`, `AfterEndActions` (design/17) |
| `resolver/` | `CardResolver` orchestrator + 9 Resolver subclasses (strategy pattern, design/21/22) |
| `game_core.py` | `GameCore` — game state machine (create, play_round, cycle management, end detection, design/35 draw) |
| `session_store.py` | `SessionStore` — pure disk-based JSON persistence (save/load/delete games) |

### 3.2 AI Layer (`src/backend/ai/`)

Pure ML — no knowledge of game rules.

| File | Responsibility |
|------|---------------|
| `registry.py` | Declarative AI registry — declares architectures (model classes); weight paths derived by `name.gen` convention (design/30) |
| `training_config.py` | `TrainingConfig` dataclass — PPO hyperparameters |
| `training_runner.py` | PPO training loop — GAE computation, policy/value loss, gradient accumulation (three-phase) |
| `training_utils.py` | Training artifact generation — CSV loss/grad logging + PNG charts |
| `rnn_v1/model.py` | Split GRU models: `GRUPolicyModel` (56→32→14) + `GRUValueModel` (56→32→1) |

- Model Architecture (design/37):
- Input: 62-d vector (8 card types × 7 segments + status × 3 per player + last action × 2 + metadata)
  - Metadata (6): HP×2, energy×2, round_in_cycle, cycle
  - Own hand counts (8): build, attack, defense, heavy, counter, energy, flex, iron_defense
  - Opponent hand counts (8): same order
  - Own status lifetimes (3): DEFENSE, GUARD, FRAGILE (0 if absent)
  - Opponent status lifetimes (3): same order
  - Own discard pile counts (8)
  - Opponent discard pile counts (8)
  - Own draw pile counts (8)
  - Opponent draw pile counts (8)
  - Last actions (2): own last policy index, opponent last policy index (-1 if first round)
- Policy: GRU(32) → Linear(32→24) → softmax (24-d: 0=pass, triplets of play/build/dismantle per card type)
- Value: GRU(32) → Linear(32→1) → scalar value estimate (no tanh)
- Mask function (injected via forward): blocks illegal actions (cards not in hand, insufficient energy for build); pass always legal
- rnn_v2 adds skip connection: Input(56)+GRU_out(32) → Concat(88) → MLP(88→64→32→14/1) (design/34)

### 3.3 App Layer (`src/backend/app/`)

Bridge between game logic and AI — the only layer that understands both.

| File | Responsibility |
|------|---------------|
| `app_manager.py` | `AppManager` — game creation, AI action selection (greedy + sampling), model lifecycle (loading/caching policy + value models), mask function construction |
| `converter.py` | `Converter` — card↔index bidirectional mapping, 42-d input vector construction (design/27), illegal action masking, accepts `GameState` dataclass (design/19) |
| `selfplay_config.py` | `SelfPlayConfig` + `SelfPlayResult` dataclasses |
| `selfplay_runner.py` | `SelfPlayRunner` — AI vs AI game loop with PPO sample generation (game→perspective→steps hierarchy, design/7) |

### 3.4 Controller Layer (`src/backend/controller/`)

Thin HTTP adapter — no business logic.

| File | Responsibility |
|------|---------------|
| `api.py` | Flask app factory + routes: `POST /api/games`, `GET /api/games/{id}`, `POST /api/games/{id}/play`, `DELETE /api/games/{id}`, `GET /api/ais` |
| `serializers.py` | CardType↔string, RoundResult→dict, GameState→dict (design/19), error response helpers |

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
New gen directories auto-discovered by filesystem scanning (design/30)
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
| Unit tests (backend) | pytest | `tests/unit/` | ~270 tests across game_core, resolver, converter, player, player_state, game_state, card, app_manager (8 test files) |
| Integration tests (backend) | pytest | `tests/integration/` | API routes + full game flows |
| Frontend tests | vitest | `src/frontend/tests/` | API client + UI component tests |
| E2E tests | (future) | `tests/e2e/` | — |

**TDD workflow** is the project standard — tests are written before implementation.

---

## 7. Key Design Decisions

1. **Pure game logic** — `game/` has zero dependencies on AI, UI, or I/O.
2. **Pure AI models** — `ai/` has zero knowledge of game rules; all card↔tensor conversion lives in `app/converter.py`.
3. **Split policy/value networks** (design/9) — avoids interference between the two heads.
4. **Mask function as closure injection** — illegal action masking is defined in `AppManager` and passed into `model.forward()`.
5. **Typed game state snapshots** (design/19) — `GameState` + `PlayerState` dataclasses replace ad-hoc dicts; all fields use typed objects (CardType, PlayerStatus).
6. **Four-phase round results** (design/17) — `RoundResult` captures `before`, `after_resolve`, `end_actions`, `after_end_actions` snapshots for unambiguous round lifecycle tracking.
7. **Disk-based persistence** — `SessionStore` uses JSON files, no database.
8. **Structured logging** — `StructuredLogger` emits machine-parseable log lines with traceId/threadId/processId for post-hoc analysis.
9. **Breaking changes are acceptable** — new game versions drop backward compatibility for old saves, weights, and samples.

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
