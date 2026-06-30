# Architecture Design

## 1. Module Breakdown

```
custom-card/
├── src/
│   ├── backend/
│   │   ├── game/              # Pure game logic — NO UI, NO I/O, NO framework imports
│   │   │   ├── card.py        # CardType enum, get_initial_hand()
│   │   │   ├── player.py      # Player dataclass
│   │   │   ├── resolver.py    # CardResolver, ResolutionResult (stateless resolution)
│   │   │   ├── game_manager.py  # GameManager, RoundResult (game state machine)
│   │   │   └── session_store.py  # SessionStore (disk-based persistence)
│   │   ├── script/            # CLI entry points — NO business logic
│   │   │   └── cli.py         # Parse args, call game logic, display results
│   │   └── controller/        # REST API for web UI — same restrictions as script/
│   │       └── api.py         # HTTP routes, request/response handling
│   └── web/                   # Web UI (HTML/CSS/JS)
│       └── (frontend assets)
```

| Module | Responsibility | Constraints | Dependencies |
|--------|---------------|-------------|-------------|
| `src/backend/game/card.py` | CardType enum, `get_initial_hand()` factory | Pure logic; no UI/I/O | none |
| `src/backend/game/player.py` | Player state container (hp, hand, history) | Pure logic; no UI/I/O | `card.py` |
| `src/backend/game/resolver.py` | Stateless card-vs-card resolution → damage | Pure logic; no side effects | `card.py` |
| `src/backend/game/game_manager.py` | Game lifecycle, round orchestration, win check | Pure logic; no UI/I/O | all above |
| `src/backend/game/session_store.py` | Disk-based game state persistence (save/load/delete) | Pure I/O; no game rules | `game_manager.py` |
| `src/backend/script/cli.py` | CLI argument parsing, calling game logic, printing output | No business logic; thin adapter only | `game/` |
| `src/backend/controller/api.py` | REST endpoints, JSON serialization, HTTP handling | No business logic; thin adapter only | `game/` |
| `src/web/` | Browser UI | No game logic; consumes REST API | none |

## 2. Key Design Decisions

**D1 — Stateless resolution engine.** Card effects are resolved by a pure function that takes card types + histories as input and returns damage values. No side effects, no state mutation. This makes the core logic trivially testable.

**D2 — Player as a data container.** `Player` is a dataclass with hp, hand (list of CardType), and history (list of CardType). No behavior beyond state access. The manager mutates it.

**D3 — Simultaneous resolution.** All cards' effects are computed independently before any HP changes are applied, matching the game rule that effects resolve "at the same time."

**D4 — Round 1 history is empty list.** Both `played_defense_last_round` and `get_consecutive_attack` operate on an empty history for round 1, naturally producing the correct edge-case behavior.

**D5 — HP clamped to 0.** HP never goes negative; `max(0, hp - damage)` is applied after resolution.

**D6 — Cards discard after resolution.** Cards are removed from hands only after damage is applied (matching PRD §4: "结算完成后弃掉"), so resolution never depends on current hand state.

**D12 — Multi-card extensibility.** All interfaces accept `list[CardType]` for cards played per round rather than a single `CardType`. The current version enforces exactly 1 card per player per round at validation, but save files, API schemas, and function signatures all support multiple cards for future gameplay variants.

**D7 — Defense is pure-block.** The Defense card never deals damage; it only prevents incoming damage. When HS hits a non-consecutive DEF, the 1 damage comes from the HS card's effect, not from the DEF card. The resolver must not double-count this damage.

**D8 — Damage is set, not summed.** When both cards' effects imply damage to the same player (only possible in HS vs Counter where Counter says "both take 1" and HS says "opponent takes 1"), those are the same damage event — use `max()`, not addition. Damage from different card effects to the same player is always either 0-vs-1 or 1-vs-1, so max() is both correct and sufficient.

**D9 — All game logic isolated in `game/`.** The `src/backend/game/` module is a pure library: no HTTP, no CLI, no DOM, no framework imports. It exposes only data types and a `GameManager` class. This makes the entire game engine testable without any I/O setup.

**D10 — `script/` and `controller/` are thin adapters.** These modules contain no business logic. They only: (a) parse/handle input (CLI args or HTTP requests), (b) delegate to `game/` classes, (c) format and return/display output. Any conditional branching in these modules is limited to input validation and output formatting — never to game rules.

**D11 — Face-down protocol is client-side.** The face-down selection protocol (buffer Player 1 → pass screen → wait for Player 2 → submit both) is handled entirely by the web UI JavaScript. The API only receives the final pair via `POST /play`. This keeps the server stateless between rounds.

## 3. Dependency Graph

Refer to [class_diagram.puml](diagrams/class_diagram.puml) for the full class diagram.

```
                    ┌──────────────────────┐
                    │  src/backend/game/   │  ← Pure logic boundary
                    │  card → player       │
                    │  game_manager         │
                    └──────────┬───────────┘
                               │ import-only (no reverse deps)
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    src/backend/script/  src/backend/controller/  src/web/
       (CLI adapter)       (REST API adapter)    (Browser UI)
```

**Rule:** Arrows point from I/O layers into `game/`. `game/` never imports from `script/`, `controller/`, or `web/`.

## 4. Key Flows

- **Round flow**: see [round_flow.puml](diagrams/round_flow.puml)
- **Game lifecycle**: see [game_lifecycle.puml](diagrams/game_lifecycle.puml)
