# Architecture Design — Human vs AI Mode

## 1. Module Overview

New module `src/backend/ai/` at the same layer as `controller/` and `script/`. It depends on `game/` for game state types, but `game/` never imports from `ai/`.

```
src/backend/
├── ai/                       # NEW — AI models and inference
│   ├── registry.py           # AI registry: list, load by name
│   └── rnn_v1/
│       ├── model.py          # GRU model definition (PyTorch)
│       └── gen1/
│           └── rnn_v1.pth    # First version weights (random init)
├── game/                     # UNCHANGED except: GameManager + SessionStore
│   ├── card.py
│   ├── player.py
│   ├── resolver.py
│   ├── game_manager.py       # + mode constructor parameter (immutable), get_state() includes round_log
│   └── session_store.py      # + mode/ai/ai_state serialization, load() returns tuple
├── controller/               # MODIFIED — new endpoints, AI integration
│   ├── api.py                # + GET /ais, modified POST /games, modified POST /play
│   └── serializers.py        # UNCHANGED
└── script/                   # UNCHANGED
```

---

## 2. Module Responsibilities

| Module | Responsibility | Constraints | Dependencies |
|--------|---------------|-------------|--------------|
| `ai/registry.py` | List registered AIs; load AI by name | Pure data; no PyTorch unless load called | `ai/rnn_v1/model.py` |
| `ai/rnn_v1/model.py` | GRU model forward pass, inference | PyTorch allowed; no HTTP/game logic | `torch` |
| `game/game_manager.py` | Game lifecycle + **mode** attribute | Pure logic + mode field | existing game/ modules |
| `game/session_store.py` | Serialize/deserialize mode + ai_state | Pure I/O; no game rules | `game_manager.py` |
| `controller/api.py` | Route AI game creation, AI card auto-selection | Thin adapter; delegates to game/ + ai/ | `game/`, `ai/` |

### Dependency Rule

```
controller/ ──imports──▶ game/     (existing)
controller/ ──imports──▶ ai/       (new)
ai/         ──imports──▶ game/     (reads CardType, game state dict)
game/       ──NEVER──▶  ai/        (strict boundary)
```

Refer to [class_diagram.puml](diagrams/class_diagram.puml).

---

## 3. Key Design Decisions

**D1 — AI is an adapter, not game logic.** `ai/` sits at the same layer as `controller/`. It reads `game/` types (CardType, state dict) but `game/` has zero knowledge of AI. The AI is invoked by the controller before calling `play_round()`.

**D2 — GameManager constructor gains `mode` parameter.** `__init__(mode="pvp")`. Set at construction, immutable thereafter. When `"pve"`, the controller auto-computes Player 2's card via AI. Default `"pvp"` preserves backward compatibility.

**D3 — AI hidden state saved alongside game state.** The GRU hidden state (list of 8 floats) persists across rounds in the save file's `ai_state` field. `SessionStore.save()` accepts an optional `ai_state` parameter; `load()` returns a `(GameManager, metadata_dict)` tuple.

**D4 — Common AI interface.** All AI implementations expose the same protocol: given game state dict + hidden state → return (CardType, new_hidden_state). See [ai_module.md](ai_module.md).

**D5 — Modified POST /play for PVE.** When `mode == "pve"`, the client sends only `cards_1`. The controller calls `ai.select_action()` to get Player 2's card, then calls `gm.play_round(cards_1, ai_card)`.

**D6 — AI registry is declarative.** A flat list of `(name, description, weight_path)`. Adding a new AI means adding one entry. No code changes to the controller.

**D7 — Backward-compatible save format.** Existing save files without `mode` default to `"pvp"`. Files without `ai_state` are valid for PVP games.

**D8 — AI state lifetime.** Hidden state is initialized (all zeros) at game creation and updated after each round. It is destroyed with the game when deleted.

---

## 4. Key Flows

- **AI game round flow**: see [ai_round_flow.puml](diagrams/ai_round_flow.puml)
- **AI inference flow**: see [ai_inference.puml](diagrams/ai_inference.puml)
