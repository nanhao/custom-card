# Architecture Design — App Bridge Layer

## 1. Motivation

The `ai/` package currently imports `game/card.py` (`CardType`), coupling AI models to game domain types. The goal is to make `ai/` a pure package — zero dependencies on `game/` or any other project module except standard libraries and PyTorch.

A new `app/` bridge package sits between `controller/` and the `game/`+`ai/` packages, owning cross-cutting orchestration that neither layer should handle directly.

## 2. Module Overview

```
src/backend/
├── app/                       # NEW — bridge layer
│   ├── __init__.py
│   ├── app_manager.py         # AppManager: game creation, AI action selection
│   └── converter.py           # Converter: one-hot ↔ card, input vector construction
├── ai/                        # REFACTORED — pure AI, no game/ dependency
│   ├── registry.py            # list_ais() + get_ai_info()
│   └── rnn_v1/
│       ├── model.py           # GRUCardModel only (class definition)
│       └── gen1/
│           └── rnn_v1.pth     # (unchanged)
├── game/                      # UNCHANGED
│   ├── card.py
│   ├── player.py
│   ├── resolver.py
│   ├── game_manager.py
│   └── session_store.py
├── controller/                # MODIFIED — delegates to app/
│   ├── api.py
│   └── serializers.py
└── script/                    # UNCHANGED
```

## 3. Module Responsibilities

| Module | Responsibility | Constraints | Dependencies |
|--------|---------------|-------------|--------------|
| `app/app_manager.py` | Game creation, AI card selection orchestration | Holds model cache; delegates to game/ + ai/ | `game/`, `ai/registry` |
| `app/converter.py` | One-hot/card conversion, input vector construction | Pure transformation; no PyTorch | `game/card.py` (CardType) |
| `ai/registry.py` | List AIs; return AI metadata for loading | No model imports or instantiation | none |
| `ai/rnn_v1/model.py` | GRUCardModel class definition only | No game logic, no card knowledge | `torch` |
| `game/*` | (unchanged) Game rules, state, persistence | No AI knowledge | (unchanged) |
| `controller/api.py` | HTTP adapter | Thin; calls `app/` for AI ops | `app/`, `game/`, `ai/registry` |

## 4. Dependency Rule

```
controller/ ──imports──▶ app/           (create_game, ai_select_action)
controller/ ──imports──▶ game/          (str_to_card, SessionStore)
controller/ ──imports──▶ ai/registry    (list_ais only)
app/         ──imports──▶ game/         (GameManager, SessionStore, CardType)
app/         ──imports──▶ ai/registry   (get_ai_info)
app/         ──imports──▶ ai/rnn_v1/    (GRUCardModel instantiation)
ai/          ──NEVER──▶  game/          *** STRICT ***
ai/          ──NEVER──▶  app/           *** STRICT ***
game/        ──NEVER──▶  ai/            (unchanged)
game/        ──NEVER──▶  app/           (unchanged)
```

Refer to [class_diagram.puml](diagrams/class_diagram.puml).

## 5. Key Design Decisions

**D1 — `ai/` is pure.** `ai/rnn_v1/model.py` contains only `GRUCardModel`. It has zero imports from `game/`, `app/`, or `controller/`. All game knowledge (card types, hand state, input semantics) lives in `app/Converter`.

**D2 — `AppManager` owns the bridge logic.** Game creation (`create_game`) and AI action selection (`ai_select_action`) are orchestrated by `AppManager`. It loads models, constructs inputs via `Converter`, passes mask closures to `model.forward()`, and converts outputs back to domain types.

**D3 — Mask function is a `forward()` parameter.** Illegal-action masking is defined as a closure in `AppManager` and passed to `model.forward(x, hidden, mask_fn)`. The model applies it internally before softmax. This keeps masking logic close to the model (it's part of the policy head) while keeping domain knowledge (which cards are legal) in the bridge layer.

**D4 — `Converter` separates transformations from models.** One-hot ↔ card string mapping, input vector construction all live in `Converter`. Model code never sees card strings or game state dicts — it only operates on tensors.

**D5 — `registry.py` no longer loads modules.** `load_ai()` is removed. `list_ais()` remains for the controller's `GET /api/ais`. A new `get_ai_info()` returns metadata (module path, class name, weight path) for `AppManager` to use when instantiating models. Model validation moves to `AppManager`.

**D6 — `init_hidden_state()` becomes a class constant.** `GRUCardModel.INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE` replaces the standalone function. The function `init_hidden_state()` is removed.

**D7 — Model cache in `AppManager`.** Loaded model instances are cached in `AppManager` to avoid repeated loading. Same caching pattern as the old `registry.load_ai()`, but owned by the bridge layer.

**D8 — `AppManager` created once at startup.** A single `AppManager` instance (holding model cache) is created at server startup, same lifetime as `SessionStore`. Controller route handlers call its methods per-request.

## 6. Key Flows

- **Game creation (POST /api/games)**: see [diagrams/endpoints/create_game.puml](diagrams/endpoints/create_game.puml)
- **Play round — PVE (POST /api/games/{id}/play)**: see [diagrams/endpoints/play_round.puml](diagrams/endpoints/play_round.puml)
- **Get state (GET /api/games/{id})**: see [diagrams/endpoints/get_state.puml](diagrams/endpoints/get_state.puml)
- **Delete game (DELETE /api/games/{id})**: see [diagrams/endpoints/delete_game.puml](diagrams/endpoints/delete_game.puml)
- **List AIs (GET /api/ais)**: see [diagrams/endpoints/list_ais.puml](diagrams/endpoints/list_ais.puml)

## 7. What Changes

| File | Change |
|------|--------|
| `ai/rnn_v1/model.py` | Remove: `_load_model`, `select_action`, `_build_input_vector`, `_mask_illegal_actions`, `_index_to_card`, `_card_to_index`, `init_hidden_state`, `CardType` import. Add: `INITIAL_HIDDEN_STATE` constant, `mask_fn` parameter in `forward()`. |
| `ai/registry.py` | Remove: `load_ai()`, `_loaded_ais` cache. Add: `get_ai_info(name)`. `list_ais()` unchanged. |
| `controller/api.py` | Remove: `load_ai` import, direct AI call in `play_round`. Replace game creation with `app_manager.create_game()`. Replace AI selection with `app_manager.ai_select_action()`. |
| `app/app_manager.py` | NEW |
| `app/converter.py` | NEW |
| `game/*` | Unchanged |
