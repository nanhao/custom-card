# Architecture Design — AI Self-Play Script

## 1. Motivation

Enable head-to-head AI self-play for data generation (PPO training samples) and strength evaluation. Two AIs from the registry play N games against each other. The script supports temperature-based action sampling for exploration, configurable per side and per step range.

## 2. Module Overview

```
src/backend/
├── scripts/
│   └── selfplay.py            # NEW — thin CLI wrapper (argparse only, no business logic)
├── app/
│   ├── app_manager.py         # MODIFIED — run_selfplay(), ai_sample_action() added
│   └── converter.py           # MODIFIED — build_input_vector(player_key)
├── ai/
│   └── rnn_v1/
│       └── model.py           # MODIFIED — forward() adds temperature, returns log_probs
└── game/                       # UNCHANGED
```

## 3. Module Responsibilities

| Module | Responsibility | Change |
|--------|---------------|--------|
| `scripts/selfplay.py` | CLI argument parsing; creates AppManager; calls `app.run_selfplay(config)` | NEW |
| `app/app_manager.py` | Self-play game loop, AI action selection, PPO sample I/O, progress logging | MODIFIED |
| `app/converter.py` | `build_input_vector()` supports player perspective | MODIFIED |
| `ai/rnn_v1/model.py` | `forward()` accepts temperature, returns `(probs, log_probs, value, new_hidden)` | MODIFIED |
| `game/*` | Unchanged | — |

## 4. Dependency Rule

```
scripts/selfplay.py ──imports──▶ app/AppManager     (run_selfplay)
scripts/selfplay.py ──imports──▶ game/SessionStore  (for AppManager constructor only)
app/                ──imports──▶ game/GameCore       (game state machine)
app/                ──imports──▶ ai/registry         (validate AI names)
app/                ──imports──▶ ai/rnn_v1/model     (unchanged)
ai/                 ──NEVER──▶  game/                (unchanged)
ai/                 ──NEVER──▶  app/                 (unchanged)
game/               ──NEVER──▶  ai/, app/            (unchanged)
```

Refer to [class_diagram.puml](diagrams/class_diagram.puml).

## 5. Key Design Decisions

**D1 — `AppManager` owns the self-play game loop.** `run_selfplay(config)` encapsulates the entire game loop: creates `GameCore` per game, iterates rounds, queries both AIs via `ai_sample_action()` / `ai_select_action()`, records PPO samples, prints progress, and returns aggregate results. The script is a thin CLI wrapper with zero business logic.

**D2 — Temperature sampling in `AppManager.ai_sample_action()`.** A new method that accepts `temperature` and `player_key`. When temperature > 0: samples from `softmax(logits/T)`. When temperature = 0: greedy argmax. Returns log_prob of the selected action (None for greedy).

**D3 — `Converter` supports player perspective.** `build_input_vector(state, player_key)` builds the 14-d input from the perspective of player `"1"` or `"2"`. Default `"2"` preserves backward compatibility.

**D4 — `forward()` returns log_probs.** `GRUCardModel.forward(x, hidden, mask_fn, temperature)` returns `(probs, log_probs, value, new_hidden)`. Temperature=0 → one-hot probs at argmax. Temperature>0 → `softmax(logits/T)`. Existing callers updated to unpack 4-tuple.

**D5 — PPO samples as JSONL.** One JSON object per line: `{state, action, reward, value, log_prob}`. Reward is the game outcome (+1 win / -1 lose / 0 draw) assigned to all steps retroactively. Samples written only when `--sample_output` is specified.

**D6 — Sampling controlled by temp_step and temp_side.** `temp_step=N` means only the first N rounds of each game use temperature sampling; round N+1 onward use greedy. `temp_side` restricts sampling to `player1`, `player2`, or `both`.

## 6. Key Flows

- **Self-play game loop**: see [selfplay_flow.puml](diagrams/selfplay_flow.puml)
- **Temperature sampling**: see [sampling_flow.puml](diagrams/sampling_flow.puml)

## 7. What Changes

| File | Change |
|------|--------|
| `scripts/selfplay.py` | NEW — thin CLI wrapper: parse args → `AppManager().run_selfplay(config)` |
| `app/app_manager.py` | ADD `run_selfplay()`, `ai_sample_action()`. ADD `SelfPlayConfig`, `SelfPlayResult`. UPDATE `ai_select_action()` for 4-tuple return |
| `app/converter.py` | ADD `player_key` parameter to `build_input_vector()` |
| `ai/rnn_v1/model.py` | ADD `temperature` parameter to `forward()`. ADD `log_probs` to return tuple |
| `game/*` | Unchanged |
