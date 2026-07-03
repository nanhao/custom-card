# Architecture Design — Sample File Restructuring

## 1. Motivation

The current PPO sample file is a flat JSONL stream: every step from all games is concatenated, with no game boundaries or perspective identifiers. During GRU training, the hidden state `h` carries context across steps. Feeding an interleaved P1→P2→P1→P2... sequence leaks information across players (who maintain separate `h1` and `h2` during self-play). Training must instead process one player's full trajectory as a sub-unit, then reset `h` to zeros before the next.

Restructure the sample output into **game → perspective → steps** hierarchy, where each perspective's step sequence is an independent sub-unit for training.

## 2. Module Overview

```
src/backend/
├── app/
│   └── selfplay_runner.py     # MODIFIED — _write_samples replaced; run() buffers games
└── scripts/
    └── selfplay.py             # UNCHANGED (thin CLI wrapper)
```

## 3. Module Responsibilities

| Module | Responsibility | Change |
|--------|---------------|--------|
| `app/selfplay_runner.py` | Restructure `_write_samples` into `_build_game_sample` (returns structured dict); `run()` collects all game samples in memory, writes single JSON array file | MODIFIED |
| `scripts/selfplay.py` | Unchanged | — |
| `app/app_manager.py` | Unchanged | — |
| `app/converter.py` | Unchanged | — |
| `game/*` | Unchanged | — |

## 4. Key Design Decisions

**D1 — Output format: single JSON array file.** Replaces JSONL with a single JSON file containing an array of game objects. Even at scale (1000 games × ~14 steps × 2 players), the file is ~2–3 MB — well within memory limits. A single structured JSON is easier to parse for training code and clearly expresses the hierarchy.

**D2 — Game → Perspective → Step hierarchy.** Each element in the top-level array is one game. Each game contains two perspectives (player "1" and player "2"). Each perspective contains an ordered list of steps. A training loop iterates: for each game, for each perspective, feed all steps sequentially while maintaining `h`, then reset `h` at the perspective boundary.

**D3 — Buffer all games in memory, write once at end.** Currently `_write_samples` writes per-game to an open file handle. The new format requires the full array structure, so `run()` collects all game dicts in a list and writes the complete JSON array after all games finish. File open is deferred to the end.

**D4 — Step-level fields unchanged.** Each step retains `{state, action, reward, value, log_prob}`. The `player` field (already present in the in-memory buffer) is preserved in the output structure via the perspective level rather than as a step-level attribute.

**D5 — Training consumption contract.** Training code reads the JSON array file. For each game, it processes two sub-units (one per perspective). Before each sub-unit, the GRU hidden state `h` is reset to `INITIAL_HIDDEN_STATE`. Steps within a sub-unit are fed sequentially, with `h` carried forward.

## 5. Data Schema

```
File (JSON array)
└── games[N]
    ├── game_id: int
    ├── winner: "1" | "2" | null
    ├── rounds: int
    └── perspectives[2]
        ├── player: "1" | "2"
        ├── ai: str
        └── steps[M]
            ├── state: [float × 14]
            ├── action: str
            ├── reward: float
            ├── value: float
            └── log_prob: float | null
```

## 6. Key Flows

- **Sample write flow**: see [sample_write_flow.puml](diagrams/sample_write_flow.puml)
- **Class relationships**: see [class_diagram.puml](diagrams/class_diagram.puml)

## 7. What Changes

| File | Change |
|------|--------|
| `app/selfplay_runner.py` | REPLACE `_write_samples()` with `_build_game_sample()`. MODIFY `run()` to collect game dicts in a list and write JSON array at end |
| All other files | Unchanged |
