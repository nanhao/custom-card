# Architecture Design — Minor Optimizations (intention #10)

## 1. Motivation

Three targeted optimizations to improve code clarity and reduce friction:
- **D1**: Rename `state` → `input` in sample files to eliminate ambiguity with GRU hidden state
- **D2**: Allow self-play sample output to filter by perspective, reducing unnecessary data
- **D3**: Simplify train.py CLI by merging output path and epoch parameters

## 2. Module Overview

```
src/backend/
├── scripts/
│   ├── selfplay.py            # MODIFIED — add --perspective arg
│   └── train.py               # MODIFIED — unified --output-dir + --epoch
├── ai/
│   ├── training_runner.py     # MODIFIED — use output_dir, unified epoch
│   └── training_config.py     # MODIFIED — output_dir, remove value_epochs
└── app/
    ├── selfplay_config.py     # MODIFIED — add perspective field
    └── selfplay_runner.py     # MODIFIED — perspective filter, state→input rename
```

## 3. Key Design Decisions

**D1 — Sample key rename: `state` → `input`.** The 14-element vector in sample step dicts is the model input, not a "state" in the RL/MDP sense. Renaming to `input` avoids confusion with GRU hidden state (`h`). This is a breaking change to the sample JSON format — existing sample files must be regenerated.

**D2 — Self-play perspective filter.** New `--perspective` CLI parameter for `selfplay.py`, controlling which player's viewpoint is recorded in the sample file. Values: `both` (default, current behavior), `ai1` (player 1 only), `ai2` (player 2 only). Filtering applies at both step-recording time (memory efficiency) and sample-building time.

**D3 — Unified train.py parameters.** Replace two output path arguments with a single `--output-dir`. Policy weights always saved as `{output_dir}/policy.pth`, value weights as `{output_dir}/value.pth`. Merge `--value-epochs` and `--epoch` into a single `--epoch` controlling both training phases. This is a deliberate simplification: it loses the ability to tune value/policy epoch counts independently (previously `value_epochs` typically > `epoch`), but for rapid experimentation the simplicity outweighs the flexibility loss.

## 4. What Changes

| File | Change |
|------|--------|
| `app/selfplay_config.py` | ADD `perspective: str = "both"` field with validation |
| `app/selfplay_runner.py` | MODIFIED: step recording filters by perspective; `_build_game_sample` filters perspectives; rename `"state"` → `"input"` |
| `ai/training_config.py` | REPLACE `policy_output_path` + `value_output_path` with `output_dir`; REMOVE `value_epochs`, keep single `epoch` |
| `ai/training_runner.py` | MODIFIED: derive output paths from `output_dir`; `_load_samples` reads `step["input"]` instead of `step["state"]`; all internal step access uses `"input"`; Phase 1 uses `epoch` instead of `value_epochs` |
| `scripts/selfplay.py` | ADD `--perspective` argument (`choices=["ai1","ai2","both"]`, default `"both"`) |
| `scripts/train.py` | REPLACE `--policy-output-path` + `--value-output-path` with `--output-dir`; REPLACE `--value-epochs` + `--epoch` with single `--epoch` |
| `config/backend.json` | REMOVE `training.value_epochs` |
| `game/*` | Unchanged |
| `app/converter.py` | Unchanged |
| `ai/rnn_v1/model.py` | Unchanged |
| `ai/registry.py` | Unchanged |

## 5. Key Flows

- **Self-play with perspective filtering**: see [selfplay_perspective_flow.puml](diagrams/selfplay_perspective_flow.puml)
- **Train CLI unified parameters**: see [train_cli_flow.puml](diagrams/train_cli_flow.puml)
- **Class relationships**: see [class_diagram.puml](diagrams/class_diagram.puml)
