# Architecture Design — PPO Training Script

## 1. Motivation

Train the rnn_v1 GRU model using PPO (Proximal Policy Optimization) on self-play sample data. The training script consumes the hierarchical sample format (game → perspective → steps) produced by the self-play script, optimizes the model, and outputs trained weights plus loss/gradient artifacts.

Design follows the existing script-business-logic separation pattern (design/5 D1).

## 2. Module Overview

```
src/backend/
├── scripts/
│   └── train.py                # NEW — thin CLI wrapper (argparse + config loading)
├── ai/
│   ├── training_runner.py      # NEW — PPO training loop, GAE, loss, optimizer, plotting
│   ├── training_config.py      # NEW — TrainingConfig dataclass
│   ├── registry.py             # UNCHANGED
│   └── rnn_v1/
│       └── model.py            # UNCHANGED
├── config/
│   └── backend.json            # MODIFIED — add "training" section with hyperparameters
└── loss/                       # NEW — output directory: loss.png, loss.csv, grad.png, grad.csv
```

## 3. Module Responsibilities

| Module | Responsibility | Change |
|--------|---------------|--------|
| `scripts/train.py` | CLI argument parsing, config loading from backend.json, delegation to TrainingRunner | NEW |
| `ai/training_runner.py` | Sample loading, PPO training loop (GAE advantage, clipped loss, optimizer step), weight saving, loss/grad plotting | NEW |
| `ai/training_config.py` | `TrainingConfig` frozen dataclass with validation | NEW |
| `ai/registry.py` | Unchanged — model class/weight resolution | — |
| `ai/rnn_v1/model.py` | Unchanged — `forward(x, h, mask_fn, temperature)` returns `(probs, log_probs, value, new_h)` | — |
| `config/backend.json` | Add `training` section: lr, gamma, gae_lambda, clip_eps, max_grad_norm, value_coef, entropy_coef, batch_size | MODIFIED |

## 4. Dependency Rule

```
scripts/train.py  ──imports──▶ ai/TrainingRunner
scripts/train.py  ──imports──▶ ai/TrainingConfig
ai/               ──imports──▶ ai/registry         (model resolution)
ai/               ──imports──▶ ai/rnn_v1/model     (GRUCardModel, constants)
ai/               ──NEVER──▶  game/                 (no game logic)
ai/               ──NEVER──▶  app/                  (no app layer)
```

Refer to [class_diagram.puml](diagrams/class_diagram.puml).

## 5. Key Design Decisions

**D1 — Training logic in `ai/` package.** Training is purely an AI concern: sample loading, forward passes, loss computation, optimization. No game logic involved. Place `TrainingRunner` in `ai/` alongside the model, not in `app/`.

**D2 — Hyperparameters in `config/backend.json`.** All PPO hyperparameters (lr, gamma, gae_lambda, clip_eps, max_grad_norm, value_coef, entropy_coef, batch_size) live in a `training` section of backend.json. The script reads them at startup. No hardcoded defaults in code.

**D3 — Script is a thin CLI wrapper.** `scripts/train.py` handles only: argument parsing, config file reading, `TrainingConfig` construction, and delegation to `TrainingRunner.run()`. Zero business logic.

**D4 — Single `.pth` output.** Trained weights saved as a single `.pth` file at `--output-path`. Compatible with `torch.load(path, weights_only=True)` as used by `AppManager._get_model()`.

**D5 — GRU hidden state resets at perspective boundaries.** Per design/7 D5: within a perspective, hidden state `h` is carried forward across steps. At each perspective boundary, `h` resets to `INITIAL_HIDDEN_STATE` (all zeros). This prevents information leakage between players.

**D6 — GPU training with CPU fallback.** Defaults to CUDA if available. Falls back to CPU with a warning. No `--device` flag needed.

**D7 — Loss/grad artifacts in fixed `loss/` directory.** Each training run produces: `loss/loss.png` (loss curve), `loss/loss.csv` (raw data), `loss/grad.png` (gradient norm curve), `loss/grad.csv` (raw data). The `loss/` directory is auto-created if it doesn't exist. Previous run's artifacts are overwritten.

**D8 — Gradient accumulation across perspectives.** Perspectives vary in length (3–7 rounds). Batching variable-length GRU sequences requires padding, which contaminates hidden state across padded timesteps. Instead, perspectives are processed sequentially: each perspective runs forward+backward independently, gradients are accumulated across `batch_size` perspectives, then a single optimizer step is taken. This avoids padding entirely and produces the same loss as true batching.

**D9 — Temperature during training.** `model.forward()` is called with `temperature=1.0` and `mask_fn=None` — standard softmax over all 4 actions, no sampling. Training uses the action already recorded in the sample; it computes `log_prob` of that action under the current policy. `mask_fn=None` is correct because the full softmax denominator (over all 4 actions) is needed for proper policy gradient computation. Sampling (temperature > 0) only occurs during self-play data collection.

**D10 — GAE for advantage estimation.** GAE uses `old_value` from the sample file (the value predicted by the behavior policy at collection time). For the terminal step in each perspective, `V(s_{t+1}) = 0` since the game has ended. Value targets: `V_target_t = A_t + old_V_t`.

**D11 — Gradient clipping.** Gradients are clipped via `torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)` before each optimizer step. Prevents training divergence from extreme importance sampling ratios. `max_grad_norm` defaults to 0.5.

**D12 — NaN detection.** After each loss computation and optimizer step, if loss or gradient norm is NaN, the update is skipped with a warning message. Training continues with the next group.

**D13 — Action-to-index mapping in `ai/` layer.** The training runner defines a hardcoded mapping `{"attack": 0, "defense": 1, "heavy": 2, "counter": 3}` matching `Converter.card_to_index()` in `app/`. This avoids importing from `app/`. The mapping is used to extract `log_probs[action_idx]` from the model output.

**D14 — Off-policy training.** `old_log_prob` and `old_value` in the sample reflect the model weights at collection time. Training uses current weights for `new_log_prob`. Users should ensure samples were generated by the same model version being trained, or understand the off-policy implications (PPO's clipping mechanism handles moderate distribution shift).

**D15 — Single-pass PPO.** Unlike standard PPO which performs multiple optimization epochs per data collection cycle, this implementation uses one gradient update per data pass (`--epoch` controls passes through the full dataset). Given the tiny model (~500 parameters) and small state space, multiple passes would quickly overfit.

## 6. PPO Algorithm Summary

```
For each epoch:
    1. Shuffle all perspectives from all sample files
    2. Process perspectives in groups of batch_size:
        a. zero_grad()
        b. For each perspective in the group (one at a time):
           - Load steps from sample JSON: state_t, action, old_log_prob, old_value, reward
           - Replace null old_log_prob with 0.0 (from greedy sampling during self-play)
           - Compute GAE using old_value from sample (D10):
             A_t = sum((gamma*lambda)^k * delta_{t+k})
             where delta_t = r_t + gamma * V_{t+1} - old_V_t
             V_{t+1} = 0 for terminal step (game ended)
             V_target_t = A_t + old_V_t
           - h = INITIAL_HIDDEN_STATE (converted to tensor on device)
           - Run GRU forward sequentially through all steps:
             * forward(state_t, h, mask_fn=None, temperature=1.0) → (probs, log_probs, value, new_h)
             * Extract new_log_prob_t via action-to-index mapping (D13)
             * h = new_h
           - Compute PPO clipped loss for this perspective:
             ratio = exp(new_log_prob - old_log_prob)
             L_clip = -sum(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
             L_value = sum((V_target - V_new)^2)
             L_entropy = -sum(p * log(p))
             L_total = (L_clip + cv * L_value - ce * L_entropy) / T
             where T = number of steps in this perspective
             (loss is averaged over steps; all T steps participate)
           - If L_total is NaN: skip, warn, continue to next perspective
           - (L_total / batch_size).backward()  ← accumulate gradients
        c. clip_grad_norm_(model.parameters(), max_grad_norm)  (D11)
        d. optimizer.step()  ← single update after accumulation
        e. Record: total loss, policy loss, value loss, entropy, gradient norm
           (averages of unscaled loss components across the group)
    3. Record epoch-level averages
4. Save weights to output_path
5. Generate loss/grad plots + CSV to loss/
   - loss/loss.csv columns: epoch,avg_loss,policy_loss,value_loss,entropy
   - loss/grad.csv columns: epoch,grad_norm
```

## 7. CLI Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--sample-path` | str | yes | File path or directory. If directory, all `.json` files inside are treated as samples. Non-JSON files are silently skipped |
| `--output-path` | str | yes | Output `.pth` file path for trained weights |
| `--model` | str | yes | AI registry name to train (e.g. `rnn_v1.1`) |
| `--epoch` | int | yes | Number of training epochs |

## 8. backend.json Extension

```json
{
  "logging": { ... },
  "training": {
    "lr": 0.001,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "max_grad_norm": 0.5,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "batch_size": 64
  }
}
```

TrainingConfig reads all these fields. Missing or invalid fields → raise ValueError.

## 9. Key Flows

- **Training flow**: see [training_flow.puml](diagrams/training_flow.puml)
- **Class relationships**: see [class_diagram.puml](diagrams/class_diagram.puml)

## 10. What Changes

| File | Change |
|------|--------|
| `scripts/train.py` | NEW — thin CLI: parse args, read config, create TrainingRunner, call run() |
| `ai/training_runner.py` | NEW — TrainingRunner: sample load, PPO loop, weight save, plotting |
| `ai/training_config.py` | NEW — TrainingConfig frozen dataclass |
| `config/backend.json` | ADD `training` section with 8 hyperparameters |
| `ai/rnn_v1/model.py` | Unchanged |
| `ai/registry.py` | Unchanged |
| `game/*` | Unchanged |
| `app/*` | Unchanged |
