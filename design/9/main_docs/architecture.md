# Architecture Design — Split Policy/Value Networks

## 1. Motivation

The current `GRUCardModel` uses a shared GRU backbone for both policy and value heads. PPG (Phasic Policy Gradient, Cobbe et al. 2020) and empirical evidence show that shared backbone causes gradient interference between policy and value objectives, leading to value_loss monotonically increasing across epochs. This design splits the model into two independent networks with identical GRU architecture, trained sequentially: value first, then policy using the trained value network's advantage estimates.

## 2. Module Overview

```
src/backend/
├── scripts/
│   └── train.py                # MODIFIED — output two .pth files
├── ai/
│   ├── training_runner.py      # MODIFIED — three-phase training
│   ├── training_config.py      # MODIFIED — output_path split, add value_epochs
│   ├── training_utils.py       # MODIFIED — loss chart splits policy/value gradients
│   ├── registry.py             # MODIFIED — split format for all entries
│   └── rnn_v1/
│       └── model.py            # MODIFIED — replace GRUCardModel with GRUPolicyModel + GRUValueModel
├── app/
│   ├── app_manager.py          # MODIFIED — two model caches, dual hidden states
│   ├── selfplay_runner.py      # MODIFIED — maintain two hidden states per player
│   └── converter.py            # UNCHANGED
├── config/
│   └── backend.json            # MODIFIED — "training" section: value_epochs, separate LRs
└── game/                       # UNCHANGED
```

## 3. Model Architecture

**GRUPolicyModel**: `Input(14) → GRU(8, 1 layer) → Linear(8→4) → softmax`
- `forward(x, hidden, mask_fn, temperature)` → `(probs, log_probs, new_hidden)`

**GRUValueModel**: `Input(14) → GRU(8, 1 layer) → Linear(8→1)`
- `forward(x, hidden)` → `(value, new_hidden)`
- No tanh activation (per PPO lesson learned)

Both networks share the same GRU backbone structure.

Constants: `INPUT_SIZE = 14`, `HIDDEN_SIZE = 8`, `NUM_CARD_TYPES = 4`.

## 4. Registry

All entries use the split format:

```python
{
    "name": "rnn_v2.1",
    "description": "Split policy/value, random weights (gen1)",
    "policy": {
        "weight_path": "src/backend/ai/rnn_v1/gen1/policy.pth",
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUPolicyModel",
    },
    "value": {
        "weight_path": "src/backend/ai/rnn_v1/gen1/value.pth",
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUValueModel",
    },
}
```

## 5. Key Design Decisions

**D1 — Identical GRU architecture for both networks.** The 14-dim state includes last-move one-hots but not full action history — the GRU is necessary for both policy and value to capture game dynamics.

**D2 — Sequential phased training: value first, then policy.** Phase 1 trains the value network in isolation to convergence. Phase 2 re-computes GAE advantages using the trained value network. Phase 3 trains the policy using these improved advantages. This ensures the policy gradient is based on the best available value estimates.

**D3 — Dual hidden states during self-play.** SelfPlayRunner maintains `h_policy` and `h_value` per player. `ai_sample_action` returns `(card_str, new_h_policy, log_prob, value, new_h_value)`. `ai_select_action` (used in PVE) only uses policy network — single hidden state, no value hidden state needed.

**D4 — Sample format unchanged.** `value` field in samples comes from GRUValueModel during self-play.

**D5 — Separate optimizers (AdamW), separate phases.** Policy and value use independent AdamW optimizers with separate learning rates and weight decay. No interleaving — value and policy are trained in distinct phases.

**D6 — Two weight files per generation.** E.g., `gen1/policy.pth` + `gen1/value.pth`.

**D7 — Loss artifact changes.** `loss/loss.csv` columns: `phase,epoch,policy_loss,value_loss,entropy`. `loss/grad.csv` columns: `phase,epoch,policy_grad_norm,value_grad_norm`. Phase column distinguishes value-training epochs from policy-training epochs.

**D8 — Gradient accumulation unchanged.** Each perspective processed independently, gradients accumulated across `batch_size` perspectives.

**D9 — GAE re-computed with trained value network.** Phase 1 uses sample `old_values` for initial GAE. After value training, Phase 2 re-runs GAE with the trained value network's predictions, producing improved advantages for Phase 3 policy training.

**D10 — Separate epoch counts.** `value_epochs` controls Phase 1 iterations. `epoch` controls Phase 3 iterations. Phase 2 is a single pass (no gradient updates).

**D11 — AdamW default betas/eps.** AdamW uses PyTorch defaults: `betas=(0.9, 0.999)`, `eps=1e-8`. These are not exposed in config — the model is small (~500 params per network) and these defaults are well-tested for PPO. Only `weight_decay` is configurable.

## 6. Training Algorithm

```
Phase 1: Value Training (value_epochs)
─────────────────────────────────────
For each value_epoch:
    1. Shuffle all perspectives
    2. Process perspectives in groups of batch_size:
        a. value_optimizer.zero_grad()
        b. For each perspective in group:
           - Compute GAE using old_value from sample → V_targets
           - Value forward: h_val=INIT, step through GRUValueModel → new_values
           - value_loss = MSE(V_targets, new_values).mean()
           - (value_coef * value_loss / batch_size).backward()
        c. clip_grad_norm_(value_params, max_grad_norm)
        d. value_optimizer.step()
    3. Record: value_loss, value_grad_norm

Phase 2: Advantage Re-computation
─────────────────────────────────
For each perspective:
    - Value forward through trained GRUValueModel → new_values
    - Re-compute GAE:
      delta_t = r_t + gamma * V_new(s_{t+1}) - V_new(s_t)
      where V_new(s_T) = 0 for terminal step (game ended)
      A_t = delta_t + gamma * gae_lambda * A_{t+1}
    - Store new advantages (no gradient computation)

Phase 3: Policy Training (epoch)
─────────────────────────────────
For each policy_epoch:
    1. Shuffle all perspectives
    2. Process perspectives in groups of batch_size:
        a. policy_optimizer.zero_grad()
        b. For each perspective in group:
           - Load pre-computed advantages from Phase 2
           - Policy forward: h_pol=INIT, step through GRUPolicyModel → new_log_probs
           - Compute entropy from probs
           - PPO clipped loss: L = -min(ratio*A, clip(ratio)*A).mean()
           - total = (L - ce * entropy) / batch_size
           - total.backward()
        c. clip_grad_norm_(policy_params, max_grad_norm)
        d. policy_optimizer.step()
    3. Record: policy_loss, entropy, policy_grad_norm

Save: policy → policy_output_path, value → value_output_path
```

## 7. CLI Arguments (train.py)

| Argument | Description |
|----------|-------------|
| `--sample-path` | File path or directory |
| `--policy-output-path` | Output .pth for policy network |
| `--value-output-path` | Output .pth for value network |
| `--model` | Registry name |
| `--value-epochs` | Number of value training epochs (Phase 1) |
| `--epoch` | Number of policy training epochs (Phase 3) |

## 8. backend.json

```json
{
  "training": {
    "value_epochs": 20,
    "value_lr": 0.001,
    "policy_lr": 0.001,
    "weight_decay": 0.01,
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

Note: `value_epochs` typically > `epoch` since value training is pure regression and benefits from more passes. `value_coef` only applies in Phase 1 (Phase 3 has no value loss term).

## 9. Key Flows

- **Training flow**: see [training_flow.puml](diagrams/training_flow.puml)
- **Self-play flow**: see [selfplay_flow.puml](diagrams/selfplay_flow.puml)
- **Class relationships**: see [class_diagram.puml](diagrams/class_diagram.puml)

## 10. What Changes

| File | Change |
|------|--------|
| `ai/rnn_v1/model.py` | REPLACE `GRUCardModel` with `GRUPolicyModel` + `GRUValueModel` |
| `ai/registry.py` | REPLACE all entries with split format |
| `ai/training_runner.py` | REWRITE for three-phase training |
| `ai/training_config.py` | REPLACE: add `value_epochs`, separate `value_lr`/`policy_lr` |
| `ai/training_utils.py` | MODIFIED: CSV adds `phase` column |
| `app/app_manager.py` | REPLACE `_get_model` with `_get_policy_model` + `_get_value_model`; update `ai_sample_action` signature |
| `app/selfplay_runner.py` | MODIFIED: dual hidden states per player |
| `scripts/train.py` | MODIFIED: add `--value-epoch` |
| `config/backend.json` | REPLACE `training` section: add `value_epochs` |
| `game/*` | Unchanged |
| `app/converter.py` | Unchanged |
