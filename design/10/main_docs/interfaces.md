# Interface Design — Minor Optimizations (intention #10)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `app/selfplay_config.py` — SelfPlayConfig

```python
@dataclass(frozen=True)
class SelfPlayConfig:
    """Immutable configuration for a self-play run.

    Attributes:
        ai1: AI name for player 1 (from registry).
        ai2: AI name for player 2 (from registry).
        game_count: Number of games to play (>=1).
        sample_output: Path for PPO sample JSON array file, or None.
        temperature: Sampling temperature (0.0 = greedy).
        temp_step: First N rounds use sampling, then greedy. None = no limit.
        temp_seed: Random seed for sampling reproducibility, or None.
        temp_side: Which side to sample — "player1", "player2", or "both".
        perspective: Which perspective(s) to include in sample output —
                     "ai1", "ai2", or "both" (default).
    """

    ai1: str
    ai2: str
    game_count: int = 100
    sample_output: str | None = None
    temperature: float = 1.0
    temp_step: int | None = None
    temp_seed: int | None = None
    temp_side: str = "both"
    perspective: str = "both"
```

Validation: `perspective` must be `"ai1"`, `"ai2"`, or `"both"`.

Note: `perspective` uses `"ai1"/"ai2"` prefix vs `temp_side` which uses `"player1"/"player2"`. This is intentional — `temp_side` controls which player's moves use temperature sampling during gameplay, while `perspective` controls which AI's viewpoint is recorded in the sample output.

---

## 2. `app/selfplay_runner.py` — SelfPlayRunner

```python
class SelfPlayRunner:
    """Execute self-play games with split policy/value networks."""

    def __init__(self, sample_action_fn):
        """sample_action_fn signature unchanged:
        (state, h_pol, h_val, ai_name, temp, player_key)
        → (card_str, new_h_pol, log_prob, value, new_h_val)
        """

    def run(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all games.

        Behavior changes:
        - Step dicts use key "input" instead of "state".
        - Only perspectives matching config.perspective are recorded
          in step buffers (in `_play_one_game`) and included in
          final game samples (in `_build_game_sample`):
          "ai1" → only player="1" perspective;
          "ai2" → only player="2" perspective;
          "both" → both perspectives (existing behavior).
        - run_start log event includes `perspective` field.

        Returns:
            SelfPlayResult with win/loss/draw counts and avg rounds.
        """
```

---

## 3. `ai/training_config.py` — TrainingConfig

```python
@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a three-phase split PPO training run.

    CLI-provided fields: sample_path, output_dir, model, epoch.
    Config-loaded fields (backend.json → training): policy_lr, value_lr,
    weight_decay, gamma, gae_lambda, clip_eps, max_grad_norm,
    value_coef, entropy_coef, batch_size.

    Attributes:
        sample_path: File path or directory for training samples.
        output_dir: Directory to save policy.pth and value.pth.
        model: AI registry name.
        epoch: Number of training epochs for both value (Phase 1)
               and policy (Phase 3) phases (>=1).
        policy_lr: Learning rate for policy optimizer.
        value_lr: Learning rate for value optimizer.
        weight_decay: Weight decay coefficient for AdamW (>=0).
        gamma: Discount factor (0 < gamma <= 1).
        gae_lambda: GAE lambda (0 <= lambda <= 1).
        clip_eps: PPO clipping epsilon (>0).
        max_grad_norm: Maximum gradient norm for clipping (>0).
        value_coef: Value loss coefficient (>=0) — Phase 1 only.
        entropy_coef: Entropy bonus coefficient (>=0) — Phase 3 only.
        batch_size: Perspectives per accumulation group (>=1).
    """

    sample_path: str
    output_dir: str
    model: str
    epoch: int
    policy_lr: float
    value_lr: float
    weight_decay: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    max_grad_norm: float
    value_coef: float
    entropy_coef: float
    batch_size: int
```

Validation: `output_dir` must be non-empty; `epoch>=1`, `batch_size>=1`,
`0<gamma<=1`, `0<=gae_lambda<=1`, `clip_eps>0`, `max_grad_norm>0`,
`policy_lr>0`, `value_lr>0`, `weight_decay>=0`, `value_coef>=0`,
`entropy_coef>=0`, `sample_path` exists.

Removed fields: `policy_output_path`, `value_output_path`, `value_epochs`.

### 3.1 Affected Internal Methods in TrainingRunner

All internal methods that access step data must use key `"input"` instead of
`"state"`: `_load_samples`, `_process_value_perspective`,
`_recompute_advantages`, `_process_policy_perspective`.

---

## 4. `ai/training_runner.py` — TrainingRunner

```python
class TrainingRunner:
    """Three-phase split PPO training orchestrator."""

    def run(self, config: TrainingConfig) -> None:
        """Execute three-phase split PPO training.

        Behavior changes:
        - Phase 1 (value training) now runs config.epoch iterations
          instead of config.value_epochs.
        - Policy weights saved to {config.output_dir}/policy.pth.
        - Value weights saved to {config.output_dir}/value.pth.
        - Sample loading reads step["input"] instead of step["state"].

        Raises:
            ValueError: If model not in registry.
            RuntimeError: If no valid training samples found.
        """
```

---

## 5. `scripts/selfplay.py` — CLI

```
Usage:
    python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1
    python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 \
        --game 200 --temperature 1 --temp_step 20 --temp_side both \
        --perspective ai1 --sample_output samples/v3_self_play.json
```

New argument:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--perspective` | `str` | `both` | Perspectives in sample file: `ai1`, `ai2`, or `both` |

When `--perspective` is set to a non-"both" value but `--sample_output` is not
specified, the script shall emit a stderr warning (perspective has no effect
without sample output).

---

## 6. `scripts/train.py` — CLI

```
Usage:
    python src/backend/scripts/train.py \
        --sample-path samples/v3_self_play.json \
        --output-dir src/backend/ai/rnn_v1/gen5 \
        --model rnn_v2.1 \
        --epoch 10
```

Changed arguments:

| Argument | Change |
|----------|--------|
| `--policy-output-path` | REMOVED — replaced by `--output-dir` |
| `--value-output-path` | REMOVED — replaced by `--output-dir` |
| `--output-dir` | NEW (required) — directory for policy.pth + value.pth |
| `--value-epochs` | REMOVED — merged into `--epoch` |
| `--epoch` | MODIFIED — now controls both Phase 1 (value) and Phase 3 (policy) |

Unchanged: `--sample-path` (required), `--model` (required).

---

## 7. Sample Format Change

Step dict key renamed:

```
# Before:
{"state": [1.0, 2.0, ...], "action": "attack", "reward": 0.0, "value": 0.35, "log_prob": -1.2}

# After:
{"input": [1.0, 2.0, ...], "action": "attack", "reward": 0.0, "value": 0.35, "log_prob": -1.2}
```

Game-level structure (`game_id`, `winner`, `rounds`, `perspectives`) unchanged.

---

## 8. `config/backend.json`

```json
{
  "training": {
    "policy_lr": 0.001,
    "value_lr": 0.001,
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

Removed: `value_epochs`. Note: `epoch` is now CLI-provided only, not in config.
