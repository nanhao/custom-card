# Interface Design — PPO Training Script

Public interfaces only. Private methods omitted per design rule.

---

## 1. `ai/training_config.py` — TrainingConfig

Immutable configuration value object.

```
@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a PPO training run.

    Attributes:
        sample_path: File path or directory. If directory, all .json files
                     inside are loaded as training samples.
        output_path: Output .pth file path for trained weights.
        model: AI registry name to train (e.g. "rnn_v1.1").
        epoch: Number of training epochs (>=1).
        lr: Learning rate for Adam optimizer.
        gamma: Discount factor for GAE (0 < gamma <= 1).
        gae_lambda: GAE lambda parameter (0 <= lambda <= 1).
        clip_eps: PPO clipping epsilon (>0).
        max_grad_norm: Maximum gradient norm for clipping (>0).
        value_coef: Value loss coefficient (>=0).
        entropy_coef: Entropy bonus coefficient (>=0).
        batch_size: Number of perspectives per training batch (>=1).
    """

    sample_path: str
    output_path: str
    model: str
    epoch: int
    lr: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    max_grad_norm: float
    value_coef: float
    entropy_coef: float
    batch_size: int
```

Validation in `__post_init__`:
- `epoch >= 1`
- `batch_size >= 1`
- `0 < gamma <= 1`
- `0 <= gae_lambda <= 1`
- `clip_eps > 0`
- `max_grad_norm > 0`
- `lr > 0`
- `value_coef >= 0`
- `entropy_coef >= 0`
- `sample_path` exists as file or directory

---

## 2. `ai/training_runner.py` — TrainingRunner

Encapsulates the entire PPO training pipeline.

```
class TrainingRunner:
    """PPO training orchestrator for GRU-based models.

    Handles sample loading, training loop, weight saving, and
    loss/gradient artifact generation.
    """

    def run(self, config: TrainingConfig) -> None:
        """Execute PPO training end-to-end.

        1. Resolve model from registry by config.model
        2. Instantiate GRUCardModel, load current weights
        3. Load all sample files from config.sample_path
        4. Detect device (cuda if available, else cpu with warning)
        5. Move model to device
        6. Create Adam optimizer with config.lr
        7. For each epoch (1..config.epoch):
           a. Shuffle all perspectives from all samples
           b. Process in groups of config.batch_size:
              - optimizer.zero_grad()
              - For each perspective in group (one at a time):
                * Reset h = INITIAL_HIDDEN_STATE
                * Forward pass through all steps (maintaining h)
                * Compute GAE advantages + value targets
                * Second forward pass to get new policy logits/values
                * Compute PPO clipped loss (policy + value + entropy)
                * (loss / batch_size).backward()  -- accumulate gradients
              - optimizer.step()  -- single update per group
              - Record: total_loss, policy_loss, value_loss, grad_norm
           c. Print epoch summary: epoch N/M | avg_loss | avg_grad_norm
        8. Save trained weights (state_dict) to config.output_path
        9. Generate loss/loss.png, loss/loss.csv, loss/grad.png, loss/grad.csv
           CSV format:
           - loss/loss.csv: epoch,avg_loss,policy_loss,value_loss,entropy
           - loss/grad.csv: epoch,grad_norm

        Perspectives vary in length (3-7 rounds). Gradient accumulation
        avoids GRU padding, which would contaminate hidden state.

        The GRU hidden state h is reset to INITIAL_HIDDEN_STATE at each
        perspective boundary (per design D5).

        model.forward() is called with temperature=1.0 (standard softmax,
        no sampling) and mask_fn=None (full distribution needed for
        log_prob computation).

        Raises:
            ValueError: If config.model not found in registry.
            FileNotFoundError: If sample_path does not exist.
            RuntimeError: If no valid training samples found.
        """
```

---

## 3. `scripts/train.py` — CLI Entry Point

Thin wrapper. No business logic — only argument parsing, config loading, and delegation.

```
"""PPO Training CLI.

Usage:
    python src/backend/scripts/train.py \
        --sample-path samples/rnn_v1_self_play.json \
        --output-path src/backend/ai/rnn_v1/gen2/rnn_v1.pth \
        --model rnn_v1.1 \
        --epoch 10

    python src/backend/scripts/train.py \
        --sample-path data/samples/ \
        --output-path output.pth \
        --model rnn_v1.1 \
        --epoch 50
"""

import argparse
from src.backend.ai.training_runner import TrainingRunner
from src.backend.ai.training_config import TrainingConfig

# 1. Parse CLI arguments (sample-path, output-path, model, epoch)
# 2. Load config/backend.json, extract "training" section
# 3. Build TrainingConfig(..., lr=..., gamma=..., ...)
# 4. runner = TrainingRunner()
# 5. runner.run(config)
```

### Progress Output Format

Per epoch (printed to stdout):
```
Epoch 3/10 | avg_loss: 0.4231 | policy_loss: 0.2104 | value_loss: 0.1872 | entropy: 0.0255 | grad_norm: 1.2341
```

---

## 4. `config/backend.json` — Training Section (Added)

```
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

All fields required. Script validates types and raises ValueError on missing or invalid fields.

---

## 5. `ai/rnn_v1/model.py` — GRUCardModel (Unchanged)

No changes. Existing interface:

```
class GRUCardModel(nn.Module):
    INITIAL_HIDDEN_STATE: list[float] = [0.0] * 8

    def forward(
        self,
        x: torch.Tensor,           # (1, 1, 14)
        hidden: torch.Tensor,      # (1, 1, 8)
        mask_fn: Callable | None = None,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Returns (probs, log_probs, value, new_hidden)
```

Training uses `forward()` with `mask_fn=None` and `temperature=1.0` to get full log_probs over all 4 actions. Temperature=1.0 is a no-op (standard softmax) — training does not sample actions; it computes log_prob of the action already recorded in the sample data.

---

## 6. `game/`, `app/` — Unchanged

No changes to game or app packages.
