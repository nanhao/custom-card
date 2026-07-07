# Interface Design — Split Policy/Value Networks

Public interfaces only. Private methods omitted per design rule.

---

## 1. `ai/rnn_v1/model.py` — GRUPolicyModel

```
class GRUPolicyModel(nn.Module):
    """GRU-based policy network for card selection.

    Architecture: Input(14) → GRU(8, 1 layer) → Linear(8→4) → softmax.
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * 8

    def forward(
        self,
        x: torch.Tensor,           # (1, 1, 14)
        hidden: torch.Tensor,      # (1, 1, 8)
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one GRU step.

        Returns:
            (probs, log_probs, new_hidden):
                probs: (1, 4) masked probability distribution.
                log_probs: (1, 4) log-probabilities.
                new_hidden: (1, 1, 8) updated GRU hidden state.
        """
```

## 2. `ai/rnn_v1/model.py` — GRUValueModel

```
class GRUValueModel(nn.Module):
    """GRU-based value network for state evaluation.

    Architecture: Input(14) → GRU(8, 1 layer) → Linear(8→1).
    No tanh — value can exceed [-1, 1].
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * 8

    def forward(
        self,
        x: torch.Tensor,           # (1, 1, 14)
        hidden: torch.Tensor,      # (1, 1, 8)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one GRU step, return scalar value estimate.

        Returns:
            (value, new_hidden):
                value: (1, 1) scalar.
                new_hidden: (1, 1, 8) updated GRU hidden state.
        """
```

## 3. `ai/training_config.py` — TrainingConfig

```
@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a three-phase split PPO training run.

    Attributes:
        sample_path: File path or directory for training samples.
        policy_output_path: Output .pth for policy network weights.
        value_output_path: Output .pth for value network weights.
        model: AI registry name.
        epoch: Number of policy training epochs — Phase 3 (>=1).
        value_epochs: Number of value training epochs — Phase 1 (>=1).
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
    policy_output_path: str
    value_output_path: str
    model: str
    epoch: int
    value_epochs: int
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

Validation: `epoch>=1`, `value_epochs>=1`, `batch_size>=1`, `0<gamma<=1`, `0<=gae_lambda<=1`, `clip_eps>0`, `max_grad_norm>0`, `policy_lr>0`, `value_lr>0`, `weight_decay>=0`, `value_coef>=0`, `entropy_coef>=0`, `sample_path` exists.

## 4. `ai/training_runner.py` — TrainingRunner

```
class TrainingRunner:
    """Three-phase split PPO training orchestrator.

    Phase 1: Train value network (value_epochs).
    Phase 2: Re-compute GAE advantages with trained value network.
    Phase 3: Train policy network (epoch).
    """

    def run(self, config: TrainingConfig) -> None:
        """Execute three-phase split PPO training.

        1. Resolve model from registry, instantiate both networks
        2. Detect device (cuda/cpu)
        3. Load samples, create separate Adam optimizers
        4. Phase 1 — Value Training (value_epochs):
           a. Each epoch: shuffle, group by batch_size
           b. Per perspective: GAE from sample old_values → V_targets
           c. Value forward → MSE(V_target, V_new) → backward
           d. optimizer.step(); record value_loss, value_grad_norm
        5. Phase 2 — Advantage Re-computation:
           a. For each perspective: value forward through trained model
           b. Re-compute GAE: delta_t = r_t + gamma*V_new(t+1) - V_new(t)
           c. Store new advantages (no_grad)
        6. Phase 3 — Policy Training (epoch):
           a. Each epoch: shuffle, group by batch_size
           b. Per perspective: policy forward → new_log_probs
           c. PPO clipped loss using Phase 2 advantages → backward
           d. optimizer.step(); record policy_loss, entropy, policy_grad_norm
        7. Save weights, generate artifacts

        Raises:
            ValueError: If model not in registry.
            RuntimeError: If no valid training samples found.
        """
```

## 5. `app/app_manager.py` — AppManager

```
class AppManager:
    """Bridge layer. Owns two model caches for split networks."""

    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store
        self._policy_models: dict[str, nn.Module] = {}
        self._value_models: dict[str, nn.Module] = {}

    def ai_select_action(
        self, state: dict, hidden_state: list[float], ai_name: str
    ) -> tuple[str, list[float]]:
        """Select AI card greedily. Only uses policy network.

        Returns (card_str, new_hidden_state).
        """

    def ai_sample_action(
        self,
        state: dict,
        hidden_state_policy: list[float],
        hidden_state_value: list[float],
        ai_name: str,
        temperature: float,
        player_key: str = "2",
    ) -> tuple[str, list[float], float | None, float, list[float]]:
        """Select AI card with temperature sampling.

        Uses both policy and value networks independently.

        Returns (card_str, new_h_policy, log_prob, value, new_h_value).
        """

    def create_game(
        self, mode: str, ai_name: str | None = None
    ) -> tuple[str, dict]:
        """Create game. PVE AI state stores policy_hidden + value_hidden."""

    def run_selfplay(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute self-play games. Delegates to SelfPlayRunner."""
```

## 6. `app/selfplay_runner.py` — SelfPlayRunner

```
class SelfPlayRunner:
    """Execute self-play games with split policy/value networks."""

    def __init__(self, sample_action_fn):
        """sample_action_fn signature:
        (state, h_pol, h_val, ai_name, temp, player_key)
        → (card_str, new_h_pol, log_prob, value, new_h_val)
        """

    def run(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all games. Maintains h_policy + h_value per player."""
```

## 7. `scripts/train.py` — CLI

```
Usage:
    python src/backend/scripts/train.py \
        --sample-path samples/v2_self_play.json \
        --policy-output-path src/backend/ai/rnn_v1/gen1/policy.pth \
        --value-output-path src/backend/ai/rnn_v1/gen1/value.pth \
        --model rnn_v2.1 \
        --value-epochs 20 \
        --epoch 10
```

## 8. `ai/registry.py` — Registry

All entries are split format:

```
_REGISTRY = [
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
    },
]

def list_ais() -> list[dict]:
    # Returns name + description only

def get_ai_info(name: str) -> dict:
    # Returns raw dict with "policy" + "value" keys
```

## 9. `ai/training_utils.py` — Artifacts

CSV columns:
- `loss/loss.csv`: `phase,epoch,policy_loss,value_loss,entropy`
- `loss/grad.csv`: `phase,epoch,policy_grad_norm,value_grad_norm`

Phase values: `"value"` for Phase 1 epochs, `"policy"` for Phase 3 epochs.

## 10. Sample Format (Unchanged)

```
[{
    "game_id": 1, "winner": "1", "rounds": 5,
    "perspectives": [{
        "player": "1", "ai": "rnn_v2.1",
        "steps": [
            {"state": [...], "action": "attack", "reward": 0.0, "value": 0.35, "log_prob": -1.2},
            ...
        ]
    }]
}]
```

`value` field comes from `GRUValueModel.forward()` during self-play.
