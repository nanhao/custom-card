# Interface Design — 分离 Policy/Value 训练 Epoch (intention #32)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `train.py` — CLI 参数变更

### 1.1 参数定义

```python
parser.add_argument(
    "--policy-epoch", type=int, required=True,
    help="Number of training epochs for policy network (Phase 3)",
)
parser.add_argument(
    "--value-epoch", type=int, default=None,
    help="Number of training epochs for value network (Phase 1). "
         "Defaults to --policy-epoch if not specified.",
)
```

### 1.2 使用方式

```python
def main() -> None:
    args = parser.parse_args()
    value_epoch = args.value_epoch if args.value_epoch is not None else args.policy_epoch

    config = TrainingConfig(
        sample_path=args.sample_path,
        output_dir=args.output_dir,
        model=args.model,
        policy_epoch=args.policy_epoch,
        value_epoch=value_epoch,
        # ... 其余字段不变
    )
```

### 1.3 Usage 示例更新

```
Usage:
    python src/backend/scripts/train.py ^
        --sample-path samples/rnn_v2_self_play.json ^
        --output-dir src/backend/ai/rnn_v1/gen2 ^
        --model rnn_v2.1 ^
        --policy-epoch 10 ^
        --value-epoch 5
```

## 2. `TrainingConfig` — 字段变更

### 2.1 字段定义（仅列出变更部分）

```python
@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a three-phase split PPO training run."""

    sample_path: str
    output_dir: str
    model: str
    policy_epoch: int      # 原 epoch，>=1，控制 Phase 3 轮数
    value_epoch: int       # 新增，>=1，控制 Phase 1 轮数
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

### 2.2 `__post_init__` 校验变更

原 `if self.epoch < 1` 拆分为两条独立校验：

```python
if self.policy_epoch < 1:
    raise ValueError("policy_epoch must be >= 1")
if self.value_epoch < 1:
    raise ValueError("value_epoch must be >= 1")
```

## 3. `BaseTrainingRunner.run()` — 循环变更

Phase 1 循环：`config.epoch` → `config.value_epoch`

```python
# Phase 1: Value Training
for epoch in range(1, config.value_epoch + 1):
    ...
    print(
        f"[Value] Epoch {epoch}/{config.value_epoch} | "
        ...
    )
```

Phase 3 循环：`config.epoch` → `config.policy_epoch`

```python
# Phase 3: Policy Training
for epoch in range(1, config.policy_epoch + 1):
    ...
    print(
        f"[Policy] Epoch {epoch}/{config.policy_epoch} | "
        ...
    )
```

## 4. Other Modules — Unchanged

| Module | Status |
|--------|--------|
| `v1_training_runner.py` | Unchanged — 不引用 `config.epoch`，仅接收 `config` 传给 `_compute_gae` |
| `v2_training_runner.py` | Unchanged — 同上 |
| `training_utils.py` | Unchanged — `epoch` 取自循环局部变量，非 `config.epoch` |
| `training_factory.py` | Unchanged |
| `app_manager.py` | Unchanged |
| `converter.py` | Unchanged |
| `selfplay_runner.py` | Unchanged |
