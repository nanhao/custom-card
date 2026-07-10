# Interface Design — Value Network Validation Set

仅列出变更的公有接口。未变更接口不重复列举。

---

## 1. `ai/training_config.py` — TrainingConfig

新增字段：

```
@dataclass(frozen=True)
class TrainingConfig:
    # ... 已有字段不变 ...

    validation_split: float = 0.1
    """验证集占比 (0 <= validation_split < 1)。默认 0.1。
        0 时验证集为空，跳过验证。"""
```

`__post_init__` 新增校验：`0 <= validation_split < 1`。

---

## 2. `ai/training_runner.py` — BaseTrainingRunner

### `run()`

流程变更：
- Phase 1 之前新增：调用 `_split_train_val()` 划分 train/val
- `_train_value_epoch()` 参数增加 `val_perspectives`
- Phase 1 记录新增 `val_value_loss` 字段

### `_split_train_val(perspectives, validation_split) -> tuple[list[dict], list[dict]]`

```
@staticmethod
def _split_train_val(
    perspectives: list[dict],
    validation_split: float,
) -> tuple[list[dict], list[dict]]:
    """按 game_id 分组后随机划分 train/val 集合。

    validation_split=0 时，验证集为空列表。
    至少保留 1 个 game 在训练集中。
    返回 (train_perspectives, val_perspectives)。
    """
```

### `_train_value_epoch()` — 签名变更

```
def _train_value_epoch(
    self,
    perspectives: list[dict],          # 训练集（仅 train 部分）
    val_perspectives: list[dict],      # 新增 — 验证集
    value_model: nn.Module,
    policy_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict:
    """返回值新增 val_value_loss 键。
    训练完成后用 val_perspectives 计算验证 loss。
    """
```

返回值新增键：`"val_value_loss"`（float，验证集为空时为 `nan`）。

### `_compute_val_loss()` — 新增方法

```
def _compute_val_loss(
    self,
    val_perspectives: list[dict],
    value_model: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    """在验证集上计算 value loss（无梯度）。

    model.eval() 下执行，复用 _process_value_perspective() 计算 MSE，
    完成后恢复 model.train()。
    返回平均 val_value_loss，验证集为空时返回 nan。
    """
```

---

## 3. `ai/sample_loader.py` — SampleLoader

### `_extract_perspectives()` — 变更

每个 perspective 字典增加 `"game_id"` 键：

```
# 变更前: {"steps": steps}
# 变更后: {"steps": steps, "game_id": game_id}
```

`game_id` 的值从 `game["game_id"]` 获取，类型为原始 JSON 中的类型（通常为 `int` 或 `str`）。

---

## 4. `ai/training_utils.py` — save_split_training_artifacts()

### 函数签名变更

```
def save_split_training_artifacts(
    phase_records: list[dict],
    output_dir: str | None = None,
) -> None:
```

### CSV 变更

`loss.csv` 表头变更为：`phase,epoch,policy_loss,value_loss,val_value_loss,entropy`

- `val_value_loss` 仅在 `phase="value"` 时有值，`phase="policy"` 时为 `nan`

### 图表变更

`loss.png` 新增验证 loss 曲线：
- 新增一条蓝色虚线（`"b--"`），label 为 `"Val Value Loss (Phase 1)"`
- 仅在 `val_value_loss` 有有效值时绘制

---

## 5. `scripts/train.py` — CLI

### 新增参数

| 参数 | 说明 |
|------|------|
| `--validation-split` | `float`，可选。验证集占比。默认使用 `backend.json` 中的 `training.validation_split` |

### `_load_training_config()` — 变更

新增 required key 校验：`"validation_split": float`。

---

## 6. `config/backend.json` — training 段

```json
{
  "training": {
    "value_lr": 0.001,
    "policy_lr": 0.001,
    "weight_decay": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "max_grad_norm": 0.5,
    "value_coef": 0.5,
    "entropy_coef": 0.1,
    "batch_size": 64,
    "validation_split": 0.1
  }
}
```
