# Interface Design — rnn_v2 Skip-Connection 网络 (intention #31)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `rnn_v2/model.py` — New Module

### 1.1 Constants

```python
INPUT_SIZE = 42           # Same as v1
HIDDEN_SIZE = 32          # GRU hidden state, same as v1
NUM_CARD_TYPES = 7        # Same as v1
POLICY_OUTPUT_SIZE = 14   # Same as v1
```

### 1.2 `GRUPolicyModelV2(nn.Module)`

```python
class GRUPolicyModelV2(nn.Module):
    INITIAL_HIDDEN_STATE: list[float]  # [0.0] * 32

    def forward(
        self,
        x: torch.Tensor,            # (1, 1, 42)
        hidden: torch.Tensor,       # (1, 1, 32)
        mask_fn: Callable | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (probs, log_probs, new_hidden)
        # probs: (1, 14), log_probs: (1, 14), new_hidden: (1, 1, 32)
```

架构：`x → GRU(42→32) → gru_out`，然后 `concat(gru_out, x) → fc1(74→64)→ReLU→fc2(64→32)→ReLU→fc_policy(32→14)→softmax`。

签名与 v1 `GRUPolicyModel.forward` 完全一致。

### 1.3 `GRUValueModelV2(nn.Module)`

```python
class GRUValueModelV2(nn.Module):
    INITIAL_HIDDEN_STATE: list[float]  # [0.0] * 32

    def forward(
        self,
        x: torch.Tensor,        # (1, 1, 42)
        hidden: torch.Tensor,   # (1, 1, 32)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (value, new_hidden)
        # value: (1, 1), new_hidden: (1, 1, 32)
```

架构：`x → GRU(42→32) → gru_out`，然后 `concat(gru_out, x) → fc1(74→64)→ReLU→fc2(64→32)→ReLU→fc_value(32→1)`。

签名与 v1 `GRUValueModel.forward` 完全一致。

---

## 2. `registry.py` — 新增 rnn_v2 条目（v1 条目保留不动）

### 2.1 `_REGISTRY` 新增条目

```python
{
    "name": "rnn_v2",
    "description": "gru skip-connection 74→64→32 head, level=3",
    "policy": {
        "module": "src.backend.ai.rnn_v2.model",
        "class": "GRUPolicyModelV2",
    },
    "value": {
        "module": "src.backend.ai.rnn_v2.model",
        "class": "GRUValueModelV2",
    },
}
```

现有 rnn_v1 条目不动。

---

## 3. `training_runner.py` — 提取继承体系，子类归入各自模型包

### 3.1 `ai/training_runner.py` — `BaseTrainingRunner`（抽象父类）

承载所有版本无关的共享逻辑：
- `run(config)` — 三阶段训练编排
- `_build_models(model_name)` — 通过 registry 动态加载模型
- `_load_samples(sample_path)` — 从 json 文件加载训练样本
- `_compute_gae(values, rewards, gamma, gae_lambda)` — GAE 计算
- `_save_weights(model, output_path)` — 权重保存
- `_train_value_epoch()` — 批处理循环（调用抽象方法 `_process_value_perspective`）
- `_train_policy_epoch()` — 批处理循环（调用抽象方法 `_process_policy_perspective`）

抽象方法（子类必须实现）：
- `_recompute_advantages(perspectives, value_model, config, device)` — Phase 2
- `_process_value_perspective(perspective, value_model, config, device)` — Phase 1 单视角前向
- `_process_policy_perspective(perspective, policy_model, config, device)` — Phase 3 单视角前向

### 3.2 `ai/rnn_v1/v1_training_runner.py` — `V1TrainingRunner(BaseTrainingRunner)`

实现 v1 的前向策略：直接访问 `model.gru` + `model.fc_value` / `model.fc_policy` 做序列 bypass（当前代码不变，仅提取到子类）。

### 3.3 `ai/rnn_v2/v2_training_runner.py` — `V2TrainingRunner(BaseTrainingRunner)`

实现 v2 的前向策略：调用完整的 `model.forward()` 按 timestep 迭代，因为 v2 的 head 有 skip connection + 多层 MLP，不能 bypass GRU 输出。

### 3.4 `ai/training_factory.py` — `TrainingRunnerFactory`（工厂类）

根据 `model_name`（如 `"rnn_v1.3"` 或 `"rnn_v2.1"`）解析版本号，返回对应的 `V1TrainingRunner` 或 `V2TrainingRunner` 实例。

```python
class TrainingRunnerFactory:
    @staticmethod
    def create(model_name: str) -> BaseTrainingRunner:
        name, _gen = parse_ai_name(model_name)
        if name == "rnn_v1":
            from src.backend.ai.rnn_v1.v1_training_runner import V1TrainingRunner
            return V1TrainingRunner()
        elif name == "rnn_v2":
            from src.backend.ai.rnn_v2.v2_training_runner import V2TrainingRunner
            return V2TrainingRunner()
        raise ValueError(f"Unknown model: {name!r}")
```

---

## 4. `app_manager.py` — NO CHANGE

`_load_model()` 已通过 `importlib.import_module(module_name)` + `getattr(module, class_name)` 动态加载，对 v1/v2 完全透明。唯一硬编码的 `from src.backend.ai.rnn_v1.model import HIDDEN_SIZE, ...` 三个常量在 v1 和 v2 中值相同（32、42、14），无需修改。

---

## 5. Other Modules — Unchanged

| Module | Status |
|--------|--------|
| `converter.py` | Unchanged — INPUT_SIZE=42, POLICY_OUTPUT_SIZE=14 不变 |
| `selfplay_runner.py` | Unchanged — 通过 `app_manager.ai_sample_action` 调用，透明适配 |
| `selfplay_config.py` | Unchanged |
| `training_config.py` | Unchanged |
| `training_utils.py` | Unchanged |
| `scripts/train.py` | MODIFIED — `TrainingRunner()` 改为 `TrainingRunnerFactory.create(config.model)` |
| `scripts/selfplay.py` | Unchanged — `--ai1 rnn_v2.1` 即可 |
| `controller/api.py` | Unchanged |
| `game/*` | Unchanged |
