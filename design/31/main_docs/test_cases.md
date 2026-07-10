# Test Cases — rnn_v2 Skip-Connection 网络 (intention #31)

## 测试范围

- **单元测试**: `tests/unit/test_model_v2.py` — GRUPolicyModelV2 / GRUValueModelV2 模型测试
- **单元测试**: `tests/unit/test_registry.py` — rnn_v2 registry 条目测试

---

## 1. GRUPolicyModelV2 — 模型结构与前向传播

### TC-PV2-01 — 常量正确性

**Given**: 从 `src.backend.ai.rnn_v2.model` 导入常量

**Then**:
- `INPUT_SIZE == 42`
- `HIDDEN_SIZE == 32`
- `POLICY_OUTPUT_SIZE == 14`
- `NUM_CARD_TYPES == 7`

---

### TC-PV2-02 — forward 输出形状正确

**Given**: `model = GRUPolicyModelV2()`；x = torch.randn(1, 1, 42)；hidden = torch.randn(1, 1, 32)

**When**: `probs, log_probs, new_hidden = model.forward(x, hidden)`

**Then**:
- `probs.shape == (1, 14)`
- `log_probs.shape == (1, 14)`
- `new_hidden.shape == (1, 1, 32)`
- `probs.sum().item()` ≈ 1.0（全概率和质量）
- `torch.exp(log_probs).sum().item()` ≈ 1.0

---

### TC-PV2-03 — temperature=0 贪婪模式

**Given**: `model = GRUPolicyModelV2()`；x = torch.randn(1, 1, 42)；hidden = torch.zeros(1, 1, 32)

**When**: `probs, log_probs, _ = model.forward(x, hidden, temperature=0.0)`

**Then**:
- `probs` 中恰好有 1 个元素为 1.0，其余为 0.0
- `log_probs` 中对应元素为 0.0，其余为 -inf

---

### TC-PV2-04 — temperature > 0 正常采样

**Given**: `model = GRUPolicyModelV2()`；x = torch.randn(1, 1, 42)；hidden = torch.zeros(1, 1, 32)

**When**: `probs, log_probs, _ = model.forward(x, hidden, temperature=1.0)`

**Then**:
- `probs` 所有元素 > 0
- `probs.sum().item()` ≈ 1.0
- `log_probs` 不包含 NaN 或 inf

---

### TC-PV2-05 — mask_fn 正确应用

**Given**: `model = GRUPolicyModelV2()`；x = torch.randn(1, 1, 42)；hidden = torch.zeros(1, 1, 32)

**When**: 使用 mask_fn 将索引 2（ATTACK play）设为 -inf，然后 forward

```python
def mask_fn(logits):
    logits[:, 2] = float("-inf")
    return logits
probs, _, _ = model.forward(x, hidden, mask_fn=mask_fn, temperature=1.0)
```

**Then**:
- `probs[0, 2].item()` ≈ 0.0（被 mask 的索引概率为 0）

---

### TC-PV2-06 — INITIAL_HIDDEN_STATE 长度正确

**Given**: `GRUPolicyModelV2.INITIAL_HIDDEN_STATE`

**Then**:
- 长度为 32
- 全部为 0.0

---

### TC-PV2-07 — hidden state 在连续 forward 中正确传递

**Given**: `model = GRUPolicyModelV2()`；x1 = torch.randn(1, 1, 42)；x2 = torch.randn(1, 1, 42)；h0 = torch.zeros(1, 1, 32)

**When**: `_, _, h1 = model.forward(x1, h0)` 然后 `_, _, h2 = model.forward(x2, h1)`

**Then**:
- `h1` 不等于 `h0`
- `h2` 不等于 `h1`
- 无 NaN

---

## 2. GRUValueModelV2 — 模型结构与前向传播

### TC-VV2-01 — forward 输出形状正确

**Given**: `model = GRUValueModelV2()`；x = torch.randn(1, 1, 42)；hidden = torch.randn(1, 1, 32)

**When**: `value, new_hidden = model.forward(x, hidden)`

**Then**:
- `value.shape == (1, 1)`
- `new_hidden.shape == (1, 1, 32)`
- value 为有限值（非 NaN、非 inf）

---

### TC-VV2-02 — INITIAL_HIDDEN_STATE 长度正确

**Given**: `GRUValueModelV2.INITIAL_HIDDEN_STATE`

**Then**:
- 长度为 32
- 全部为 0.0

---

### TC-VV2-03 — value 无 tanh 约束

**Given**: `model = GRUValueModelV2()`；x = torch.ones(1, 1, 42) * 100；hidden = torch.zeros(1, 1, 32)

**When**: `value, _ = model.forward(x, hidden)`

**Then**:
- `value` 为有限值
- `value` 可能超出 [-1, 1]（无 tanh 限制）

---

### TC-VV2-04 — hidden state 在连续 forward 中正确传递

**Given**: `model = GRUValueModelV2()`；x1 = torch.randn(1, 1, 42)；x2 = torch.randn(1, 1, 42)；h0 = torch.zeros(1, 1, 32)

**When**: `_, h1 = model.forward(x1, h0)` 然后 `_, h2 = model.forward(x2, h1)`

**Then**:
- `h1` 不等于 `h0`
- `h2` 不等于 `h1`
- 无 NaN

---

## 3. Registry — rnn_v2 条目

### TC-RV2-01 — rnn_v2 条目存在

**Given**: 导入 `get_ai_info`

**When**: `info = get_ai_info("rnn_v2.1")`

**Then**:
- `info["name"] == "rnn_v2"`
- `info["policy"]["module"] == "src.backend.ai.rnn_v2.model"`
- `info["policy"]["class"] == "GRUPolicyModelV2"`
- `info["value"]["module"] == "src.backend.ai.rnn_v2.model"`
- `info["value"]["class"] == "GRUValueModelV2"`
- `info["policy"]["weight_path"]` 以 `rnn_v2/gen1/policy.pth` 结尾
- `info["value"]["weight_path"]` 以 `rnn_v2/gen1/value.pth` 结尾

---

### TC-RV2-02 — 动态导入 rnn_v2 模型类

**Given**: 无权重文件存在

**When**: 从 registry 动态导入 module 和 class 并实例化

```python
import importlib
info = get_ai_info("rnn_v2.1")
policy_module = importlib.import_module(info["policy"]["module"])
policy_class = getattr(policy_module, info["policy"]["class"])
model = policy_class()
```

**Then**:
- `model` 成功实例化
- `isinstance(model, nn.Module)` 为 True
- `model.forward` 可调用

---

### TC-RV2-03 — list_ais 包含 rnn_v2 条目

**Given**: `rnn_v2/gen1/` 目录存在（含 policy.pth 和 value.pth）

**When**: `result = list_ais()`

**Then**:
- 列表中存在 `name == "rnn_v2.1"` 的条目

---

## 4. Skip Connection — 架构专项验证

### TC-SC-01 — 不同 input 产生不同 head 输入

**Given**: `model = GRUPolicyModelV2()`；x_a = torch.zeros(1, 1, 42)；x_b = torch.ones(1, 1, 42)；h0 = torch.zeros(1, 1, 32)

**When**: `probs_a, _, _ = model.forward(x_a, h0)` 然后 `probs_b, _, _ = model.forward(x_b, h0)`

**Then**:
- `probs_a` 不等于 `probs_b`（skip connection 使 head 对不同 input 敏感）

---

### TC-SC-02 — 手牌段变化影响策略输出

**Given**: `model = GRUPolicyModelV2()`；x_normal 和 x_no_hand 仅在索引 6-12（我方手牌）有差异；h0 = torch.zeros(1, 1, 32)

**When**: 分别 forward，比较 probs

**Then**:
- 两个 probs 分布不同（证明 head 能感知手牌段差异）

---

## 5. 与 v1 的兼容性验证

### TC-CP-01 — v1 input shape 在 v2 上正常工作

**Given**: 使用与 v1 测试相同的 input tensor 形状

**When**: `GRUPolicyModelV2().forward(x, hidden)` 和 `GRUValueModelV2().forward(x, hidden)`

**Then**:
- 输出形状与 v1 完全一致
- 无维度错误

---

### TC-CP-02 — v2 模型可被 v1 调用方透明替换

**Given**: app_manager 使用 `get_ai_info("rnn_v2.1")` 获取模型类

**When**: 按 registry 信息动态加载并调用 forward

**Then**:
- 无需修改 Converter、mask_fn 逻辑
- 输出可正常用于 action selection
