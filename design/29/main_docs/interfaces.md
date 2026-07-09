# Interface Design — Training Performance Optimization (intention #29)

Public interfaces only. Private methods omitted per design rule.

---

## 1. Public API — Unchanged

本次优化仅修改 `TrainingRunner` 的私有方法，**无公开接口变更**。

| Component | Status |
|-----------|--------|
| `TrainingRunner.run(config)` | Unchanged — 签名、行为、输出均不变 |
| `TrainingConfig` | Unchanged |
| `GRUPolicyModel.forward()` | Unchanged — 保留单步接口，推理侧继续使用 |
| `GRUValueModel.forward()` | Unchanged — 保留单步接口，推理侧继续使用 |
| `train.py` CLI | Unchanged |
| `backend.json` 配置 | Unchanged |

## 2. Affected Internal Methods

### 2.1 `_process_value_perspective`

**Before**: 逐步调用 `value_model(x_t, h)`，`x_t` 为 `(1, 1, 42)`，每步创建新 tensor。
```python
for t in range(T):
    x = torch.tensor(states[t], ...).view(1, 1, 42)
    val, h = value_model(x, h)
    new_vals.append(val.squeeze())
v_new = torch.stack(new_vals)  # (T,)
```

**After**: 一次调用 `value_model.gru(x_seq, h_init)`，`x_seq` 为 `(T, 1, 42)`。
```python
x = torch.tensor(states, ...).view(T, 1, 42)
gru_out, _ = value_model.gru(x, h_init)           # (T, 1, 32)
v_new = value_model.fc_value(gru_out.squeeze(1)).squeeze()  # (T,)
```

### 2.2 `_recompute_advantages`

**Before**: 逐步 `value_model(x_t, h)` + `float(val.item())`，每次 GPU→CPU 同步。
```python
for t in range(T):
    val, h = value_model(x_t, h)
    new_values.append(float(val.item()))
```

**After**: 全序列 GRU + `tolist()` 批量取回。
```python
gru_out, _ = value_model.gru(x_seq, h_init)  # (T, 1, 32)
new_values = value_model.fc_value(gru_out.squeeze(1)).squeeze().tolist()
```

### 2.3 `_process_policy_perspective`

**Before**: 逐步 `policy_model(x_t, h)` → 收集 `log_probs[0, action]` 和 `probs`。
```python
for t in range(T):
    probs, log_probs, h = policy_model(x_t, h)
    new_log_probs_tensors.append(log_probs[0, action])
    all_probs.append(probs)
new_log = torch.stack(new_log_probs_tensors)
```

**After**: 全序列 `policy_model.gru(x_seq, h_init)` → `fc_policy` → `log_softmax` → 索引提取。
```python
gru_out, _ = policy_model.gru(x_seq, h_init)         # (T, 1, 32)
policy_logits = policy_model.fc_policy(gru_out.squeeze(1))  # (T, 14)
log_probs = torch.log_softmax(policy_logits, dim=-1)  # (T, 14)
new_log = log_probs[torch.arange(T), action_indices]  # (T,)
```

### 2.4 `_load_samples`

增加 `print()` 进度输出：加载文件路径、解析游戏数、验证进度百分比、加载完成汇总。

### 2.5 `_recompute_advantages`

增加 Phase 2 开始时的 `print()` 提示。

## 3. Behavioral Equivalence

所有修改在数学上与原始步进计算等价：
- GRU 的递归定义确保 `gru(x[:t+1])` 第 t 步输出 = 逐步调用的第 t 步输出（相同初始 hidden state）；
- `log_softmax` + 索引提取 等价于 逐步 `log_softmax` + 逐步索引；
- 熵计算从多次 `sum()` 改为 `sum(dim=-1).mean()` 等价（线性求和可交换）。
