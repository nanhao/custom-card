# 设计 #43 — 架构概览

## 变更范围

| 层面 | 模块 | 变更类型 |
|------|------|----------|
| AI 层 | `ai/atte_v1/model.py` | 修改 — `_AtteV1Body` 新增 `forward_pre_gru()` / `forward_post_gru()` |
| Training 层 | `training/atte_v1/atte_v1_training_runner.py` | 修改 — 新增 `_forward_value_sequence()` / `_forward_policy_sequence()`，重写 `_process_value_perspective()` / `_process_policy_perspective()` / `_recompute_advantages()` |

## 核心变更摘要

### 问题

当前 `AtteV1TrainingRunner` 在每个 step 单独调用 `value_model(**kwargs)`，每次完整执行 嵌入→注意力→GRU→拼接→FC。一条 48 step 的 perspective 调用 48 次 GRU forward（每次 seq_len=1），总计 ~1008ms。

对比 rnn_v2 的 `V2TrainingRunner`：将全部 step 的 input 拼为 `(T,1,62)` 张量，一次 GRU 调用处理整条序列。

### 方案

将 `_AtteV1Body.forward()` 拆为三个步骤，暴露给训练器：

```
forward_pre_gru()     → 嵌入 + 注意力 + 状态特征 → gru_input (80-d) + 中间结果
GRU(gru_inputs_stack) → (T,1,32)                    ← 训练器一次调用，展开全部 T 步
forward_post_gru()    → 手牌注意力 + 拼接 + FC → out (64-d)
```

训练器流程变更：
1. 循环计算每个 step 的 `forward_pre_gru()`，收集所有 `gru_input`
2. 拼成 `(T,1,80)` 张量，调用 `body.gru()` 一次
3. 循环每个 step，用对应 `h_out[t]` 调用 `forward_post_gru()`

预期：单条 perspective 从 ~1008ms 降至 ~200ms（省去 47/48 次 GRU 调用和注意力重复计算）。

### 推理不变

`AtteV1PolicyModel.forward()` 和 `AtteV1ValueModel.forward()` 保持不变，仅内部委托给 body 的新方法。推理路径（`ai_select_action` / `ai_sample_action`）不受影响。

## 文件变更清单

```
修改:
  src/backend/ai/atte_v1/model.py
  src/backend/training/atte_v1/atte_v1_training_runner.py
```

## 交互图

- 训练批量前向流程：[training_batch_flow.puml](diagrams/training_batch_flow.puml)

## 类图与依赖图

- 类图：[class_diagram.puml](diagrams/class_diagram.puml)
