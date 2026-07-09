# Architecture Design — Training Performance Optimization (intention #29)

## 1. Motivation

单 epoch 训练耗时过长，根因是 GRU 模型在每个 perspective 内逐时间步调用：
56,000+ 次 `torch.tensor()` 创建 + CPU→GPU 传输 + 单步 GRU forward，
GPU 并行能力完全空转。

本次优化将 GRU 调用从逐步改为全序列一次性输入，使 GRU 内部自动展开时间步，
同时为加载阶段和 Phase 2 增加进度提示。

## 2. Module Overview

```
src/backend/ai/
├── training_runner.py        # MODIFIED — 三个内部方法改为序列批量化 + 进度打印
└── rnn_v1/model.py           # Unchanged — 模型 forward 仍保留单步接口（推理侧使用）
```

## 3. Key Design Decisions

**D1 — Sequence batching: 单 perspective 全序列 GRU。**

PyTorch GRU（`batch_first=False`）原生支持 `(T, batch, input_size)` 的序列输入。
改造前每个 perspective 内 `for t in range(T): model(x_t, h)` → 改造后 `model.gru(x_seq, h_init)`，
`x_seq` 形状 `(T, 1, 42)`，GRU 一次调用返回全部时间步输出 `(T, 1, 32)`。

绕过 `model.forward()` 直接调用 `model.gru` + `model.fc_*` 的原因：
- `model.forward()` 接口设计为单步 `(1, 1, 42)`，内部有 `squeeze(0)` 假设；
- 推理侧（self-play）仍然需要单步接口；
- 直接调用底层组件避免修改模型公开 API，变更完全收敛在 `training_runner.py` 内。

**不跨 perspective 批量**：不同 perspective 序列长度不同，跨 perspective 批量需要
`pack_padded_sequence` + padding + mask，增加复杂度。单 perspective 全序列的收益已足够
（将 GRU 调用次数从 steps 总数降至 perspectives 总数，约 28x 减少）。

**D2 — 加载阶段进度提示**：`_load_samples` 原完全静默，大数据集加载时用户无法判断是否卡死。
增加：加载文件提示、解析后游戏数、每 10% 进度打印、完成时汇总 perspective/step 数。
Phase 2 同理增加开始提示。

## 4. What Changes

| File | Change |
|------|--------|
| `ai/training_runner.py` | MODIFIED: `_process_value_perspective` — 全序列 GRU 替代逐步循环；
| | `_recompute_advantages` — 全序列 GRU 替代逐步循环 + `tolist()` 替代 `item()` |
| | `_process_policy_perspective` — 全序列 GRU 替代逐步循环，熵计算改为张量操作 |
| | `_load_samples` — 增加加载进度打印 |
| | `_recompute_advantages` — 增加 Phase 2 开始提示 |
| `ai/rnn_v1/model.py` | Unchanged |
| `scripts/train.py` | Unchanged |
| `ai/training_config.py` | Unchanged |
| `config/backend.json` | Unchanged |

## 5. Key Flows

- **Training loop with sequence batching**: see [training_flow.puml](diagrams/training_flow.puml)
