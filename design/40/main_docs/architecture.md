# 设计 #40 — 架构概览

## 变更范围

| 层面 | 模块 | 变更类型 |
|------|------|----------|
| AI 层 | `ai/atte_v1/model.py` | **新增** — AtteV1PolicyModel + AtteV1ValueModel |
| AI 层 | `ai/atte_v1/constants.py` | **新增** — 模型维度常量 |
| AI 层 | `ai/registry.py` | 修改 — 注册 atte_v1 条目 |
| AI Adapter 层 | `ai_adapter/atte_v1_converter.py` | **新增** — AtteV1Converter |
| AI Adapter 层 | `ai_adapter/atte_v1_ai.py` | **新增** — AtteV1Adapter |
| AI Adapter 层 | `ai_adapter/ai_registry.py` | 修改 — 工厂支持多态构造 |
| Training 层 | `training/atte_v1/atte_v1_training_runner.py` | **新增** |
| Training 层 | `training/atte_v1/sample_collector.py` | **新增** |
| Training 层 | `training/training_factory.py` | 修改 |
| App 层 | `app/selfplay_runner.py` | 修改 |

## 核心变更摘要

### 1. 新模型架构：Attention + GRU

- **输入**：卡牌 action ID 序列（双方手牌/弃牌堆/抽牌堆）+ 标量 + 状态序列。
- **Embedding**：e1 vocab=25（24 action + 1 PAD，dim=32）、e2 vocab=3（dim=8）、e3 vocab=20（dim=2）。
  - **设计决策**：意图定义 e1 词汇表为 24，设计扩展为 25。原因：首回合 last_action=-1 需要特殊 token，
    直接放入 e1 比额外维护零向量更简洁，且仅增加 32 个参数（<0.1% 总量）。
- **序数编码**：动作在「手牌+弃牌堆+抽牌堆」中的出现次序 → e3 嵌入，截断上限 19。
  - **序数上限分析**：初始牌组单卡最多 3 张。每 3 回合一轮，BUILD 每次消耗 1-2 能量（能量上限 5，每轮自然恢复有限），
    加上 DISMANTLE 可移除卡牌。单卡在全部牌堆中的最大副本数不会超过 ~15（3 初始 + 极限构建），20 留有充足余量。
  - **数据可见性前提**：序数计算依赖对方弃牌堆和抽牌堆的可见性。当前游戏规则（design/34）已确保双方弃牌堆/抽牌堆
    互相可见，GameStateSnapshot 提供完整数据。
- **注意力**：Q1/Q2/Q3 均为 34-d，对 emb2(34-d) 做缩放点积注意力。
- **空序列**：空序列→零向量。全空输入（无手牌/弃牌/抽牌/状态）时，仅标量特征驱动输出。
- **last_action=-1**：映射为 e1 PAD token（index 24）。
- **时序建模**：GRU(32-d)，输入 80-d。
- **输出特征**：拼接 169-d → FC+ReLU → out(64-d)。
- **Policy Head**：64→32→24，末层与 e1 共享权重。
- **Value Head**：64→32→1，无 tanh。

### 2. 权重共享梯度流

Policy Head 末层 FC(32→24) 与 e1.weight(25,32) 前 24 行转置共享：
- **Forward**：e1 用于 embedding lookup（card_ids → 32-d vectors）；Policy Head FC 用于 logits 计算。
- **Backward**：e1.weight 同时接收两路梯度——embedding lookup 的梯度（作用于全部 25 行）和 Policy Head
  FC 的梯度（仅作用于前 24 行，第 25 行 PAD 不受 Policy Head 影响）。PyTorch autograd 自动累加两路梯度，
  无需特殊处理，但需确保 optimizer 只注册一次参数。

### 3. 状态特征 Duration FC 处理

当玩家同时有多个状态（如 DEFENSE(2) + FRAGILE(1)）：
1. 每个 duration 独立输入 FC(1→32→16)。
2. 对每个 status embedding S 做逐元素变换：S' = S * w + b。
3. 所有 S' 求和池化得到 SS（8-d）。
多个状态的不同 duration 会通过 w/b 产生不同的调制效果，再通过 sum pool 聚合。

### 4. 训练三阶段流程

```
Phase 1: Value Training
  样本加载 → AtteV1Input 反序列化
  → 逐步 card 序列 → embedding → attention → GRU → value MSE loss
  → 仅更新 value_model

Phase 2: Advantage Re-computation
  用更新后的 value_model 重新前向（no_grad）
  → GAE(γ, λ) → perspective["new_advantages"] / perspective["new_value_targets"]

Phase 3: Policy Training
  逐步 card 序列 → embedding → attention → GRU → log_softmax
  → PPO clip loss + entropy bonus
  → 仅更新 policy_model
```

### 5. 性能评估

- 单次 forward ~25K FLOPs（rnn_v2 的 5×）
- 自对弈吞吐量预计从 ~200 samples/sec 降至 ~40-50 samples/sec
- `torch.compile` 预计恢复至 ~60-80；batch=4 批量推理可额外 2×
- 样本存储：~25-40 KB/局（rnn_v2 的 ~7.5 KB 的 4-5×）

### 6. 权重文件

- `policy_e1_e2.pth` + `policy.pth`
- `value_e1_e2.pth` + `value.pth`

## 文件变更清单

```
新增:
  src/backend/ai/atte_v1/__init__.py
  src/backend/ai/atte_v1/model.py
  src/backend/ai/atte_v1/constants.py
  src/backend/ai_adapter/atte_v1_converter.py
  src/backend/ai_adapter/atte_v1_ai.py
  src/backend/training/atte_v1/__init__.py
  src/backend/training/atte_v1/atte_v1_training_runner.py
  src/backend/training/atte_v1/sample_collector.py

修改:
  src/backend/ai/registry.py
  src/backend/ai_adapter/ai_registry.py
  src/backend/training/training_factory.py
  src/backend/app/selfplay_runner.py
```

## 交互图

- 前向传播流程：[atte_v1_forward_flow.puml](diagrams/atte_v1_forward_flow.puml)

## 类图与依赖图

- 模型架构图：[atte_v1_architecture.puml](diagrams/atte_v1_architecture.puml)
- 类图：[class_diagram.puml](diagrams/class_diagram.puml)
- 依赖图：[dependency_diagram.puml](diagrams/dependency_diagram.puml)
