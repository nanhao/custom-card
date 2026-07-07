# Architecture Design — Input Reordering (intention #13)

## 1. Motivation

当前 input 向量（设计#11 D5）将卡牌计数放在索引 0-6，元数据（HP/回合/轮数）嵌在中间。未来新增卡牌类型时，元数据索引会整体后移，增加维护负担。

本次将元数据前置、卡牌类型特征后置，使得：
- **新增卡牌时**：元数据索引 0-3 固定不变，仅尾部维度增长
- **代码可读性**：索引语义分组清晰（元数据 → 卡牌计数 → one-hot 历史）

## 2. Input Vector Specification

维度总数保持不变：**INPUT_SIZE = 25**。

### 新索引映射（0-based）

| 索引 | 内容 | 说明 |
|------|------|------|
| 0 | 我方当前血量 | 不归一化（延续 D8） |
| 1 | 对方当前血量 | 不归一化 |
| 2 | 当前轮内回合数 | 1-5 |
| 3 | 当前轮数 | cycle number |
| 4 | 我方手牌攻击数 | |
| 5 | 我方手牌防御数 | |
| 6 | 我方手牌重击数 | |
| 7 | 我方手牌反击数 | |
| 8 | 我方手牌聚能数 | |
| 9 | 我方手牌构建数 | |
| 10 | 我方手牌攻守兼备数 | |
| 11-17 | 我方上回合出牌 | one-hot (7 类: attack→flex) |
| 18-24 | 对方上回合出牌 | one-hot (7 类: attack→flex) |

### 与旧索引对照

| 旧索引 (设计#11) | 新索引 (设计#13) | 内容 |
|---|---|---|
| 0-6 | 4-10 | 卡牌计数 |
| 7-8 | 0-1 | HP |
| 9-15 | 11-17 | one-hot 己方 |
| 16-22 | 18-24 | one-hot 对方 |
| 23 | 2 | round_in_cycle |
| 24 | 3 | cycle |

## 3. Module Impact

```
src/backend/
├── app/converter.py        # MODIFIED — build_input_vector 重排输出顺序
├── ai/rnn_v1/model.py      # MODIFIED — 注释更新（INPUT_SIZE/HIDDEN_SIZE/NUM_CARD_TYPES 不变）
├── ai/training_runner.py   # UNCHANGED — 仅校验 INPUT_SIZE，不依赖索引语义
├── app/app_manager.py      # UNCHANGED — 通过 Converter 获取 input，不直接依赖索引
├── app/selfplay_runner.py  # UNCHANGED — 同上
└── game/                   # UNCHANGED
```

## 4. Key Design Decisions

**D1 — 元数据前置。** 将不随卡牌类型数量变化的四个维度（HP×2, round_in_cycle, cycle）放在索引 0-3。无论未来新增多少卡牌类型，这四个索引永远不变。

**D2 — 破坏性变更。** 旧模型权重（设计#11 gen1-gen4）和旧样本文件（设计#11 格式）均与新 input 顺序不兼容。需重新生成随机权重 + 重新采集样本。

**D3 — GRU 架构不变。** INPUT_SIZE=25, HIDDEN_SIZE=32, NUM_CARD_TYPES=7 全部保持不变。`GRUPolicyModel` 和 `GRUValueModel` 内部代码无改动，仅注释反映新索引语义。

**D4 — Converter 是唯一实现变更点。** 所有 input 构造均通过 `Converter.build_input_vector()`，其他模块不直接拼装 input 向量。重排仅需修改 Converter 一处。

## 5. Diagrams

- **类图**: 见 [class_diagram.puml](diagrams/class_diagram.puml)
- **Input 构造流程**: 见 [input_construction_flow.puml](diagrams/input_construction_flow.puml)
