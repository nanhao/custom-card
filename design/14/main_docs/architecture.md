# Architecture Design — Input/Output Redesign (intention #14)

## 1. Motivation

设计 #13 的 25-d input 使用 one-hot 编码上回合出牌（14 维），policy 输出分两段（7+7）。存在两个问题：

1. **维度浪费**：one-hot 用 14 维表达等价于 2 个整数的信息，且新增卡牌时 one-hot 段持续膨胀
2. **输出语义割裂**：14-d 分两段，"出攻击"和"出构建+构建攻击"在向量中相距 7 个位置，梯度信号不聚焦

本次重新设计：

- **Input 15-d**：上回合出牌改为 policy index（-1 表示无历史），新增双方能量信息
- **Policy 输出 13-d**：interleaved 结构，奇数位为纯出牌，偶数位为"出构建+构建 X"

## 2. Input Vector Specification

**INPUT_SIZE = 15**

| 索引 | 内容 | 范围 | 说明 |
|------|------|------|------|
| 0 | 我方血量 | ≥0 | 不归一化 |
| 1 | 对方血量 | ≥0 | 不归一化 |
| 2 | 我方能量 | 0-5 | |
| 3 | 对方能量 | 0-5 | |
| 4 | 当前回合数 | 1-5 | round_in_cycle |
| 5 | 当前轮数 | ≥1 | cycle |
| 6 | 手牌-攻击数 | ≥0 | |
| 7 | 手牌-防御数 | ≥0 | |
| 8 | 手牌-重击数 | ≥0 | |
| 9 | 手牌-反击数 | ≥0 | |
| 10 | 手牌-聚能数 | ≥0 | |
| 11 | 手牌-构建数 | ≥0 | |
| 12 | 手牌-攻守兼备数 | ≥0 | |
| 13 | 我方上回合 action | -1 或 0-12 | policy index，首回合=-1 |
| 14 | 对方上回合 action | -1 或 0-12 | policy index，首回合=-1 |

## 3. Policy Output Specification

**Output = 13-d**，interleaved 结构：

| 索引 | 含义 | 卡牌 | 构建目标 |
|------|------|------|----------|
| 0 | 出构建+构建构建 | build | build |
| 1 | 出攻击 | attack | — |
| 2 | 出构建+构建攻击 | build | attack |
| 3 | 出防御 | defense | — |
| 4 | 出构建+构建防御 | build | defense |
| 5 | 出重击 | heavy | — |
| 6 | 出构建+构建重击 | build | heavy |
| 7 | 出反击 | counter | — |
| 8 | 出构建+构建反击 | build | counter |
| 9 | 出聚能 | energy | — |
| 10 | 出构建+构建聚能 | build | energy |
| 11 | 出攻守兼备 | flex | — |
| 12 | 出构建+构建攻守兼备 | build | flex |

规律：偶数索引 = 出构建并构建某卡，奇数索引 = 纯出某卡。

## 4. Key Design Decisions

**D1 — 能量信息加入 input。** 索引 2-3 新增双方能量。能量影响"是否该聚能""能构建什么"等关键决策，仅靠 mask_fn 屏蔽不够。

**D2 — 上回合 action 用 policy index。** 从 14-d one-hot 压缩为 2-d 整数。新增动作时仅 policy 输出维度增长，input 尾部索引 13-14 不变。首回合无历史填 -1。

**D3 — Policy 输出 interleaved。** "出 X"和"出构建+构建 X"相邻排列，梯度信号更聚焦。

**D4 — 破坏性变更。** 旧权重（设计#13 gen1-gen4）和旧样本文件均不兼容。需重新生成权重 + 采集样本。

**D5 — Converter 集中编解码。** 新增 `action_to_policy_index` / `policy_index_to_action` 方法，统一管理 13-d 输出与游戏动作的互转。

**D6 — Sample action 改为整数。** 训练时直接使用 policy index，不再存 card 字符串。

**D7 — GRU hidden_size=32 不变。** 虽然 input 从 25 降到 15，但游戏复杂度未变。

**D8 — Game state 的 history 补全构建目标。** Player 新增 `build_history` 字段，与 `history` 并行记录每回合的构建目标（无构建时为 null）。新增 `PlayerHistoryItem` dataclass（`card: CardType, build_target: CardType | None`），`get_state()` 输出的 history 从 `list[str]` 改为 `list[PlayerHistoryItem]`。Converter 从 state 即可推导任意上回合 action 的 policy index，无需额外参数。

## 5. Module Impact

```
src/backend/
├── app/converter.py           # MODIFIED — 15-d input + action 编解码
├── ai/rnn_v1/model.py         # MODIFIED — INPUT_SIZE=15, policy output=13
├── app/app_manager.py         # MODIFIED — mask_fn 13-d interleaved, action 解释
├── ai/training_runner.py      # MODIFIED — ACTION_TO_INDEX 13-d, action 整数校验
├── app/selfplay_runner.py     # MODIFIED — sample action 整数格式
├── game/player.py             # MODIFIED — 新增 build_history 字段
├── game/game_core.py          # MODIFIED — play_round 记录 build_target，get_state 输出新格式
└── game/session_store.py      # MODIFIED — 序列化/反序列化 build_history
```

## 6. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **Input 构造流程**: [input_construction_flow.puml](diagrams/input_construction_flow.puml)
- **Policy 输出解码流程**: [policy_decode_flow.puml](diagrams/policy_decode_flow.puml)
