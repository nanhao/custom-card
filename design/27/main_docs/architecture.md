# Architecture Design — 双方手牌可见 + 42维输入 (intention #27)

## 1. Motivation

当前设计 #14 的 15-d input 仅包含我方手牌信息，对方手牌对 AI 不可见。同时缺少状态持续回合、弃牌堆/抽牌堆分布等信息，限制了 AI 的策略深度。

本次变更：
- **游戏机制**：双方互相可见手牌（前端移除隐藏显示）
- **Input 42-d**：新增对方手牌计数、双方状态持续回合、我方弃牌堆/抽牌堆分布

## 2. Input Vector Specification

**INPUT_SIZE = 42**

| 索引 | 内容 | 范围 | 说明 |
|------|------|------|------|
| 0 | 我方血量 | ≥0 | |
| 1 | 对方血量 | ≥0 | |
| 2 | 我方能量 | 0-5 | |
| 3 | 对方能量 | 0-5 | |
| 4 | 当前回合数 | 1-3 | round_in_cycle，design/26 D1 |
| 5 | 当前轮数 | ≥1 | cycle |
| 6 | 我方手牌-攻击数 | ≥0 | |
| 7 | 我方手牌-防御数 | ≥0 | |
| 8 | 我方手牌-重击数 | ≥0 | |
| 9 | 我方手牌-防守反击数 | ≥0 | |
| 10 | 我方手牌-聚能数 | ≥0 | |
| 11 | 我方手牌-构建数 | ≥0 | |
| 12 | 我方手牌-攻守兼备数 | ≥0 | |
| 13 | 对方手牌-攻击数 | ≥0 | |
| 14 | 对方手牌-防御数 | ≥0 | |
| 15 | 对方手牌-重击数 | ≥0 | |
| 16 | 对方手牌-防守反击数 | ≥0 | |
| 17 | 对方手牌-聚能数 | ≥0 | |
| 18 | 对方手牌-构建数 | ≥0 | |
| 19 | 对方手牌-攻守兼备数 | ≥0 | |
| 20 | 我方防御持续回合 | ≥0 | 0=无该状态 |
| 21 | 我方联防持续回合 | ≥0 | 0=无该状态 |
| 22 | 我方脆弱持续回合 | ≥0 | 0=无该状态 |
| 23 | 对方防御持续回合 | ≥0 | 0=无该状态 |
| 24 | 对方联防持续回合 | ≥0 | 0=无该状态 |
| 25 | 对方脆弱持续回合 | ≥0 | 0=无该状态 |
| 26 | 我方弃牌堆-攻击数 | ≥0 | |
| 27 | 我方弃牌堆-防御数 | ≥0 | |
| 28 | 我方弃牌堆-重击数 | ≥0 | |
| 29 | 我方弃牌堆-防守反击数 | ≥0 | |
| 30 | 我方弃牌堆-聚能数 | ≥0 | |
| 31 | 我方弃牌堆-构建数 | ≥0 | |
| 32 | 我方弃牌堆-攻守兼备数 | ≥0 | |
| 33 | 我方抽牌堆-攻击数 | ≥0 | |
| 34 | 我方抽牌堆-防御数 | ≥0 | |
| 35 | 我方抽牌堆-重击数 | ≥0 | |
| 36 | 我方抽牌堆-防守反击数 | ≥0 | |
| 37 | 我方抽牌堆-聚能数 | ≥0 | |
| 38 | 我方抽牌堆-构建数 | ≥0 | |
| 39 | 我方抽牌堆-攻守兼备数 | ≥0 | |
| 40 | 我方上回合出牌 | -1 或 0-12 | policy index，首回合=-1 |
| 41 | 对方上回合出牌 | -1 或 0-12 | policy index，首回合=-1 |

## 3. Key Design Decisions

**D1 — INPUT_SIZE 从 15 改为 42。** 新增对方手牌（7 维）、双方状态持续回合（6 维）、我方弃牌堆（7 维）+ 抽牌堆（7 维）分布。破坏性变更：旧模型权重不兼容，需重新训练。

**D2 — 对方手牌直接取自 GameState。** `PlayerState.hand` 始终包含双方手牌的完整 `list[CardType]`，无需修改 game 层。

**D3 — 状态持续回合从 PlayerStatus.lifetime 提取。** 遍历 `PlayerState.statuses`，按 `StatusType` 查找 lifetime。未拥有的状态类型返回 0。三种状态类型：DEFENSE、GUARD、FRAGILE。

**D4 — 弃牌堆/抽牌堆计数需要完整卡牌列表。** 当前 `PlayerState.draw_pile` / `discard_pile` 仅在 `player_perspective` 匹配时返回完整列表（否则为 None）。Converter 调用方需确保传入 state 时 perspective 匹配（当前 SelfPlayRunner 和 AppManager 已按此方式使用）。

**D5 — 上回合出牌编码不变。** 沿用 policy index（0-12），首回合填 -1。与设计 #14 的 `action_to_policy_index` 完全兼容。

**D6 — GRU hidden_size=32 不变。** 输入从 15 扩到 42，新增的是"更多维度的完整信息"而非"更复杂的游戏规则"。游戏机制本身未变，hidden_size 保持不变。第一个 Linear 层的 input_size 随 INPUT_SIZE 变更。

**D7 — mask_fn 逻辑不变。** mask 依赖我方手牌和能量判断合法动作，与 input 向量结构无关。

## 4. Module Impact

```
src/backend/
├── ai/rnn_v1/model.py         # MODIFIED — INPUT_SIZE=42
├── app/converter.py           # MODIFIED — build_input_vector 42-d
├── game/game_state.py         # NO CHANGE — hand 已暴露双方
├── app/app_manager.py         # NO CHANGE — mask_fn 不变
├── app/selfplay_runner.py     # NO CHANGE — 自动适配新 input 维度
└── ai/training_runner.py      # NO CHANGE — 自动适配

src/frontend/
└── src/GameBoard.jsx           # MODIFIED — 移除对手手牌隐藏显示
```

## 5. Frontend Impact

- **PVP 模式**：对手手牌从牌背（🂠）改为显示真实卡牌类型和数量
- **PVE 模式**：AI 手牌从牌背改为显示真实卡牌
- `GameBoard.jsx` 中 `showOppHidden` 分支改为复用 `showP2Select` 的卡牌网格显示逻辑
- App.jsx 中 "Both selections are hidden" 提示文字需调整

## 6. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **Input 构造流程**: [input_construction_flow.puml](diagrams/input_construction_flow.puml)
