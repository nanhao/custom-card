# Architecture Design — 回合与抽牌参数调整 (intention #26)

## 1. Motivation

当前每轮（cycle）5 回合、每轮开始抽 7 张牌。本次将回合数缩减为 3、抽牌数缩减为 5，以加快游戏节奏。

## 2. Key Design Decisions

**D1 — ROUNDS_PER_CYCLE = 3。** 从 5 改为 3。每轮结束时（第 3 回合后）触发 bleed（双方各扣 1 HP）并开始新轮。

**D2 — CARDS_PER_CYCLE = 5。** 新增类常量，替代 `_start_cycle()` 中硬编码的 `7`。每轮开始时从抽牌堆抽 5 张到手牌。

**D3 — 初始牌堆保持 7 张不变。** 开局时 7 张牌入抽牌堆 → 抽 5 张到手牌，剩余 2 张留在抽牌堆。后续轮次中，随着出牌回收，抽牌堆总量会增加，每轮仍抽 5 张。

**D4 — 不涉及 AI 模型变更。** 输入向量维度（15-d）和输出维度（13-d）不变。`round_in_cycle` 的取值范围从 1-5 变为 1-3，但这仅影响输入值的分布，不影响模型结构。

**D5 — 破坏性变更。** 旧存档（saves/*.json）中 `round_in_cycle` 可能记录值为 4 或 5，与新版规则兼容性不保证。建议旧存档废弃。

## 3. Module Impact

```
src/backend/
├── game/game_core.py      # MODIFIED — ROUNDS_PER_CYCLE=3, CARDS_PER_CYCLE=5, _start_cycle() draw=5
├── game/game_state.py     # MODIFIED — round_in_cycle docstring 1-5 → 1-3
└── game/round_result.py   # MODIFIED — cycle_ended docstring 更新

src/frontend/
└── src/GameBoard.jsx      # MODIFIED — 显示文字 "/5" → "/3"
```

## 4. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **Cycle 流程**: [cycle_flow.puml](diagrams/cycle_flow.puml)
