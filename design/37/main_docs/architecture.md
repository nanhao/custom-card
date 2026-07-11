# 设计 #37 — 架构概览

## 变更范围

本次变更涉及三个层面：

| 层面 | 模块 | 变更类型 |
|------|------|----------|
| 游戏层 | `CardType`, `CardDef`, `Player`, `ResolveContext`, `BuildResolver`, 新增 `DismantleResolver` `StrongDefenseResolver`, `GameCore` | 新增 + 修改 |
| AI/App 层 | `ai/constants.py`, `rnn_v1/model.py`, `Converter`, `AppManager`, `PlayerHistoryItem` | 修改 |
| 前端 | `GameBoard.jsx`, `App.jsx` | 修改 |

## 核心变更摘要

### 1. 新增卡牌：严防死守 (`IRON_DEFENSE`)
- 效果1：防御状态持续回合数 +2 (P11)
- 效果2：若上回合有防御/联防状态，联防状态 +1 (P12)
- 构建消耗：2 能量，标签："防"
- **不加入初始抽牌堆**，仅可通过构建获得

### 2. 构建卡改造：构建/拆解
- BUILD 卡变为二选一卡，保持 `CardType.BUILD` 不变
- **构建模式**：与原来相同但新卡放入**抽牌堆**（非弃牌堆）
- **拆解模式**：从弃牌堆选一张卡，**移出游戏**（不消耗能量）
- UI 弹出选择框让玩家二选一

### 3. 模型输入输出扩展
- 输入：56维 → 62维（8 卡类型 × 7 段统计）
- 输出：14维 → 24维（每卡增加拆解维度，1 + 2×BUILD + 21×其他7卡 = 24）

## 文件变更清单

```
修改:
  src/backend/game/card_type.py          — 新增 IRON_DEFENSE 枚举
  src/backend/game/card_def.py           — 更新 BUILD + 新增 IRON_DEFENSE
  src/backend/game/player.py             — PlayerHistoryItem 加 dismantle_target
  src/backend/game/resolver/base.py      — ResolveContext 加 dismantle_target
  src/backend/game/resolver/build_resolver.py — 构建卡放入抽牌堆
  src/backend/game/resolver/__init__.py  — 注册新 Resolver
  src/backend/game/game_core.py          — play_round 加 dismantle_target 参数
  src/backend/ai/constants.py            — INPUT_SIZE/POLICY_OUTPUT_SIZE/NUM_CARD_TYPES
  src/backend/ai/rnn_v1/model.py         — 更新注释、输入输出维度
  src/backend/app/converter.py           — _CARD_MAP 重构为三元组
  src/backend/app/app_manager.py         — mask_fn 支持 24-d 拆解维度
  src/backend/controller/serializers.py  — 如有硬编码索引需更新
  src/frontend/src/GameBoard.jsx         — 新卡显示 + BUILD 选择框
  src/frontend/src/App.jsx               — 构建/拆解交互

新增:
  src/backend/game/resolver/dismantle_resolver.py      — 拆解结算
  src/backend/game/resolver/strong_defense_resolver.py — +2 防御
```

## 交互图

- 构建/拆解结算流程：[build_dismantle_flow.puml](diagrams/build_dismantle_flow.puml)
- play_round 新流程：[play_round_flow.puml](diagrams/play_round_flow.puml)

## 类图与依赖图

- 类图：[class_diagram.puml](diagrams/class_diagram.puml)
- 依赖图：[dependency_diagram.puml](diagrams/dependency_diagram.puml)
