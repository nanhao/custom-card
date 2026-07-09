# Design #28: Policy 网络 14 维输出 + 不出牌

## 概述

将 policy 网络输出从 13 维扩展为 14 维，新增 0 号位"不出牌"（pass），并在游戏逻辑、前后端完整支持玩家选择不出牌。

## 背景

当前设计（design/14）中，policy 输出 13 维，覆盖了直接出牌和"构建+目标"的组合，但未提供"不出牌"选项。玩家每回合必须出牌，缺乏策略灵活性。

本设计在 13 维基础上新增"不出牌"选项，同时保持已有维度的语义不变。

## Policy 输出向量定义（14 维）

| 维度 | 含义 |
|------|------|
| 0 | 不出牌 |
| 1 | 出构建，构建构建 |
| 2 | 出攻击 |
| 3 | 出构建，构建攻击 |
| 4 | 出防御 |
| 5 | 出构建，构建防御 |
| 6 | 出重击 |
| 7 | 出构建，构建重击 |
| 8 | 出防守反击 |
| 9 | 出构建，构建防守反击 |
| 10 | 出聚能 |
| 11 | 出构建，构建聚能 |
| 12 | 出攻守平衡 |
| 13 | 出构建，构建攻守平衡 |

偶数位（2,4,6,8,10,12）= 纯出牌，奇数位（1,3,5,7,9,11,13）= 构建+目标。与原来 13 维相比，所有非 pass 索引统一 +1（原 0→1, 原 1→2, …, 原 12→13）。

## 不出牌（Pass）的游戏规则

- **不消耗手牌**：不出牌不从手牌移除任何卡牌
- **不进入弃牌堆**：不出牌不产生弃牌
- **对手照常结算**：对手的卡牌正常结算，不受 pass 影响
- **是否受伤取决于状态**：pass 不提供任何防御效果，是否受伤由玩家当前状态（防御/联防）决定
- **重置连续攻击计数**：pass 打断连续攻击链
- **双方都 pass**：双方都不掉血，回合正常推进，轮次正常推进
- **无限制**：不限制连续 pass 次数（轮末流血机制自然防止无限循环）

## 影响范围

```
model.py        → POLICY_OUTPUT_SIZE 13→14，fc_policy Linear(32→14)
converter.py    → _CARD_MAP 索引重排，pass 映射
game_core.py    → play_round 支持空卡牌列表，_validate 跳过空列表
app_manager.py  → _make_mask_fn 适配 14 维，pass 始终合法
player.py       → PlayerHistoryItem.card 允许 None（表示 pass）
round_result.py → to_dict 处理空 cards_1/cards_2
api.py          → play 端点接受空 cards_1/cards_2
serializers.py  → 序列化兼容空卡牌列表
App.jsx         → 新增 onPass 回调
GameBoard.jsx   → 新增 Pass 按钮，reveal 阶段显示 pass
```

## 关键流程

参见 [pass_game_flow.puml](diagrams/pass_game_flow.puml) 和 [class_diagram.puml](diagrams/class_diagram.puml)。
