# Design #28: 接口设计

本文档仅描述对外暴露的公共接口变更。私有方法不在设计范围内。

---

## 1. model.py — GRUPolicyModel

### 常量变更

```
POLICY_OUTPUT_SIZE = 14  # 原 13
```

### 构造器变更

```
GRUPolicyModel.__init__():
    self.fc_policy = nn.Linear(HIDDEN_SIZE, POLICY_OUTPUT_SIZE)  # 32→14
```

### forward() 不变

签名和输出语义不变，仅 logits 维度从 `(1,13)` 变为 `(1,14)`。

---

## 2. converter.py — Converter

### _CARD_MAP 重排

```python
# 新：（CardType, play_index, build_index）
(CardType.BUILD,   None, 1),   # 原 (..., 0)
(CardType.ATTACK,  2, 3),      # 原 (1, 2)
(CardType.DEFENSE, 4, 5),      # 原 (3, 4)
(CardType.HEAVY,   6, 7),      # 原 (5, 6)
(CardType.COUNTER, 8, 9),      # 原 (7, 8)
(CardType.ENERGY,  10, 11),    # 原 (9, 10)
(CardType.FLEX,    12, 13),    # 原 (11, 12)
```

派生表 `_PLAY_TO_INDEX`, `_BUILD_TO_INDEX`, `_INDEX_TO_ACTION` 自动同步重排。

### _INDEX_TO_ACTION 新增 pass 映射

```
0 → (None, None)  # pass
1 → (BUILD, BUILD)
2 → (ATTACK, None)
3 → (BUILD, ATTACK)
...
13 → (BUILD, FLEX)
```

### action_to_policy_index() — 签名扩展

```
action_to_policy_index(card: CardType | None, build_target: CardType | None) -> int
```

card 参数允许 `None`。`card=None, build_target=None` 返回 0（pass）。

### policy_index_to_action() — 返回值扩展

```
policy_index_to_action(index: int) -> tuple[CardType | None, CardType | None]
```

index=0 返回 `(None, None)` 表示 pass。

### build_input_vector() 不变

输入向量保持 42 维。仅 `_action_item_to_index` 内部处理 pass（返回 0）。

---

## 3. game_core.py — GameCore

### play_round() — cards 参数可为空列表

```
play_round(
    cards_1: list[CardType],   # 空列表 = pass
    cards_2: list[CardType],   # 空列表 = pass
    build_target_1: CardType | None = None,
    build_target_2: CardType | None = None,
) -> RoundResult
```

card 列表为空时表示该玩家不出牌：
- 不消费手牌、不弃牌
- 不为其构建 ResolverTask
- 不记录 `PlayerHistoryItem`（或记录 card=None 的 pass entry）
- 对手的卡牌正常构建 ResolverTask 并结算

### _validate() — 空列表跳过校验

空 cards_1 或 cards_2 时跳过手牌校验和 BUILD 校验。

---

## 4. player.py — PlayerHistoryItem

```
PlayerHistoryItem:
    card: CardType | None   # 原 CardType，None 表示 pass
    build_target: CardType | None
```

### get_consecutive_attack_count() 不变

遍历逻辑中 `item.card != CardType.ATTACK` 自然成立（None != ATTACK），因此 pass 自动重置连续攻击计数。

---

## 5. app_manager.py — AppManager

### _make_mask_fn() — 适配 14 维

- 索引 0（pass）：始终合法，永不屏蔽
- 偶数索引（2,4,6,8,10,12）：屏蔽手牌中没有的卡
- 奇数索引（1,3,5,7,9,11,13）：屏蔽无法构建的情况（无 BUILD 卡或能量不足）

### ai_select_action() / ai_sample_action() 不变

签名不变，内部通过 `policy_index_to_action(0)` 自动返回 `(None, None)` 表示 pass。

---

## 6. round_result.py — RoundResult

### to_dict() — 兼容空 cards

`cards_1` 和 `cards_2` 序列化时，空列表保持为空数组 `[]`。

---

## 7. controller/api.py — play_round 端点

### POST /api/games/{id}/play

- `cards_1` 和 `cards_2` 允许为空数组 `[]`
- 空数组表示该玩家不出牌，不再返回 `MISSING_CARDS_1` 错误
- PVE 模式下 AI 可能返回 pass，`cards_2` 相应为空数组

### AI 输出校验调整

AI 返回 `card_type=None` 时视为合法 pass，不再报 `AI_INFERENCE_ERROR`。

---

## 8. Frontend 接口

### GameBoard props 新增

```
onPass: () => void   # 用户点击 Pass 按钮时调用
```

### App.jsx — handlePass

```
handlePass() → 调用 doPlayRound 但 cards_1=[]，buildTarget=null
```

### PVP 模式

- P1 选择阶段：Pass 按钮和卡牌按钮并列显示
- P2 选择阶段（pvp_pass）：也需要 Pass 按钮

### reveal 阶段

- 一方 pass 时，结果面板中该方显示为"不出牌"而非卡牌图标
- 双方 pass 时同理

---

## 9. serializers.py

### round_result_to_dict() 兼容空卡牌

`cards_1` 和 `cards_2` 为空数组时序列化为 `[]`。

### str_to_card() 不变

不需要新增 pass 的字符串映射，pass 通过空数组 `[]` 表达。
