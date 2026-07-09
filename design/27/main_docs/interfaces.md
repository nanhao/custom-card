# Interface Design — 42维输入 (intention #27)

## 1. Converter

### 1.1 build_input_vector

```
build_input_vector(state: GameState, player_key: str = "2") -> list[float]
```

**行为**：从游戏状态构造 42-d input 向量。

**参数**：
- `state`: `GameState` 数据类实例
- `player_key`: 视角标识，`"1"` 或 `"2"`，默认 `"2"`

**返回值**：长度 42 的浮点数列表

**索引映射**：

| 索引 | 取值来源 | 类型 |
|------|----------|------|
| 0 | `state.players[player_key].hp` | float |
| 1 | `state.players[opponent_key].hp` | float |
| 2 | `state.players[player_key].energy` | float |
| 3 | `state.players[opponent_key].energy` | float |
| 4 | `state.round_in_cycle` | float, 1-3 |
| 5 | `state.cycle` | float |
| 6-12 | 我方手牌各类型计数 | float, _CARD_TYPE_ORDER 顺序 |
| 13-19 | 对方手牌各类型计数 | float, _CARD_TYPE_ORDER 顺序 |
| 20-22 | 我方状态持续回合（DEFENSE, GUARD, FRAGILE） | float, 0=无该状态 |
| 23-25 | 对方状态持续回合（DEFENSE, GUARD, FRAGILE） | float, 0=无该状态 |
| 26-32 | 我方弃牌堆各类型计数 | float, _CARD_TYPE_ORDER 顺序 |
| 33-39 | 我方抽牌堆各类型计数 | float, _CARD_TYPE_ORDER 顺序 |
| 40 | 我方上回合 action 的 policy index | float, -1 或 0-12 |
| 41 | 对方上回合 action 的 policy index | float, -1 或 0-12 |

**异常**：
- `ValueError`: `player_key` 不是 `"1"` 或 `"2"`

**前置条件**：
- `state.players[player_key].draw_pile` 和 `discard_pile` 必须不为 None（即 `player_perspective` 需匹配）
- 状态持续回合从 `PlayerState.statuses` 按 `StatusType` 查找，未拥有的类型返回 0.0
- 手牌/弃牌堆/抽牌堆计数按 `Converter._CARD_TYPE_ORDER` 顺序排列（attack=0, defense=1, heavy=2, counter=3, energy=4, build=5, flex=6）

**与设计 #14 的差异**：
- 返回值长度：15 → 42
- 手牌计数段从 7 维扩展为 14 维（我方+对方）
- 新增状态持续回合段 6 维
- 新增弃牌堆+抽牌堆段 14 维
- 上回合 action 索引从 13-14 移至 40-41

### 1.2 新增内部方法：_count_by_card_type

```
_count_by_card_type(cards: list[CardType]) -> list[float]
```

**行为**：统计卡牌列表中各类型数量，按 `_CARD_TYPE_ORDER` 顺序输出。

**参数**：
- `cards`: 卡牌列表

**返回值**：长度 7 的浮点数列表

### 1.3 新增内部方法：_extract_status_lifetime

```
_extract_status_lifetime(statuses: list[PlayerStatus], status_type: StatusType) -> float
```

**行为**：从玩家状态列表中查找指定类型的剩余持续回合。

**参数**：
- `statuses`: 玩家状态列表
- `status_type`: 要查找的状态类型

**返回值**：lifetime 值（float），未找到返回 0.0

### 1.4 保留不变的接口

- `action_to_policy_index(card, build_target)` — 不变
- `policy_index_to_action(index)` — 不变
- `build_cost(card_type)` — 不变
- `ALL_CARD_TYPES` — 不变
- `_CARD_MAP` — 不变
- `NUM_CARD_TYPES` — 不变

## 2. GRUPolicyModel

### 2.1 常量

```python
INPUT_SIZE = 42  # 从 15 变更
```

### 2.2 构造函数

```
GRUPolicyModel()
```

变更：`gru.input_size` = **42**（原 15）。`fc_policy` 输出维度 **13** 不变。

### 2.3 forward

```
forward(x, hidden, mask_fn=None, temperature=1.0) -> (probs, log_probs, new_hidden)
```

**变更**：
- `x` 形状：`(1, 1, 42)`（原 `(1, 1, 15)`）
- 其他签名和返回值不变

## 3. GRUValueModel

### 3.1 常量

```python
INPUT_SIZE = 42  # 从 15 变更
```

### 3.2 构造函数

```
GRUValueModel()
```

变更：`gru.input_size` = **42**（原 15）。

### 3.3 forward

```
forward(x, hidden) -> (value, new_hidden)
```

**变更**：
- `x` 形状：`(1, 1, 42)`（原 `(1, 1, 15)`）

## 4. GameState / PlayerState — 无需变更

`PlayerState.hand` 始终包含双方手牌的完整 `list[CardType]`。`PlayerState.statuses` 包含完整的 `list[PlayerStatus]`（含 lifetime）。`PlayerState.draw_pile` / `discard_pile` 在 `player_perspective` 匹配时返回完整列表。

以上字段现行实现已满足 42-d input 的所有数据需求，无需修改 game 层。

## 5. 与设计 #14 差异总结

| 项目 | 设计 #14 | 设计 #27 |
|------|----------|----------|
| INPUT_SIZE | 15 | 42 |
| 手牌信息 | 仅我方 7 维 | 我方+对方 14 维 |
| 状态持续回合 | 无 | 双方各 3 维 = 6 维 |
| 弃牌堆/抽牌堆 | 无 | 我方各 7 维 = 14 维 |
| 上回合 action | 索引 13-14 | 索引 40-41 |
| Policy output dim | 13 | 13（不变） |
| HIDDEN_SIZE | 32 | 32（不变） |
| Converter 内部方法 | — | 新增 _count_by_card_type, _extract_status_lifetime |
