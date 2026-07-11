# Interface Design — 56维输入 + 牌堆互相可见 (intention #34)

## 1. Converter

### 1.1 build_input_vector

```
build_input_vector(state: GameState, player_key: str = "2") -> list[float]
```

**行为**：从游戏状态构造 56-d input 向量。

**参数**：
- `state`: `GameState` 数据类实例
- `player_key`: 视角标识，`"1"` 或 `"2"`，默认 `"2"`

**返回值**：长度 56 的浮点数列表

**索引映射**：

| 索引 | 取值来源 | 变更 |
|------|----------|------|
| 0 | `state.players[player_key].hp` | |
| 1 | `state.players[opponent_key].hp` | |
| 2 | `state.players[player_key].energy` | |
| 3 | `state.players[opponent_key].energy` | |
| 4 | `state.round_in_cycle` | |
| 5 | `state.cycle` | |
| 6-12 | 我方手牌各类型计数 | |
| 13-19 | 对方手牌各类型计数 | |
| 20-22 | 我方状态持续回合（DEFENSE, GUARD, FRAGILE） | |
| 23-25 | 对方状态持续回合 | |
| 26-32 | 我方弃牌堆各类型计数 | |
| 33-39 | 对方弃牌堆各类型计数 | **新增** |
| 40-46 | 我方抽牌堆各类型计数 | 从 33-39 偏移 |
| 47-53 | 对方抽牌堆各类型计数 | **新增** |
| 54 | 我方上回合 action 的 policy index | 从 40 偏移 |
| 55 | 对方上回合 action 的 policy index | 从 41 偏移 |

**异常**：
- `ValueError`: `player_key` 不是 `"1"` 或 `"2"`

**前置条件变更**（per D4）：
- ~~`state.players[player_key].draw_pile` 和 `discard_pile` 必须不为 None~~（已移除：piles 始终非空）
- ~~`state.players[opponent_key].draw_pile` 和 `discard_pile` 必须不为 None~~（已移除：piles 始终非空）
- 状态持续回合从 `PlayerState.statuses` 按 `StatusType` 查找，未拥有的类型返回 0.0
- 手牌/弃牌堆/抽牌堆计数按 `Converter._CARD_TYPE_ORDER` 顺序排列

**与设计 #27 的差异**：
- 返回值长度：42 → 56
- 索引 33-39：我方抽牌堆 → 对方弃牌堆（新增）
- 索引 40-46：last action → 我方抽牌堆（偏移）
- 索引 47-53：新增对方抽牌堆
- 索引 54-55：last action 从 40-41 偏移
- 移除 pile-is-None 的前置条件检查

### 1.2 新增内部方法：无

无需新增方法。现有 `_count_by_card_type` 和 `_extract_status_lifetime` 直接复用于新增的对方牌堆段。

### 1.3 保留不变的接口

- `action_to_policy_index(card, build_target)` — 不变
- `policy_index_to_action(index)` — 不变
- `build_cost(card_type)` — 不变
- `ALL_CARD_TYPES` — 不变
- `_CARD_MAP` — 不变
- `NUM_CARD_TYPES` — 不变

---

## 2. GRUPolicyModel (rnn_v1)

### 2.1 常量

```
INPUT_SIZE = 56  # 从 42 变更
```

### 2.2 构造函数

```
GRUPolicyModel()
```

变更：`gru.input_size` = **56**（原 42）。`fc_policy` 输出维度 **14** 不变。

### 2.3 forward

```
forward(x, hidden, mask_fn=None, temperature=1.0) -> (probs, log_probs, new_hidden)
```

变更：
- `x` 形状：`(1, 1, 56)`（原 `(1, 1, 42)`）
- 其他签名和返回值不变

---

## 3. GRUValueModel (rnn_v1)

### 3.1 常量

```
INPUT_SIZE = 56  # 从 42 变更
```

### 3.2 构造函数

```
GRUValueModel()
```

变更：`gru.input_size` = **56**（原 42）。

### 3.3 forward

```
forward(x, hidden) -> (value, new_hidden)
```

变更：
- `x` 形状：`(1, 1, 56)`（原 `(1, 1, 42)`）

---

## 4. GRUPolicyModelV2 (rnn_v2)

### 4.1 常量

```
INPUT_SIZE = 56          # 从 42 变更
SKIP_INPUT_SIZE = 88     # 从 74 变更 (56 + 32)
```

### 4.2 构造函数

```
GRUPolicyModelV2()
```

变更：
- `gru.input_size` = **56**（原 42）
- `fc1` = `Linear(88, 64)`（原 `Linear(74, 64)`）
- `fc2`、`fc_policy` 不变

### 4.3 forward

```
forward(x, hidden, mask_fn=None, temperature=1.0) -> (probs, log_probs, new_hidden)
```

变更：
- `x` 形状：`(1, 1, 56)`（原 `(1, 1, 42)`）

---

## 5. GRUValueModelV2 (rnn_v2)

### 5.1 常量

```
INPUT_SIZE = 56
SKIP_INPUT_SIZE = 88
```

### 5.2 构造函数

```
GRUValueModelV2()
```

变更：
- `gru.input_size` = **56**（原 42）
- `fc1` = `Linear(88, 64)`（原 `Linear(74, 64)`）

### 5.3 forward

```
forward(x, hidden) -> (value, new_hidden)
```

变更：
- `x` 形状：`(1, 1, 56)`（原 `(1, 1, 42)`）

---

## 6. GameCore

### 6.1 get_state

```
get_state(player_perspective: str | None = None) -> GameState
```

变更（per D3）：
- `player_perspective` 参数保留但行为统一——所有玩家始终看到双方完整 `draw_pile` 和 `discard_pile`
- 不再有条件分支：`PlayerState.draw_pile` / `discard_pile` 始终为完整 `list[CardType]`
- 向后兼容：调用方传入 `player_perspective` 不报错，行为与不传一致

---

## 7. GameState / PlayerState

### 7.1 PlayerState 字段变更

| 字段 | 原类型 | 新类型 |
|------|--------|--------|
| `draw_pile` | `list[CardType] \| None` | `list[CardType]` |
| `discard_pile` | `list[CardType] \| None` | `list[CardType]` |

### 7.2 GameState.to_dict 变更

`_serialize_player` 中：始终序列化 `draw_pile` 和 `discard_pile`（不再检查 `is not None`）。

---

## 8. Frontend

### 8.1 GameBoard props 变更

```
GameBoard({ ..., pileViewer, onViewPiles, onClosePiles, ... })
```

- `pileViewer`: 从 `{ draw_pile, discard_pile }` 扩展为 `{ pid, draw_pile, discard_pile }`，增加 `pid` 标识查看的是哪个玩家
- `onViewPiles`: 签名从 `() => void` 改为 `(pid: string) => void`，接收玩家标识

### 8.2 App.handleViewPiles 变更

```
handleViewPiles(pid: string) -> void
```

直接从 `state.players[pid]` 读取 `draw_pile` 和 `discard_pile`（不再调用 API）。`getMyPiles` API 可移除。

### 8.3 api.js 变更

- `getState`: 不变（返回已包含双方完整牌堆）
- `getMyPiles`: 可移除（不再需要单独的 perspective 查询）

---

## 9. 与设计 #27 差异总结

| 项目 | 设计 #27 | 设计 #34 |
|------|----------|----------|
| INPUT_SIZE | 42 | 56 |
| 对方弃牌堆 | 无 | 7 维（索引 33-39） |
| 我方抽牌堆 | 索引 33-39 | 索引 40-46 |
| 对方抽牌堆 | 无 | 7 维（索引 47-53） |
| last action | 索引 40-41 | 索引 54-55 |
| Policy output dim | 14 | 14（不变） |
| HIDDEN_SIZE | 32 | 32（不变） |
| rnn_v2 SKIP_INPUT_SIZE | 74 | 88 |
| pile visibility | perspective 控制 | 始终全部可见 |
| pile type | list[CardType] \| None | list[CardType] |
| Converter pile None 检查 | 有 | 无 |
| 前端 pile viewer | 仅 P1 | 双方均可查看 |
| input 定义来源 | 代码注释 | intention/input.md |
