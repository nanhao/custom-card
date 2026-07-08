# Interface Design — Input/Output Redesign (intention #14)

## 1. Converter

### 1.1 build_input_vector

```
build_input_vector(state: dict, player_key: str = "2") -> list[float]
```

**行为**：从游戏状态构造 15-d input 向量。

**参数**：
- `state`: `GameCore.get_state()` 返回的游戏状态字典
- `player_key`: 视角标识，`"1"` 或 `"2"`，默认 `"2"`

**返回值**：长度 15 的浮点数列表

**索引映射**：

| 索引 | 取值来源 | 类型 |
|------|----------|------|
| 0 | `state["players"][player_key]["hp"]` | float |
| 1 | `state["players"][opponent_key]["hp"]` | float |
| 2 | `state["players"][player_key]["energy"]` | float, 0-5 |
| 3 | `state["players"][opponent_key]["energy"]` | float, 0-5 |
| 4 | `state["round_in_cycle"]` | float, 1-5 |
| 5 | `state["cycle"]` | float |
| 6-12 | `state["players"][player_key]["hand"]` 中各卡牌类型计数 | float |
| 13 | 我方上回合 action 的 policy index，从 `history[-1]` 通过 `action_to_policy_index` 计算 | int, -1 或 0-12 |
| 14 | 对方上回合 action 的 policy index，从 `history[-1]` 通过 `action_to_policy_index` 计算 | int, -1 或 0-12 |

**异常**：
- `ValueError`: `player_key` 不是 `"1"` 或 `"2"`

**前置条件**：
- `state` 必须包含 `players`、`round_in_cycle`、`cycle` 字段
- `players` 的 `history` 格式为 `list[PlayerHistoryItem]`（见 §5 Game State）
- 若玩家无出牌历史（`history` 为空），索引 13/14 = -1.0

### 1.2 action_to_policy_index

```
action_to_policy_index(card_str: str, build_target: str | None) -> int
```

**行为**：将游戏动作（卡牌 + 可选构建目标）映射为 policy 输出下标（0-12）。

**参数**：
- `card_str`: 打出的卡牌，如 `"attack"`、`"build"`
- `build_target`: 构建目标，仅 `card_str == "build"` 时有意义

**返回值**：policy index，0-12

**映射表**：

| card_str | build_target | 返回值 |
|----------|-------------|--------|
| build | build | 0 |
| attack | — | 1 |
| build | attack | 2 |
| defense | — | 3 |
| build | defense | 4 |
| heavy | — | 5 |
| build | heavy | 6 |
| counter | — | 7 |
| build | counter | 8 |
| energy | — | 9 |
| build | energy | 10 |
| flex | — | 11 |
| build | flex | 12 |

**异常**：
- `ValueError`: card_str 不是合法卡牌，或 build_target 不合法

### 1.3 policy_index_to_action

```
policy_index_to_action(index: int) -> tuple[str, str | None]
```

**行为**：将 policy 输出下标解码为 (card_str, build_target) 元组。

**参数**：
- `index`: policy 下标，0-12

**返回值**：`(card_str, build_target)`，偶数索引时 card_str = `"build"`，奇数索引时 build_target = `None`

**异常**：
- `IndexError`: index 不在 [0, 12]

### 1.4 移除的接口

以下旧接口移除：
- `card_to_index` — 被 `action_to_policy_index` 替代
- `index_to_card` — 被 `policy_index_to_action` 替代
- `build_cost_for_index` — 迁移至 `AppManager._make_mask_fn` 内部

### 1.5 保留的类属性

- `ALL_CARD_TYPES`: frozenset 不变
- `INPUT_SIZE`: int = **15**（从 25 变更）
- `NUM_CARD_TYPES`: int = **7**（不变）

## 2. GRUPolicyModel

### 2.1 构造函数

```
GRUPolicyModel()
```

变更：`fc_policy` 输出维度从 `NUM_CARD_TYPES * 2 (=14)` 改为 `13`。

### 2.2 forward

```
forward(x, hidden, mask_fn=None, temperature=1.0) -> (probs, log_probs, new_hidden)
```

**变更**：
- `x` 形状：`(1, 1, 15)`（原 `(1, 1, 25)`）
- `probs` 形状：`(1, 13)`（原 `(1, 14)`）
- 其他签名不变

### 2.3 INITIAL_HIDDEN_STATE

```
INITIAL_HIDDEN_STATE: list[float] = [0.0] * 32
```

不变。

## 3. GRUValueModel

### 3.1 构造函数

```
GRUValueModel()
```

变更：`gru.input_size` = **15**（原 25）。

### 3.2 forward

```
forward(x, hidden) -> (value, new_hidden)
```

**变更**：
- `x` 形状：`(1, 1, 15)`

## 4. AppManager

### 4.1 ai_select_action

```
ai_select_action(state, hidden_state, ai_name) -> (card_str, build_target, new_hidden)
```

**变更**：
- 内部调用 `Converter.build_input_vector` 返回 15-d 向量
- action 解码改用 `Converter.policy_index_to_action`
- mask_fn 改为 13-d interleaved 格式

### 4.2 ai_sample_action

```
ai_sample_action(state, h_pol, h_val, ai_name, temperature, player_key="2")
    -> (card_str, build_target, new_h_pol, log_prob, value, new_h_val)
```

**变更**：同 `ai_select_action`

### 4.3 _make_mask_fn（私有）

**行为**：对 13-d interleaved 输出做两段屏蔽。

- **奇数索引（纯出牌）**：屏蔽不在手牌中的卡牌；若能量=0 则屏蔽 build（索引 0,2,4,6,8,10,12）
- **偶数索引（出构建+构建 X）**：若手牌无 build 或能量不足，全屏蔽；否则按构建消耗逐个屏蔽

## 5. PlayerHistoryItem — 新增领域类

位于 `game/player.py`，与 `Player` 同模块。

```
@dataclass
class PlayerHistoryItem:
    """Single entry in a player's action history.

    Records both the card played and the build target (if any).
    """
    card: CardType              # 打出的卡牌
    build_target: CardType | None  # 构建目标，非构建动作为 None
```

## 6. Game State — history 格式变更

`GameCore.get_state()` 返回的 `players["1"/"2"]["history"]` 格式变更：

**旧格式**（设计 #13）：
```
list[str]，如 ["attack", "build", "defense"]
```

**新格式**（设计 #14）：
```
list[PlayerHistoryItem]，如 [
    PlayerHistoryItem(card=CardType.ATTACK, build_target=None),
    PlayerHistoryItem(card=CardType.BUILD,  build_target=CardType.ATTACK),
    PlayerHistoryItem(card=CardType.DEFENSE, build_target=None),
]
```

此变更由 Player 新增的 `build_history: list[CardType | None]` 字段支撑。`play_round()` 中与 `history` 同步记录。`get_state()` 将内部两个列表 zip 为 `PlayerHistoryItem` 列表输出。

## 7. 与旧版差异总结

| 项目 | 设计 #13 | 设计 #14 |
|------|----------|----------|
| INPUT_SIZE | 25 | 15 |
| Policy output dim | 14 | 13 |
| 能量在 input 中 | 无 | 索引 2-3 |
| 上回合出牌编码 | one-hot (14-d) | policy index (2-d) |
| Sample action 格式 | 字符串 | 整数 policy index |
| Converter.card_to_index | 存在 | 移除 |
| Converter.index_to_card | 存在 | 移除 |
| State history 格式 | list[str] | list[PlayerHistoryItem] |
| Player 字段 | 无 build_history | 新增 build_history |
| HIDDEN_SIZE | 32 | 32（不变） |
