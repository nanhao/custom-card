# Interface Design — Input Reordering (intention #13)

## 1. Converter

唯一变更的公有接口。`Converter` 类的其他公有方法（`card_to_index`, `index_to_card`, `build_cost_for_index`）不变。

### 1.1 build_input_vector

```
build_input_vector(state: dict, player_key: str = "2") -> list[float]
```

**行为**：从游戏状态构造 25-d input 向量，维度顺序为本设计定义的新顺序。

**参数**：
- `state`: `GameCore.get_state()` 返回的游戏状态字典
- `player_key`: 视角标识，`"1"` 或 `"2"`，默认 `"2"`

**返回值**：长度 25 的浮点数列表

**索引映射（0-based）**：

| 索引 | 取值来源 | 类型 |
|------|----------|------|
| 0 | `state["players"][player_key]["hp"]` | float |
| 1 | `state["players"][opponent_key]["hp"]` | float |
| 2 | `state["round_in_cycle"]` | float, 1-5 |
| 3 | `state["cycle"]` | float |
| 4-10 | `state["players"][player_key]["hand"]` 中各卡牌类型的计数 | float, ≥0 |
| 11-17 | `state["players"][player_key]["history"][-1]` 的 one-hot | 0.0 or 1.0 |
| 18-24 | `state["players"][opponent_key]["history"][-1]` 的 one-hot | 0.0 or 1.0 |

**异常**：
- `ValueError`: `player_key` 不是 `"1"` 或 `"2"`

**前置条件**：
- `state` 必须包含 `players`、`round_in_cycle`、`cycle` 字段
- 若玩家无出牌历史（`history` 为空），对应 one-hot 段全为 0.0

### 1.2 与旧版差异

仅返回值顺序变化，参数、异常、前置条件均不变。调用方（`AppManager.ai_select_action`, `AppManager.ai_sample_action`, `SelfPlayRunner._play_one_game`）无需修改。
