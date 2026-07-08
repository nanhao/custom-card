# Interface Design — GameState 重构 (intention #20)

## 1. PlayerState — 新增数据类

位于 `game/game_state.py`。

```
@dataclass
class PlayerState:
    """Read-only snapshot of a single player's state at a point in time.

    Attributes:
        hp: Current HP (≥ 0).
        energy: Current energy (0–5).
        statuses: Active statuses with lifetimes.
        hand: Cards currently in hand.
        history: Full action history (card + build_target per round).
        draw_pile: Cards in draw pile, or None if hidden from this perspective.
        discard_pile: Cards in discard pile, or None if hidden.
        draw_pile_count: Count of cards in draw pile (always visible).
        discard_pile_count: Count of cards in discard pile (always visible).
    """
    hp: int
    energy: int
    statuses: list[PlayerStatus]
    hand: list[CardType]
    history: list[PlayerHistoryItem]
    draw_pile: list[CardType] | None = None
    discard_pile: list[CardType] | None = None
    draw_pile_count: int = 0
    discard_pile_count: int = 0
```

**字段说明**：
- `hp: int` — 当前 HP，≥ 0
- `energy: int` — 当前能量，0–5
- `statuses: list[PlayerStatus]` — 生效中的状态列表（使用 `PlayerStatus` 对象，非 dict）
- `hand: list[CardType]` — 手牌（使用 `CardType` 枚举，非字符串）
- `history: list[PlayerHistoryItem]` — 完整行动历史
- `draw_pile: list[CardType] | None` — 抽牌堆内容；不可见时为 `None`
- `discard_pile: list[CardType] | None` — 弃牌堆内容；不可见时为 `None`
- `draw_pile_count: int` — 抽牌堆数量，始终可见
- `discard_pile_count: int` — 弃牌堆数量，始终可见

## 2. GameState — 新增数据类

位于 `game/game_state.py`。

```
@dataclass
class GameState:
    """Read-only snapshot of the full game state.

    Attributes:
        round: Current round number (1-based).
        cycle: Current cycle number (1-based).
        round_in_cycle: Round position within current cycle (1–5).
        players: Player states keyed by "1" and "2".
        is_over: True if game has ended.
        winner: "1", "2", "draw", or None if game ongoing.
        mode: "pvp" or "pve".
        round_log: Complete round result history.
    """
    round: int
    cycle: int
    round_in_cycle: int
    players: dict[str, PlayerState]
    is_over: bool
    winner: str | None
    mode: str
    round_log: list[RoundResult]
```

### 2.1 to_dict — 序列化

```
@staticmethod
def to_dict(gs: "GameState") -> dict:
    """Serialize a GameState to a JSON-safe dict.

    Converts CardType → string via .value, PlayerStatus → dict via .to_dict(),
    PlayerHistoryItem → {"card": str, "build_target": str|None},
    RoundResult → dict via RoundResult.to_dict().

    Returns:
        dict with nested players and round_log structures,
        matching the existing JSON API schema.
    """
```

**序列化规则**：
| 字段 | 内部类型 | 序列化后 |
|------|---------|---------|
| `hand` | `list[CardType]` | `[c.value, ...]` |
| `statuses` | `list[PlayerStatus]` | `[ps.to_dict(), ...]` |
| `history` | `list[PlayerHistoryItem]` | `[{"card": h.card.value, "build_target": h.build_target.value if h.build_target else None}, ...]` |
| `draw_pile` | `list[CardType] \| None` | `[c.value, ...]` 或省略 |
| `discard_pile` | `list[CardType] \| None` | `[c.value, ...]` 或省略 |
| `draw_pile_count` | `int` | `int` |
| `discard_pile_count` | `int` | `int` |
| `round_log` | `list[RoundResult]` | `[RoundResult.to_dict(rr), ...]` |

## 3. GameCore — 变更

位于 `game/game_core.py`。

### 3.1 get_state 签名与行为变更

```
get_state(player_perspective: str | None = None) -> GameState
```

**旧**：返回 `dict`，内部用嵌套函数 `_player_dict()` 构造玩家状态。  
**新**：返回 `GameState` 实例，直接构造 `PlayerState` 对象。

**行为不变**：`player_perspective` 参数逻辑保持一致 — 指定视角时暴露该玩家的牌堆内容，否则仅暴露计数。

### 3.2 移除 `_player_dict` 内部函数

当前 `get_state()` 内定义的 `_player_dict(p, show_piles) -> dict` 函数移除，逻辑迁移至 `PlayerState` 构造。

## 4. Converter — 变更

位于 `app/converter.py`。

### 4.1 build_input_vector 签名变更

```
build_input_vector(state: GameState, player_key: str = "2") -> list[float]
```

**旧**：`state: dict`。  
**新**：`state: GameState`。

**内部适配**：
- `state["players"][player_key]["hp"]` → `state.players[player_key].hp`
- `state["players"][player_key]["hand"]` → `state.players[player_key].hand`（`list[CardType]`）
- 迭代 `hand` 时通过 `card.value` 获取字符串用于 `_CARD_TYPE_ORDER` 查找
- `state.get("round_in_cycle", 1)` → `state.round_in_cycle`
- `this_player.get("history", [])` → `this_player.history`
- `_action_item_to_index` 签名不变（已接收 `PlayerHistoryItem`）

### 4.2 _action_item_to_index

无变更 — 已使用 `PlayerHistoryItem.card.value`。

## 5. AppManager — 变更

位于 `app/app_manager.py`。

### 5.1 create_game 签名变更

```
create_game(mode: str, ai_name: str | None = None) -> tuple[str, GameState]
```

**旧**：返回 `(game_id, dict)`。  
**新**：返回 `(game_id, GameState)`。

### 5.2 ai_select_action 签名变更

```
ai_select_action(state: GameState, hidden_state: list[float], ai_name: str) -> tuple[str, str | None, list[float]]
```

**旧**：`state: dict`。  
**新**：`state: GameState`。

**内部适配**：
- `state["players"]["2"]["energy"]` → `state.players["2"].energy`
- `state["players"]["2"]["hand"]` → `state.players["2"].hand`

### 5.3 ai_sample_action 签名变更

```
ai_sample_action(state: GameState, hidden_state_policy: list[float], hidden_state_value: list[float], ai_name: str, temperature: float, player_key: str = "2")
    -> tuple[str, str | None, list[float], float | None, float, list[float]]
```

**旧**：`state: dict`。  
**新**：`state: GameState`。内部适配同 `ai_select_action`。

### 5.4 _make_mask_fn 签名变更

```
_make_mask_fn(hand: list[CardType], energy: int) -> Callable
```

**旧**：`hand: list[str]`。  
**新**：`hand: list[CardType]`。

**内部适配**：`legal_cards = set(hand)` 中元素的 `.value` 比较需要改为 `card.value`。调用 `Converter._PLAY_TO_INDEX` 和 `Converter._CARD_MAP` 时需传入 `card.value`（因为这些映射表使用字符串键）。

## 6. Controller Serializers — 变更

位于 `controller/serializers.py`。

### 6.1 新增 game_state_to_dict

```
def game_state_to_dict(gs: GameState) -> dict:
    """Serialize a GameState to a JSON-safe dict.

    Thin wrapper that delegates to GameState.to_dict().
    """
    return GameState.to_dict(gs)
```

## 7. Controller API — 变更

位于 `controller/api.py`。

### 7.1 create_game 路由

`app_manager.create_game()` 返回 `(game_id, GameState)`。需先调用 `game_state_to_dict(state)`，再注入 `ai` 字段：

```python
game_id, gs = app_manager.create_game(mode=mode, ai_name=ai_name)
state_dict = game_state_to_dict(gs)
state_dict["ai"] = ai_name
return jsonify({"game_id": game_id, "state": state_dict}), 201
```

### 7.2 get_state 路由

`gm.get_state()` 返回 `GameState`。需先 `to_dict()` 再注入 `ai`：

```python
gs = gm.get_state(player_perspective=perspective)
state_dict = game_state_to_dict(gs)
state_dict["ai"] = metadata["ai"] if metadata else None
```

### 7.3 play_round 路由

同 `get_state` — `gm.get_state()` 返回 `GameState`，`to_dict()` 后注入 `ai`。

### 7.4 移除自定义 JSON encoder 中的 PlayerHistoryItem 处理

`GameState.to_dict()` 已将 `PlayerHistoryItem` 转为 dict，不再需要 Flask 自定义 encoder。

## 8. SelfPlayRunner — 变更

位于 `app/selfplay_runner.py`。

### 8.1 _play_one_game 中 pre_state 使用

`pre_state = gm.get_state()` 返回 `GameState`。  
`pre_state["round"]` → `pre_state.round`。

`pre_state` 传给 `self._sample_action()` 即 `AppManager.ai_sample_action`，已适配 `GameState`。

## 9. Frontend

无变更 — 接收的 JSON 格式不变。

## 10. 与旧版差异总结

| 项目 | 旧 | 新 |
|------|-----|-----|
| get_state 返回类型 | `dict` | `GameState` |
| 玩家状态结构 | 嵌套 `_player_dict()` 函数动态构造 dict | `PlayerState` dataclass |
| hand 类型 | `list[str]` | `list[CardType]` |
| statuses 类型 | `list[dict]`（`ps.to_dict()`） | `list[PlayerStatus]` |
| history 类型 | `list[PlayerHistoryItem]`（不变） | `list[PlayerHistoryItem]`（不变） |
| 牌堆内容 | 条件性地添加 `draw_pile`/`discard_pile` 字符串列表 | `draw_pile: list[CardType] \| None` |
| Converter 入参 | `state: dict` | `state: GameState` |
| AppManager 入参 | `state: dict` | `state: GameState` |
| Flask 自定义 encoder | 处理 `PlayerHistoryItem` | 移除（GameState.to_dict() 已处理） |
| 序列化入口 | `jsonify(state)` | `jsonify(game_state_to_dict(gs))` |
