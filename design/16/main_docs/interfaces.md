# Interface Design — Player 历史记录合并 (intention #16)

## 1. Player — 变更

位于 `game/player.py`。

### 1.1 删除字段

移除 `build_history: list[CardType | None]`。

### 1.2 history 字段类型变更

```
# 旧
history: list[CardType] = field(default_factory=list)

# 新
history: list[PlayerHistoryItem] = field(default_factory=list)
```

### 1.3 其他字段和方法不变

`hp`、`energy`、`statuses`、`hand`、`draw_pile`、`discard_pile` 字段不变。
`play_card`、`play_cards`、`has_card`、`has_cards`、`take_damage`、`add_energy`、`has_status` 签名和行为不变。

### 1.4 PlayerHistoryItem — 不变

```python
@dataclass
class PlayerHistoryItem:
    """Single entry in a player's action history.
    Attributes:
        card: The card played this round.
        build_target: The card built via BUILD, or None.
    """
    card: CardType
    build_target: CardType | None = None
```

## 2. GameCore — 变更

位于 `game/game_core.py`。

### 2.1 play_round — 历史记录合并

**Step 8 变更（记录历史）**：

```python
# 旧（两个平行列表）
self.player_1.history.extend(cards_1)
self.player_2.history.extend(cards_2)
self.player_1.build_history.append(build_target_1)
self.player_2.build_history.append(build_target_2)

# 新（合并为 PlayerHistoryItem）
self.player_1.history.append(PlayerHistoryItem(cards_1[0], build_target_1))
self.player_2.history.append(PlayerHistoryItem(cards_2[0], build_target_2))
```

### 2.2 get_state — 简化

`_player_dict()` 中 history 序列化从 zip 操作简化为直接使用列表：

```python
# 旧
"history": [
    PlayerHistoryItem(card=c, build_target=bt)
    for c, bt in zip(p.history, p.build_history)
],

# 新
"history": p.history,
```

由于 `p.history` 已经是 `list[PlayerHistoryItem]`，而 `PlayerHistoryItem` 在 API 的 JSON 编码中已有处理（`api.py` 中的 `_custom_default`），故直接返回即可。

### 2.3 其他方法不变

`start_game`、`_start_cycle`、`_validate`、`_process_build`、`_tick_status_lifetimes`、`_check_game_end` 签名和行为不变。

## 3. CardResolver — 变更

位于 `game/resolver.py`。

### 3.1 get_consecutive_attack 签名变更

```python
# 旧
@staticmethod
def get_consecutive_attack(history: list[CardType]) -> int:

# 新
@staticmethod
def get_consecutive_attack(history: list[PlayerHistoryItem]) -> int:
```

**行为不变**：从 `item.card` 提取卡牌类型，仅 `CardType.ATTACK` 计入连击计数，其他类型中断连击。

GameCore 中调用方式不变，仍传入 `self.player_1.history`（类型已变为 `list[PlayerHistoryItem]`）。

### 3.2 resolve — 不变

`resolve()` 签名和行为不变。

## 4. SessionStore — 变更

位于 `game/session_store.py`。

### 4.1 _serialize_player — 序列化

```python
# 旧
"history": [_card_to_str(c) for c in player.history],
"build_history": [
    _card_to_str(bt) if bt is not None else None
    for bt in player.build_history
],

# 新
"history": [
    {"card": _card_to_str(item.card),
     "build_target": _card_to_str(item.build_target) if item.build_target else None}
    for item in player.history
],
```

移除 `build_history` 字段。

### 4.2 _deserialize_player — 反序列化

```python
# 旧
required = ("hp", "energy", "statuses", "hand", "draw_pile",
            "discard_pile", "history", "build_history")
...
history=[_str_to_card(s) for s in data["history"]],
build_history=[
    _str_to_card(s) if s is not None else None
    for s in data["build_history"]
],

# 新
required = ("hp", "energy", "statuses", "hand", "draw_pile",
            "discard_pile", "history")
...
history=[
    PlayerHistoryItem(
        card=_str_to_card(d["card"]),
        build_target=_str_to_card(d["build_target"]) if d.get("build_target") else None,
    )
    for d in data["history"]
],
```

移除 `build_history` 字段。

## 5. 不受影响的模块

### 5.1 Converter (`app/converter.py`)

- `_action_item_to_index()` 原本就接收 `PlayerHistoryItem`，不变
- `build_input_vector()` 通过 `get_state()` 获取 history（字典列表格式），不变

### 5.2 Controller (`controller/api.py`)

- `PlayerHistoryItem` JSON 编码 (`_custom_default`) 已存在且格式不变

### 5.3 Frontend (`GameBoard.jsx`)

- 已通过 `typeof c === 'string' ? c : c.card` 兼容新旧格式，不变

## 6. 与旧版差异总结

| 项目 | 旧 | 新 |
|------|-----|-----|
| Player.history 类型 | `list[CardType]` | `list[PlayerHistoryItem]` |
| Player.build_history | 存在 | 删除 |
| play_round 历史记录 | 分别 append 两个列表 | 合并 append 一个 PlayerHistoryItem |
| get_state history | zip 两个列表临时构造 | 直接返回 Player.history |
| 序列化 history | `["attack", "defense"]` | `[{"card":"attack","build_target":null}]` |
| 序列化 build_history | 独立字段 | 已合并到 history |
| get_consecutive_attack 参数 | `list[CardType]` | `list[PlayerHistoryItem]` |
