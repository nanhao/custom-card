# Interface Design — PlayerState 重构 (intention #15)

## 1. StatusType — 新增枚举

位于 `game/player_state.py`。

| 成员 | 值 | 说明 |
|------|-----|------|
| `DEFENSE` | `"defense"` | 防御状态，阻挡攻击伤害 |
| `GUARD` | `"guard"` | 联防状态，阻挡重击伤害 |

## 2. PlayerStatus — 新增数据类

位于 `game/player_state.py`。

```
@dataclass
class PlayerStatus:
    """Single status instance on a player.

    Attributes:
        status_type: The type of status (DEFENSE or GUARD).
        lifetime: Remaining rounds before expiry. > 0 means active.
    """
    status_type: StatusType
    lifetime: int
```

**字段**：
- `status_type: StatusType` — 状态类型
- `lifetime: int` — 剩余有效回合数，> 0 表示生效中

## 3. Player — 变更

位于 `game/player.py`。

### 3.1 statuses 字段类型变更

```
# 旧
statuses: list[str] = field(default_factory=list)

# 新
statuses: list[PlayerStatus] = field(default_factory=list)
```

### 3.2 has_status 签名变更

```
has_status(status_type: StatusType) -> bool
```

**行为**：检查 `statuses` 中是否存在指定类型的有效状态（lifetime > 0）。

**参数**：
- `status_type: StatusType` — 要检查的状态类型

**返回值**：`True` 如果 player 拥有该状态

### 3.3 其他方法不变

`play_card`、`play_cards`、`has_card`、`has_cards`、`take_damage`、`add_energy` 签名和行为不变。

## 4. RoundResult — 变更

位于 `game/game_core.py`。

### 4.1 字段类型变更

| 字段 | 旧类型 | 新类型 |
|------|--------|--------|
| `statuses_1` | `list[str]` | `list[PlayerStatus]` |
| `statuses_2` | `list[str]` | `list[PlayerStatus]` |
| `after_statuses_1` | `list[str]` | `list[PlayerStatus]` |
| `after_statuses_2` | `list[str]` | `list[PlayerStatus]` |

其他字段不变。

## 5. ResolutionResult — 变更

位于 `game/resolver.py`。

### 5.1 字段类型变更

| 字段 | 旧类型 | 新类型 |
|------|--------|--------|
| `statuses_granted_1` | `list[str]` | `list[StatusType]` |
| `statuses_granted_2` | `list[str]` | `list[StatusType]` |

**注意**：`statuses_granted` 仅包含状态类型，不包含生存期。生存期由 GameCore 在 `play_round()` 中统一设为 1。

## 6. CardResolver — 变更

位于 `game/resolver.py`。

### 6.1 resolve 签名变更

```
resolve(
    cards_1: list[CardType],
    cards_2: list[CardType],
    player_1_statuses: list[PlayerStatus],   # 旧: list[str]
    player_2_statuses: list[PlayerStatus],   # 旧: list[str]
    p1_attack_streak: int,
    p2_attack_streak: int,
) -> ResolutionResult
```

**行为不变**。内部检查从 `"defense" in statuses` 改为使用 `StatusType.DEFENSE`。

### 6.2 内部状态字符串替换

| 旧字符串 | 新枚举 |
|----------|--------|
| `"defense"` | `StatusType.DEFENSE` |
| `"联防"` | `StatusType.GUARD` |

## 7. GameCore — 变更

位于 `game/game_core.py`。

### 7.1 移除字段

- `_status_lifetimes_1: list[int]` — 已进入 PlayerStatus.lifetime
- `_status_lifetimes_2: list[int]` — 同上

### 7.2 _tick_status_lifetimes 行为变更

```
_tick_status_lifetimes() -> None
```

**行为**：遍历 `player.statuses` 中的每个 PlayerStatus，将其 `lifetime -= 1`，移除 `lifetime <= 0` 的状态。

比旧实现更简洁：不再需要维护独立数组的索引对齐。

### 7.3 play_round 中状态赋值变更

授予新状态时，构造 PlayerStatus 对象：

```python
# 旧
self.player_1.statuses = list(resolution.statuses_granted_1)
self._status_lifetimes_1 = [1] * len(resolution.statuses_granted_1)

# 新
self.player_1.statuses = [
    PlayerStatus(st, lifetime=1) for st in resolution.statuses_granted_1
]
```

### 7.4 get_state 中 statuses 序列化

`get_state()` 返回的 statuses 从 `list[str]` 改为：

```json
[{"type": "defense", "lifetime": 1}]
```

## 8. SessionStore — 变更

位于 `game/session_store.py`。

### 8.1 序列化变更

- Player 的 `statuses` 序列化：`[{"type": s.status_type.value, "lifetime": s.lifetime} for s in player.statuses]`
- 移除 `_serialize_game_core` 中的 `status_lifetimes_1` / `status_lifetimes_2` 字段

### 8.2 反序列化变更

- Player 的 `statuses` 反序列化：`[PlayerStatus(StatusType(d["type"]), d["lifetime"]) for d in data["statuses"]]`
- 移除 `_deserialize_game_core` 中对 `status_lifetimes_*` 的读取

## 9. Controller Serializers — 变更

位于 `controller/serializers.py`。

### 9.1 round_result_to_dict

RoundResult 中 statuses 字段的 JSON 序列化适配 `list[PlayerStatus]` 格式：

```json
"statuses_1": [{"type": st.status_type.value, "lifetime": st.lifetime} for st in rr.statuses_1]
```

## 10. Frontend — 变更

### 10.1 GameBoard.jsx — statuses 渲染

```jsx
// 旧
const statusTags = (p.statuses || []).map((s, i) => {
  const tagCls = s === '联防' ? 'tag-guard' : 'tag-defense';
  return <span key={i} className={`status-tag ${tagCls}`}>{s}</span>;
});

// 新
const statusTags = (p.statuses || []).map((s, i) => {
  const tagCls = s.type === 'guard' ? 'tag-guard' : 'tag-defense';
  return <span key={i} className={`status-tag ${tagCls}`}>{s.type}</span>;
});
```

## 11. 与旧版差异总结

| 项目 | 旧 | 新 |
|------|-----|-----|
| Player.statuses 类型 | `list[str]` | `list[PlayerStatus]` |
| has_status 参数 | `str` | `StatusType` |
| GameCore 生存期跟踪 | 独立 `_status_lifetimes_1/2` | PlayerStatus.lifetime 内聚 |
| RoundResult statuses | `list[str]` | `list[PlayerStatus]` |
| ResolutionResult statuses_granted | `list[str]` | `list[StatusType]` |
| 状态字符串 | `"defense"`, `"联防"` | `StatusType.DEFENSE`, `StatusType.GUARD` |
| 序列化格式 | `["defense"]` | `[{"type":"defense","lifetime":1}]` |
| status_lifetimes 独立序列化 | 有 | 无 |
