# Architecture Design — PlayerState 重构 (intention #15)

## 1. Motivation

当前玩家状态（`"defense"`、`"联防"`）以裸字符串字面量形式分散在代码各处：
- Player 中 `statuses: list[str]`
- GameCore 中 `_status_lifetimes_1/2: list[int]` 作为独立并行数组跟踪生存期
- RoundResult、ResolutionResult 中均为 `list[str]`
- CardResolver 中遍布 `"defense" in statuses`、`"联防" not in statuses` 的硬编码检查
- 前端 `s === '联防'` 直接比较字符串

本次重构引入 `StatusType` 枚举和 `PlayerStatus` 类，将所有状态字符串集中管理，并将生存期封装进 PlayerStatus 内部。

## 2. Key Design Decisions

**D1 — 新增 `StatusType` 枚举。** 定义所有可能的状态类型。当前两种：`DEFENSE`（防御）、`GUARD`（联防）。后续新增状态只需扩展枚举。

**D2 — 新增 `PlayerStatus` 数据类。** 包含 `status_type: StatusType` 和 `lifetime: int`。生存期不再由 GameCore 维护独立数组，而是内聚在状态实例中。

**D3 — Player.statuses 类型变更。** `list[str]` → `list[PlayerStatus]`。`has_status` 参数改为 `StatusType`。

**D4 — 移除 GameCore._status_lifetimes_*。** 生存期已进入 PlayerStatus，不再需要并行数组。`_tick_status_lifetimes` 直接操作 `player.statuses` 中的 PlayerStatus 对象。

**D5 — RoundResult / ResolutionResult 类型变更。** RoundResult 的 statuses 字段改为 `list[PlayerStatus]`；ResolutionResult 的 statuses_granted 改为 `list[StatusType]`（仅类型，不含生存期，生存期由 GameCore 在 play_round 中统一设为 1）。

**D6 — 序列化格式变更。** 持久化和 API 中，PlayerStatus 序列化为 `{"type": "defense", "lifetime": 1}`。移除 `status_lifetimes_1/2` 独立字段。**不保留旧存档兼容性。**

**D7 — 新模块 `game/player_state.py`。** 集中存放 `StatusType` 和 `PlayerStatus`，与 Player 分离以保持职责清晰。

**D8 — CardResolver 改用枚举。** 所有 `"defense"`、`"联防"` 字符串替换为 `StatusType.DEFENSE`、`StatusType.GUARD`。`resolve()` 的 `player_*_statuses` 参数从 `list[str]` 改为 `list[PlayerStatus]`。

## 3. Module Impact

```
src/backend/
├── game/player_state.py       # NEW — StatusType 枚举 + PlayerStatus 数据类
├── game/player.py             # MODIFIED — statuses 类型、has_status 签名
├── game/game_core.py          # MODIFIED — 移除 _status_lifetimes_*, 更新 play_round/get_state
├── game/resolver.py           # MODIFIED — 字符串 → 枚举, 参数类型变更
├── game/session_store.py      # MODIFIED — 序列化/反序列化 PlayerStatus, 移除 status_lifetimes
├── controller/serializers.py  # MODIFIED — RoundResult 序列化适配
└── (frontend) GameBoard.jsx   # MODIFIED — statuses 格式适配
```

## 4. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **状态生命周期流程**: [status_lifecycle_flow.puml](diagrams/status_lifecycle_flow.puml)
