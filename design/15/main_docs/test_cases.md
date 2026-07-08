# Test Cases — PlayerState 重构 (intention #15)

## 测试范围

- **单元测试**: `tests/unit/test_player_state.py` — StatusType / PlayerStatus 基础行为
- **单元测试**: `tests/unit/test_player.py` — has_status 签名变更
- **单元测试**: `tests/unit/test_game_core.py` — _tick_status_lifetimes + play_round 状态变更
- **单元测试**: `tests/unit/test_resolver.py` — CardResolver 使用枚举
- **单元测试**: `tests/unit/test_converter.py` — 确认不受影响

---

## 1. StatusType 枚举

### TC-ST-01 — 枚举成员存在

**Given**: 导入 `StatusType`

**Then**: `StatusType.DEFENSE` 和 `StatusType.GUARD` 存在，`.value` 分别为 `"defense"` 和 `"guard"`

---

### TC-ST-02 — StatusType 从字符串构造

**Given**: `StatusType("defense")` 和 `StatusType("guard")`

**Then**: 分别返回 `StatusType.DEFENSE` 和 `StatusType.GUARD`

---

### TC-ST-03 — 非法字符串抛异常

**Given**: `StatusType("invalid")`

**Then**: 抛出 `ValueError`

---

## 2. PlayerStatus 数据类

### TC-PS-01 — 构造与字段访问

**Given**: `PlayerStatus(status_type=StatusType.DEFENSE, lifetime=1)`

**Then**:
- `ps.status_type == StatusType.DEFENSE`
- `ps.lifetime == 1`

---

### TC-PS-02 — default_factory 返回空列表

**Given**: `PlayerStatus` 无默认构造参数（无 default_factory）

**Then**: 所有字段必须显式赋值

---

### TC-PS-03 — 相等性比较

**Given**: `PlayerStatus(StatusType.DEFENSE, 1)` 和另一个相同值的 `PlayerStatus(StatusType.DEFENSE, 1)`

**Then**: 两者相等（dataclass 值相等）

---

### TC-PS-04 — lifetime 边界值 0

**Given**: `PlayerStatus(StatusType.GUARD, lifetime=0)`

**Then**: `.lifetime == 0`（表示已过期，将在下一轮 tick 时被移除）

---

## 3. Player.has_status

### TC-HS-01 — 拥有指定状态返回 True

**Given**: `player.statuses = [PlayerStatus(StatusType.DEFENSE, 1)]`

**When**: `player.has_status(StatusType.DEFENSE)`

**Then**: `True`

---

### TC-HS-02 — 不拥有指定状态返回 False

**Given**: `player.statuses = [PlayerStatus(StatusType.DEFENSE, 1)]`

**When**: `player.has_status(StatusType.GUARD)`

**Then**: `False`

---

### TC-HS-03 — 状态列表为空返回 False

**Given**: `player.statuses = []`

**When**: `player.has_status(StatusType.DEFENSE)`

**Then**: `False`

---

### TC-HS-04 — lifetime=0 的状态视为不拥有

**Given**: `player.statuses = [PlayerStatus(StatusType.DEFENSE, 0)]`

**When**: `player.has_status(StatusType.DEFENSE)`

**Then**: `False`（lifetime=0 表示已过期）

---

### TC-HS-05 — 同时拥有多个状态仍能正确检查

**Given**: `player.statuses = [PlayerStatus(StatusType.DEFENSE, 1), PlayerStatus(StatusType.GUARD, 1)]`

**When**: `player.has_status(StatusType.DEFENSE)` 和 `player.has_status(StatusType.GUARD)`

**Then**: 两者均为 `True`

---

## 4. GameCore — 状态生命周期

### TC-GS-01 — play_round 后 statuses 为 PlayerStatus 列表

**Given**: GameCore 启动，P1 手牌 [ATTACK]，P2 手牌 [DEFENSE]

**When**: `play_round([ATTACK], [DEFENSE])`

**Then**:
- `gm.player_2.statuses` 是 `list[PlayerStatus]`
- `gm.player_2.statuses[0].status_type == StatusType.DEFENSE`
- `gm.player_2.statuses[0].lifetime == 1`

---

### TC-GS-02 — _tick_status_lifetimes 减少 lifetime 并移除过期状态

**Given**: player 的 `statuses = [PlayerStatus(DEFENSE, 2), PlayerStatus(GUARD, 1)]`

**When**: `_tick_status_lifetimes()` 执行

**Then**:
- DEFENSE 的 lifetime 从 2 → 1，保留
- GUARD 的 lifetime 从 1 → 0，被移除
- statuses = `[PlayerStatus(DEFENSE, 1)]`

---

### TC-GS-03 — 单状态 lifetime=1 在 tick 后被移除

**Given**: player 的 `statuses = [PlayerStatus(DEFENSE, 1)]`

**When**: `_tick_status_lifetimes()` 执行

**Then**: statuses = `[]`

---

### TC-GS-04 — 状态不跨回合保留（每回合需重新授予）

**Given**: Round 1: P1 出 ATTACK，P2 出 DEFENSE（P2 获得 DEFENSE）

**When**: Round 2: P1 出 ATTACK，P2 出 ATTACK

**Then**: Round 2 结束时 P2.statuses 不包含 DEFENSE（因为新回合未被重新授予）

---

### TC-GS-05 — 移除 _status_lifetimes_* 后 GameCore 能正常工作

**Given**: GameCore 实例

**Then**: `gm._status_lifetimes_1` 和 `gm._status_lifetimes_2` 不存在（AttributeError）

---

### TC-GS-06 — RoundResult 包含 PlayerStatus 对象

**Given**: P1 出 DEFENSE，P2 出 ATTACK

**When**: `play_round([DEFENSE], [ATTACK])`

**Then**:
- `result.statuses_1[0]` 是 `PlayerStatus` 实例
- `result.statuses_1[0].status_type == StatusType.DEFENSE`
- `result.statuses_1[0].lifetime == 1`

---

### TC-GS-07 — 联防在上一轮有防御时被授予

**Given**: Round 1: P1 出 DEFENSE（获得 DEFENSE）；Round 2：P1 出 DEFENSE（获得 联防）

**When**: Round 2: P1 出 DEFENSE，P2 出 ATTACK

**Then**:
- `result.statuses_1` 包含 `PlayerStatus(StatusType.GUARD, 1)`
- 同时包含 `PlayerStatus(StatusType.DEFENSE, 1)`（DEFENSE 被重新授予）

---

### TC-GS-08 — 联防在上一轮无防御时不被授予

**Given**: Round 1: P1 出 ATTACK（无 DEFENSE）；Round 2：P1 出 DEFENSE

**When**: Round 2: P1 出 DEFENSE，P2 出 ATTACK

**Then**:
- `result.statuses_1` 包含 `PlayerStatus(StatusType.DEFENSE, 1)`
- `result.statuses_1` 不包含 GUARD 类型

---

## 5. CardResolver — 枚举替换

### TC-CR-01 — grant_defense 使用 StatusType.DEFENSE

**Given**: card1=DEFENSE, card2=ATTACK

**When**: `CardResolver.resolve([DEFENSE], [ATTACK], [], [], 0, 0)`

**Then**: `result.statuses_granted_1` 包含 `StatusType.DEFENSE`

---

### TC-CR-02 — grant_guard 使用 StatusType.GUARD

**Given**: card1=DEFENSE, card2=ATTACK, player_1_statuses 包含 DEFENSE（表示上轮有防御）

**When**: `CardResolver.resolve([DEFENSE], [ATTACK], [PlayerStatus(DEFENSE,1)], [], 0, 0)`

**Then**: `result.statuses_granted_1` 包含 `StatusType.GUARD`

---

### TC-CR-03 — ATTACK 伤害检查用 StatusType.DEFENSE 判断阻挡

**Given**: card1=ATTACK, card2=DEFENSE

**When**: `CardResolver.resolve([ATTACK], [DEFENSE], [], [], 0, 0)`

**Then**: P1 造成的伤害被阻挡（`result.damage_2 == 0`），因为 P2 的 DEFENSE 在优先级 11 先生效

---

### TC-CR-04 — HEAVY 伤害检查用 StatusType.GUARD 判断阻挡

**Given**: card1=HEAVY, card2=DEFENSE（P2 上轮有防御，本轮也出防御 → 获得 联防）

**When**: `CardResolver.resolve([HEAVY], [DEFENSE], [], [PlayerStatus(DEFENSE,1)], 0, 0)`

**Then**: P1 的 HEAVY 伤害被联防阻挡（`result.damage_2 == 0`）

---

### TC-CR-05 — FLEX 伤害检查用 StatusType.DEFENSE 判断阻挡

**Given**: card1=FLEX, card2=DEFENSE

**When**: `CardResolver.resolve([FLEX], [DEFENSE], [], [], 0, 0)`

**Then**: P1 的 FLEX 伤害被阻挡

---

## 6. 序列化 / 反序列化

### TC-SE-01 — PlayerStatus 序列化格式

**Given**: `PlayerStatus(StatusType.DEFENSE, 1)`

**When**: 序列化

**Then**: JSON 为 `{"type": "defense", "lifetime": 1}`

---

### TC-SE-02 — PlayerStatus 反序列化

**Given**: JSON `{"type": "defense", "lifetime": 1}`

**When**: 反序列化为 PlayerStatus

**Then**: `PlayerStatus(StatusType.DEFENSE, 1)`

---

### TC-SE-03 — GUARD 状态序列化

**Given**: `PlayerStatus(StatusType.GUARD, 2)`

**When**: 序列化

**Then**: JSON 为 `{"type": "guard", "lifetime": 2}`

---

### TC-SE-04 — 存档不包含 status_lifetimes 字段

**Given**: play_round 后保存游戏

**When**: 读取存档 JSON

**Then**: 存档顶层无 `status_lifetimes_1` 和 `status_lifetimes_2` 字段

---

### TC-SE-05 — 存档中 statuses 格式正确

**Given**: play_round(DEFENSE, ATTACK) 后保存

**When**: 读取存档 JSON 中 `player_1.statuses`

**Then**: 为 `[{"type": "defense", "lifetime": 1}]`

---

### TC-SE-06 — 保存后加载 statuses 恢复正确

**Given**: play_round(DEFENSE, ATTACK) → 保存 → 加载

**When**: 加载后 `gm.player_1.statuses`

**Then**: `[PlayerStatus(StatusType.DEFENSE, 1)]`

---

### TC-SE-07 — RoundResult 序列化后 statuses 为字典列表

**Given**: play_round(DEFENSE, ATTACK)

**When**: `round_result_to_dict(result)["statuses_1"]`

**Then**: `[{"type": "defense", "lifetime": 1}]`

---

### TC-SE-08 — get_state 返回的 statuses 格式

**Given**: play_round(DEFENSE, ATTACK)

**When**: `gm.get_state()["players"]["1"]["statuses"]`

**Then**: `[{"type": "defense", "lifetime": 1}]`

---

## 7. Converter — 确认不受影响

### TC-CV-01 — build_input_vector 不受 statuses 变更影响

**Given**: 重构后的 game state（statuses 格式为 `[{"type": "defense", "lifetime": 1}]`）

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 返回 15-d 向量，索引 0-14 的取值逻辑不变（statuses 不在 input vector 中）

---

## 8. Frontend — 状态渲染

### TC-FE-01 — 防御状态渲染

**Given**: state.players["1"].statuses = `[{"type": "defense", "lifetime": 1}]`

**Then**: 渲染 "defense" 标签，className 为 `tag-defense`

---

### TC-FE-02 — 联防状态渲染

**Given**: state.players["1"].statuses = `[{"type": "guard", "lifetime": 1}]`

**Then**: 渲染 "guard" 标签，className 为 `tag-guard`

---

### TC-FE-03 — 无状态时不渲染标签

**Given**: state.players["1"].statuses = `[]`

**Then**: 不渲染任何 status-tag 元素
