# Test Cases — RoundResult 字段层次化重组 (intention #17)

## 测试范围

- **单元测试**: `tests/unit/test_game_core.py` — RoundResult 新结构验证、before/after_resolve/end_actions/after_end_actions 各阶段数据正确性
- **前端测试**: `src/frontend/tests/App.test.jsx` — mock 数据适配新格式

---

## 1. RoundResult — 嵌套结构基本验证

### TC-RR-01 — RoundResult 包含所有嵌套对象

**Given**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.before` 类型为 `BeforeResolve`，无默认值
- `result.after_resolve` 类型为 `AfterResolve`，无默认值
- `result.end_actions` 类型为 `EndActions`，无默认值
- `result.after_end_actions` 类型为 `AfterEndActions`，无默认值

---

### TC-RR-02 — 顶层字段正确

**Given**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.round == 1`
- `result.cycle == 1`
- `result.round_in_cycle == 1`
- `result.cards_1 == [CardType.ATTACK]`
- `result.cards_2 == [CardType.DEFENSE]`

---

### TC-RR-03 — 旧扁平字段全部消失

**Given**: play_round([ATTACK], [DEFENSE])

**Then**: 访问以下任一旧字段均抛出 `AttributeError`：
- `result.damage_1`
- `result.damage_2`
- `result.hp_1_after`
- `result.hp_2_after`
- `result.energy_1_after`
- `result.energy_2_after`
- `result.statuses_1`
- `result.statuses_2`
- `result.details`
- `result.is_game_over`
- `result.winner`
- `result.cycle_ended`
- `result.bleed_damage_1`
- `result.bleed_damage_2`
- `result.after_cycle`
- `result.after_round_in_cycle`
- `result.after_hp_1`
- `result.after_hp_2`
- `result.after_statuses_1`
- `result.after_statuses_2`

---

## 2. before — 结算前快照

### TC-BF-01 — before 记录结算前 HP

**Given**: P1 HP=10, P2 HP=5，双方手牌 [ATTACK]、[DEFENSE]

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.before.hp_1 == 10`
- `result.before.hp_2 == 5`

---

### TC-BF-02 — before 记录结算前能量

**Given**: P1 能量=2, P2 能量=0，手牌 [ATTACK]、[DEFENSE]

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.before.energy_1 == 2`
- `result.before.energy_2 == 0`

---

### TC-BF-03 — before 记录结算前状态（防御拷贝）

**Given**: P2 有 DEFENSE 状态（lifetime=1），手牌 [ATTACK]、[ATTACK]

**When**: play_round([ATTACK], [ATTACK])

**Then**:
- `len(result.before.statuses_2) == 1`
- `result.before.statuses_2[0].status_type == StatusType.DEFENSE`
- `len(result.before.statuses_1) == 0`

---

### TC-BF-04 — before 是防御性拷贝，不随 Player 状态变更而变化

**Given**: 采集 before 后 Player 状态被修改

**Then**: `result.before` 中的 statuses 是独立副本，不受后续状态变更影响

---

## 3. after_resolve — 结算后快照

### TC-AR-01 — after_resolve 记录结算后 HP

**Given**: P1 HP=10, P2 HP=5；ATTACK vs DEFENSE（P2 无伤）

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.after_resolve.hp_1 == 10`（未受伤）
- `result.after_resolve.hp_2 == 5`（DEFENSE 阻挡）

---

### TC-AR-02 — after_resolve 记录结算后能量

**Given**: P1 能量=1, P2 能量=0；BUILD(target=ATTACK) vs ATTACK

**When**: play_round([BUILD], [ATTACK], build_target_1=ATTACK)

**Then**:
- `result.after_resolve.energy_1 == 0`（消耗 1 能量）
- `result.after_resolve.energy_2 == 0`

---

### TC-AR-03 — after_resolve 记录结算后的 statuses

**Given**: P1 DEFENSE vs P2 ATTACK（P1 获 DEFENSE）

**When**: play_round([DEFENSE], [ATTACK])

**Then**:
- `len(result.after_resolve.statuses_1) == 1`
- `result.after_resolve.statuses_1[0].status_type == StatusType.DEFENSE`
- `result.after_resolve.statuses_1[0].lifetime == 1`

---

### TC-AR-04 — after_resolve 记录 details

**Given**: ATTACK vs DEFENSE

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `len(result.after_resolve.details) >= 1`
- details 包含描述攻击被阻挡的文本

---

### TC-AR-05 — after_resolve 结算伤害致死 → is_game_over=True

**Given**: P2 HP=1，手牌 ATTACK vs ENERGY（不阻挡）

**When**: play_round([ATTACK], [ENERGY])

**Then**:
- `result.after_resolve.is_game_over == True`
- `result.after_resolve.winner == "1"`
- `result.after_resolve.hp_2 == 0`

---

### TC-AR-06 — after_resolve 非致死 → is_game_over=False

**Given**: 双方 HP=20，ATTACK vs DEFENSE

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.after_resolve.is_game_over == False`
- `result.after_resolve.winner is None`

---

## 4. end_actions — 回合结束动作

### TC-EA-01 — 非周期结束回合 → cycle_ended=False

**Given**: round_in_cycle=3（非第 5 回合），ATTACK vs DEFENSE

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.end_actions.cycle_ended == False`
- `result.end_actions.bleed_damage_1 == 0`
- `result.end_actions.bleed_damage_2 == 0`

---

### TC-EA-02 — 周期第 5 回合 → cycle_ended=True，bleed=1

**Given**: round_in_cycle=5, P1/P2 HP=10，ATTACK vs DEFENSE

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.end_actions.cycle_ended == True`
- `result.end_actions.bleed_damage_1 == 1`
- `result.end_actions.bleed_damage_2 == 1`

---

## 5. after_end_actions — 最终状态

### TC-AE-01 — after_end_actions 记录推进后的 round 计数器

**Given**: 第 1 回合，非周期结束

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.after_end_actions.round == 2`
- `result.after_end_actions.round_in_cycle == 2`

---

### TC-AE-02 — after_end_actions 周期结束时计数器正确推进

**Given**: round_in_cycle=5（最后回合），round=5

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.after_end_actions.round == 6`
- `result.after_end_actions.cycle == 2`（新周期）
- `result.after_end_actions.round_in_cycle == 1`

---

### TC-AE-03 — after_end_actions 记录最终 HP（含流血）

**Given**: round_in_cycle=5, P1 HP=10, P2 HP=10；ATTACK vs ATTACK（各受 1 伤 + 1 流血）

**When**: play_round([ATTACK], [ATTACK])

**Then**:
- `result.after_end_actions.hp_1 == 8`（10 - 1 攻击 - 1 流血）
- `result.after_end_actions.hp_2 == 8`

---

### TC-AE-04 — after_end_actions 记录最终 energy

**Given**: P1 能量=0, P2 能量=0；ENERGY vs ATTACK

**When**: play_round([ENERGY], [ATTACK])

**Then**:
- `result.after_end_actions.energy_1 == 1`（ENERGY +1）

---

### TC-AE-05 — after_end_actions 记录最终 statuses（tick 后）

**Given**: P1 DEFENSE vs P2 ATTACK（P1 获 DEFENSE lifetime=1）

**When**: play_round([DEFENSE], [ATTACK])

**Then**:
- `result.after_end_actions.statuses_1 == []` （lifetime=1 被 tick 移除）

---

### TC-AE-06 — after_end_actions 记录新周期手牌

**Given**: round_in_cycle=5

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `len(result.after_end_actions.hand_cards_1) == 7`（新周期抽 7 张）
- `len(result.after_end_actions.hand_cards_2) == 7`

---

### TC-AE-07 — after_end_actions 非周期结束时 hand_cards 为空

**Given**: round_in_cycle=3（非周期结束）

**When**: play_round([ATTACK], [DEFENSE])

**Then**:
- `result.after_end_actions.hand_cards_1 == []`（非周期结束不抽新手牌）
- `result.after_end_actions.hand_cards_2 == []`

---

### TC-AE-08 — after_end_actions 流血致死 → is_game_over=True

**Given**: round_in_cycle=5, P1 HP=1, P2 HP=1；DEFENSE vs DEFENSE（双方无伤，但流血 1 致死）

**When**: play_round([DEFENSE], [DEFENSE])

**Then**:
- `result.after_resolve.is_game_over == False`（结算时未死）
- `result.after_end_actions.is_game_over == True`
- `result.after_end_actions.winner == "draw"`

---

### TC-AE-09 — 结算时已致死 → after_end_actions 复用判定

**Given**: P2 HP=1；ATTACK vs ENERGY

**When**: play_round([ATTACK], [ENERGY])

**Then**:
- `result.after_resolve.is_game_over == True`
- `result.after_end_actions.is_game_over == True`
- `result.after_end_actions.winner == "1"`

---

## 6. round_result_to_dict — 序列化验证

### TC-SD-01 — 顶层字段序列化

**Given**: play_round([ATTACK], [DEFENSE])

**When**: `round_result_to_dict(result)`

**Then**:
- `d["round"] == 1`
- `d["cards_1"] == ["attack"]`
- `d["cards_2"] == ["defense"]`

---

### TC-SD-02 — before 序列化

**Given**: play_round([ATTACK], [DEFENSE])

**When**: `round_result_to_dict(result)`

**Then**:
- `d["before"]["hp_1"]` 为 int
- `d["before"]["statuses_1"]` 为 list
- `d["before"]["energy_1"]` 为 int

---

### TC-SD-03 — after_resolve 序列化包含 is_game_over

**Given**: play_round([ATTACK], [ENERGY])（P2 HP=1，致死）

**When**: `round_result_to_dict(result)`

**Then**:
- `d["after_resolve"]["is_game_over"] == True`
- `d["after_resolve"]["winner"] == "1"`

---

### TC-SD-04 — end_actions 序列化

**Given**: play_round([ATTACK], [DEFENSE])（非周期结束）

**When**: `round_result_to_dict(result)`

**Then**:
- `d["end_actions"]["cycle_ended"] == False`
- `d["end_actions"]["bleed_damage_1"] == 0`

---

### TC-SD-05 — after_end_actions 序列化包含 hand_cards

**Given**: round_in_cycle=5 的 play_round

**When**: `round_result_to_dict(result)`

**Then**:
- `d["after_end_actions"]["hand_cards_1"]` 为 list[str]，长度 7
- `d["after_end_actions"]["hand_cards_2"]` 为 list[str]，长度 7

---

## 7. get_state — round_log 变更

### TC-GS-01 — get_state 返回的 round_log 为新嵌套格式

**Given**: play_round([ATTACK], [DEFENSE])

**When**: `gm.get_state()["round_log"]`

**Then**:
- `round_log[0]` 包含 `"before"`、`"after_resolve"`、`"end_actions"`、`"after_end_actions"` 键
- `round_log[0]` 不包含旧扁平字段（如 `"hp_1_after"`、`"is_game_over"`）

---

## 8. 被删除的测试用例

旧测试用例中直接断言 RoundResult 扁平字段的部分，需适配为新嵌套结构：

- `test_game_core.py::TestCycleMechanics::test_cycle_end_bleed` — `result.hp_1_after` → `result.after_resolve.hp_1`
- `test_game_core.py::TestCycleMechanics::test_cycle_end_bleed` — `result.after_hp_1` → `result.after_end_actions.hp_1`
- `test_game_core.py::TestCycleMechanics::test_cycle_end_bleed` — `result.cycle_ended` → `result.end_actions.cycle_ended`
- `test_game_core.py::TestCycleMechanics::test_cycle_end_bleed` — `result.bleed_damage_1` → `result.end_actions.bleed_damage_1`
- `test_game_core.py::TestCycleMechanics::test_cycle_end_bleed` — `result.after_cycle` → `result.after_end_actions.cycle`
- `test_game_core.py::TestStatusLifecycle` 中所有断言 `result.statuses_1` → `result.after_resolve.statuses_1`
- `test_game_core.py::TestStatusLifecycle` 中所有断言 `result.after_statuses_1` → `result.after_end_actions.statuses_1`

前端 `App.test.jsx` 中 mock 的 `round_result` 对象需改为新嵌套格式。
