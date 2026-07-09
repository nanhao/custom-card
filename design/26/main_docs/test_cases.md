# Test Cases — 回合与抽牌参数调整 (intention #26)

## 测试范围

- **单元测试**: `tests/unit/test_game_core.py` — cycle 机制 + 抽牌逻辑
- **单元测试**: `tests/unit/test_game_state.py` — GameState 构造与序列化
- **集成测试**: `tests/integration/test_controller_api.py` — API 返回手牌数
- **前端测试**: `src/frontend/tests/App.test.jsx` — 回合显示文本

---

## 1. 初始状态

### TC-INIT-01 — start_game 初始手牌 = 5

**Given**: 调用 `GameCore(mode="pvp").start_game()`

**Then**:
- P1.hand 长度 = 5
- P2.hand 长度 = 5
- round_number = 1, cycle_number = 1, round_in_cycle = 1

---

### TC-INIT-02 — 初始抽牌堆剩余 2 张

**Given**: `GameCore(mode="pvp").start_game()`

**Then**:
- P1.draw_pile 长度 = 2（7 张初始牌 - 5 张抽入手牌）
- P2.draw_pile 长度 = 2

---

### TC-INIT-03 — 初始手牌内容正确

**Given**: `GameCore(mode="pvp").start_game()`

**Then**: P1.hand + P1.draw_pile 的并集等于 `get_initial_draw_pile()` 返回的 7 张牌（2×攻击 + 其余各 1）

---

## 2. 回合内流程

### TC-ROUND-01 — 回合 1-2 无 bleed

**Given**: 设置 `round_in_cycle = 1`，双方手牌 = [attack]，HP = 10

**When**: `play_round([attack], [attack])`

**Then**:
- P1/P2 都承受攻击伤害
- `result.end_actions.cycle_ended` = False
- `result.end_actions.bleed_damage_1` = 0
- 无额外 HP 损失

---

### TC-ROUND-02 — 回合 2（mid-cycle，同 TC-ROUND-01）

**Given**: 设置 `round_in_cycle = 2`

**When**: `play_round`

**Then**: `result.end_actions.cycle_ended` = False

---

## 3. 轮末结算

### TC-CYCLE-01 — 第 3 回合触发 bleed

**Given**: `round_in_cycle = 3`，双方手牌 = [attack]，HP = 10

**When**: `play_round([attack], [attack])`

**Then**:
- `result.end_actions.cycle_ended` = True
- `result.end_actions.bleed_damage_1` = 1
- `result.end_actions.bleed_damage_2` = 1
- P1/P2 HP 先受攻击伤害再扣 bleed = 10 → 攻击伤害 → 再 -1（bleed）
- `result.after_end_actions.cycle` = 2
- `result.after_end_actions.round_in_cycle` = 1

---

### TC-CYCLE-02 — bleed 导致双亡（draw）

**Given**: `round_in_cycle = 3`，双方 hand = [defense]，HP = 1

**When**: `play_round([defense], [defense])`

**Then**:
- `gm.is_over` = True
- `gm.winner` = "draw"
- 双方 HP = 0

---

### TC-CYCLE-03 — bleed 导致单方死亡

**Given**: `round_in_cycle = 3`，P1 hand = [defense], HP = 2；P2 hand = [attack], HP = 1

**When**: `play_round([defense], [attack])`

**Then**:
- P2 受攻击伤害（从 P1）→ HP 变化；再受 bleed 1 → HP = 0（死亡）
- `gm.is_over` = True
- `gm.winner` = "1"

---

## 4. 新轮的抽牌

### TC-DRAW-01 — 新轮抽 5 张

**Given**: `round_in_cycle = 5`（当前 cycle 最后一回合，ROUNDS_PER_CYCLE = 5 时），改为 ROUNDS_PER_CYCLE=3 后模拟第 3 回合：手牌含 2 张、弃牌堆有足够牌

**When**: `play_round` 触发 `_start_cycle()`

**Then**:
- P1.hand 长度 = 5
- P2.hand 长度 = 5

---

### TC-DRAW-02 — 手牌回收后再抽 5 张

**Given**: 模拟 cycle 结束：P1 hand = [attack, defense]，discard_pile = [heavy] × 10

**When**: `_start_cycle()` 执行

**Then**:
- P1 discard_pile = []（清空，已移入 draw_pile）
- P1 hand 长度 = 5
- P1.draw_pile + P1.hand + P1.discard_pile 总和 = 回收的总牌数（2+10=12），hand 中 5 张，draw_pile 中 7 张

---

### TC-DRAW-03 — 抽牌堆不足 5 张时抽全部

**Given**: P1 draw_pile = [attack, defense]，discard_pile = []，hand = []（模拟极端场景）

**When**: `_start_cycle()` 执行

**Then**: P1.hand 长度 = 2（min(5, 2) = 2）

---

### TC-DRAW-04 — 抽牌堆为空时手牌为空

**Given**: P1 draw_pile = []，discard_pile = []，hand = []

**When**: `_start_cycle()` 执行

**Then**: P1.hand 长度 = 0

---

## 5. GameState

### TC-GS-01 — round_in_cycle 取值 1-3

**Given**: 任意 game state

**Then**: `state.round_in_cycle` ∈ {1, 2, 3}

---

### TC-GS-02 — to_dict 序列化包含 round_in_cycle

**Given**: `round_in_cycle = 2` 的 GameState

**When**: `GameState.to_dict(gs)`

**Then**: `result["round_in_cycle"]` = 2

---

## 6. 历史记录不变性

### TC-HIST-01 — PlayerHistoryItem 格式不受影响

**Given**: 任何 play_round 调用

**Then**: `player.history` 中的元素仍然是 `PlayerHistoryItem` 类型，card 和 build_target 字段正确

---

### TC-HIST-02 — 跨轮历史保留

**Given**: 两轮共 6 回合的游戏

**Then**: `player.history` 长度 ≥ 历史回合数，所有条目格式正确

---

## 7. 前端显示

### TC-FE-01 — 选牌阶段显示 "/3"

**Given**: 游戏进行中，`round_in_cycle = 2`

**Then**: UI 中显示 "Round 2/3"

---

### TC-FE-02 — 结算弹窗显示 "/3"

**Given**: 回合结果返回，`rr.round_in_cycle = 1`

**Then**: 结算弹窗中显示 "Round 1/3"

---

## 8. Regression — 未改动的机制

### TC-REG-01 — status lifetime 正常 tick

**Given**: defense card played → status lifetime = 1

**When**: 下一回合

**Then**: status 过期移除（与 ROUNDS_PER_CYCLE 无关）

---

### TC-REG-02 — 能量系统不受影响

**Given**: P1 energy = 2

**When**: play energy card

**Then**: energy +1 = 3（与 cycle 参数无关）

---

### TC-REG-03 — 构建消耗不变

**Given**: P1 手牌含 build，energy = 1

**When**: `play_round([build], [...], build_target_1=attack)`

**Then**: build_cost 按 CardDef 定义扣除，不受 cycle 参数影响
