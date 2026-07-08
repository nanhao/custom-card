# Test Cases — Player 历史记录合并 (intention #16)

## 测试范围

- **单元测试**: `tests/unit/test_player.py` — Player.history 类型变更、build_history 删除、PlayerHistoryItem 行为不变
- **单元测试**: `tests/unit/test_game_core.py` — play_round 历史记录、get_state 格式、攻击连击跨周期保留
- **单元测试**: `tests/unit/test_resolver.py` — get_consecutive_attack 适配
- **单元测试**: `tests/unit/test_converter.py` — 确认不受影响

---

## 1. Player — history 类型变更和 build_history 删除

### TC-PL-H01 — 新 Player 的 history 为空列表

**Given**: `Player()`

**Then**: `p.history == []`

---

### TC-PL-H02 — history 元素类型为 PlayerHistoryItem

**Given**: `Player(history=[PlayerHistoryItem(CardType.ATTACK, None)])`

**Then**:
- `len(p.history) == 1`
- `isinstance(p.history[0], PlayerHistoryItem) == True`
- `p.history[0].card == CardType.ATTACK`
- `p.history[0].build_target is None`

---

### TC-PL-H03 — Player 没有 build_history 属性

**Given**: `Player()`

**Then**: 访问 `p.build_history` 抛出 `AttributeError`

---

### TC-PL-H04 — PlayerHistoryItem 构造与字段访问

**Given**: `PlayerHistoryItem(card=CardType.BUILD, build_target=CardType.ATTACK)`

**Then**:
- `item.card == CardType.BUILD`
- `item.build_target == CardType.ATTACK`

---

### TC-PL-H05 — PlayerHistoryItem 默认 build_target 为 None

**Given**: `PlayerHistoryItem(card=CardType.ATTACK)`

**Then**: `item.build_target is None`

---

### TC-PL-H06 — PlayerHistoryItem 相等性比较

**Given**: `PlayerHistoryItem(CardType.ATTACK, None)` 和另一个相同值的 `PlayerHistoryItem(CardType.ATTACK, None)`

**Then**: 两者相等

---

### TC-PL-H07 — PlayerHistoryItem 不等比较

**Given**: `PlayerHistoryItem(CardType.ATTACK, None)` 和 `PlayerHistoryItem(CardType.DEFENSE, None)`

**Then**: 两者不相等

---

## 2. GameCore — play_round 历史记录

### TC-GC-H01 — 非 BUILD 卡牌记录到 history

**Given**: GameCore 启动，P1 手牌 [ATTACK]，P2 手牌 [DEFENSE]

**When**: `play_round([ATTACK], [DEFENSE])`

**Then**:
- `len(gm.player_1.history) == 1`
- `gm.player_1.history[0] == PlayerHistoryItem(CardType.ATTACK, None)`
- `len(gm.player_2.history) == 1`
- `gm.player_2.history[0] == PlayerHistoryItem(CardType.DEFENSE, None)`

---

### TC-GC-H02 — BUILD 卡牌记录 build_target 到 history

**Given**: GameCore 启动，P1 手牌 [BUILD]，P1 能量充足，P2 手牌 [ATTACK]

**When**: `play_round([BUILD], [ATTACK], build_target_1=ATTACK)`

**Then**:
- `gm.player_1.history[0] == PlayerHistoryItem(CardType.BUILD, CardType.ATTACK)`
- `gm.player_2.history[0] == PlayerHistoryItem(CardType.ATTACK, None)`

---

### TC-GC-H03 — 多回合 history 累积

**Given**: 连续三个回合，P1 出 [ATTACK, BUILD(target=HEAVY), DEFENSE]，P2 出 [DEFENSE, ATTACK, ATTACK]

**When**: 每回合调用 `play_round()`

**Then**:
- `gm.player_1.history == [PlayerHistoryItem(ATTACK, None), PlayerHistoryItem(BUILD, HEAVY), PlayerHistoryItem(DEFENSE, None)]`
- `gm.player_2.history == [PlayerHistoryItem(DEFENSE, None), PlayerHistoryItem(ATTACK, None), PlayerHistoryItem(ATTACK, None)]`

---

### TC-GC-H04 — 刷新周期后 history 保留

**Given**: Round 4: P1 连续出 ATTACK 两次，P2 出 DEFENSE

**When**: Round 5 触发新周期（round_in_cycle 从 5 重置到 1）

**Then**: `gm.player_1.history` 仍保留所有历史条目（通过 `item.card` 可访问）

---

## 3. GameCore — get_state 格式

### TC-GS-H01 — get_state history 条目为字典

**Given**: play_round([ATTACK], [DEFENSE])

**When**: `gm.get_state()["players"]["1"]["history"]`

**Then**:
- `history[0]["card"] == "attack"`
- `history[0]["build_target"] is None`

---

### TC-GS-H02 — get_state BUILD 记录包含 build_target

**Given**: play_round([BUILD], [ATTACK], build_target_1=ATTACK)

**When**: `gm.get_state()["players"]["1"]["history"]`

**Then**:
- `history[0]["card"] == "build"`
- `history[0]["build_target"] == "attack"`

---

### TC-GS-H03 — get_state 无玩家视角时返回对手 count（非历史内容）

**Given**: play_round([ATTACK], [DEFENSE])

**When**: `gm.get_state()` 不传入 player_perspective

**Then**: 对手的 history 格式不变（仍为字典列表格式），`draw_pile_count` 和 `discard_pile_count` 展示计数而非内容

---

## 4. CardResolver — get_consecutive_attack 适配

### TC-CA-H01 — 空历史返回 0

**Given**: `history: list[PlayerHistoryItem] = []`

**When**: `CardResolver.get_consecutive_attack(history)`

**Then**: `0`

---

### TC-CA-H02 — 末尾一个 ATTACK 返回 1

**Given**: `history = [PlayerHistoryItem(CardType.ATTACK, None)]`

**When**: `CardResolver.get_consecutive_attack(history)`

**Then**: `1`

---

### TC-CA-H03 — 连续两个 ATTACK 返回 2

**Given**: `history = [PlayerHistoryItem(ATTACK, None), PlayerHistoryItem(ATTACK, None)]`

**When**: `CardResolver.get_consecutive_attack(history)`

**Then**: `2`

---

### TC-CA-H04 — 末尾为 DEFENSE 中断攻击连击

**Given**: `history = [PlayerHistoryItem(ATTACK, None), PlayerHistoryItem(DEFENSE, None)]`

**When**: `CardResolver.get_consecutive_attack(history)`

**Then**: `0`（末尾不是 ATTACK）

---

### TC-CA-H05 — DEFENSE 后面的 ATTACK 重新计数

**Given**: `history = [PlayerHistoryItem(DEFENSE, None), PlayerHistoryItem(ATTACK, None), PlayerHistoryItem(ATTACK, None)]`

**When**: `CardResolver.get_consecutive_attack(history)`

**Then**: `2`

---

### TC-CA-H06 — BUILD 中断攻击连击

**Given**: `history = [PlayerHistoryItem(ATTACK, None), PlayerHistoryItem(BUILD, ATTACK)]`

**When**: `CardResolver.get_consecutive_attack(history)`

**Then**: `0`（BUILD 不是 ATTACK）

---

### TC-CA-H07 — 连续三个 ATTACK 返回 3

**Given**: `history = [PlayerHistoryItem(ATTACK, None), PlayerHistoryItem(ATTACK, None), PlayerHistoryItem(ATTACK, None)]`

**When**: `CardResolver.get_consecutive_attack(history)`

**Then**: `3`

---

## 5. SessionStore — 序列化/反序列化

### TC-SE-H01 — 序列化后无 build_history 字段

**Given**: play_round([ATTACK], [DEFENSE]) 后保存游戏

**When**: 读取存档 JSON 中 `player_1`

**Then**: 存档的 player 对象中无 `build_history` 键

---

### TC-SE-H02 — history 序列化格式

**Given**: play_round([ATTACK], [DEFENSE]) 后保存游戏

**When**: 读取存档 JSON 中 `player_1.history`

**Then**: `[{"card": "attack", "build_target": null}]`

---

### TC-SE-H03 — BUILD history 序列化包含 build_target

**Given**: play_round([BUILD], [ATTACK], build_target_1=COUNTER) 后保存

**When**: 读取存档 JSON 中 `player_1.history`

**Then**: `[{"card": "build", "build_target": "counter"}]`

---

### TC-SE-H04 — 非 BUILD history 序列化 build_target 为 null

**Given**: play_round([FLEX], [HEAVY]) 后保存

**When**: 读取存档 JSON 中 `player_1.history[0]`

**Then**: `{"card": "flex", "build_target": null}`

---

### TC-SE-H05 — 保存后加载 history 恢复正确

**Given**: 连续两回合 play_round([ATTACK], [DEFENSE]) 和 play_round([BUILD], [ATTACK], ATTACK)

**When**: 保存 → 加载

**Then**:
- `gm.player_1.history[0] == PlayerHistoryItem(CardType.ATTACK, None)`
- `gm.player_1.history[1] == PlayerHistoryItem(CardType.BUILD, CardType.ATTACK)`

---

### TC-SE-H06 — 多回合 history 保存和恢复

**Given**: 三个回合不同操作后保存

**When**: 加载游戏

**Then**: history 包含所有三个回合的完整 PlayerHistoryItem，且顺序不变

---

## 6. Converter — 确认不受影响

### TC-CV-H01 — build_input_vector 使用 get_state 的 history

**Given**: play_round([ATTACK], [DEFENSE]) → get_state()

**When**: `Converter.build_input_vector(state, player_key="1")`

**Then**: 返回 15-d 向量，索引 13（own last action）对应 ATTACK 的 policy index

---

### TC-CV-H02 — _action_item_to_index 正常处理 PlayerHistoryItem

**Given**: `PlayerHistoryItem(CardType.BUILD, CardType.ATTACK)`

**When**: `Converter._action_item_to_index(item)`

**Then**: 返回 BUILD+ATTACK 对应的 policy index

---

## 7. 被删除的测试用例

以下旧测试用例与 `build_history` 相关，应被移除：

- `test_player.py` 中的 `TestBuildHistory` 类（TC-BH-01 ~ TC-BH-04）
- `test_game_core.py` 中直接断言 `gm.player_*.build_history` 的用例

**替代策略**：原来验证 build_history 行为的测试用例，应改为验证 `PlayerHistoryItem.build_target` 字段。
