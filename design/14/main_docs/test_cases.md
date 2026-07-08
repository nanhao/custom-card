# Test Cases — Input/Output Redesign (intention #14)

## 测试范围

- **单元测试**: `tests/unit/test_converter.py` — Converter 15-d input 构造 + action 编解码
- **单元测试**: `tests/unit/test_player.py` — Player.build_history 字段正确性
- **单元测试**: `tests/unit/test_game_core.py` — get_state() 新 history 格式

---

## 1. build_input_vector

### TC-IV-01 — 15-d 索引顺序正确性（player_key="2"，有纯出牌历史）

**Given**: state 中 P2 HP=15, energy=3; P1 HP=8, energy=1; round_in_cycle=3, cycle=2; P2 手牌 [attack, defense, heavy, counter, energy, build, flex]; P2 history=[PlayerHistoryItem(CardType.ATTACK,None)], P1 history=[PlayerHistoryItem(CardType.DEFENSE,None)]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- len = 15
- 索引 0 = 15.0（我方 HP）
- 索引 1 = 8.0（对方 HP）
- 索引 2 = 3.0（我方 energy）
- 索引 3 = 1.0（对方 energy）
- 索引 4 = 3.0（round_in_cycle）
- 索引 5 = 2.0（cycle）
- 索引 6-12 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]（手牌计数）
- 索引 13 = 1.0（attack 对应 policy index 1）
- 索引 14 = 3.0（defense 对应 policy index 3）

---

### TC-IV-02 — 视角互换（player_key="1"）

**Given**: 同 TC-IV-01 的 state

**When**: `Converter.build_input_vector(state, player_key="1")`

**Then**:
- 索引 0 = 8.0（P1 HP）
- 索引 1 = 15.0（P2 HP）
- 索引 2 = 1.0（P1 energy）
- 索引 3 = 3.0（P2 energy）
- 索引 13 = 3.0（P1 上回出了 defense → index 3）
- 索引 14 = 1.0（P2 上回出了 attack → index 1）

---

### TC-IV-03 — 首回合无出牌历史

**Given**: P2 手牌 [attack], P1 手牌 [defense]; 双方 history 为空

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 13 = -1.0（无历史）
- 索引 14 = -1.0（无历史）

---

### TC-IV-04 — 上回合出的是构建类动作

**Given**: P2 history=[PlayerHistoryItem(CardType.BUILD,CardType.ATTACK)], P1 history=[PlayerHistoryItem(CardType.BUILD,CardType.DEFENSE)]（policy index 分别为 2 和 4）

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 13 = 2.0
- 索引 14 = 4.0

---

### TC-IV-05 — 空手牌

**Given**: P2 手牌为空

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 6-12 全部为 0.0

---

### TC-IV-06 — player_key 非法值抛异常

**Given**: 任意合法 state

**When**: `Converter.build_input_vector(state, player_key="3")` 或 `player_key="0"` 或 `player_key=""`

**Then**: 抛出 `ValueError`

---

### TC-IV-07 — HP=0 边界

**Given**: P2 HP=0, P1 HP=1

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 0 = 0.0, 索引 1 = 1.0

---

### TC-IV-08 — 能量满值边界

**Given**: P2 energy=5, P1 energy=0

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 2 = 5.0, 索引 3 = 0.0

---

### TC-IV-09 — 大轮数 / 大回合数

**Given**: cycle=100, round_in_cycle=5

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 4 = 5.0, 索引 5 = 100.0

---

## 2. action_to_policy_index

### TC-AP-01 — 纯出牌映射

**Given**: 各种纯出牌

**When**: `Converter.action_to_policy_index(card_str, None)`

**Then**:

| card_str | 返回 |
|----------|------|
| attack | 1 |
| defense | 3 |
| heavy | 5 |
| counter | 7 |
| energy | 9 |
| flex | 11 |

---

### TC-AP-02 — 出构建映射

**Given**: card_str="build"

**When**: `Converter.action_to_policy_index("build", build_target)`

**Then**:

| build_target | 返回 |
|-------------|------|
| build | 0 |
| attack | 2 |
| defense | 4 |
| heavy | 6 |
| counter | 8 |
| energy | 10 |
| flex | 12 |

---

### TC-AP-03 — 非法 card_str 抛异常

**Given**: card_str="invalid"

**When**: `Converter.action_to_policy_index("invalid", None)`

**Then**: 抛出 `ValueError`

---

### TC-AP-04 — 非 build 时 build_target 不为 None

**Given**: card_str="attack", build_target="defense"（非法组合）

**When**: `Converter.action_to_policy_index("attack", "defense")`

**Then**: 抛出 `ValueError`

---

### TC-AP-05 — build 时 build_target 为 None

**Given**: card_str="build", build_target=None

**When**: `Converter.action_to_policy_index("build", None)`

**Then**: 抛出 `ValueError`

---

## 3. policy_index_to_action

### TC-PI-01 — 奇数索引（纯出牌）

**Given**: index ∈ {1, 3, 5, 7, 9, 11}

**When**: `Converter.policy_index_to_action(index)`

**Then**:
- card_str = 对应卡牌名
- build_target = None

---

### TC-PI-02 — 偶数索引（出构建）

**Given**: index ∈ {0, 2, 4, 6, 8, 10, 12}

**When**: `Converter.policy_index_to_action(index)`

**Then**:
- card_str = "build"
- build_target = 对应卡牌名

---

### TC-PI-03 — 索引越界抛异常

**Given**: index = -1 或 index = 13

**When**: `Converter.policy_index_to_action(index)`

**Then**: 抛出 `IndexError`

---

### TC-PI-04 — 往返一致性

**Given**: 所有合法 (card_str, build_target) 组合

**When**: `policy_index_to_action(action_to_policy_index(card, bt))`

**Then**: 返回值 == (card_str, build_target)

---

## 4. Player.build_history

### TC-BH-01 — 初始状态 build_history 为空

**Given**: 新建 Player 实例

**Then**: `player.build_history` 为 `[]`

---

### TC-BH-02 — play_round 后 build_history 与 history 等长

**Given**: 创建 GameCore，play_round([attack], [defense])

**Then**: P1.history 长度 = P1.build_history 长度 = 1；P2 同理

---

### TC-BH-03 — 非 build 动作记录 null

**Given**: play_round([attack], [defense])

**Then**: P1.build_history[0] = None, P2.build_history[0] = None

---

### TC-BH-04 — build 动作记录构建目标

**Given**: P1 手牌含 build + 能量足够；play_round([build], [attack], build_target_1=attack)

**Then**: P1.build_history[0] = CardType.ATTACK

---

## 5. GameCore.get_state() history 格式

### TC-GS-01 — 空历史返回空列表

**Given**: 新游戏尚未 play_round

**When**: `gc.get_state()`

**Then**: players["1"]["history"] = []

---

### TC-GS-02 — 纯出牌 history 含 build_target=null

**Given**: play_round([attack], [defense])

**When**: `gc.get_state()`

**Then**: players["1"]["history"][-1] = PlayerHistoryItem(CardType.ATTACK, None)

---

### TC-GS-03 — 构建类 history 含 build_target

**Given**: P1 手牌含 build + energy≥1；play_round([build], [attack], build_target_1=attack)

**When**: `gc.get_state()`

**Then**: players["1"]["history"][-1] = PlayerHistoryItem(CardType.BUILD, CardType.ATTACK)

---

### TC-GS-04 — 多轮累计正确

**Given**: 连续 play_round 3 次，分别出 attack、defense、build+attack

**When**: `gc.get_state()`

**Then**: players["1"]["history"] 长度为 3，对应 card 和 build_target 均正确
