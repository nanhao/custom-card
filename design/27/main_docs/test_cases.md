# Test Cases — 42维输入 (intention #27)

## 测试范围

- **单元测试**: `tests/unit/test_converter.py` — Converter 42-d input 构造
- **单元测试**: `tests/unit/test_model.py` — INPUT_SIZE 常量验证

---

## 1. build_input_vector — 42-d 向量构造

### TC-IV-01 — 42维长度正确性

**Given**: 正常 GameState（P2 视角，双方手牌各含 attack+defense，各有 history）

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 返回值长度 = 42

---

### TC-IV-02 — 基础元数据段（索引 0-5）

**Given**: P2 HP=15, energy=3; P1 HP=8, energy=1; round_in_cycle=2, cycle=3

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 0 = 15.0（我方 HP）
- 索引 1 = 8.0（对方 HP）
- 索引 2 = 3.0（我方 energy）
- 索引 3 = 1.0（对方 energy）
- 索引 4 = 2.0（round_in_cycle）
- 索引 5 = 3.0（cycle）

---

### TC-IV-03 — 我方手牌计数段（索引 6-12）

**Given**: P2 手牌 [attack, attack, defense, heavy, counter, energy, build, flex]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 6 = 2.0（attack×2）
- 索引 7 = 1.0（defense）
- 索引 8 = 1.0（heavy）
- 索引 9 = 1.0（counter）
- 索引 10 = 1.0（energy）
- 索引 11 = 1.0（build）
- 索引 12 = 1.0（flex）

---

### TC-IV-04 — 对方手牌计数段（索引 13-19）

**Given**: P2 视角，P1 手牌 [defense, defense, heavy]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 13 = 0.0（attack）
- 索引 14 = 2.0（defense×2）
- 索引 15 = 1.0（heavy）
- 索引 16-19 = 0.0

---

### TC-IV-05 — 视角互换后手牌段对调

**Given**: P1 手牌 [attack], P2 手牌 [defense]

**When**: `Converter.build_input_vector(state, player_key="1")`

**Then**:
- 索引 6 = 1.0（我方=P1，attack）
- 索引 7-12 = 0.0
- 索引 13 = 0.0
- 索引 14 = 1.0（对方=P2，defense）

---

### TC-IV-06 — 我方状态持续回合段（索引 20-22）

**Given**: P2 statuses 含 PlayerStatus(DEFENSE, lifetime=2), PlayerStatus(FRAGILE, lifetime=1)；无 GUARD

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 20 = 2.0（DEFENSE）
- 索引 21 = 0.0（GUARD 无）
- 索引 22 = 1.0（FRAGILE）

---

### TC-IV-07 — 对方状态持续回合段（索引 23-25）

**Given**: P1 statuses 含 PlayerStatus(GUARD, lifetime=3)

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 23 = 0.0（对方 DEFENSE 无）
- 索引 24 = 3.0（对方 GUARD）
- 索引 25 = 0.0（对方 FRAGILE 无）

---

### TC-IV-08 — 双方均无任何状态

**Given**: P2 statuses=[], P1 statuses=[]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 20-25 全部为 0.0

---

### TC-IV-09 — 我方弃牌堆计数段（索引 26-32）

**Given**: P2 discard_pile = [attack, attack, defense, defense, defense, flex]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 26 = 2.0（attack×2）
- 索引 27 = 3.0（defense×3）
- 索引 28 = 0.0
- 索引 29 = 0.0
- 索引 30 = 0.0
- 索引 31 = 0.0
- 索引 32 = 1.0（flex）

---

### TC-IV-10 — 我方抽牌堆计数段（索引 33-39）

**Given**: P2 draw_pile = [energy, energy, build]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 33-36 = 0.0
- 索引 37 = 2.0（energy×2）
- 索引 38 = 1.0（build）
- 索引 39 = 0.0

---

### TC-IV-11 — 弃牌堆和抽牌堆均为空

**Given**: P2 discard_pile=[], draw_pile=[]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 26-39 全部为 0.0

---

### TC-IV-12 — 上回合 action 段（索引 40-41）

**Given**: P2 history=[PlayerHistoryItem(CardType.ATTACK, None)]；P1 history=[PlayerHistoryItem(CardType.BUILD, CardType.DEFENSE)]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 40 = 1.0（ATTACK play → policy index 1）
- 索引 41 = 4.0（BUILD+DEFENSE → policy index 4）

---

### TC-IV-13 — 首回合无出牌历史

**Given**: 双方 history 均为空

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 40 = -1.0
- 索引 41 = -1.0

---

### TC-IV-14 — player_key 非法值抛异常

**Given**: 任意合法 GameState

**When**: `Converter.build_input_vector(state, player_key="3")` 或 `"0"` 或 `""`

**Then**: 抛出 `ValueError`

---

### TC-IV-15 — 空手牌

**Given**: P2 hand=[], P1 hand=[]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 6-19 全部为 0.0

---

### TC-IV-16 — HP=0 边界

**Given**: P2 HP=0, P1 HP=1

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 0 = 0.0, 索引 1 = 1.0

---

### TC-IV-17 — 能量满值边界

**Given**: P2 energy=5, P1 energy=0

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 2 = 5.0, 索引 3 = 0.0

---

### TC-IV-18 — 大轮数

**Given**: cycle=100, round_in_cycle=3

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 4 = 3.0, 索引 5 = 100.0

---

### TC-IV-19 — 手牌含同一卡牌多张

**Given**: P2 hand=[attack, attack, attack, attack, attack]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 6 = 5.0, 索引 7-12 = 0.0

---

### TC-IV-20 — 视角互换全量验证

**Given**: P1 HP=10, energy=2, hand=[heavy, counter], statuses=[PlayerStatus(DEFENSE, lifetime=1)]；P2 HP=20, energy=4, hand=[attack], statuses=[PlayerStatus(GUARD, lifetime=2), PlayerStatus(FRAGILE, lifetime=3)]

**When**: `Converter.build_input_vector(state, player_key="1")`

**Then**:
- 索引 0 = 10.0（P1 HP）
- 索引 1 = 20.0（P2 HP）
- 索引 2 = 2.0（P1 energy）
- 索引 3 = 4.0（P2 energy）
- 索引 6 = 0.0, 索引 7 = 0.0, 索引 8 = 1.0, 索引 9 = 1.0, 索引 10-12 = 0.0（P1 手牌）
- 索引 13 = 1.0, 索引 14-19 = 0.0（P2 手牌）
- 索引 20 = 1.0, 索引 21 = 0.0, 索引 22 = 0.0（P1 状态）
- 索引 23 = 0.0, 索引 24 = 2.0, 索引 25 = 3.0（P2 状态）

---

## 2. _count_by_card_type（新增内部方法）

### TC-CC-01 — 正常计数

**Given**: cards = [CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE, CardType.FLEX]

**When**: `Converter._count_by_card_type(cards)`

**Then**: [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

---

### TC-CC-02 — 空列表

**Given**: cards = []

**When**: `Converter._count_by_card_type(cards)`

**Then**: [0.0] × 7

---

### TC-CC-03 — CardType 注入测试（含非合法值）

**Given**: cards = [CardType.ATTACK, "invalid"]

**When**: `Converter._count_by_card_type(cards)`

**Then**: 抛出 `KeyError` 或 `AttributeError`

---

## 3. _extract_status_lifetime（新增内部方法）

### TC-ES-01 — 找到目标状态

**Given**: statuses = [PlayerStatus(DEFENSE, lifetime=3)]

**When**: `Converter._extract_status_lifetime(statuses, StatusType.DEFENSE)`

**Then**: 返回 3.0

---

### TC-ES-02 — 未找到目标状态

**Given**: statuses = [PlayerStatus(DEFENSE, lifetime=2)]

**When**: `Converter._extract_status_lifetime(statuses, StatusType.GUARD)`

**Then**: 返回 0.0

---

### TC-ES-03 — 空状态列表

**Given**: statuses = []

**When**: `Converter._extract_status_lifetime(statuses, StatusType.FRAGILE)`

**Then**: 返回 0.0

---

### TC-ES-04 — 多状态中查找

**Given**: statuses = [PlayerStatus(DEFENSE, lifetime=1), PlayerStatus(GUARD, lifetime=2), PlayerStatus(FRAGILE, lifetime=3)]

**When**: `Converter._extract_status_lifetime(statuses, StatusType.FRAGILE)`

**Then**: 返回 3.0

---

## 4. INPUT_SIZE 常量

### TC-IS-01 — model.py INPUT_SIZE = 42

**Given**: 导入 `from src.backend.ai.rnn_v1.model import INPUT_SIZE`

**Then**: `INPUT_SIZE == 42`

---

### TC-IS-02 — GRUPolicyModel 接受 42-d 输入

**Given**: `model = GRUPolicyModel()`；x = torch.randn(1, 1, 42)；hidden = torch.randn(1, 1, 32)

**When**: `model.forward(x, hidden)`

**Then**: probs.shape = (1, 13)

---

### TC-IS-03 — GRUValueModel 接受 42-d 输入

**Given**: `model = GRUValueModel()`；x = torch.randn(1, 1, 42)；hidden = torch.randn(1, 1, 32)

**When**: `model.forward(x, hidden)`

**Then**: value.shape = (1, 1)

---

## 5. 往返一致性

### TC-RT-01 — 上回合 action 编解码往返

**Given**: play_round([CardType.ATTACK], [CardType.DEFENSE]) 后的 GameState

**When**: 对 state 构造 input vector，取出索引 40 和 41 的值，分别调用 `Converter.policy_index_to_action(int(index_40))` 和 `Converter.policy_index_to_action(int(index_41))`

**Then**: 索引 40 解码为 (CardType.ATTACK, None)；索引 41 解码为 (CardType.DEFENSE, None)
