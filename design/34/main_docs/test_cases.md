# Test Cases — 56维输入 + 牌堆互相可见 (intention #34)

## 测试范围

- **单元测试**: `tests/unit/test_converter.py` — Converter 56-d input 构造（含对方牌堆段）
- **单元测试**: `tests/unit/test_model.py` — INPUT_SIZE 常量验证（56）
- **单元测试**: `tests/unit/test_game_state.py` — PlayerState piles 始终非空
- **单元测试**: `tests/unit/test_game_core.py` — get_state 始终暴露双方完整 piles
- **前端测试**: `src/frontend/tests/App.test.jsx` — 双方 pile viewer 交互

---

## 1. build_input_vector — 56-d 向量构造

### TC-IV-01 — 56维长度正确性

**Given**: 正常 GameState（P2 视角，双方各有完整 hand/discard_pile/draw_pile/history）

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 返回值长度 = 56

---

### TC-IV-02 — 基础元数据段（索引 0-5）不变

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

### TC-IV-03 — 我方手牌计数段（索引 6-12）不变

**Given**: P2 手牌 [attack, attack, defense, heavy, counter, energy, build, flex]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 6 = 2.0, 7 = 1.0, 8 = 1.0, 9 = 1.0, 10 = 1.0, 11 = 1.0, 12 = 1.0

---

### TC-IV-04 — 对方手牌计数段（索引 13-19）不变

**Given**: P2 视角，P1 手牌 [defense, defense, heavy]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 13 = 0.0, 14 = 2.0, 15 = 1.0, 16-19 = 0.0

---

### TC-IV-05 — 视角互换后手牌段对调

**Given**: P1 手牌 [attack], P2 手牌 [defense]

**When**: `Converter.build_input_vector(state, player_key="1")`

**Then**: 索引 6 = 1.0（我方=P1 attack）, 索引 13 = 0.0, 索引 14 = 1.0（对方=P2 defense）

---

### TC-IV-06 — 我方状态持续回合段（索引 20-22）

**Given**: P2 statuses 含 PlayerStatus(DEFENSE, lifetime=2), PlayerStatus(FRAGILE, lifetime=1)

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 20 = 2.0, 21 = 0.0, 22 = 1.0

---

### TC-IV-07 — 对方状态持续回合段（索引 23-25）

**Given**: P1 statuses 含 PlayerStatus(GUARD, lifetime=3)

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 23 = 0.0, 24 = 3.0, 25 = 0.0

---

### TC-IV-08 — 双方均无任何状态

**Given**: P2 statuses=[], P1 statuses=[]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 20-25 全部为 0.0

---

### TC-IV-09 — 我方弃牌堆计数段（索引 26-32）不变

**Given**: P2 discard_pile = [attack, attack, defense, defense, defense, flex]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 26 = 2.0, 27 = 3.0, 28 = 0.0, 29 = 0.0, 30 = 0.0, 31 = 0.0, 32 = 1.0

---

### TC-IV-10 — 对方弃牌堆计数段（索引 33-39）新增

**Given**: P2 视角，P1 discard_pile = [heavy, energy, energy, build]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 33 = 0.0（对方弃牌堆 attack=0）
- 索引 34 = 0.0（defense=0）
- 索引 35 = 1.0（heavy=1）
- 索引 36 = 0.0（counter=0）
- 索引 37 = 2.0（energy=2）
- 索引 38 = 1.0（build=1）
- 索引 39 = 0.0（flex=0）

---

### TC-IV-11 — 对方弃牌堆为空

**Given**: P1 discard_pile=[]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 33-39 全部为 0.0

---

### TC-IV-12 — 我方抽牌堆计数段（索引 40-46）偏移验证

**Given**: P2 draw_pile = [energy, energy, build]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 40-43 = 0.0
- 索引 44 = 2.0（energy×2）
- 索引 45 = 1.0（build）
- 索引 46 = 0.0

**注意**: 与设计 #27 对比，抽牌堆从索引 33-39 移至 40-46。

---

### TC-IV-13 — 对方抽牌堆计数段（索引 47-53）新增

**Given**: P2 视角，P1 draw_pile = [counter, counter, flex, flex, flex]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 47-49 = 0.0
- 索引 50 = 2.0（counter=2）
- 索引 51 = 0.0
- 索引 52 = 0.0
- 索引 53 = 3.0（flex=3）

---

### TC-IV-14 — 对方抽牌堆为空

**Given**: P1 draw_pile=[]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 47-53 全部为 0.0

---

### TC-IV-15 — 上回合 action 段（索引 54-55）偏移验证

**Given**: P2 history=[PlayerHistoryItem(CardType.ATTACK, None)]；P1 history=[PlayerHistoryItem(CardType.BUILD, CardType.DEFENSE)]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 54 = 2.0（ATTACK play → policy index 2）
- 索引 55 = 5.0（BUILD+DEFENSE → policy index 5）

**注意**: 与设计 #27 对比，last action 从索引 40-41 移至 54-55。

---

### TC-IV-16 — 首回合无出牌历史

**Given**: 双方 history 均为空

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 54 = -1.0, 索引 55 = -1.0

---

### TC-IV-17 — player_key 非法值抛异常

**Given**: 任意合法 GameState

**When**: `Converter.build_input_vector(state, player_key="3")` 或 `"0"` 或 `""`

**Then**: 抛出 `ValueError`

---

### TC-IV-18 — 空手牌

**Given**: P2 hand=[], P1 hand=[]

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 6-19 全部为 0.0

---

### TC-IV-19 — HP=0 边界

**Given**: P2 HP=0, P1 HP=1

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 0 = 0.0, 索引 1 = 1.0

---

### TC-IV-20 — 视角互换全量验证（56-d）

**Given**: P1 HP=10, energy=2, hand=[heavy, counter], statuses=[PlayerStatus(DEFENSE, lifetime=1)], discard_pile=[attack], draw_pile=[energy]；P2 HP=20, energy=4, hand=[attack], statuses=[PlayerStatus(GUARD, lifetime=2)], discard_pile=[defense, defense], draw_pile=[flex]

**When**: `Converter.build_input_vector(state, player_key="1")`

**Then**:
- 索引 0 = 10.0, 1 = 20.0（HP）
- 索引 2 = 2.0, 3 = 4.0（energy）
- 索引 6-12（P1 手牌）: heavy=1, counter=1, 其余=0
- 索引 13-19（P2 手牌）: attack=1, 其余=0
- 索引 20-22（P1 状态）: DEFENSE=1, GUARD=0, FRAGILE=0
- 索引 23-25（P2 状态）: DEFENSE=0, GUARD=2, FRAGILE=0
- 索引 26-32（P1 弃牌堆）: attack=1, 其余=0
- 索引 33-39（P2 弃牌堆）: defense=2, 其余=0
- 索引 40-46（P1 抽牌堆）: energy=1, 其余=0
- 索引 47-53（P2 抽牌堆）: flex=1, 其余=0

---

### TC-IV-21 — 构建后对方弃牌堆增加验证

**Given**: P2 视角。游戏执行 play_round: P1 打 BUILD+ATTACK，P2 打 DEFENSE

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 33（对方弃牌堆 attack 计数）= 1.0（构建的 ATTACK 进入 P1 弃牌堆）
- 索引 26（我方弃牌堆 defense 计数）= 1.0（P2 弃掉的 DEFENSE）

---

## 2. INPUT_SIZE 常量

### TC-IS-01 — constants.py INPUT_SIZE = 56

**Given**: 导入 `from src.backend.ai.constants import INPUT_SIZE`

**Then**: `INPUT_SIZE == 56`

---

### TC-IS-02 — rnn_v1 GRUPolicyModel 接受 56-d 输入

**Given**: `model = GRUPolicyModel()`；x = torch.randn(1, 1, 56)；hidden = torch.randn(1, 1, 32)

**When**: `model.forward(x, hidden)`

**Then**: probs.shape = (1, 14)

---

### TC-IS-03 — rnn_v1 GRUValueModel 接受 56-d 输入

**Given**: `model = GRUValueModel()`；x = torch.randn(1, 1, 56)；hidden = torch.randn(1, 1, 32)

**When**: `model.forward(x, hidden)`

**Then**: value.shape = (1, 1)

---

### TC-IS-04 — rnn_v2 GRUPolicyModelV2 接受 56-d 输入，SKIP_INPUT_SIZE=88

**Given**: `model = GRUPolicyModelV2()`；x = torch.randn(1, 1, 56)；hidden = torch.randn(1, 1, 32)

**When**: `model.forward(x, hidden)`

**Then**: probs.shape = (1, 14)；model.fc1.in_features == 88

---

### TC-IS-05 — rnn_v2 GRUValueModelV2 接受 56-d 输入，fc1 in_features=88

**Given**: `model = GRUValueModelV2()`

**Then**: model.fc1.in_features == 88

---

## 3. GameState — 牌堆互相可见

### TC-GS-01 — get_state 始终包含双方完整 draw_pile

**Given**: 正常进行中的游戏（至少 1 回合），P1 draw_pile 有卡牌

**When**: `gm.get_state()`（不传 player_perspective）

**Then**: `state.players["1"].draw_pile` 为完整 `list[CardType]`（非 None），`state.players["2"].draw_pile` 同理

---

### TC-GS-02 — get_state 始终包含双方完整 discard_pile

**Given**: 游戏中 P1 打出 ATTACK

**When**: `gm.get_state()`

**Then**: `state.players["1"].discard_pile` 包含 ATTACK（非 None），`state.players["2"].discard_pile` 也为非 None list

---

### TC-GS-03 — to_dict 始终序列化双方 piles

**Given**: 普通 GameState

**When**: `GameState.to_dict(gs)`

**Then**: 返回 dict 中 `players["1"]` 和 `players["2"]` 均包含 `"draw_pile"` 和 `"discard_pile"` 键，值为卡牌字符串数组

---

### TC-GS-04 — player_perspective 参数兼容但行为统一

**Given**: 正常游戏

**When**: `gm.get_state(player_perspective="1")` 和 `gm.get_state()`

**Then**: 两次返回的 `PlayerState.draw_pile` / `discard_pile` 均非 None 且内容相同

---

## 4. Converter — 移除 pile None 检查

### TC-CN-01 — 双方 piles 为空时不抛异常

**Given**: 刚创建的游戏（第一轮开始前），双方 draw_pile 有卡、discard_pile 为空

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 正常返回 56-d 向量，不抛出 ValueError

---

### TC-CN-02 — player_perspective 不匹配时也不抛异常

**Given**: 正常游戏状态（piles 均非空）

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 正常返回，不因 pile 为 None 抛异常

---

## 5. 往返一致性

### TC-RT-01 — 上回合 action 编解码往返

**Given**: play_round([CardType.ATTACK], [CardType.DEFENSE]) 后的 GameState

**When**: 对 state 构造 input vector，取出索引 54 和 55 的值，分别调用 `Converter.policy_index_to_action(int(index_54))` 和 `Converter.policy_index_to_action(int(index_55))`

**Then**: 索引 54 解码为 (CardType.ATTACK, None)；索引 55 解码为 (CardType.DEFENSE, None)

---

## 6. 前端 pile viewer 交互

### TC-FE-01 — P1 点击己方牌堆查看器

**Given**: P1 的 draw_pile=[attack, defense], discard_pile=[heavy]

**When**: 用户点击 P1 状态栏牌堆计数

**Then**: 弹出 pileViewer 显示 "Your Piles"，draw_pile 显示 attack 和 defense，discard_pile 显示 heavy

---

### TC-FE-02 — 点击对方牌堆查看器

**Given**: P2 的 draw_pile=[energy], discard_pile=[flex, flex]

**When**: 用户点击对方状态栏牌堆计数

**Then**: 弹出 pileViewer 显示 "Opponent's Piles"，展示对方牌堆内容

---

### TC-FE-03 — 空牌堆显示

**Given**: 双方 discard_pile=[]

**When**: 用户点击牌堆查看器

**Then**: discard_pile 区域显示 "Empty"

---

### TC-FE-04 — 双方牌堆计数在状态栏中均可点击

**Given**: 正常游戏状态

**Then**: P1 状态栏中 Deck/Disc 计数显示可点击样式（cursor: pointer, title attribute）；P2 同理
