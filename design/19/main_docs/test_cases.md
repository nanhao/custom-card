# Test Cases — GameState 重构 (intention #20)

## 测试范围

- **单元测试**: `tests/unit/test_game_state.py` — PlayerState / GameState 构造与序列化
- **单元测试**: `tests/unit/test_game_core.py` — get_state() 返回 GameState
- **单元测试**: `tests/unit/test_converter.py` — build_input_vector 接受 GameState
- **单元测试**: `tests/unit/test_app_manager.py` — 方法接受 GameState
- **集成测试**: `tests/integration/test_controller_api.py` — API 响应 JSON 格式不变

---

## 1. PlayerState 构造

### TC-PS-01 — 基本构造

**Given**: `PlayerState(hp=20, energy=3, statuses=[], hand=[], history=[])`

**Then**:
- `.hp == 20`
- `.energy == 3`
- `.statuses == []`
- `.hand == []`
- `.history == []`
- `.draw_pile is None`
- `.discard_pile is None`
- `.draw_pile_count == 0`
- `.discard_pile_count == 0`

---

### TC-PS-02 — 含卡牌和状态的构造

**Given**: `PlayerState(hp=15, energy=2, statuses=[PlayerStatus(StatusType.DEFENSE, 1)], hand=[CardType.ATTACK, CardType.DEFENSE], history=[PlayerHistoryItem(CardType.ATTACK, None)], draw_pile=[CardType.BUILD], discard_pile=[CardType.ENERGY], draw_pile_count=5, discard_pile_count=1)`

**Then**:
- `.hp == 15`
- `.energy == 2`
- `.statuses == [PlayerStatus(StatusType.DEFENSE, 1)]`
- `.hand == [CardType.ATTACK, CardType.DEFENSE]` — 是 `list[CardType]`，不是字符串列表
- `.history[0].card == CardType.ATTACK`
- `.history[0].build_target is None`
- `.draw_pile == [CardType.BUILD]`
- `.discard_pile == [CardType.ENERGY]`
- `.draw_pile_count == 5`
- `.discard_pile_count == 1`

---

### TC-PS-03 — 手牌中不包含字符串

**Given**: `PlayerState(hp=20, energy=0, statuses=[], hand=[CardType.ATTACK], history=[])`

**When**: 检查 `.hand[0]`

**Then**: `.hand[0]` 是 `CardType` 实例，`isinstance(ps.hand[0], CardType)` 为 `True`；不是 `str`

---

### TC-PS-04 — statuses 中不包含 dict

**Given**: `PlayerState(hp=20, energy=0, statuses=[PlayerStatus(StatusType.DEFENSE, 1)], hand=[], history=[])`

**When**: 检查 `.statuses[0]`

**Then**: `.statuses[0]` 是 `PlayerStatus` 实例；不是 `dict`

---

### TC-PS-05 — draw_pile 为 None 的默认状态

**Given**: `PlayerState(hp=20, energy=0, statuses=[], hand=[], history=[])` 使用默认参数

**Then**:
- `.draw_pile is None`
- `.discard_pile is None`
- `.draw_pile_count == 0`
- `.discard_pile_count == 0`

---

## 2. GameState 构造

### TC-GS-01 — 基本构造

**Given**: 两个 `PlayerState`、`round=1`、`cycle=1`、`round_in_cycle=1`、`is_over=False`、`winner=None`、`mode="pvp"`、`round_log=[]`

**Then**:
- `.round == 1`
- `.cycle == 1`
- `.round_in_cycle == 1`
- `.players["1"]` 和 `.players["2"]` 是 `PlayerState` 实例
- `.is_over is False`
- `.winner is None`
- `.mode == "pvp"`
- `.round_log == []`

---

### TC-GS-02 — 游戏结束状态

**Given**: `is_over=True`、`winner="1"`

**Then**: `.is_over is True`、`.winner == "1"`

---

### TC-GS-03 — 平局状态

**Given**: `winner="draw"`

**Then**: `.winner == "draw"`

---

## 3. GameState.to_dict 序列化

### TC-TD-01 — 空状态序列化

**Given**: `GameState(round=1, cycle=1, round_in_cycle=1, players={"1": ps1, "2": ps2}, is_over=False, winner=None, mode="pvp", round_log=[])`，其中 `ps1`、`ps2` 为初始状态（hp=20, energy=0, 空手牌等）

**When**: `GameState.to_dict(gs)`

**Then**: 返回 dict，结构为：
- `"round": 1`
- `"cycle": 1`
- `"round_in_cycle": 1`
- `"players"."1"."hp": 20`
- `"players"."1"."energy": 0`
- `"players"."1"."statuses": []`
- `"players"."1"."hand": []`
- `"players"."1"."history": []`
- `"players"."1"."draw_pile_count": 0`
- `"players"."1"."discard_pile_count": 0`
- `"players"."2"` 同理
- `"is_over": False`
- `"winner": None`
- `"mode": "pvp"`
- `"round_log": []`

---

### TC-TD-02 — 含卡牌的状态序列化

**Given**: `ps1.hand = [CardType.ATTACK, CardType.DEFENSE]`，`ps1.draw_pile = [CardType.BUILD]`

**When**: `GameState.to_dict(gs)`

**Then**:
- `"players"."1"."hand" == ["attack", "defense"]` — 字符串数组
- `"players"."1"."draw_pile" == ["build"]`
- 数组中各元素为 `str` 类型

---

### TC-TD-03 — 含状态的状态序列化

**Given**: `ps1.statuses = [PlayerStatus(StatusType.DEFENSE, 1)]`

**When**: `GameState.to_dict(gs)`

**Then**: `"players"."1"."statuses" == [{"type": "defense", "lifetime": 1}]`

---

### TC-TD-04 — 含历史的状态序列化

**Given**: `ps1.history = [PlayerHistoryItem(CardType.ATTACK, None), PlayerHistoryItem(CardType.BUILD, CardType.COUNTER)]`

**When**: `GameState.to_dict(gs)`

**Then**: `"players"."1"."history" == [{"card": "attack", "build_target": None}, {"card": "build", "build_target": "counter"}]`

---

### TC-TD-05 — draw_pile 为 None 时不输出 draw_pile 键

**Given**: `ps1.draw_pile = None`，`ps1.draw_pile_count = 5`

**When**: `GameState.to_dict(gs)`

**Then**:
- `"players"."1"` 中不含 `"draw_pile"` 键
- `"players"."1"."draw_pile_count" == 5`

---

### TC-TD-06 — round_log 序列化

**Given**: `round_log` 包含一个 `RoundResult`（P1=ATTACK, P2=DEFENSE）

**When**: `GameState.to_dict(gs)`

**Then**: `"round_log"[0]` 结构 = `RoundResult.to_dict()` 的输出，含 `before`/`after_resolve`/`end_actions`/`after_end_actions`

---

### TC-TD-07 — to_dict 不修改原 GameState

**Given**: 一个 `GameState` 实例

**When**: 调用 `GameState.to_dict(gs)` 两次

**Then**: 两次返回的 dict 内容相同；原 `GameState` 实例字段未被修改

---

## 4. GameCore.get_state

### TC-GC-01 — 返回 GameState 实例

**Given**: `GameCore` 启动后

**When**: `gs = gm.get_state()`

**Then**: `isinstance(gs, GameState)` 为 `True`；不是 `dict`

---

### TC-GC-02 — 基本字段正确

**Given**: `GameCore` 启动后

**When**: `gs = gm.get_state()`

**Then**:
- `gs.round == 1`
- `gs.cycle == 1`
- `gs.round_in_cycle == 1`
- `gs.is_over is False`
- `gs.winner is None`
- `gs.mode == "pvp"`

---

### TC-GC-03 — 玩家状态字段正确

**Given**: `GameCore` 启动后

**When**: `gs = gm.get_state()`

**Then**:
- `gs.players["1"].hp == 20`
- `gs.players["1"].energy == 0`
- `gs.players["1"].hand` 长度为 7（初始抽 7 张）
- `gs.players["2"].hp == 20`

---

### TC-GC-04 — hand 中不含字符串

**Given**: `GameCore` 启动后

**When**: `gs = gm.get_state()`

**Then**: `gs.players["1"].hand` 中每个元素是 `CardType` 实例；没有一个元素是 `str`

---

### TC-GC-05 — player_perspective="1" 时暴露 P1 牌堆

**Given**: `GameCore` 启动后

**When**: `gs = gm.get_state(player_perspective="1")`

**Then**:
- `gs.players["1"].draw_pile is not None` — P1 可见牌堆内容
- `gs.players["1"].discard_pile is not None`
- `gs.players["2"].draw_pile is None` — P2 牌堆隐藏
- `gs.players["2"].discard_pile is None`
- `gs.players["2"].draw_pile_count` 和 `gs.players["2"].discard_pile_count` 有值

---

### TC-GC-06 — player_perspective=None 时两方牌堆均隐藏

**Given**: `GameCore` 启动后

**When**: `gs = gm.get_state()`（默认 perspective=None）

**Then**:
- `gs.players["1"].draw_pile is None`
- `gs.players["2"].draw_pile is None`
- 两方 `draw_pile_count`、`discard_pile_count` 有值

---

### TC-GC-07 — play_round 后 state 反映变化

**Given**: P1=ATTACK, P2=DEFENSE，`play_round()` 后

**When**: `gs = gm.get_state()`

**Then**:
- `gs.round == 2`（回合推进）
- `gs.players["1"].hand` 长度比上一回合少 1（卡牌已打出）
- `gs.players["2"].statuses` 包含 `PlayerStatus(StatusType.DEFENSE, 1)`

---

### TC-GC-08 — game over 后 state 反映结果

**Given**: 一系列 round 导致 P2 HP=0

**When**: `gs = gm.get_state()`

**Then**:
- `gs.is_over is True`
- `gs.winner == "1"`

---

## 5. Converter.build_input_vector

### TC-CV-01 — 接受 GameState 不抛异常

**Given**: `gm.get_state()` 返回的 `GameState`

**When**: `Converter.build_input_vector(gs, player_key="2")`

**Then**: 返回 15 个 float 的列表，无异常

---

### TC-CV-02 — 输入向量第 0–1 位为 HP

**Given**: `gs.players["2"].hp = 18`，`gs.players["1"].hp = 15`

**When**: `vec = Converter.build_input_vector(gs, player_key="2")`

**Then**: `vec[0] == 18.0`（自己的 HP）、`vec[1] == 15.0`（对手的 HP）

---

### TC-CV-03 — 输入向量第 2–3 位为能量

**Given**: `gs.players["2"].energy = 3`，`gs.players["1"].energy = 1`

**When**: `vec = Converter.build_input_vector(gs, player_key="2")`

**Then**: `vec[2] == 3.0`、`vec[3] == 1.0`

---

### TC-CV-04 — 输入向量第 4–5 位为 round/cycle

**Given**: `gs.round_in_cycle = 3`，`gs.cycle = 2`

**When**: `vec = Converter.build_input_vector(gs, player_key="2")`

**Then**: `vec[4] == 3.0`、`vec[5] == 2.0`

---

### TC-CV-05 — 输入向量第 6–12 位为手牌计数

**Given**: `gs.players["2"].hand = [CardType.ATTACK, CardType.ATTACK, CardType.DEFENSE]`

**When**: `vec = Converter.build_input_vector(gs, player_key="2")`

**Then**: 每个 `CardType` 对应索引的计数正确（例如 ATTACK=2, DEFENSE=1, 其余=0）

---

### TC-CV-06 — player_key="1" 产生正确视角

**Given**: `gs.players["1"].hp = 10`，`gs.players["2"].hp = 20`

**When**: `vec = Converter.build_input_vector(gs, player_key="1")`

**Then**: `vec[0] == 10.0`、`vec[1] == 20.0`

---

## 6. AppManager

### TC-AM-01 — create_game 返回 GameState

**Given**: `AppManager` 实例

**When**: `game_id, gs = app_manager.create_game(mode="pvp")`

**Then**: `isinstance(gs, GameState)` 为 `True`

---

### TC-AM-02 — _make_mask_fn 接受 list[CardType]

**Given**: `hand = [CardType.ATTACK, CardType.DEFENSE]`，`energy = 3`

**When**: `mask_fn = AppManager._make_mask_fn(hand, energy)`

**Then**: 返回可调用的 mask 函数，不抛异常

---

### TC-AM-03 — mask_fn 正确屏蔽不合法卡牌

**Given**: `hand = [CardType.ATTACK]`，`energy = 5`

**When**: `mask_fn(logits)` 作用于 13-d 向量

**Then**: DEFENSE 对应的 play index 被设为 `-inf`；ATTACK 对应的 play index 保持不变

---

## 7. Controller API 集成测试

### TC-API-01 — GET /api/games/{id} 返回 JSON 格式不变

**Given**: 创建游戏后

**When**: `GET /api/games/{game_id}`

**Then**: 响应 JSON 结构与重构前一致：
- `state.round` - int
- `state.players."1".hand` - 字符串数组
- `state.players."1".statuses` - `[{"type": ..., "lifetime": ...}]` 格式
- `state.players."1".history` - `[{"card": ..., "build_target": ...}]` 格式
- `state.players."1".draw_pile_count` - int
- 无 `state.players."1".draw_pile`（默认 perspective 不暴露牌堆内容）

---

### TC-API-02 — GET /api/games/{id}?perspective=1 暴露牌堆

**Given**: 创建游戏后

**When**: `GET /api/games/{game_id}?perspective=1`

**Then**:
- `state.players."1".draw_pile` 为字符串数组
- `state.players."1".discard_pile` 为字符串数组
- `state.players."2".draw_pile` 不存在

---

### TC-API-03 — POST /api/games 创建游戏返回格式不变

**Given**: `POST /api/games {"mode": "pvp"}`

**Then**: 响应包含 `game_id`（str）和 `state`（dict，结构同 TC-API-01）

---

### TC-API-04 — POST /api/games/{id}/play 返回格式不变

**Given**: 创建游戏后，`POST /api/games/{id}/play {"cards_1": ["attack"], "cards_2": ["defense"]}`

**Then**: 响应包含 `round_result` 和 `state`，两者 JSON 结构与重构前一致

---

## 8. 前端兼容性

### TC-FE-01 — 前端渲染不变

**Given**: 重构后的 API 返回的 JSON

**When**: 前端 `GameBoard` 渲染

**Then**: 所有 UI 元素正常显示（手牌、血量、能量、状态标签、历史芯片、回合日志）
