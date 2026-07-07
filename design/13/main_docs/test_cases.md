# Test Cases — Input Reordering (intention #13)

## 测试范围

- **单元测试**: `tests/unit/test_converter.py` — Converter.build_input_vector 新索引顺序验证
- **无需集成测试**: 无跨模块交互变更

---

## TC-IV-01 — 索引顺序正确性（player_key="2"）

**Given**: state 中 P2 血量 15，P1 血量 8，round_in_cycle=3，cycle=2，P2 手牌 [attack, defense, heavy, counter, energy, build, flex]，双方 history 均为空。

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 0 = 15.0
- 索引 1 = 8.0
- 索引 2 = 3.0
- 索引 3 = 2.0
- 索引 4-10 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- 索引 11-17 = [0.0]×7（无出牌历史）
- 索引 18-24 = [0.0]×7（无出牌历史）

---

## TC-IV-02 — 索引顺序正确性（player_key="1" 视角互换）

**Given**: 同 TC-IV-01 的 state。

**When**: `Converter.build_input_vector(state, player_key="1")`

**Then**:
- 索引 0 = 8.0（P1 血量）
- 索引 1 = 15.0（P2 血量）
- 索引 2 = 3.0
- 索引 3 = 2.0
- 索引 4-10 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]（P1 手牌）
- 索引 11-17 = [0.0]×7
- 索引 18-24 = [0.0]×7

---

## TC-IV-03 — one-hot 历史正确记录

**Given**: P2 上次出了 attack，P1 上次出了 defense。

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 11 = 1.0（attack），索引 12-17 = 0.0
- 索引 18 = 0.0，索引 19 = 1.0（defense），索引 20-24 = 0.0

---

## TC-IV-04 — 空手牌 / 部分手牌

**Given**: P2 手牌仅 [attack, attack]，其余类型为 0。

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 4 = 2.0
- 索引 5-10 = [0.0]×6

---

## TC-IV-05 — player_key 非法值抛异常

**Given**: 任意合法 state。

**When**: `Converter.build_input_vector(state, player_key="3")`

**Then**: 抛出 `ValueError`

---

## TC-IV-06 — HP 为 0 的边界

**Given**: P2 血量 0，P1 血量 1。

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**:
- 索引 0 = 0.0
- 索引 1 = 1.0

---

## TC-IV-07 — 返回值长度恒为 25

**Given**: 任意合法 state，任意合法 player_key。

**When**: `Converter.build_input_vector(state, player_key)`

**Then**: 返回值长度为 25（等于 `Converter.INPUT_SIZE`）
