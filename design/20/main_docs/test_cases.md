# Test Cases — Converter 类型化重构 (intention #20)

## 测试范围

- **单元测试**: `tests/unit/test_converter.py` — Converter 全部公共方法
- **单元测试**: `tests/unit/test_app_manager.py` — `_make_mask_fn` CardType 适配
- **集成测试**: `tests/integration/test_controller_api.py` — AI 输出验证适配

---

## 1. Converter 常量

### TC-CT-01 — `INPUT_SIZE` 从 `ai.rnn_v1.model` 导入

**Given**: `from src.backend.ai.rnn_v1.model import INPUT_SIZE`

**Then**: `Converter.INPUT_SIZE` 不存在（已移除）；直接使用 `INPUT_SIZE`，值为 `15`

---

### TC-CT-02 — `POLICY_OUTPUT_SIZE` 从 `ai.rnn_v1.model` 导入

**Given**: `from src.backend.ai.rnn_v1.model import POLICY_OUTPUT_SIZE`

**Then**: `Converter.POLICY_OUTPUT_SIZE` 不存在（已移除）；直接使用 `POLICY_OUTPUT_SIZE`，值为 `13`

---

## 2. `_CARD_MAP` 与映射表

### TC-CM-01 — `_CARD_MAP` 元素为 `CardType` 枚举

**Given**: `Converter._CARD_MAP`

**Then**: 每个元组的第一个元素是 `CardType` 实例（如 `CardType.ATTACK`），不是 `str`

---

### TC-CM-02 — `_CARD_MAP` 不含 `build_cost` 列

**Given**: `Converter._CARD_MAP`

**Then**: 每个元组长度为 `3`（`card_type, play_index, build_index`），不含 `build_cost`

---

### TC-CM-03 — `ALL_CARD_TYPES` 元素为 `CardType`

**Given**: `Converter.ALL_CARD_TYPES`

**Then**: 集合中的每个元素是 `CardType` 实例；`CardType.ATTACK in ALL_CARD_TYPES` 为 `True`；`"attack" in ALL_CARD_TYPES` 为 `False`

---

### TC-CM-04 — `_PLAY_TO_INDEX` 键为 `CardType`

**Given**: `Converter._PLAY_TO_INDEX`

**Then**: 键为 `CardType.ATTACK`、`CardType.DEFENSE` 等（不含 `CardType.BUILD`）；`CardType.ATTACK` 映射到奇数索引 `1`

---

### TC-CM-05 — `_BUILD_TO_INDEX` 键为 `CardType`

**Given**: `Converter._BUILD_TO_INDEX`

**Then**: 键包含全部 7 种 `CardType`；`CardType.ATTACK` 映射到偶数索引 `2`

---

### TC-CM-06 — `_INDEX_TO_ACTION` 值为 `CardType`

**Given**: `Converter._INDEX_TO_ACTION[1]`

**Then**: 返回 `(CardType.ATTACK, None)`，第一个元素是 `CardType` 实例

---

### TC-CM-07 — `_BUILD_COST` 派生自 `get_card_def()`

**Given**: `Converter._BUILD_COST`

**Then**: `Converter._BUILD_COST[CardType.ATTACK] == 1`（与 `get_card_def(CardType.ATTACK).build_cost` 一致）；`Converter._BUILD_COST[CardType.COUNTER] == 2`；`Converter._BUILD_COST[CardType.FLEX] == 2`

---

## 3. `build_input_vector`

### TC-IV-01 — 内部使用 `CardType` 键查找计数

**Given**: `GameState`，P2 手牌为 `[CardType.ATTACK, CardType.DEFENSE]`

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 返回 15 个 float 的列表；索引 6（ATTACK 计数）= `1.0`；索引 7（DEFENSE 计数）= `1.0`；不再需要 `card.value` 字符串转换

---

### TC-IV-02 — `NUM_CARD_TYPES` 保留为 `7`

**Given**: `Converter.NUM_CARD_TYPES`

**Then**: 值为 `7`（保持不变）

---

### TC-IV-03 — 全量手牌计数正确（7 种各一张）

**Given**: P2 手牌 = `[CardType.ATTACK, CardType.DEFENSE, CardType.HEAVY, CardType.COUNTER, CardType.ENERGY, CardType.BUILD, CardType.FLEX]`

**When**: `Converter.build_input_vector(state, player_key="2")`

**Then**: 索引 6~12 全部为 `1.0`

---

## 4. `action_to_policy_index`

### TC-AP-01 — 纯打出动作接受 `CardType`

**Given**: `card=CardType.ATTACK`, `build_target=None`

**When**: `Converter.action_to_policy_index(card, None)`

**Then**: 返回 `1`（奇数索引）

---

### TC-AP-02 — 构建动作接受 `CardType`

**Given**: `card=CardType.BUILD`, `build_target=CardType.ATTACK`

**When**: `Converter.action_to_policy_index(card, build_target)`

**Then**: 返回 `2`（偶数索引）

---

### TC-AP-03 — 传入 `str` 抛出 `TypeError`（或 `ValueError`）

**Given**: `card="attack"`（字符串）

**When**: `Converter.action_to_policy_index("attack", None)`

**Then**: 因类型不匹配抛出异常（`_PLAY_TO_INDEX` 键为 `CardType`，`"attack"` 不在其中）

---

### TC-AP-04 — 所有 6 种纯打出卡牌映射正确

**Given**: 遍历 `[ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX]`

**When**: `action_to_policy_index(card, None)`

**Then**: 分别返回 `1, 3, 5, 7, 9, 11`

---

### TC-AP-05 — 所有 7 种构建目标映射正确

**Given**: 遍历全部 7 种 `CardType` 作为 `build_target`，`card=CardType.BUILD`

**When**: `action_to_policy_index(CardType.BUILD, target)`

**Then**: 分别返回 `0, 2, 4, 6, 8, 10, 12`

---

### TC-AP-06 — 非 BUILD 卡带 `build_target` 抛出异常

**Given**: `card=CardType.ATTACK`, `build_target=CardType.DEFENSE`

**When**: `action_to_policy_index(card, build_target)`

**Then**: 抛出 `ValueError`

---

### TC-AP-07 — BUILD 卡无 `build_target` 抛出异常

**Given**: `card=CardType.BUILD`, `build_target=None`

**When**: `action_to_policy_index(card, None)`

**Then**: 抛出 `ValueError`

---

## 5. `policy_index_to_action`

### TC-PI-01 — 奇数索引返回纯打出动作

**Given**: `index=1`

**When**: `Converter.policy_index_to_action(1)`

**Then**: 返回 `(CardType.ATTACK, None)`

---

### TC-PI-02 — 偶数索引返回构建动作

**Given**: `index=2`

**When**: `Converter.policy_index_to_action(2)`

**Then**: 返回 `(CardType.BUILD, CardType.ATTACK)`

---

### TC-PI-03 — 越界索引抛出 `IndexError`

**Given**: `index=13` 或 `index=-1`

**When**: `Converter.policy_index_to_action(index)`

**Then**: 抛出 `IndexError`

---

### TC-PI-04 — 往返一致性

**Given**: `(CardType.ATTACK, None)` 等所有 13 种合法动作

**When**: `policy_index_to_action(action_to_policy_index(card, bt))`

**Then**: 返回原始 `(card, bt)`

---

## 6. `build_cost`

### TC-BC-01 — 从 `CardDef` 获取 `build_cost`

**Given**: `card_type=CardType.COUNTER`

**When**: `Converter.build_cost(CardType.COUNTER)`

**Then**: 返回 `2`（与 `get_card_def(CardType.COUNTER).build_cost` 一致）

---

### TC-BC-02 — 所有 7 种卡牌 `build_cost` 正确

**Given**: 遍历全部 7 种 `CardType`

**When**: `Converter.build_cost(card_type)`

**Then**: ATTACK=1, DEFENSE=1, HEAVY=1, COUNTER=2, ENERGY=1, BUILD=1, FLEX=2

---

### TC-BC-03 — 传入 `str` 类型不匹配

**Given**: `card_type="attack"`（字符串）

**When**: `Converter.build_cost("attack")`

**Then**: 因 `_BUILD_COST` 键为 `CardType`，返回 `999`（sentinel）或抛出 `KeyError`

---

## 7. `_action_item_to_index`

### TC-AI-01 — 直接使用 `CardType` 传递

**Given**: `PlayerHistoryItem(card=CardType.ATTACK, build_target=None)`

**When**: `Converter._action_item_to_index(item)`

**Then**: 返回 `1`；不再需要 `item.card.value` 转换

---

### TC-AI-02 — 构建历史

**Given**: `PlayerHistoryItem(card=CardType.BUILD, build_target=CardType.COUNTER)`

**When**: `Converter._action_item_to_index(item)`

**Then**: 返回 `8`（build+counter 的偶数索引）

---

## 8. AppManager `_make_mask_fn`

### TC-MK-01 — `legal_cards` 为 `set[CardType]`

**Given**: `hand=[CardType.ATTACK, CardType.DEFENSE]`, `energy=5`

**When**: `AppManager._make_mask_fn(hand, energy)`

**Then**: 返回可调用的 mask 函数；内部 `legal_cards` 为 `{CardType.ATTACK, CardType.DEFENSE}`，不含 `.value` 转换

---

### TC-MK-02 — 直接使用 `CardType` 键查找 `_PLAY_TO_INDEX`

**Given**: `hand=[CardType.ATTACK]`, `energy=5`，logits 为全 0 的 13-d tensor

**When**: `mask_fn(logits)`

**Then**: DEFENSE 对应的 odd index `3` 被设为 `-inf`；ATTACK 对应的 odd index `1` 保持原值；不再使用 `card.value` 转换

---

### TC-MK-03 — 能量不足时屏蔽构建目标

**Given**: `hand=[CardType.BUILD, CardType.COUNTER]`, `energy=1`（COUNTER 构建费 2 能量，ATTACK 构建费 1 能量）

**When**: `mask_fn(logits)`

**Then**: COUNTER 对应的 build index `8` 被设为 `-inf`（能量不足）；但其他 build index 因 BUILD 不在手或能量不足也应被屏蔽

---

### TC-MK-04 — `Converter.build_cost()` 接收 `CardType`

**Given**: mask_fn 内部调用 `Converter.build_cost(card_type)`，其中 `card_type` 来自 `_CARD_MAP` 元组的第一元素

**When**: 遍历 `_CARD_MAP` 各元素

**Then**: `card_type` 为 `CardType` 实例，直接传递给 `build_cost()`，无需转换

---

## 9. Controller API

### TC-API-01 — `ALL_CARD_TYPES` 检查适配

**Given**: AI 返回 `card_type=CardType.ATTACK`

**When**: `card_type not in Converter.ALL_CARD_TYPES`

**Then**: 检查通过（`CardType.ATTACK` 在 `frozenset[CardType]` 中）；不再需要字符串比较

---

### TC-API-02 — 无需 `str_to_card` 转换

**Given**: `ai_select_action` 返回 `(CardType.ATTACK, None, hidden_state)`

**When**: 构造 `cards_2`

**Then**: 直接使用 `[card_type]`，不再调用 `str_to_card(card_str)`

---

## 10. SelfPlayRunner

### TC-SP-01 — `action_to_policy_index` 适配

**Given**: `ai_sample_action` 返回 `card_type=CardType.ATTACK`, `bt=None`

**When**: `Converter.action_to_policy_index(card_type, bt)`

**Then**: 返回 `1`；`card_type` 和 `bt` 已是 `CardType` 实例，无需转换

---

## 11. 回归验证

### TC-RG-01 — 现有 `build_input_vector` 用例全部通过

**Given**: `tests/unit/test_converter.py` 中所有 `TestBuildInputVector15D` 用例

**When**: 运行测试

**Then**: 全部通过（仅内部实现改为 `CardType` 键查找，行为不变）

---

### TC-RG-02 — 现有 `action_to_policy_index` 用例适配后通过

**Given**: `tests/unit/test_converter.py` 中所有 `TestActionToPolicyIndex` 用例，参数从 `str` 改为 `CardType`

**When**: 运行测试

**Then**: 全部通过

---

### TC-RG-03 — 现有 `policy_index_to_action` 用例适配后通过

**Given**: `tests/unit/test_converter.py` 中所有 `TestPolicyIndexToAction` 用例，断言从 `str` 改为 `CardType`

**When**: 运行测试

**Then**: 全部通过

---

### TC-RG-04 — 现有 `build_cost` 用例适配后通过

**Given**: `tests/unit/test_converter.py` 中所有 `TestBuildCost` 用例，参数从 `str` 改为 `CardType`

**When**: 运行测试

**Then**: 全部通过；`build_cost("nonexistent")` 用例改为 `build_cost` 传入不存在于 `_BUILD_COST` 的任意 `CardType` 实例时返回 `999`

---

### TC-RG-05 — 全量后端测试通过

**Given**: `python -m pytest tests/`

**When**: 运行全部测试

**Then**: 所有测试通过，无回归
