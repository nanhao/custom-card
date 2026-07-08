# Interface Design — Converter 类型化重构 (intention #20)

## 1. Converter — 核心变更

位于 `app/converter.py`。

### 1.1 常量：移除 `INPUT_SIZE` / `POLICY_OUTPUT_SIZE`

移除类属性 `INPUT_SIZE` 和 `POLICY_OUTPUT_SIZE`，改为从 `ai.rnn_v1.model` 导入：

```
from src.backend.ai.rnn_v1.model import INPUT_SIZE, POLICY_OUTPUT_SIZE
```

`NUM_CARD_TYPES` 保留于 Converter（用于 `build_input_vector` 内部分配计数数组）。

### 1.2 `_CARD_MAP` — 类型化 + 移除 `build_cost`

旧：

```
_CARD_MAP: list[tuple[str, int | None, int, int]] = [
    ("attack",  1, 1, 2),
    ("defense", 3, 1, 4),
    ...
]
```

新：

```
_CARD_MAP: list[tuple[CardType, int | None, int]] = [
    (CardType.ATTACK,  1, 2),
    (CardType.DEFENSE, 3, 4),
    (CardType.HEAVY,   5, 6),
    (CardType.COUNTER, 7, 8),
    (CardType.ENERGY,  9, 10),
    (CardType.BUILD,   None, 0),
    (CardType.FLEX,    11, 12),
]
```

元组语义变为 `(card_type, play_index, build_index)`，`build_cost` 列移除。

### 1.3 派生映射表

| 映射表 | 旧类型 | 新类型 |
|--------|--------|--------|
| `ALL_CARD_TYPES` | `frozenset[str]` | `frozenset[CardType]` |
| `_CARD_TYPE_ORDER` | `dict[str, int]` | `dict[CardType, int]` |
| `_PLAY_TO_INDEX` | `dict[str, int]` | `dict[CardType, int]` |
| `_BUILD_TO_INDEX` | `dict[str, int]` | `dict[CardType, int]` |
| `_INDEX_TO_ACTION` | `dict[int, tuple[str, str\|None]]` | `dict[int, tuple[CardType, CardType\|None]]` |
| `_BUILD_COST` | `dict[str, int]` | `dict[CardType, int]` |

**`_BUILD_COST` 派生方式变更**：

旧：`{c[0]: c[2] for c in _CARD_MAP}`
新：`{c[0]: get_card_def(c[0]).build_cost for c in _CARD_MAP}`

### 1.4 `build_input_vector` — 内部适配

```
build_input_vector(state: GameState, player_key: str = "2") -> list[float]
```

签名不变。内部变更：

- `_CARD_TYPE_ORDER` 键改为 `CardType`，不再需要 `card.value` 转换
- 直接 `Converter._CARD_TYPE_ORDER.get(card)`（`card` 已是 `CardType`）

### 1.5 `action_to_policy_index` — 参数类型化

旧：

```
action_to_policy_index(card_str: str, build_target: str | None) -> int
```

新：

```
action_to_policy_index(card: CardType, build_target: CardType | None) -> int
```

内部比较 `card == CardType.BUILD` 替代 `card_str == "build"`。

### 1.6 `policy_index_to_action` — 返回值类型化

旧：

```
policy_index_to_action(index: int) -> tuple[str, str | None]
```

新：

```
policy_index_to_action(index: int) -> tuple[CardType, CardType | None]
```

### 1.7 `build_cost` — 参数类型化

旧：

```
build_cost(card_str: str) -> int
```

新：

```
build_cost(card_type: CardType) -> int
```

内部从 `_BUILD_COST.get(card_type, 999)` 取值。`_BUILD_COST` 派生自 `get_card_def()`。

### 1.8 `_action_item_to_index` — 适配

```
_action_item_to_index(item: PlayerHistoryItem) -> int
```

签名不变。内部不再需要 `item.card.value` 转换，直接传递 `item.card`（已是 `CardType`）。

## 2. AppManager — 变更

位于 `app/app_manager.py`。

### 2.1 导入变更

新增导入：

```
from src.backend.ai.rnn_v1.model import HIDDEN_SIZE, INPUT_SIZE, POLICY_OUTPUT_SIZE
```

旧导入移除 `Converter.INPUT_SIZE` 和 `Converter.POLICY_OUTPUT_SIZE` 的使用（`POLICY_OUTPUT_SIZE` 已从 model 导入）。

### 2.2 `ai_select_action` / `ai_sample_action`

方法签名不变。内部 `Converter.INPUT_SIZE` → `INPUT_SIZE`。

`Converter.policy_index_to_action()` 现在返回 `(CardType, CardType | None)`，调用方直接获得 `CardType`，无需后续 `str_to_card()` 转换。

### 2.3 `_make_mask_fn` — `CardType` 适配

```
_make_mask_fn(hand: list[CardType], energy: int) -> Callable
```

签名不变（design/19 已改为 `list[CardType]`）。内部变更：

- `legal_cards = set(hand)` → 直接是 `set[CardType]`，无需 `.value`
- `Converter._PLAY_TO_INDEX` 键为 `CardType`，直接使用 `card` 而非 `card.value`
- `Converter._CARD_MAP` 元组第一元素为 `CardType`，直接使用
- `Converter.build_cost()` 参数为 `CardType`，直接传递

## 3. SelfPlayRunner — 变更

位于 `app/selfplay_runner.py`。

`Converter.action_to_policy_index(card1, bt1)` — `card1`、`bt1` 来自 `ai_sample_action` 的返回值。由于 `ai_sample_action` 内部调用 `Converter.policy_index_to_action()`，返回值已变为 `CardType`，此处直接适配。

## 4. Controller API — 变更

位于 `controller/api.py`。

旧：

```
if card_str not in Converter.ALL_CARD_TYPES:
```

新：

```
if card_type not in Converter.ALL_CARD_TYPES:
```

`card_type` 来自 `ai_select_action` 返回值（已是 `CardType`）。`ALL_CARD_TYPES` 为 `frozenset[CardType]`。

同步移除 `str_to_card(card_str)` 调用，因为 `ai_select_action` 返回值已是 `CardType`。

## 5. 与旧版差异总结

| 项目 | 旧 | 新 |
|------|-----|-----|
| `_CARD_MAP` 元素类型 | `tuple[str, int\|None, int, int]` | `tuple[CardType, int\|None, int]` |
| `_CARD_MAP` `build_cost` | 第三列硬编码 | 移除，由 `get_card_def()` 派生 |
| `ALL_CARD_TYPES` | `frozenset[str]` | `frozenset[CardType]` |
| `action_to_policy_index` 参数 | `card_str: str` | `card: CardType` |
| `policy_index_to_action` 返回 | `tuple[str, str\|None]` | `tuple[CardType, CardType\|None]` |
| `build_cost` 参数 | `card_str: str` | `card_type: CardType` |
| `INPUT_SIZE` / `POLICY_OUTPUT_SIZE` | Converter 类属性 | 从 `ai.rnn_v1.model` 导入 |
| `build_input_vector` 内部 | `card.value` 查找 `str` 键 | 直接 `CardType` 键查找 |
| `AppManager` mask_fn | `card.value` 转换后查找 | 直接 `CardType` 查找 |
