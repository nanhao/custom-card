# Architecture Design — Converter 类型化重构 (intention #20)

## 1. Motivation

`converter.py` 存在三个问题：

1. **字符串字面值替代 `CardType` 枚举**：`_CARD_MAP` 使用 `"attack"`、`"defense"` 等字符串，而非已导入的 `CardType` 枚举。与 design/19 引入的 `GameState`（手牌已是 `list[CardType]`）不一致。

2. **`build_cost` 重复定义**：`_CARD_MAP` 的第三列存储 `build_cost`（1 或 2），但 `CardDef`（`game/card.py`）已有 `build_cost` 字段，构成数据重复。

3. **`INPUT_SIZE` / `POLICY_OUTPUT_SIZE` 位置不当**：这两个 AI 模型维度常量定义在 converter.py，但 `ai/rnn_v1/model.py` 已有相同定义。按分层架构，AI 层拥有模型维度，App 层应引用而非定义。

## 2. Key Design Decisions

**D1 — Converter 内部映射全部使用 `CardType`。** `_CARD_MAP` 从 `tuple[str, int|None, int, int]` 改为 `tuple[CardType, int|None, int]`（移除 `build_cost` 列）。所有派生映射表（`_PLAY_TO_INDEX`、`_BUILD_TO_INDEX` 等）键从 `str` 改为 `CardType`。

**D2 — Converter 公共 API 类型化。** `action_to_policy_index` 参数、`policy_index_to_action` 返回值、`build_cost` 参数、`ALL_CARD_TYPES` 元素类型全部从 `str` 改为 `CardType`。与 design/19 引入的 `GameState` 类型方向一致。

**D3 — `build_cost` 从 `_CARD_MAP` 移除，改由 `get_card_def()` 派生。** `_BUILD_COST` 字典从 `{c[0]: c[2] for c in _CARD_MAP}` 改为 `{ct: get_card_def(ct).build_cost for ct in ...}`，消除与 `CardDef.build_cost` 的重复。

**D4 — `INPUT_SIZE` / `POLICY_OUTPUT_SIZE` 从 converter.py 移除。** Converter 从 `ai.rnn_v1.model` 导入这两个常量。`app_manager.py` 同步改为从 model.py 导入 `INPUT_SIZE`，不再引用 `Converter.INPUT_SIZE`。

**D5 — 不保留对旧 `str` 接口的兼容。** 遵循项目"重构不保留向后兼容"原则，所有消费者同步更新。

## 3. Module Impact

```
src/backend/
├── app/converter.py            # MODIFIED — D1~D4 核心重构
├── app/app_manager.py          # MODIFIED — INPUT_SIZE 导入源、_make_mask_fn CardType 适配
├── app/selfplay_runner.py      # MODIFIED — action_to_policy_index 参数适配
├── controller/api.py           # MODIFIED — ALL_CARD_TYPES 类型适配
├── ai/rnn_v1/model.py          # 无变更（常量已存在）
└── game/card.py                # 无变更
```

```
tests/
└── unit/test_converter.py      # MODIFIED — 所有 str 字面值改为 CardType
```

## 4. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **Converter 映射构建与调用流程**: [converter_flow.puml](diagrams/converter_flow.puml)
