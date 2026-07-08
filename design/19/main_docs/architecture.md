# Architecture Design — GameState 重构 (intention #20)

## 1. Motivation

当前 `GameCore.get_state()` 返回一个嵌套 `dict`，其中：
- `hand` 用 `c.value` 存为字符串（如 `"attack"`）
- `statuses` 用 `ps.to_dict()` 存为字典（如 `{"type": "defense", "lifetime": 1}`）
- `history` 直接存 `PlayerHistoryItem` 对象，依赖 Flask 自定义 JSON encoder 序列化

这违反了项目的 **dataclass 规范**（"禁止使用 dict 存储具有固定字段的数据"），且类型不安全。

本次重构引入 `GameState` 和 `PlayerState` 两个 dataclass，将 `get_state()` 返回值从无类型 dict 替换为类型化实例。**GameState 类中禁用卡片、状态的字符串/dict 表示，必须使用 `CardType`、`PlayerStatus`、`PlayerHistoryItem` 等类型类。**

## 2. Key Design Decisions

**D1 — 新增 `game/game_state.py` 模块。** 集中定义 `GameState` 和 `PlayerState` 两个 dataclass，遵循 "game 层纯逻辑" 原则，不依赖 AI、UI、I/O。

**D2 — `PlayerState` 替代 `_player_dict` 内部函数。** 当前 `get_state()` 内部用嵌套函数 `_player_dict()` 动态构造玩家状态 dict。重构后用 `PlayerState` dataclass 实例替代。

**D3 — 牌堆视角差异化。** 当前 `player_perspective` 参数控制牌堆内容显示还是仅显示计数。重构后 `PlayerState` 同时持有 `draw_pile: list[CardType] | None` 和 `draw_pile_count: int`。当 pile 内容不可见时，`draw_pile` 为 `None`，`draw_pile_count` 始终有值。

**D4 — 序列化模式遵循现有惯例。** `GameState` 提供 `to_dict()` 静态方法（类比 `RoundResult.to_dict()`），在 API 边界（`serializers.py`）调用。`Converter` 和 `AppManager` 直接使用类型化的 `GameState`，避免 dict 的字符串还原。

**D5 — 不保留对旧 `get_state` dict 返回值的兼容。** 遵循项目 "重构不保留向后兼容" 原则，所有消费者同步更新。

**D6 — `Converter.build_input_vector` 签名变更。** 从 `(state: dict, ...)` 改为 `(state: GameState, ...)`。内部通过 `card.value` 获取字符串用于字符串键的查找表（`_CARD_TYPE_ORDER` 等），不做 Converter 内部数据结构的额外变更。

**D7 — `AppManager._make_mask_fn` 接收 `list[CardType]`。** 签名从 `(hand: list[str], energy: int)` 改为 `(hand: list[CardType], energy: int)`。

**D8 — `GameState` 不含 `ai` 字段。** `ai` 是控制器层注入的元数据，不属于游戏状态。`to_dict()` 后由调用方注入 `"ai"` 键。

## 3. Module Impact

```
src/backend/
├── game/game_state.py          # NEW — GameState + PlayerState dataclass
├── game/game_core.py           # MODIFIED — get_state() 返回 GameState
├── app/converter.py            # MODIFIED — build_input_vector 接受 GameState
├── app/app_manager.py          # MODIFIED — 方法签名 state: GameState
├── app/selfplay_runner.py      # MODIFIED — pre_state 类型适配
├── controller/serializers.py   # MODIFIED — 新增 game_state_to_dict()
├── controller/api.py           # MODIFIED — get_state() 结果先 to_dict()
└── (无前端变更)                 # JSON 格式兼容，前端无需修改
```

## 4. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **get_state 流程交互图**: [get_state_flow.puml](diagrams/get_state_flow.puml)
