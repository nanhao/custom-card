# Architecture Design — Player 历史记录合并 (intention #16)

## 1. Motivation

当前 `Player` 使用两个平行列表分别记录每回合的卡牌和 BUILD 目标：

- `history: list[CardType]` — 每回合打出的卡牌
- `build_history: list[CardType | None]` — 每回合的 BUILD 目标（非 BUILD 回合填 None）

这两个列表通过索引隐式关联，在 `get_state()` 中需要 `zip(history, build_history)` 来构造 `PlayerHistoryItem`，增加了维护成本和出错风险。而 `PlayerHistoryItem` 这个 dataclass 已经存在，包含了 `card` 和 `build_target` 两个字段，天然适合作为历史记录的基本单元。

本次重构将两个平行列表合并为一个 `list[PlayerHistoryItem]`，消除数据冗余和隐式索引依赖。

## 2. Key Design Decisions

**D1 — Player.history 类型变更。** `list[CardType]` → `list[PlayerHistoryItem]`。`PlayerHistoryItem` 已存在于 `game/player.py`，无需新增模块。每个条目同时包含 `card` 和 `build_target`。

**D2 — 删除 Player.build_history 字段。** 该字段的功能已被 `PlayerHistoryItem.build_target` 完全替代。删除此字段后，不再需要维护两个列表的索引对齐关系。

**D3 — play_round 中历史记录合并。** 将两行分别 append 的代码合并为一个 `history.append(PlayerHistoryItem(card, build_target))`。

**D4 — get_state 简化。** 移除 `zip(p.history, p.build_history)` 逻辑，直接使用 `p.history`（列表中已是 `PlayerHistoryItem` 实例）。

**D5 — CardResolver.get_consecutive_attack 适配。** 参数类型从 `list[CardType]` 改为 `list[PlayerHistoryItem]`。内部通过 `item.card` 提取卡牌类型进行攻击连击计数，行为不变。

**D6 — 序列化格式变更。** 持久化中 history 从 `["attack", "defense", ...]` 改为 `[{"card": "attack", "build_target": null}, ...]`；移除 `build_history` 独立字段。**不保留旧存档兼容性。**

**D7 — 前端无需变更。** 前端 `GameBoard.jsx` 已通过 `typeof c === 'string' ? c : c.card` 兼容字符串和对象两种 history 格式，`get_state()` 返回的 history 条目原本就是 `PlayerHistoryItem` 序列化后的字典，格式不变。

**D8 — Converter 无需变更。** `Converter._action_item_to_index()` 原本就接收 `PlayerHistoryItem`；`build_input_vector()` 通过 `get_state()` 接口获取 history（已是字典格式），不受影响。

## 3. Module Impact

```
src/backend/
├── game/player.py             # MODIFIED — history 类型变更，删除 build_history
├── game/game_core.py          # MODIFIED — play_round 合并历史记录，get_state 简化
├── game/resolver.py           # MODIFIED — get_consecutive_attack 参数类型变更
├── game/session_store.py      # MODIFIED — 序列化/反序列化格式变更，移除 build_history
├── controller/api.py          # 不变 — PlayerHistoryItem JSON 编码已存在
├── app/converter.py           # 不变 — 接口层不变
```

## 4. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **历史记录流程**: [history_flow.puml](diagrams/history_flow.puml)
