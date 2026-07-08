# Architecture Design — RoundResult 字段层次化重组 (intention #17)

## 1. Motivation

当前 `RoundResult` 所有字段扁平分布在顶层，按 Phase 1/2/3 注释区分，但结构上无强制层次约束。随着三阶段快照逐渐增多（Phase 1 结算快照、Phase 2 周期结束动作、Phase 3 最终状态），字段命名被迫使用 `after_` 前缀区分不同阶段的同名数据（如 `hp_1_after` vs `after_hp_1`），大幅降低了可读性和可维护性。

本次重构将三个阶段的字段分别收敛到嵌套 dataclass 中，同时新增 `before` 阶段快照，使 RoundResult 形成四个清晰的语义层。

## 2. Key Design Decisions

**D1 — 四层嵌套结构。** `RoundResult` 包含四个嵌套 dataclass：
- `BeforeResolve` — 结算前状态（HP、能量、状态）
- `AfterResolve` — 卡牌结算后状态 + 游戏结束判定
- `EndActions` — 回合结束动作（周期流血）
- `AfterEndActions` — 回合结束处理后状态（含游戏结束判定 + 手牌）

**D2 — 顶层仅保留回合标识和出牌信息。** `round`、`cycle`、`round_in_cycle`、`cards_1`、`cards_2` 保留在顶层，因为它们是最基本的回合描述信息，不属于任何阶段快照。

**D3 — 游戏结束判定出现于两个位置。** `AfterResolve.is_game_over` 判断结算伤害是否致死；`AfterEndActions.is_game_over` 判断周期流血是否致死。两者独立，不互相覆盖。

**D4 — `after_end_actions` 包含 `hand_cards_1`/`hand_cards_2`。** 新增字段，记录回合结束后双方手牌内容（若周期结束则展示洗牌后的新手牌）。该字段仅在 API 序列化层消费（前端展示手牌数量）。

**D5 — 不保留旧存档兼容性。** 序列化格式从扁平字典变为嵌套结构，旧存档无法还原。项目策略允许 breaking changes。

**D6 — `end_actions` 中的 `bleed_damage` 仅当 `cycle_ended=True` 时有意义。** `cycle_ended=False` 时 `bleed_damage_1`/`bleed_damage_2` 为 0。

## 3. Module Impact

```
src/backend/
├── game/game_core.py          # MODIFIED — RoundResult 重构 + play_round 赋值路径变更
├── controller/serializers.py  # MODIFIED — round_result_to_dict 适配嵌套结构
├── controller/api.py          # 不变 — 仅调用 round_result_to_dict
├── app/selfplay_runner.py     # MODIFIED — 日志中访问路径变更
├── app/converter.py           # 不变 — 不依赖 RoundResult
├── game/session_store.py      # MODIFIED — round_log 序列化格式变更
src/frontend/
├── src/GameBoard.jsx          # MODIFIED — round_log 访问路径变更
├── src/App.jsx                # MODIFIED — round_result 字段访问路径变更
├── tests/App.test.jsx         # MODIFIED — mock 数据适配新格式
tests/
├── unit/test_game_core.py     # MODIFIED — 断言适配新结构
├── integration/test_controller_api.py  # 不变 — assert 路径适配
```

## 4. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **play_round 流程**: [round_flow.puml](diagrams/round_flow.puml)
