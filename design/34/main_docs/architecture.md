# Architecture Design — 56维输入 + 牌堆互相可见 (intention #34)

## 1. Motivation

当前设计 #27 的 42-d input 仅包含我方弃牌堆和抽牌堆分布信息，对方牌堆不可见。同时游戏机制中只有请求方牌堆内容暴露，对手牌堆仅显示计数。

本次变更：
- **Input 56-d**：新增对方弃牌堆（7 维）+ 对方抽牌堆（7 维），input.md 成为模型输入唯一定义文件
- **游戏机制**：双方弃牌堆和抽牌堆互相可见

## 2. Input Vector Specification

**INPUT_SIZE = 56**（由 `intention/input.md` 唯一定义）

| 索引 | 内容 | 范围 | 说明 |
|------|------|------|------|
| 0 | 我方血量 | ≥0 | |
| 1 | 对方血量 | ≥0 | |
| 2 | 我方能量 | 0-5 | |
| 3 | 对方能量 | 0-5 | |
| 4 | 当前回合数 | 1-3 | round_in_cycle |
| 5 | 当前轮数 | ≥1 | cycle |
| 6 | 我方手牌-攻击数 | ≥0 | |
| 7 | 我方手牌-防御数 | ≥0 | |
| 8 | 我方手牌-重击数 | ≥0 | |
| 9 | 我方手牌-防守反击数 | ≥0 | |
| 10 | 我方手牌-聚能数 | ≥0 | |
| 11 | 我方手牌-构建数 | ≥0 | |
| 12 | 我方手牌-攻守兼备数 | ≥0 | |
| 13 | 对方手牌-攻击数 | ≥0 | |
| 14 | 对方手牌-防御数 | ≥0 | |
| 15 | 对方手牌-重击数 | ≥0 | |
| 16 | 对方手牌-防守反击数 | ≥0 | |
| 17 | 对方手牌-聚能数 | ≥0 | |
| 18 | 对方手牌-构建数 | ≥0 | |
| 19 | 对方手牌-攻守兼备数 | ≥0 | |
| 20 | 我方防御持续回合 | ≥0 | 0=无该状态 |
| 21 | 我方联防持续回合 | ≥0 | 0=无该状态 |
| 22 | 我方脆弱持续回合 | ≥0 | 0=无该状态 |
| 23 | 对方防御持续回合 | ≥0 | 0=无该状态 |
| 24 | 对方联防持续回合 | ≥0 | 0=无该状态 |
| 25 | 对方脆弱持续回合 | ≥0 | 0=无该状态 |
| 26 | 我方弃牌堆-攻击数 | ≥0 | |
| 27 | 我方弃牌堆-防御数 | ≥0 | |
| 28 | 我方弃牌堆-重击数 | ≥0 | |
| 29 | 我方弃牌堆-防守反击数 | ≥0 | |
| 30 | 我方弃牌堆-聚能数 | ≥0 | |
| 31 | 我方弃牌堆-构建数 | ≥0 | |
| 32 | 我方弃牌堆-攻守兼备数 | ≥0 | |
| 33 | 对方弃牌堆-攻击数 | ≥0 | **新增** |
| 34 | 对方弃牌堆-防御数 | ≥0 | **新增** |
| 35 | 对方弃牌堆-重击数 | ≥0 | **新增** |
| 36 | 对方弃牌堆-防守反击数 | ≥0 | **新增** |
| 37 | 对方弃牌堆-聚能数 | ≥0 | **新增** |
| 38 | 对方弃牌堆-构建数 | ≥0 | **新增** |
| 39 | 对方弃牌堆-攻守兼备数 | ≥0 | **新增** |
| 40 | 我方抽牌堆-攻击数 | ≥0 | 从索引 33 偏移 |
| 41 | 我方抽牌堆-防御数 | ≥0 | |
| 42 | 我方抽牌堆-重击数 | ≥0 | |
| 43 | 我方抽牌堆-防守反击数 | ≥0 | |
| 44 | 我方抽牌堆-聚能数 | ≥0 | |
| 45 | 我方抽牌堆-构建数 | ≥0 | |
| 46 | 我方抽牌堆-攻守兼备数 | ≥0 | |
| 47 | 对方抽牌堆-攻击数 | ≥0 | **新增** |
| 48 | 对方抽牌堆-防御数 | ≥0 | **新增** |
| 49 | 对方抽牌堆-重击数 | ≥0 | **新增** |
| 50 | 对方抽牌堆-防守反击数 | ≥0 | **新增** |
| 51 | 对方抽牌堆-聚能数 | ≥0 | **新增** |
| 52 | 对方抽牌堆-构建数 | ≥0 | **新增** |
| 53 | 对方抽牌堆-攻守兼备数 | ≥0 | **新增** |
| 54 | 我方上回合出牌 | -1 或 0-13 | policy index，首回合=-1 |
| 55 | 对方上回合出牌 | -1 或 0-13 | policy index，首回合=-1 |

## 3. Key Design Decisions

**D1 — INPUT_SIZE 从 42 改为 56。** 新增对方弃牌堆（7 维）+ 对方抽牌堆（7 维）。我方抽牌堆从索引 33-39 移至 40-46，last action 从 40-41 移至 54-55。破坏性变更：旧模型权重不兼容，全部删除。

**D2 — input.md 成为 model input 唯一定义文件。** `ai/constants.py` 中的 `INPUT_SIZE` 以及其他代码中的 input 维度描述均对齐 `intention/input.md`。

**D3 — 牌堆互相可见：`player_perspective` 参数废弃。** `GameCore.get_state()` 不再需要 `player_perspective` 参数——所有玩家随时能看到双方完整的抽牌堆和弃牌堆内容。`PlayerState.draw_pile` / `discard_pile` 始终为 `list[CardType]`（不再是 `list[CardType] | None`）。保留参数向后兼容但不产生行为差异。

**D4 — Converter 移除 pile-is-None 检查。** 由于 D3 保证 piles 始终非空，`build_input_vector` 中针对 `draw_pile is None` / `discard_pile is None` 的 ValueError 检查和前置条件不再需要。

**D5 — GRU hidden_size=32 不变。** 输入从 42 扩到 56，新增维度是"我方已知的对方牌堆分布"而非更复杂的游戏规则。hidden_size 保持不变。第一个 Linear 层的 input_size 随 INPUT_SIZE 变更。

**D6 — rnn_v2 SKIP_INPUT_SIZE 更新。** `SKIP_INPUT_SIZE = INPUT_SIZE + HIDDEN_SIZE = 56 + 32 = 88`（原 74）。三个 Linear 层中 fc1 输入维度相应变更。

**D7 — 旧权重文件全部删除。** 所有 gen 目录下的 `.pth` 文件不兼容新 input 维度，按架构 §7 D9 直接删除。

**D8 — 前端展示双方完整牌堆。** 状态栏中双方均显示可点击的牌堆计数，点击弹出牌堆查看器展示完整卡牌列表。无需 `perspective` 查询参数。

## 4. Module Impact

```
src/backend/
├── ai/constants.py            # MODIFIED — INPUT_SIZE = 56
├── ai/rnn_v1/model.py         # MODIFIED — GRU input_size=56, 注释更新
├── ai/rnn_v2/model.py         # MODIFIED — GRU input_size=56, SKIP_INPUT_SIZE=88, fc1=Linear(88→64)
├── app/converter.py           # MODIFIED — build_input_vector 56-d, 新增对方牌堆段, 移除 None 检查
├── game/game_core.py          # MODIFIED — get_state() 始终暴露双方 piles
├── game/game_state.py         # MODIFIED — draw_pile/discard_pile 类型从 Optional 变非 Optional; to_dict() 总是序列化 piles
├── controller/api.py          # MODIFIED — 移除 perspective 查询参数逻辑（保留兼容但行为统一）
├── app/selfplay_runner.py     # NO CHANGE — 自动适配新 input 维度
├── app/app_manager.py         # NO CHANGE — mask_fn 不变
└── ai/training_runner.py      # NO CHANGE — 自动适配

src/frontend/
├── src/App.jsx                # MODIFIED — handleViewPiles 支持双方; 移除 getMyPiles(已不需要)
├── src/GameBoard.jsx          # MODIFIED — 状态栏双方牌堆均可点击; pileViewer 展示对方牌堆内容
├── src/api.js                 # MODIFIED — getState 可用于获取牌堆; getMyPiles 可移除或简化
└── tests/App.test.jsx         # MODIFIED — 更新 pile viewer 相关测试

src/backend/ai/rnn_v*/gen*/*.pth  # DELETED — 旧权重不兼容
```

## 5. Frontend Impact

- **状态栏**：双方玩家行中的 `Deck`/`Disc` 计数均变为可点击，点击弹出牌堆查看器
- **牌堆查看器**：弹出层标题根据查看的玩家动态显示（"Your Piles" / "Opponent's Piles"），内容来自 `state.players[pid].draw_pile` 和 `discard_pile`
- **API 调用简化**：`getState()` 返回的状态已包含双方完整牌堆，无需单独的 `getMyPiles` 接口和 `perspective` 参数
- **PVP 模式**：pass screen 提示文字中移除仅自己可见牌堆的相关提示

## 6. Diagrams

- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
- **Input 构造流程**: [input_construction_flow.puml](diagrams/input_construction_flow.puml)
- **牌堆可见性流程**: [pile_visibility_flow.puml](diagrams/pile_visibility_flow.puml)
