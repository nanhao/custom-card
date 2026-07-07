# Architecture Design — Game v2 (intention #11)

## 1. Motivation

全面升级游戏机制：引入能量/牌堆/轮次/状态/优先级五大系统，卡牌从4种扩至7种，HP从3升至20。AI模型同步适配：input 14→25，output 4→14（7选卡+7选构建目标），hidden 8→32。

本次是破坏性变更——旧存档、旧模型权重、旧样本文件均不再兼容。

## 2. Module Impact

```
src/backend/
├── game/
│   ├── card.py           # REWRITE — CardType 4→7, 新增 build_cost/priority
│   ├── player.py         # REWRITE — hp默认20, +energy/statuses/draw_pile/discard_pile
│   ├── resolver.py       # REWRITE — 优先级分层结算, 效果级粒度, 状态系统
│   ├── game_core.py      # REWRITE — cycle机制, 牌堆洗牌, cycle末扣血, 无回合上限
│   └── session_store.py  # MODIFIED — 序列化新增字段, ai_state 8→32
├── ai/
│   ├── rnn_v1/model.py   # REWRITE — INPUT_SIZE=25, HIDDEN_SIZE=32, policy output 14
│   ├── registry.py       # MODIFIED — 新增 rnn_v2.1 条目
│   ├── training_runner.py# MODIFIED — ACTION_TO_INDEX 7值, policy 14-d, +build_target
│   ├── training_utils.py # MODIFIED — 与 training_runner.py 同步
│   └── training_config.py# UNCHANGED
├── app/
│   ├── converter.py      # REWRITE — 7卡牌映射, INPUT_SIZE=25, +round_in_cycle/cycle, +build_target索引
│   └── app_manager.py    # MODIFIED — hidden_state 8→32, mask_fn 双段(手牌+构建目标)
├── controller/
│   └── api.py            # MODIFIED — PVE: ai_state 8→32, 序列化新增game字段
│   └── serializers.py    # MODIFIED — CardType 映射扩展
└── config/
    └── backend.json      # MODIFIED — 新增模型常量(可选), 训练参数不变
```

## 3. Key Design Decisions

**D1 — CardType 重新定义。** 从4值枚举扩展为7值（attack/defense/heavy/counter/energy/build/flex），每个卡牌新增 `priority: int` 和 `build_cost: int` 属性。在 `card.py` 中定义 `CardDef` 数据类承载元数据，`CardType` 枚举仅做标识。

**D2 — Player 新增能量/状态/牌堆字段。** `energy: int = 0`（上限5），`statuses: list[str] = []`（防御/联防），`draw_pile: list[CardType]`，`discard_pile: list[CardType]`。HP默认值从3改为20。`history` 字段保留但用于跨轮连续攻击计数。`hand` 初始为空。

**D3 — Resolver 改为优先级分层结算。** 不再以整张卡为单位结算，而是将每张卡拆解为若干 `Effect`（优先级+效果函数）。结算流程：收集全部效果 → 按优先级分组（11先于10）→ 同优先级内同时结算伤害 → 汇总后统一扣血。状态授予在同优先级内线性执行。

**D4 — GameCore 新增 cycle（轮）机制。** 术语：`round`=回合（全局递增编号），`cycle`=轮（5回合组成）。`MAX_ROUNDS` 移除，新增 `ROUNDS_PER_CYCLE = 5`、`cycle_number`、`round_in_cycle`(1-5)。`round_number` 保留不动（全局回合计数）。`start_game()` 初始化牌堆并触发 `_start_cycle()`（回收手牌+洗牌+抽7张）。`play_round()` 完成后检查是否满5回合→若满则触发 cycle 末扣血+下一 cycle。`_check_game_end()` 仅检查 HP=0。

**D5 — 模型维度硬提升+双输出头。** `GRUPolicyModel`：Input(25)→GRU(32)→Linear(32→14)→softmax。前7维选要出的卡，后7维选构建目标（仅在出构建卡时使用）。mask_fn 需双段屏蔽：**推理时**，前段屏蔽手牌中没有的卡 + energy=0时屏蔽BUILD，后段屏蔽能量不足以构建的卡；**训练时**，前段仍屏蔽不在手牌中的卡（物理不可能动作），后段不屏蔽（让模型学习构建目标选择）。`GRUValueModel`：Input(25)→GRU(32)→Linear(32→1)。INITIAL_HIDDEN_STATE 长度变为32。旧权重完全不兼容。

**D6 — Converter 同步扩展。** `NUM_CARD_TYPES=7`，`INPUT_SIZE=25`。卡牌映射表新增 energy(4)/build(5)/flex(6)。`build_input_vector` 新增维度23-24：当前轮内回合序号(1-5)和当前 cycle 编号。card_to_index/index_to_card 映射仅针对前7维(选卡)，构建目标的7维复用同一映射。

**D7 — 破坏性变更，不保留向后兼容。** 旧存档格式、旧模型权重、旧样本文件均废弃。selfplay脚本输出新格式样本（25-d input），train脚本仅接受新格式。

**D8 — HP 输入不归一化。** Converter 中 HP 值（0-20）直接传入模型，不做 [0,1] 归一化。理由：GRU 层可自适应数值尺度，保留物理含义便于调试；若训练中发现收敛问题可后续调整。

**D9 — BUILD 可递归构建。** 构建卡可以构建 BUILD 自身（build_cost=1），允许牌组中出现多张 BUILD 卡。这是设计意图，增加策略深度（但过度构建会稀释攻击防御牌比例，自然平衡）。

**D10 — 安全假设。** `get_state` 返回双方手牌内容，依赖调用方按玩家视角过滤。当前热座/单机模式无需服务端过滤；若未来扩展网络对战，需在 API 层增加身份鉴权和响应过滤。

## 4. What Changes

| 文件 | 变更 |
|------|------|
| `game/card.py` | CardType 扩展7值 + CardDef(effects双优先级+build_cost) |
| `game/player.py` | +energy/statuses/draw_pile/discard_pile, hp默认20, hand初始空 |
| `game/resolver.py` | 改为 Effect 粒度优先级分层结算 |
| `game/game_core.py` | +cycle机制, +牌堆洗牌, +cycle末扣血, -MAX_ROUNDS |
| `game/session_store.py` | 序列化新增 energy/statuses/piles, ai_state 长度8→32 |
| `ai/rnn_v1/model.py` | INPUT_SIZE=25, HIDDEN_SIZE=32, policy output 14(7选卡+7构建目标) |
| `ai/registry.py` | 新增 rnn_v2.1 条目(gen1随机权重) |
| `ai/training_runner.py` | ACTION_TO_INDEX 7值, policy 14-d, 样本含 build_target |
| `app/converter.py` | 7卡牌, INPUT_SIZE=25, 新增 round_in_cycle/cycle/build_target 映射 |
| `app/app_manager.py` | hidden_state 8→32, mask_fn 双段:手牌(0-6)+构建目标(7-13) |
| `controller/api.py` | ai_state 初始化 8→32 |
| `controller/serializers.py` | CardType 映射扩展 |

## 5. Key Flows

- **回合结算流程**: 见 [round_flow.puml](diagrams/round_flow.puml)
- **游戏生命周期**: 见 [game_lifecycle.puml](diagrams/game_lifecycle.puml)
- **类图及依赖关系**: 见 [class_diagram.puml](diagrams/class_diagram.puml)
