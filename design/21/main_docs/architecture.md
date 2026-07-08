# Architecture Design — Resolver 策略模式重构 (intention #21)

## 1. Motivation

`_process_effect` 是一个模块级函数，178 行 if-else 链对 7 种 EffectType 分别硬编码处理逻辑。每增加一种效果或修改现有逻辑都需要改动这个巨型函数，违反开闭原则。

本次重构将效果结算逻辑改为**策略模式**：为每张卡牌的每个效果创建独立的 Resolver 子类，通过 `CardResolverRegistry` 注册。`EffectType` 枚举被消除。

## 2. Module Impact

| 文件 | 变更 |
|------|------|
| `game/resolver.py` | **重写** — Resolver 基类 + 8 个具体 Resolver + CardResolverRegistry + ResolveDetails + CardResolver 编排器 |
| `game/card.py` | **修改** — 移除 EffectType 枚举、CardDef.effects、CardDef.priority |
| `game/game_core.py` | **修改** — 新增结算上下文管理（get_resolve_*），play_round() 改为委派 Resolver 直接操作状态 |
| `game/player.py` | **修改** — 新增 grant_status() 方法 |

不需要变更的文件：`controller/`、`ai/`、`app/`、`frontend/`、`log/`、`config/` 等（resolver 内部重构，外部 API 不变）。

## 3. Key Design Decisions

**D1 — 有状态结算。** Resolver 不再返回累积值（dmg、energy、statuses），而是直接调用 GameCore 暴露的 player 方法修改状态。GameCore 在调用每个 Resolver 前设置结算上下文（source_player、opponent_card、attack_streak、had_defense_pre），Resolver 通过 `game_core.get_resolve_*` 读取。

**D2 — EffectType 枚举消除。** 当前 7 个 EffectType（DAMAGE、GRANT_DEFENSE 等）被移除。效果逻辑封装在 8 个 Resolver 子类中。CardDef 不再需要 `effects` 和 `priority` 字段，只保留 `card_type`、`build_cost`、`display_name`、`icon`。效果组合关系由 CardResolverRegistry 隐式表达。

**D3 — CardResolverRegistry。** 模块级注册表，建立 `CardType -> list[Resolver]` 映射。每张卡注册 1~2 个 Resolver。Resolver 实例在模块加载时注册（无状态单例），通过 `get_priority()` 排序后依次调用。

**D4 — 结算编排保留在 CardResolver。** `CardResolver.resolve()` 仍然是结算入口，但内部逻辑改为：从注册表查询 → 按 priority 排序 → 设置 GameCore 上下文 → 调用 Resolver。`ResolutionResult` 仍然作为返回值（从 GameCore 状态汇总），`details` 来自 `ResolveDetails`。

**D5 — 破坏性变更。** 内部重构，不保持旧 resolver.py 的向后兼容。测试文件 `test_resolver.py` 需要适配新的有状态 API。外部模块的接口不变（CardResolver.resolve() 仍接收相同参数返回相同类型）。

## 4. CardResolverRegistry 映射表

| CardType | Resolver(s) | Priority |
|----------|------------|----------|
| ATTACK | AttackDamageResolver | 10 |
| DEFENSE | GrantDefenseResolver, GrantGuardResolver | 11, 11 |
| HEAVY | HeavyDamageResolver | 10 |
| COUNTER | GrantDefenseResolver, ReflectResolver | 11, 10 |
| ENERGY | GainEnergyResolver | 10 |
| BUILD | BuildResolver | 10 |
| FLEX | GrantDefenseResolver, FlexDamageResolver | 11, 10 |

共 8 个具体 Resolver 类，7 个 CardType 条目。

## 5. Key Flows

- **结算流程**: 见 [resolve_flow.puml](diagrams/resolve_flow.puml)
- **类图及依赖关系**: 见 [class_diagram.puml](diagrams/class_diagram.puml)
