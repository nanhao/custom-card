# Architecture Design — 消除 CardResolverRegistry (intention #22)

## 1. Motivation

当前设计（design/21）中，`CardType → list[Resolver]` 的映射关系定义在独立的 `CardResolverRegistry` 类中（`resolver/registry.py`），通过模块级注册建立关联。CardDef 本身不携带效果信息。

本次重构将映射关系内聚到 `CardDef` 内部：每张卡牌的定义直接包含其结算器列表。`CardResolverRegistry` 类及 `registry.py` 文件被删除。同时，为解决 `card.py` ↔ `resolver/` 之间的潜在循环依赖，将原 `card.py` 拆分为 `card_type.py` 和 `card_def.py` 两个文件。

## 2. Module Impact

| 文件 | 变更 |
|------|------|
| `game/card_type.py` | **新建** — 仅含 `CardType` 枚举（从 `card.py` 移出） |
| `game/card_def.py` | **新建** — `CardDef`（新增 `resolvers` 字段）+ `_CARD_DEFS` + `get_card_def()` + `get_initial_draw_pile()` |
| `game/card.py` | **删除** — 拆分为 `card_type.py` + `card_def.py` |
| `game/resolver/registry.py` | **删除** — 功能内聚到 `CardDef.resolvers` |
| `game/resolver/__init__.py` | **修改** — 移除 `CardResolverRegistry` 的 re-export；新增 `get_card_def` 的 re-export |
| `game/resolver/orchestrator.py` | **修改** — 通过 `get_card_def(card_type).resolvers` 获取结算器列表，替代 `CardResolverRegistry.get_resolvers()` |
| `game/resolver/base.py` | **修改** — import 路径 `game.card.CardType` → `game.card_type.CardType` |
| 所有 `from game.card import CardType` | **修改** — 改为 `from game.card_type import CardType`（共 17 个文件） |
| 所有 `from game.card import CardDef, get_card_def, get_initial_draw_pile` | **修改** — 改为 `from game.card_def import ...`（共 6 个文件） |
| `tests/unit/test_card.py` | **修改** — import 路径更新，测试用例适配 `resolvers` 字段 |
| `tests/unit/test_resolver.py` | **修改** — 移除 `CardResolverRegistry` 引用，注册表测试替换为 CardDef.resolvers 测试 |

不需要变更：`controller/`、`ai/`、`app/`、`frontend/`、`log/`、`config/`（仅 import 路径更新）。

## 3. Key Design Decisions

**D1 — CardType 独立文件。** `CardType` 枚举从 `card.py` 移出到 `card_type.py`，打破 `card.py` ↔ `resolver/` 的潜在循环依赖链。`card_type.py` 不依赖 game 包内任何其他模块。

**D2 — CardDef.resolvers 字段。** `CardDef` 新增 `resolvers: tuple[type[Resolver], ...]` 字段，直接存储该卡牌的结算器类引用（非实例）。结算时由 `orchestrator.py` 实例化。Resolver 都是无状态类，每次实例化开销可忽略。

**D3 — CardResolverRegistry 删除。** 注册表类及 `registry.py` 文件整体删除。映射关系从 _CARD_DEFS 表的 `resolvers` 字段读取，数据集中在一处，避免注册表与 CardDef 定义不同步的风险。

**D4 — 编排器接口不变。** `CardResolver.resolve(player_1, player_2, cards_1, cards_2)` 签名不变，内部改为通过 `get_card_def(card_type).resolvers` 遍历结算器。返回 `ResolutionResult` 接口不变。对外部调用方（`game_core.py`）透明。

**D5 — 破坏性变更。** `CardResolverRegistry` 被删除，依赖它的代码（测试、编排器）需同步修改。外部模块 GameCore、AppManager、API 等不受影响（仅 import 路径更新）。

## 4. CardDef 中的 Resolver 映射表

| CardType | resolvers (type[Resolver]) |
|----------|---------------------------|
| ATTACK | `(AttackDamageResolver,)` |
| DEFENSE | `(GrantDefenseResolver, GrantGuardResolver)` |
| HEAVY | `(HeavyDamageResolver,)` |
| COUNTER | `(GrantDefenseResolver, ReflectResolver)` |
| ENERGY | `(GainEnergyResolver,)` |
| BUILD | `(BuildResolver,)` |
| FLEX | `(GrantDefenseResolver, FlexDamageResolver)` |

## 5. Key Flows

- **结算流程**: 见 [resolve_flow.puml](diagrams/resolve_flow.puml)
- **类图及依赖关系**: 见 [class_diagram.puml](diagrams/class_diagram.puml)
