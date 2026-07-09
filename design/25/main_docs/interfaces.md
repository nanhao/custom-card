# Interfaces — BuildResolver 结算逻辑迁移 (intention #25)

## 1. ResolveContext

### 新增字段

`resolver/base.py` — `ResolveContext` dataclass 新增字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `build_target` | `CardType \| None` | `None` | 当 `card_played_by_owner` 为 BUILD 时，表示玩家选择的构建目标卡牌类型；其他卡牌类型时为 None |

完整接口（变更后）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `card_owner` | `Player` | 打出该卡牌的玩家 |
| `opponent` | `Player` | 对手玩家 |
| `card_played_by_owner` | `CardType` | 玩家打出的卡牌类型 |
| `card_played_by_opponent` | `CardType` | 对手打出的卡牌类型 |
| `build_target` | `CardType \| None` | 构建目标（仅 BUILD 卡时非 None） |

## 2. BuildResolver

`resolver/build_resolver.py` — `BuildResolver` 接口不变，`resolve()` 实现变更：

### resolve(ctx, details)

**前置条件**：`ctx.card_played_by_owner == CardType.BUILD`（由 GameCore 保证，仅 BUILD 卡会实例化此 Resolver）。

**行为**：
1. 若 `ctx.build_target` 为 `None`，直接返回（防御性处理）
2. 通过 `get_card_def(ctx.build_target)` 获取目标卡牌的 `build_cost`
3. 从 `ctx.card_owner.energy` 扣除 `build_cost`（不低于 0）
4. 将 `ctx.build_target` 加入 `ctx.card_owner.discard_pile`
5. 向 `details` 记录结算详情

**后置条件**：
- `ctx.card_owner.energy` 减少 `build_cost`
- `ctx.card_owner.discard_pile` 末尾新增 `ctx.build_target`

## 3. GameCore.play_round()

### 变更 1：任务创建时传入 build_target

`play_round()` 中创建 `ResolveContext` 时，根据当前处理的卡牌类型条件传入 `build_target`：

- 处理 `cards_1` 时：若 `card_type == CardType.BUILD`，设置 `build_target=build_target_1`，否则 `None`
- 处理 `cards_2` 时：若 `card_type == CardType.BUILD`，设置 `build_target=build_target_2`，否则 `None`

### 变更 2：删除 _process_build() 调用

删除 `play_round()` 中第 6 步的 BUILD 处理：

```
- self._process_build(cards_1[0], build_target_1, self.player_1)
- self._process_build(cards_2[0], build_target_2, self.player_2)
```

### 变更 3：删除 _process_build() 方法

整体删除 `GameCore._process_build()` 方法（当前第 437–454 行）。

### play_round() 接口不变

`play_round()` 方法签名不变：

```
play_round(
    self,
    cards_1: list[CardType],
    cards_2: list[CardType],
    build_target_1: CardType | None = None,
    build_target_2: CardType | None = None,
) -> RoundResult
```
