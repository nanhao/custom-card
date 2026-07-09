# Test Cases — BuildResolver 结算逻辑迁移 (intention #25)

---

## 1. ResolveContext — build_target 字段

### TC-CTX-01: 默认值为 None
- **Given**: 创建 `ResolveContext(card_owner=p1, opponent=p2, card_played_by_owner=CardType.ATTACK, card_played_by_opponent=CardType.DEFENSE)`
- **Then**: `ctx.build_target` 为 `None`

### TC-CTX-02: 显式设置 build_target
- **Given**: 创建 `ResolveContext(card_owner=p1, opponent=p2, card_played_by_owner=CardType.BUILD, card_played_by_opponent=CardType.ATTACK, build_target=CardType.HEAVY)`
- **Then**: `ctx.build_target == CardType.HEAVY`

---

## 2. BuildResolver — 结算逻辑

### TC-BLD-01: 构建 ATTACK（cost=1），能量充足
- **Given**: `ctx.card_owner.energy = 2`，`ctx.build_target = CardType.ATTACK`（build_cost=1），`card_owner.discard_pile` 为空
- **When**: `BuildResolver().resolve(ctx, details)`
- **Then**: `card_owner.energy == 1`；`CardType.ATTACK in card_owner.discard_pile`；details 包含 "BUILD ATTACK (cost 1)"

### TC-BLD-02: 构建 HEAVY（cost=2），能量刚好
- **Given**: `ctx.card_owner.energy = 2`，`ctx.build_target = CardType.HEAVY`（build_cost=2），`card_owner.discard_pile` 为空
- **When**: `BuildResolver().resolve(ctx, details)`
- **Then**: `card_owner.energy == 0`；`CardType.HEAVY in card_owner.discard_pile`；details 包含 "BUILD HEAVY (cost 2)"

### TC-BLD-03: 构建 FLEX（cost=2），能量充足且有其他 discard
- **Given**: `ctx.card_owner.energy = 5`，`ctx.build_target = CardType.FLEX`（build_cost=2），`card_owner.discard_pile = [CardType.ATTACK]`
- **When**: `BuildResolver().resolve(ctx, details)`
- **Then**: `card_owner.energy == 3`；`card_owner.discard_pile == [CardType.ATTACK, CardType.FLEX]`

### TC-BLD-04: build_target 为 None（防御性处理）
- **Given**: `ctx.card_owner.energy = 2`，`ctx.build_target = None`，`card_owner.discard_pile` 为空
- **When**: `BuildResolver().resolve(ctx, details)`
- **Then**: `card_owner.energy == 2`（不变）；`card_owner.discard_pile` 为空（不变）；details 为空或包含 "BUILD — no target"

### TC-BLD-05: 结算详情记录验证
- **Given**: `ctx.card_owner.energy = 3`，`ctx.build_target = CardType.ENERGY`（build_cost=1）
- **When**: `BuildResolver().resolve(ctx, details)`
- **Then**: `details.details` 中包含一条形如 "BUILD energy (cost 1)" 的记录

### TC-BLD-06: priority 为 10
- **Given**: `BuildResolver()`
- **When**: 调用 `.get_priority()`
- **Then**: 返回 `10`

---

## 3. GameCore — 完整 BUILD 流程（回归）

### TC-GC-BLD-01: 通过 play_round 构建 ATTACK
- **Given**: `GameCore.start_game()`；P1 energy=2, hand=[BUILD]；P2 hand=[ATTACK]
- **When**: `play_round([BUILD], [ATTACK], build_target_1=CardType.ATTACK)`
- **Then**: P1 energy 减少 1（ATTACK build_cost=1）；P1 discard_pile 包含 ATTACK；BUILD 卡和 ATTACK 卡各被弃置；返回 RoundResult 完整

### TC-GC-BLD-02: 通过 play_round 构建 HEAVY（cost=2）
- **Given**: `GameCore.start_game()`；P1 energy=2, hand=[BUILD]；P2 hand=[DEFENSE]
- **When**: `play_round([BUILD], [DEFENSE], build_target_1=CardType.HEAVY)`
- **Then**: P1 energy 减少 2；P1 discard_pile 包含 HEAVY；P2 获得 DEFENSE（不受 BUILD 影响）

### TC-GC-BLD-03: BUILD 能量不足验证仍抛异常（_validate 不变）
- **Given**: `GameCore.start_game()`；P1 energy=0, hand=[BUILD]；P2 hand=[ATTACK]
- **When**: `play_round([BUILD], [ATTACK])`
- **Then**: 抛出 `ValueError`，匹配 "energy is 0"

### TC-GC-BLD-04: BUILD 缺少 build_target 验证仍抛异常
- **Given**: `GameCore.start_game()`；P1 energy=1, hand=[BUILD]；P2 hand=[ATTACK]
- **When**: `play_round([BUILD], [ATTACK])`（不传 build_target_1）
- **Then**: 抛出 `ValueError`，匹配 "build_target_1 is required"

### TC-GC-BLD-05: BUILD HEAVY cost>energy 验证仍抛异常
- **Given**: `GameCore.start_game()`；P1 energy=1, hand=[BUILD]；P2 hand=[ATTACK]
- **When**: `play_round([BUILD], [ATTACK], build_target_1=CardType.HEAVY)`
- **Then**: 抛出 `ValueError`，匹配 "cost 2 > energy 1"

### TC-GC-BLD-06: 非 BUILD 卡牌 build_target 被忽略
- **Given**: `GameCore.start_game()`；P1 energy=5, hand=[ATTACK]；P2 hand=[DEFENSE]
- **When**: `play_round([ATTACK], [DEFENSE], build_target_1=CardType.HEAVY)`
- **Then**: 正常结算（ATTACK vs DEFENSE）；P1 energy 不变（build_target 被忽略）；无异常抛出

### TC-GC-BLD-07: _process_build 方法已被删除
- **Given**: `GameCore` 实例
- **When**: 尝试访问 `gm._process_build`
- **Then**: 抛出 `AttributeError`

### TC-GC-BLD-08: 历史记录中 build_target 正确
- **Given**: `GameCore.start_game()`；P1 energy=2, hand=[BUILD]；P2 hand=[ATTACK]
- **When**: `play_round([BUILD], [ATTACK], build_target_1=CardType.ATTACK)`
- **Then**: P1 history 末尾为 `PlayerHistoryItem(CardType.BUILD, CardType.ATTACK)`

### TC-GC-BLD-09: 双方同时 BUILD
- **Given**: `GameCore.start_game()`；P1 energy=2, hand=[BUILD]；P2 energy=2, hand=[BUILD]
- **When**: `play_round([BUILD], [BUILD], build_target_1=CardType.ATTACK, build_target_2=CardType.DEFENSE)`
- **Then**: P1 energy 减少 1，discard_pile 包含 ATTACK；P2 energy 减少 1，discard_pile 包含 DEFENSE
