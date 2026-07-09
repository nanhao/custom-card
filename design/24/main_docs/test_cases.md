# Test Cases — 卡牌定义对齐 card.md (intention #24)

---

## 1. StatusType 枚举 — FRAGILE

### TC-ST-01: StatusType 包含 FRAGILE
- **Given**: 导入 `StatusType`
- **Then**: `StatusType.FRAGILE` 存在，`value` 为 `"fragile"`

### TC-ST-02: StatusType 成员数
- **Given**: 遍历 `StatusType`
- **Then**: 成员数为 3（DEFENSE, GUARD, FRAGILE）

### TC-ST-03: FRAGILE 不等于 DEFENSE 或 GUARD
- **Given**: `StatusType.FRAGILE`
- **Then**: `!= StatusType.DEFENSE` 且 `!= StatusType.GUARD`

---

## 2. PlayerStatus — FRAGILE 状态

### TC-PS-01: 创建 FRAGILE 状态
- **Given**: `PlayerStatus(StatusType.FRAGILE, lifetime=2)`
- **Then**: `status_type == StatusType.FRAGILE`, `lifetime == 2`

### TC-PS-02: FRAGILE to_dict()
- **Given**: `PlayerStatus(StatusType.FRAGILE, lifetime=2)`
- **When**: 调用 `.to_dict()`
- **Then**: 返回 `{"type": "fragile", "lifetime": 2}`

---

## 3. Player.grant_status() — lifetime 参数

### TC-GS-01: DEFENSE 默认 lifetime=1（向后兼容）
- **Given**: 新 Player，调用 `grant_status(StatusType.DEFENSE)`
- **Then**: 状态 `lifetime == 1`；其余代码无变更

### TC-GS-02: FRAGILE 指定 lifetime=2
- **Given**: 新 Player，调用 `grant_status(StatusType.FRAGILE, lifetime=2)`
- **Then**: 状态 `lifetime == 2`；`has_status(StatusType.FRAGILE)` 为 True

### TC-GS-03: GUARD 默认 lifetime=1（向后兼容）
- **Given**: 新 Player，调用 `grant_status(StatusType.GUARD)`
- **Then**: 状态 `lifetime == 1`

---

## 4. CardDef 对齐

### TC-CD-01: HEAVY build_cost 为 2
- **Given**: `get_card_def(CardType.HEAVY)`
- **Then**: `cd.build_cost == 2`

### TC-CD-02: HEAVY resolvers 含 GrantFragileResolver
- **Given**: `get_card_def(CardType.HEAVY)`
- **Then**: `cd.resolvers == (HeavyDamageResolver, GrantFragileResolver)`

### TC-CD-03: COUNTER display_name 为 "防守反击"
- **Given**: `get_card_def(CardType.COUNTER)`
- **Then**: `cd.display_name == "防守反击"`

### TC-CD-04: 其余 CardDef build_cost 不变
- **Given**: `get_card_def(CardType.ATTACK)`, `DEFENSE`, `COUNTER`, `ENERGY`, `BUILD`, `FLEX`
- **Then**: build_cost 分别为 1, 1, 2, 1, 1, 2

### TC-CD-05: 其余 CardDef resolvers 数量不变
- **Given**: `get_card_def(CardType.ATTACK)` 等（除 HEAVY 外）
- **Then**: ATTACK: 1; DEFENSE: 2; COUNTER: 2; ENERGY: 1; BUILD: 1; FLEX: 2

---

## 5. GrantFragileResolver

### TC-GF-01: priority 为 10
- **Given**: `GrantFragileResolver()`
- **When**: 调用 `.get_priority()`
- **Then**: 返回 10

### TC-GF-02: 对方无 GUARD 时赋予 FRAGILE
- **Given**: `ctx.opponent` 无 GUARD，无 FRAGILE
- **When**: `GrantFragileResolver().resolve(ctx, details)`
- **Then**: `ctx.opponent.has_status(StatusType.FRAGILE)` 为 True；lifetime == 2；details 包含 "gains FRAGILE (2 rounds)"

### TC-GF-03: 对方有 GUARD 时被阻挡
- **Given**: `ctx.opponent` 有 GUARD 状态
- **When**: `GrantFragileResolver().resolve(ctx, details)`
- **Then**: `ctx.opponent.has_status(StatusType.FRAGILE)` 为 False；details 包含 "blocked by opponent GUARD"

### TC-GF-04: 对方无 GUARD 但已有 FRAGILE
- **Given**: `ctx.opponent` 已有 FRAGILE（lifetime=1），无 GUARD
- **When**: `GrantFragileResolver().resolve(ctx, details)`
- **Then**: opponent 新增第二个 FRAGILE（lifetime=2）；共 2 个 FRAGILE 状态

---

## 6. GrantDefenseResolver — FRAGILE 阻挡

### TC-GD-01: 持有 FRAGILE 时被阻挡
- **Given**: `ctx.card_owner` 有 FRAGILE 状态
- **When**: `GrantDefenseResolver().resolve(ctx, details)`
- **Then**: `ctx.card_owner.has_status(StatusType.DEFENSE)` 为 False；details 包含 "DEFENSE blocked by FRAGILE"

### TC-GD-02: 无 FRAGILE 时正常赋予 DEFENSE
- **Given**: `ctx.card_owner` 无 FRAGILE，无 DEFENSE
- **When**: `GrantDefenseResolver().resolve(ctx, details)`
- **Then**: `ctx.card_owner.has_status(StatusType.DEFENSE)` 为 True；details 包含 "gains DEFENSE"

---

## 7. GrantGuardResolver — FRAGILE 阻挡

### TC-GG-01: 持有 FRAGILE 时被阻挡
- **Given**: `ctx.card_owner` 有 FRAGILE 和 pre-round DEFENSE
- **When**: `GrantGuardResolver().resolve(ctx, details)`
- **Then**: `ctx.card_owner.has_status(StatusType.GUARD)` 为 False；details 包含 "GUARD blocked by FRAGILE"

### TC-GG-02: 无 FRAGILE 且 pre-round DEFENSE 时正常赋予 GUARD
- **Given**: `ctx.card_owner` 无 FRAGILE，有 pre-round DEFENSE，无 GUARD
- **When**: `GrantGuardResolver().resolve(ctx, details)`
- **Then**: `ctx.card_owner.has_status(StatusType.GUARD)` 为 True；details 包含 "gains GUARD"

### TC-GG-03: 无 FRAGILE 但无 pre-round DEFENSE 时不赋予
- **Given**: `ctx.card_owner` 无 FRAGILE，无 DEFENSE
- **When**: `GrantGuardResolver().resolve(ctx, details)`
- **Then**: `ctx.card_owner.has_status(StatusType.GUARD)` 为 False

---

## 8. 优先级交互 — 同回合 FRAGILE 不阻挡当回合防御

### TC-PR-01: HEAVY vs DEFENSE — 同回合防御先于脆弱生效
- **Given**: P1 出 DEFENSE（GrantGuardResolver 12, GrantDefenseResolver 11）；P2 出 HEAVY（HeavyDamageResolver 10, GrantFragileResolver 10）；双方无预存状态
- **When**: `CardResolver.resolve(tasks)`
- **Then**: P1 获得 DEFENSE 状态（priority 11 先于 10 执行）；P1 同时获得 FRAGILE 状态（lifetime=2）；P1 HP 不变（HEAVY 被 DEFENSE 阻挡？不，HEAVY 只看 GUARD）；P1 HP 减少 1（HEAVY damage，无 GUARD）

### TC-PR-02: HEAVY vs 无防御 — FRAGILE 正常赋予
- **Given**: P1 出 ATTACK；P2 出 HEAVY；双方无预存状态
- **When**: `CardResolver.resolve(tasks)`
- **Then**: P1 HP 减少 1（HEAVY damage）；P1 获得 FRAGILE（lifetime=2）；P2 HP 减少 1（ATTACK damage）

### TC-PR-03: 下回合 FRAGILE 阻挡 DEFENSE 获得
- **Given**: 上一回合 P1 被赋予 FRAGILE（lifetime=2）；本回合开始前 tick 使 lifetime 变为 1；P1 出 DEFENSE
- **When**: `CardResolver.resolve(tasks)`
- **Then**: `GrantDefenseResolver.resolve()` 检测到 FRAGILE → 跳过；P1 未获得 DEFENSE 状态；P1 HP 按对方卡牌结算

### TC-PR-04: 两回合后 FRAGILE 自动过期
- **Given**: 第一回合 P1 被赋予 FRAGILE（lifetime=2）；第二回合后 tick 使 lifetime 变为 1；第三回合开始前 tick 使 lifetime 变为 0 被移除
- **Then**: 第三回合 P1 出 DEFENSE 可正常获得 DEFENSE 状态

---

## 9. CardResolver 回归测试

### TC-CR-01: ATTACK vs DEFENSE（不变）
- **Given**: P1 ATTACK，P2 DEFENSE；无预存状态
- **When**: `CardResolver.resolve(tasks)`
- **Then**: P2 获得 DEFENSE；P1 ATTACK 被阻挡；P2 HP 不变

### TC-CR-02: HEAVY vs DEFENSE — 仅 DEFENSE 不阻挡 HEAVY
- **Given**: P1 HEAVY，P2 DEFENSE；P2 无 GUARD
- **When**: `CardResolver.resolve(tasks)`
- **Then**: P2 HP 减少 1；P2 获得 FRAGILE（lifetime=2）；P2 同时获得 DEFENSE

### TC-CR-03: COUNTER display_name 不影响结算
- **Given**: P1 ATTACK，P2 COUNTER（防守反击）
- **When**: `CardResolver.resolve(tasks)`
- **Then**: 结算结果与改名前的 "反击" 完全一致：P2 获得 DEFENSE；P1 受到反弹伤害 1

### TC-CR-04: HEAVY build_cost=2 影响构建验证
- **Given**: P1 energy=1, hand=[BUILD], build_target=HEAVY
- **When**: `play_round([BUILD], [ATTACK], build_target_1=CardType.HEAVY)`
- **Then**: 抛出 ValueError（build_cost 2 > energy 1）

### TC-CR-05: HEAVY build_cost=2 构建成功
- **Given**: P1 energy=2, hand=[BUILD], build_target=HEAVY
- **When**: `play_round([BUILD], [ATTACK], build_target_1=CardType.HEAVY)`
- **Then**: P1 energy 减少 2；P1 discard_pile 增加 HEAVY

---

## 10. 完整游戏流程回归

### TC-E2E-01: 单回合完整结算
- **Given**: 新游戏 start_game()；P1 hand=[ATTACK], P2 hand=[HEAVY]
- **When**: `play_round([ATTACK], [HEAVY])`
- **Then**: P1 HP 减少 1（HEAVY damage）；P2 HP 减少 1（ATTACK damage）；P1 获得 FRAGILE（lifetime=2）；返回 RoundResult 完整

### TC-E2E-02: 多回合 FRAGILE 生命周期
- **Given**: 新游戏；R1: P1 DEFENSE, P2 HEAVY → P1 获得 FRAGILE（lifetime=2）+ DEFENSE
- **When**: R2: P1 DEFENSE, P2 ATTACK
- **Then**: R2 中 P1 因 FRAGILE 无法获得 DEFENSE；P1 受到 ATTACK 伤害；R2 结束后 tick → FRAGILE lifetime 变为 0 被移除

### TC-E2E-03: HEAVY vs 连续 DEFENSE（GUARD 保护）
- **Given**: R1: P1 DEFENSE, P2 ATTACK → P1 获得 DEFENSE
- **When**: R2: P1 DEFENSE, P2 HEAVY
- **Then**: GrantGuardResolver 检测到 P1 pre-round DEFENSE → 赋予 GUARD；GrantDefenseResolver 赋予 DEFENSE；HeavyDamageResolver 检测到 P1 GUARD → 阻挡伤害；GrantFragileResolver 检测到 P1 GUARD → 阻挡 FRAGILE；P1 HP 不变，无 FRAGILE

### TC-E2E-04: Converter build_cost 自动传播
- **Given**: `Converter.build_cost(CardType.HEAVY)`
- **Then**: 返回 2（从 CardDef.build_cost 动态派生，无需修改 Converter）
