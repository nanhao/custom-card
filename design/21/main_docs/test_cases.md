# Test Cases — Resolver 策略模式重构 (intention #21)

---

## 1. CardResolverRegistry

### TC-RG-01: 注册后查询返回正确顺序
- **Given**: CardResolverRegistry 已完成模块级注册
- **When**: `get_resolvers(CardType.ATTACK)`
- **Then**: 返回列表包含 1 个 Resolver（AttackDamageResolver）

### TC-RG-02: 多 Resolver 的卡牌查询
- **Given**: CardResolverRegistry 已完成注册
- **When**: `get_resolvers(CardType.DEFENSE)`
- **Then**: 返回 2 个 Resolver（GrantDefenseResolver 和 GrantGuardResolver）

### TC-RG-03: 所有 CardType 都有注册
- **Given**: CardResolverRegistry 已完成注册
- **When**: 遍历全部 7 个 CardType 调用 `get_resolvers()`
- **Then**: 每个 CardType 返回非空列表；ATTACK=1, DEFENSE=2, HEAVY=1, COUNTER=2, ENERGY=1, BUILD=1, FLEX=2

### TC-RG-04: 重复注册同一 CardType 追加而非覆盖
- **Given**: 对一个 CardType 调用 `register()` 两次传入不同 Resolver
- **When**: `get_resolvers()`
- **Then**: 返回 2 个 Resolver

---

## 2. Resolver 基类

### TC-RS-BASE-01: Resolver 为抽象类不可直接实例化
- **Given**: 导入 Resolver
- **When**: 尝试 `Resolver()`
- **Then**: 抛出 TypeError

### TC-RS-BASE-02: 具体子类必须实现 get_priority 和 resolve
- **Given**: 定义未实现抽象方法的子类
- **When**: 尝试实例化
- **Then**: 抛出 TypeError

---

## 3. AttackDamageResolver（ATTACK, priority=10）

### TC-ATK-01: 基础攻击（streak=0）
- **Given**: P1 出 ATTACK, P2 无 DEFENSE; attack_streak=0
- **When**: AttackDamageResolver.resolve(game_core, details)
- **Then**: P2 HP 减少 1（streak+1=1）; details 包含 "attack" 描述

### TC-ATK-02: 连续攻击（streak=2）
- **Given**: P1 出 ATTACK, P2 无 DEFENSE; attack_streak=2
- **When**: resolve()
- **Then**: P2 HP 减少 min(2+1, 3)=3

### TC-ATK-03: 连续攻击上限（streak=5）
- **Given**: P1 出 ATTACK, P2 无 DEFENSE; attack_streak=5
- **When**: resolve()
- **Then**: P2 HP 减少 min(5+1, 3)=3（上限为 3）

### TC-ATK-04: 被防御阻挡
- **Given**: P1 出 ATTACK, P2 已有 DEFENSE 状态
- **When**: resolve()
- **Then**: P2 HP 不变（damage=0）; details 包含 "blocked" 描述

### TC-ATK-05: 攻击方不同（P2 出 ATTACK）
- **Given**: P2 出 ATTACK, P1 无 DEFENSE; attack_streak=0, source="2"
- **When**: resolve()
- **Then**: P1 HP 减少 1

---

## 4. GrantDefenseResolver（DEFENSE/COUNTER/FLEX, priority=11）

### TC-GD-01: 为 P1 授予 DEFENSE
- **Given**: P1 出 DEFENSE; source="1"
- **When**: GrantDefenseResolver.resolve(game_core, details)
- **Then**: P1 statuses 包含 lifetime=1 的 DEFENSE; details 包含 "defense" 描述

### TC-GD-02: 为 P2 授予 DEFENSE
- **Given**: P2 出 DEFENSE; source="2"
- **When**: resolve()
- **Then**: P2 statuses 包含 lifetime=1 的 DEFENSE

### TC-GD-03: 通过 COUNTER 卡触发
- **Given**: P1 出 COUNTER（注册了 GrantDefenseResolver）
- **When**: GrantDefenseResolver.resolve()
- **Then**: P1 获得 DEFENSE 状态

### TC-GD-04: 通过 FLEX 卡触发
- **Given**: P1 出 FLEX（注册了 GrantDefenseResolver）
- **When**: GrantDefenseResolver.resolve()
- **Then**: P1 获得 DEFENSE 状态

---

## 5. GrantGuardResolver（DEFENSE, priority=11）

### TC-GG-01: 已有 DEFENSE 时授予 GUARD
- **Given**: P1 出 DEFENSE; had_defense_pre=True; P1 当前无 GUARD
- **When**: GrantGuardResolver.resolve(game_core, details)
- **Then**: P1 statuses 包含 GUARD; details 包含 "GUARD" 描述

### TC-GG-02: 无预先 DEFENSE 时不授予 GUARD
- **Given**: P1 出 DEFENSE; had_defense_pre=False
- **When**: resolve()
- **Then**: P1 statuses 不包含 GUARD

### TC-GG-03: 已有 GUARD 时不重复授予
- **Given**: P1 出 DEFENSE; had_defense_pre=True; P1 已有 GUARD
- **When**: resolve()
- **Then**: P1 statuses 中 GUARD 数量为 1（不重复）

---

## 6. HeavyDamageResolver（HEAVY, priority=10）

### TC-HV-01: 无 GUARD 时造成伤害
- **Given**: P1 出 HEAVY; P2 无 GUARD
- **When**: HeavyDamageResolver.resolve(game_core, details)
- **Then**: P2 HP 减少 1

### TC-HV-02: 被 GUARD 阻挡
- **Given**: P1 出 HEAVY; P2 有 GUARD 状态
- **When**: resolve()
- **Then**: P2 HP 不变（damage=0）

### TC-HV-03: 不被 DEFENSE 阻挡
- **Given**: P1 出 HEAVY; P2 有 DEFENSE 状态但无 GUARD
- **When**: resolve()
- **Then**: P2 HP 减少 1（DEFENSE 挡不住 HEAVY）

---

## 7. ReflectResolver（COUNTER, priority=10）

### TC-RF-01: 反弹 ATTACK
- **Given**: P1 出 COUNTER, P2 出 ATTACK
- **When**: ReflectResolver.resolve(game_core, details)
- **Then**: P2 HP 减少 1（反弹伤害）; P1 HP 不变

### TC-RF-02: 反弹 HEAVY
- **Given**: P1 出 COUNTER, P2 出 HEAVY
- **When**: resolve()
- **Then**: P2 HP 减少 1（反弹伤害）

### TC-RF-03: 对手出 DEFENSE 不反弹
- **Given**: P1 出 COUNTER, P2 出 DEFENSE
- **When**: resolve()
- **Then**: P2 HP 不变（无反弹）

### TC-RF-04: 对手出 ENERGY 不反弹
- **Given**: P1 出 COUNTER, P2 出 ENERGY
- **When**: resolve()
- **Then**: P2 HP 不变（无反弹）

---

## 8. GainEnergyResolver（ENERGY, priority=10）

### TC-GE-01: 增加 1 点能量
- **Given**: P1 energy=0; source="1"
- **When**: GainEnergyResolver.resolve(game_core, details)
- **Then**: P1 energy=1; details 包含 "energy" 描述

### TC-GE-02: 能量已达上限时 clamped
- **Given**: P1 energy=5; source="1"
- **When**: resolve()
- **Then**: P1 energy=5（Player.add_energy 的 clamping 由 Player 层保证）

---

## 9. FlexDamageResolver（FLEX, priority=10）

### TC-FD-01: 对手无 DEFENSE 时造成伤害
- **Given**: P1 出 FLEX; P2 无 DEFENSE 状态
- **When**: FlexDamageResolver.resolve(game_core, details)
- **Then**: P2 HP 减少 1

### TC-FD-02: 对手有 DEFENSE 时被阻挡
- **Given**: P1 出 FLEX; P2 有 DEFENSE 状态
- **When**: resolve()
- **Then**: P2 HP 不变

---

## 10. BuildResolver（BUILD, priority=10）

### TC-BD-01: 不修改游戏状态
- **Given**: P1 出 BUILD; P1 HP=20, energy=0, statuses=[]
- **When**: BuildResolver.resolve(game_core, details)
- **Then**: P1 HP=20, energy=0, statuses=[]; P2 HP=20; details 包含 "BUILD" 描述

---

## 11. 优先级排序（Priority Ordering）

### TC-PO-01: priority 11 先于 10 执行
- **Given**: P1 出 DEFENSE（priority 11+11）, P2 出 ATTACK（priority 10）
- **When**: CardResolver.resolve() 整体编排
- **Then**: P1 先获得 DEFENSE → P2 ATTACK 被阻挡; P1 HP 不变

### TC-PO-02: 同优先级内的顺序不影响最终结果
- **Given**: P1 出 DEFENSE（两个 priority 11 的 Resolver）; 两个 Resolver 互相独立
- **When**: 以不同顺序调用两个 Resolver
- **Then**: 最终结果相同（GrantDefense 和 GrantGuard 独立）

---

## 12. CardResolver 编排

### TC-CR-01: resolve() 返回 ResolutionResult 含 details
- **Given**: 创建 GameCore 实例，调用 start_game(); cards_1=[ATTACK], cards_2=[DEFENSE]
- **When**: CardResolver.resolve(game_core, cards_1, cards_2, 0, 0, False, False)
- **Then**: 返回 ResolutionResult，其 details 列表非空；GameCore 玩家状态已被修改

### TC-CR-02: 双方各出一张卡的完整结算
- **Given**: P1 出 ATTACK, P2 出 COUNTER; 双方无预先状态
- **When**: CardResolver.resolve()
- **Then**: P2 获得 DEFENSE（COUNTER 的 GrantDefenseResolver）; P1 受到反弹伤害 1（ReflectResolver）; P1 ATTACK 被 P2 DEFENSE 阻挡; details 记录完整

### TC-CR-03: get_consecutive_attack 不变
- **Given**: history=[ATTACK, ATTACK, DEFENSE, ATTACK] 对应的 PlayerHistoryItem 列表
- **When**: CardResolver.get_consecutive_attack(history)
- **Then**: 返回 1（仅末尾连续 ATTACK 计数）

### TC-CR-04: 空 history
- **Given**: history=[]
- **When**: CardResolver.get_consecutive_attack([])
- **Then**: 返回 0

---

## 13. Player.grant_status()

### TC-PS-01: 授予状态
- **Given**: Player(statuses=[])
- **When**: grant_status(StatusType.DEFENSE)
- **Then**: statuses=[PlayerStatus(DEFENSE, lifetime=1)]

### TC-PS-02: 多次授予追加而非覆盖
- **Given**: Player(statuses=[PlayerStatus(DEFENSE, lifetime=1)])
- **When**: grant_status(StatusType.GUARD)
- **Then**: statuses 包含 DEFENSE 和 GUARD 各一个

---

## 14. CardDef 变更

### TC-CD-01: CardDef 不再有 effects 字段
- **Given**: 导入 CardDef 类
- **Then**: CardDef 字段不包含 effects; 包含 card_type, build_cost, display_name, icon

### TC-CD-02: CardDef 不再有 priority 字段
- **Given**: get_card_def(CardType.ATTACK)
- **Then**: 返回的 CardDef 无 priority 属性

### TC-CD-03: EffectType 枚举已移除
- **Given**: 尝试从 card 模块导入 EffectType
- **Then**: 抛出 ImportError

### TC-CD-04: get_card_def 仍返回正确的 build_cost
- **Given**: get_card_def(CardType.COUNTER)
- **Then**: build_cost=2
