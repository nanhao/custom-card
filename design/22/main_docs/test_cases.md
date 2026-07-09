# Test Cases — 消除 CardResolverRegistry (intention #22)

---

## 1. CardType 枚举（card_type.py）

### TC-CT-01: import 路径变更后行为一致
- **Given**: 导入 `from src.backend.game.card_type import CardType`
- **Then**: CardType 包含 7 个成员，值与重构前完全一致

### TC-CT-02: 旧路径不可导入
- **Given**: 尝试 `from src.backend.game.card import CardType`
- **Then**: 抛出 ImportError（card.py 已删除）

---

## 2. CardDef（card_def.py）

### TC-CD-01: CardDef 包含 resolvers 字段
- **Given**: `get_card_def(CardType.ATTACK)`
- **Then**: `cd.resolvers` 为 `(AttackDamageResolver,)` 元组

### TC-CD-02: 每个 CardType 的 resolvers 数量正确
- **Given**: 遍历全部 7 个 CardType
- **Then**:
  - ATTACK: 1, DEFENSE: 2, HEAVY: 1, COUNTER: 2, ENERGY: 1, BUILD: 1, FLEX: 2

### TC-CD-03: 每个 CardType 的 resolvers 类型正确
- **Given**: 遍历全部 7 个 CardType 调用 `get_card_def()`
- **Then**: resolvers 元组中的类与映射表一致（ATTACK→AttackDamageResolver, DEFENSE→(GrantDefenseResolver, GrantGuardResolver), HEAVY→HeavyDamageResolver, COUNTER→(GrantDefenseResolver, ReflectResolver), ENERGY→GainEnergyResolver, BUILD→BuildResolver, FLEX→(GrantDefenseResolver, FlexDamageResolver)）

### TC-CD-04: CardDef 字段完整
- **Given**: `get_card_def(CardType.ATTACK)`
- **Then**: 包含 `card_type`, `build_cost`, `display_name`, `icon`, `resolvers` 五个字段

### TC-CD-05: CardDef 无 effects 字段
- **Given**: `get_card_def(CardType.ATTACK)`
- **When**: 访问 `cd.effects`
- **Then**: 抛出 AttributeError

### TC-CD-06: CardDef 无 priority 字段
- **Given**: `get_card_def(CardType.ATTACK)`
- **When**: 访问 `cd.priority`
- **Then**: 抛出 AttributeError

### TC-CD-07: CardDef 不可变
- **Given**: `get_card_def(CardType.ATTACK)`
- **When**: 尝试 `cd.build_cost = 5`
- **Then**: 抛出 FrozenInstanceError

### TC-CD-08: resolvers 不可变
- **Given**: `get_card_def(CardType.ATTACK)`
- **When**: 尝试修改 `cd.resolvers`
- **Then**: 抛出异常（tuple 不可变）

### TC-CD-09: resolver 类可实例化
- **Given**: `get_card_def(CardType.ATTACK).resolvers[0]`
- **When**: 调用 `resolver_cls()`
- **Then**: 返回 AttackDamageResolver 实例，`get_priority()` 返回 10

### TC-CD-10: build_cost 不变
- **Given**: `get_card_def(CardType.COUNTER)`, `get_card_def(CardType.FLEX)`
- **Then**: build_cost 分别为 2, 2；ATTACK 等为 1

---

## 3. CardResolverRegistry 删除

### TC-RG-01: CardResolverRegistry 不可导入
- **Given**: 尝试 `from src.backend.game.resolver.registry import CardResolverRegistry`
- **Then**: registry.py 不存在，导入失败（ModuleNotFoundError 或 ImportError）

### TC-RG-02: resolver.__init__ 不再导出 CardResolverRegistry
- **Given**: 尝试 `from src.backend.game.resolver import CardResolverRegistry`
- **Then**: 抛出 ImportError

### TC-RG-03: registrar 文件已删除
- **Given**: 检查 `src/backend/game/resolver/registry.py`
- **Then**: 文件不存在

---

## 4. CardResolver 编排器（回归测试）

### TC-CR-01: resolve() 返回 ResolutionResult 含 details
- **Given**: 创建 GameCore 实例，调用 `start_game()`; `cards_1=[ATTACK]`, `cards_2=[DEFENSE]`
- **When**: `CardResolver.resolve(player_1, player_2, cards_1, cards_2)`
- **Then**: 返回 ResolutionResult，details 列表非空；Player 状态已被修改

### TC-CR-02: ATTACK vs DEFENSE 结算正确
- **Given**: P1 出 ATTACK，P2 出 DEFENSE; 双方无预先状态
- **When**: `CardResolver.resolve(...)`
- **Then**: P2 获得 DEFENSE 状态; P1 ATTACK 被 DEFENSE 阻挡; P2 HP 不变

### TC-CR-03: ATTACK + COUNTER 结算正确
- **Given**: P1 出 ATTACK，P2 出 COUNTER; 双方无预先状态
- **When**: `CardResolver.resolve(...)`
- **Then**: P2 获得 DEFENSE 状态; P1 ATTACK 被阻挡; P1 受到反弹伤害 1（ReflectResolver）

### TC-CR-04: HEAVY vs DEFENSE 结算正确
- **Given**: P1 出 HEAVY，P2 出 DEFENSE; P2 无 GUARD
- **When**: `CardResolver.resolve(...)`
- **Then**: P2 HP 减少 1（HEAVY 不被 DEFENSE 阻挡）

### TC-CR-05: HEAVY vs GUARD 结算正确
- **Given**: P1 出 HEAVY，P2 有 GUARD 状态
- **When**: `CardResolver.resolve(...)`
- **Then**: P2 HP 不变（HEAVY 被 GUARD 阻挡）

### TC-CR-06: 优先级排序（priority 12 > 11 > 10）
- **Given**: P1 出 DEFENSE（GrantGuardResolver priority=12, GrantDefenseResolver priority=11），P2 出 ATTACK（AttackDamageResolver priority=10）
- **When**: `CardResolver.resolve(...)`
- **Then**: GrantGuardResolver 先执行（在 GrantDefenseResolver 前检查 DEFENSE 存在性）; P1 获得 DEFENSE; P2 ATTACK 被阻挡

### TC-CR-07: FLEX vs 无 DEFENSE 结算正确
- **Given**: P1 出 FLEX，P2 无 DEFENSE
- **When**: `CardResolver.resolve(...)`
- **Then**: P1 获得 DEFENSE; P2 HP 减少 1（FlexDamageResolver）

### TC-CR-08: FLEX vs 有 DEFENSE 结算正确
- **Given**: P1 出 FLEX，P2 有 DEFENSE 状态
- **When**: `CardResolver.resolve(...)`
- **Then**: P1 获得 DEFENSE; P2 HP 不变（FlexDamageResolver 被阻挡）

### TC-CR-09: ENERGY 结算正确
- **Given**: P1 energy=0，出 ENERGY
- **When**: `CardResolver.resolve(...)`
- **Then**: P1 energy=1

### TC-CR-10: BUILD 不修改状态
- **Given**: P1 出 BUILD; P1 HP=20, energy=0, statuses=[]
- **When**: `CardResolver.resolve(...)`
- **Then**: P1 HP=20, energy=0, statuses=[]; P2 HP=20

### TC-CR-11: get_consecutive_attack 不变
- **Given**: history 包含 `[ATTACK, ATTACK, DEFENSE, ATTACK]` 对应的 PlayerHistoryItem
- **When**: `CardResolver.get_consecutive_attack(history)`
- **Then**: 返回 1（仅末尾连续 ATTACK）

### TC-CR-12: 空 history 返回 0
- **Given**: history=[]
- **When**: `CardResolver.get_consecutive_attack([])`
- **Then**: 返回 0

---

## 5. Import 路径变更验证

### TC-IM-01: card_type 模块存在
- **Given**: 导入 `src.backend.game.card_type`
- **Then**: 模块存在，导出 `CardType`

### TC-IM-02: card_def 模块存在
- **Given**: 导入 `src.backend.game.card_def`
- **Then**: 模块存在，导出 `CardDef`, `get_card_def`, `get_initial_draw_pile`

### TC-IM-03: card 模块已删除
- **Given**: 尝试 `import src.backend.game.card`
- **Then**: 抛出 ModuleNotFoundError

### TC-IM-04: get_initial_draw_pile 可正常调用
- **Given**: 从 `card_def` 导入并调用
- **Then**: 返回 7 张卡列表，2×ATTACK + 5×其他

### TC-IM-05: controller/serializers 可正常导入
- **Given**: 导入 `src.backend.controller.serializers`
- **Then**: 无 ImportError（import 路径已更新至 card_type）

### TC-IM-06: app/converter 可正常导入
- **Given**: 导入 `src.backend.app.converter`
- **Then**: 无 ImportError

### TC-IM-07: app/app_manager 可正常导入
- **Given**: 导入 `src.backend.app.app_manager`
- **Then**: 无 ImportError

---

## 6. 完整游戏流程回归测试

### TC-E2E-01: 单回合 ATTACK vs ATTACK
- **Given**: 新游戏 start_game(); P1 hand=[ATTACK], P2 hand=[ATTACK]; 各自 draw_pile 填充
- **When**: `play_round([ATTACK], [ATTACK])`
- **Then**: 双方 HP 各减少 1; 返回 RoundResult 完整

### TC-E2E-02: 单回合 DEFENSE vs ATTACK
- **Given**: 新游戏 start_game(); P1 hand=[DEFENSE], P2 hand=[ATTACK]
- **When**: `play_round([DEFENSE], [ATTACK])`
- **Then**: P1 获得 DEFENSE; P1 HP 不变（ATTACK 被阻挡）

### TC-E2E-03: 多回合游戏正常进行
- **Given**: 新游戏，手动设置 hand 为不同卡牌
- **When**: 连续多回合 `play_round()`
- **Then**: 每回合返回 RoundResult; cycle 机制正常（第 5 回合后 bleed + 新 cycle）

### TC-E2E-04: BUILD 构建流程正常
- **Given**: P1 hand=[BUILD], energy=2; build_target=COUNTER
- **When**: `play_round([BUILD], [ATTACK], build_target_1=COUNTER)`
- **Then**: P1 energy 减少 2; P1 discard_pile 增加 COUNTER
