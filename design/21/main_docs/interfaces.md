# Interface Design — Resolver 策略模式重构 (intention #21)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `game/resolver.py` — Resolver 基类 + 具体实现

### 1.1 ResolveDetails

```python
@dataclass
class ResolveDetails:
    """结算细节收集器，作为 ResolutionResult.details 的数据源。"""
    details: list[str] = field(default_factory=list)

    def add(self, detail: str) -> None:
        """记录一条结算细节。空字符串会被忽略。"""
```

### 1.2 Resolver（抽象基类）

```python
class Resolver(ABC):
    """卡牌效果结算器的抽象基类。

    每个子类封装一个具体的效果结算逻辑。子类是无状态的单例，
    通过 CardResolverRegistry 注册。
    """

    @abstractmethod
    def get_priority(self) -> int:
        """返回结算优先级（10 或 11）。"""

    @abstractmethod
    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        """执行效果结算，直接修改 GameCore 的玩家状态。

        通过 game_core.get_resolve_* 方法读取结算上下文。
        结算细节通过 details.add() 记录。
        """
```

### 1.3 具体 Resolver 子类

8 个子类，均实现 `get_priority()` 和 `resolve()`：

| Resolver | 注册的 CardType | Priority | 结算逻辑 |
|----------|----------------|----------|---------|
| `AttackDamageResolver` | ATTACK | 10 | 根据连续攻击计数造成伤害(streak+1, 上限3)，被 DEFENSE 阻挡 |
| `GrantDefenseResolver` | DEFENSE, COUNTER, FLEX | 11 | 为出牌玩家授予 DEFENSE 状态 |
| `GrantGuardResolver` | DEFENSE | 11 | 若玩家回合前已有 DEFENSE，授予 GUARD 状态（不重复） |
| `HeavyDamageResolver` | HEAVY | 10 | 造成 1 点伤害，被 GUARD 阻挡 |
| `ReflectResolver` | COUNTER | 10 | 若对手出 ATTACK 或 HEAVY，反弹 1 点伤害 |
| `GainEnergyResolver` | ENERGY | 10 | 为出牌玩家增加 1 点能量 |
| `FlexDamageResolver` | FLEX | 10 | 若对手无 DEFENSE，造成 1 点伤害 |
| `BuildResolver` | BUILD | 10 | 无即时效果，仅记录细节 |

所有子类接口一致：
```python
class AttackDamageResolver(Resolver):
    def get_priority(self) -> int: ...

class GrantDefenseResolver(Resolver):
    def get_priority(self) -> int: ...

# ... 其余同理
```

### 1.4 CardResolverRegistry

```python
class CardResolverRegistry:
    """CardType -> list[Resolver] 的注册表。

    模块加载时注册全部映射。所有方法为 classmethod，全局共享。
    """

    @classmethod
    def register(cls, card_type: CardType, resolver: Resolver) -> None:
        """将 resolver 注册到指定 card_type 下。"""

    @classmethod
    def get_resolvers(cls, card_type: CardType) -> list[Resolver]:
        """返回指定 card_type 的所有已注册 Resolver（无特定顺序）。
        调用方按 get_priority() 排序后依次调用。
        """
```

### 1.5 CardResolver（编排器）

```python
class CardResolver:
    """Priority-layered 结算编排器。

    淘汰了旧的 _process_effect 模块级函数，改为从 CardResolverRegistry
    查询 Resolver 并按 priority 排序后委派结算。
    """

    @staticmethod
    def get_consecutive_attack(history: list[PlayerHistoryItem]) -> int:
        """统计 history 末尾连续 ATTACK 数量。接口不变。"""

    @staticmethod
    def resolve(
        game_core: "GameCore",
        cards_1: list[CardType],
        cards_2: list[CardType],
        p1_streak: int,
        p2_streak: int,
        p1_had_defense_pre: bool,
        p2_had_defense_pre: bool,
    ) -> ResolutionResult:
        """编排双方卡牌的结算。

        流程：
        1. 遍历 cards_1 和 cards_2 中的每张卡
        2. 从 CardResolverRegistry 查询对应 Resolver 列表
        3. 按 get_priority() 排序（11 先于 10）
        4. 为每个 Resolver 设置 GameCore 结算上下文
        5. 调用 resolver.resolve(game_core, details)
        6. 从 ResolveDetails 汇总返回 ResolutionResult
        """
```

### 1.6 ResolutionResult

```python
@dataclass
class ResolutionResult:
    """结算结果。状态变更已直接作用于 GameCore 玩家，
    此处仅承载 details（来自 ResolveDetails）。
    """
    details: list[str]
```

---

## 2. `game/card.py` — CardDef 简化

```python
@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata（effects/priority 已移除）。"""
    card_type: CardType
    build_cost: int
    display_name: str
    icon: str
    # effects 和 priority 字段已移除 — 效果逻辑迁移至 Resolver 子类

# EffectType 枚举已移除

def get_card_def(card_type: CardType) -> CardDef:
    """接口不变，返回简化后的 CardDef。"""

def get_initial_draw_pile() -> list[CardType]:
    """接口不变。"""
```

---

## 3. `game/game_core.py` — 结算上下文 + play_round 变更

### 3.1 GameCore 新增方法

```python
class GameCore:
    """新增结算上下文查询方法（供 Resolver 调用）。"""

    def get_resolve_source(self) -> str:
        """返回当前正在结算的玩家标识："1" 或 "2"。"""

    def get_resolve_opponent_card(self) -> CardType:
        """返回对手本回合打出的卡。"""

    def get_resolve_attack_streak(self) -> int:
        """返回当前玩家的连续攻击计数。"""

    def get_resolve_had_defense_pre(self) -> bool:
        """返回当前玩家在回合前是否已有 DEFENSE 状态。"""

    # play_round() 内部流程变更：
    # - 旧：Call CardResolver.resolve() → 返回 ResolutionResult → 逐项 apply
    # - 新：Call CardResolver.resolve(game_core=...) → Resolver 直接修改状态 → 读状态构建 RoundResult
```

### 3.2 play_round 流程变更

旧流程（4-6 步）：
```
resolve → apply_damage → apply_energy → apply_statuses
```

新流程：
```
resolve → (Resolvers directly modify player state; no separate apply step)
```

`BeforeResolve`、`AfterResolve`、`EndActions`、`AfterEndActions` 和 `RoundResult` 接口不变。

---

## 4. `game/player.py` — Player 新增方法

```python
@dataclass
class Player:
    # ... existing fields unchanged ...

    def grant_status(self, status_type: StatusType) -> None:
        """授予一个状态，lifetime=1。
        直接追加到 statuses 列表，不做去重检查（调用方负责）。
        """

    # take_damage、add_energy、has_status、has_cards 等接口不变
```
