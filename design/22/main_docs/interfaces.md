# Interface Design — 消除 CardResolverRegistry (intention #22)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `game/card_type.py` — CardType 枚举（新建）

```python
class CardType(Enum):
    """Seven card types available in the v2 game."""
    ATTACK = "attack"
    DEFENSE = "defense"
    HEAVY = "heavy"
    COUNTER = "counter"
    ENERGY = "energy"
    BUILD = "build"
    FLEX = "flex"
```

内容与原 `card.py` 中完全一致，仅独立为一个文件。

---

## 2. `game/card_def.py` — CardDef + 映射表（新建）

### 2.1 CardDef

```python
@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata.
    
    Attributes:
        card_type: CardType enum member.
        build_cost: Energy cost (1 or 2) to construct via BUILD.
        display_name: Human-readable Chinese name.
        icon: Emoji icon.
        resolvers: Resolver classes for this card, in registration order.
    """
    card_type: CardType
    build_cost: int
    display_name: str
    icon: str
    resolvers: tuple[type["Resolver"], ...]
```

对比 design/21：新增 `resolvers: tuple[type[Resolver], ...]` 字段。

### 2.2 _CARD_DEFS 静态表

```python
_CARD_DEFS: dict[CardType, CardDef] = {
    CardType.ATTACK: CardDef(
        card_type=CardType.ATTACK, build_cost=1,
        display_name="攻击", icon="⚔️",
        resolvers=(AttackDamageResolver,),
    ),
    CardType.DEFENSE: CardDef(
        card_type=CardType.DEFENSE, build_cost=1,
        display_name="防御", icon="🛡️",
        resolvers=(GrantDefenseResolver, GrantGuardResolver),
    ),
    CardType.HEAVY: CardDef(
        card_type=CardType.HEAVY, build_cost=1,
        display_name="重击", icon="💥",
        resolvers=(HeavyDamageResolver,),
    ),
    CardType.COUNTER: CardDef(
        card_type=CardType.COUNTER, build_cost=2,
        display_name="反击", icon="↩️",
        resolvers=(GrantDefenseResolver, ReflectResolver),
    ),
    CardType.ENERGY: CardDef(
        card_type=CardType.ENERGY, build_cost=1,
        display_name="聚能", icon="⚡",
        resolvers=(GainEnergyResolver,),
    ),
    CardType.BUILD: CardDef(
        card_type=CardType.BUILD, build_cost=1,
        display_name="构建", icon="🔨",
        resolvers=(BuildResolver,),
    ),
    CardType.FLEX: CardDef(
        card_type=CardType.FLEX, build_cost=2,
        display_name="攻守兼备", icon="⚖️",
        resolvers=(GrantDefenseResolver, FlexDamageResolver),
    ),
}
```

### 2.3 公共函数

```python
def get_card_def(card_type: CardType) -> CardDef:
    """Return the CardDef for a given CardType.
    Now includes resolver class references in the returned value.
    """

def get_initial_draw_pile() -> list[CardType]:
    """Return the standard 7-card initial draw pile (fresh copy).
    接口不变。
    """
```

---

## 3. `game/resolver/orchestrator.py` — 编排器变更

### 3.1 CardResolver.resolve() 签名变更

```python
class CardResolver:
    """Priority-layered resolution orchestrator."""

    @staticmethod
    def get_consecutive_attack(history: list[PlayerHistoryItem]) -> int:
        """Count consecutive ATTACK cards from the end of history. 接口不变。"""

    @staticmethod
    def resolve(
        tasks: list[tuple[Resolver, ResolveContext]],
    ) -> ResolutionResult:
        """Orchestrate resolution of pre-built (resolver, context) tasks.

        变更点：不再自行构建 tasks，改为接受调用方（GameCore.play_round）
        已构建好的 (Resolver, ResolveContext) 列表。仅负责按 priority
        排序并委托执行。不再依赖 card_def 模块。
        """
```

### 3.2 GameCore.play_round() 变更

```python
class GameCore:
    def play_round(self, cards_1, cards_2, build_target_1=None, build_target_2=None):
        # Step 2: Build resolver tasks from CardDef.resolvers (design/22).
        # GameCore owns the mapping lookup; CardResolver only orchestrates.
        tasks: list[tuple[Resolver, ResolveContext]] = []

        for card_type in cards_1:
            ctx = ResolveContext(
                card_owner=self.player_1, opponent=self.player_2,
                card_played_by_owner=card_type,
                card_played_by_opponent=cards_2[0],
            )
            for resolver_cls in get_card_def(card_type).resolvers:
                tasks.append((resolver_cls(), ctx))

        for card_type in cards_2:
            ctx = ResolveContext(
                card_owner=self.player_2, opponent=self.player_1,
                card_played_by_owner=card_type,
                card_played_by_opponent=cards_1[0],
            )
            for resolver_cls in get_card_def(card_type).resolvers:
                tasks.append((resolver_cls(), ctx))

        resolution = CardResolver.resolve(tasks)
```

核心逻辑变更：

```
# 旧 (design/21):
# CardResolver.resolve() 内部调用 CardResolverRegistry.get_resolvers(card_type)

# 新 (design/22):
# GameCore.play_round() 从 get_card_def(card_type).resolvers 构建 tasks
# CardResolver.resolve(tasks) 仅排序并执行 — 不 import card_def
```

### 3.3 import 变更

```python
# orchestrator.py — 移除:
from src.backend.game.resolver.registry import CardResolverRegistry
# 不再需要 lazy import card_def

# game_core.py — 新增:
from src.backend.game.resolver.base import ResolveContext, Resolver

---

## 4. `game/resolver/__init__.py` — Re-export 变更

```python
# 移除:
from src.backend.game.resolver.registry import CardResolverRegistry

# 保持:
from src.backend.game.resolver.base import (ResolveContext, ResolveDetails, ResolutionResult, Resolver)
from src.backend.game.resolver.orchestrator import CardResolver

# 新增:
from src.backend.game.card_def import get_card_def
```

---

## 5. `game/resolver/registry.py` — 删除

`CardResolverRegistry` 类及文件整体删除。功能由 `CardDef.resolvers` 替代。

---

## 6. Import 路径变更总表

| 原 import | 新 import |
|-----------|----------|
| `from src.backend.game.card import CardType` | `from src.backend.game.card_type import CardType` |
| `from src.backend.game.card import CardDef` | `from src.backend.game.card_def import CardDef` |
| `from src.backend.game.card import get_card_def` | `from src.backend.game.card_def import get_card_def` |
| `from src.backend.game.card import get_initial_draw_pile` | `from src.backend.game.card_def import get_initial_draw_pile` |
| `from src.backend.game.resolver import CardResolverRegistry` | **删除** — 使用 `get_card_def(x).resolvers` 替代 |
