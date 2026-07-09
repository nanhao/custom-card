# Interface Design — 卡牌定义对齐 card.md (intention #24)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `game/player_state.py` — StatusType 枚举

### 1.1 StatusType 新增 FRAGILE

```python
class StatusType(Enum):
    DEFENSE = "defense"  # 防御状态
    GUARD = "guard"      # 联防状态
    FRAGILE = "fragile"  # 脆弱状态 — 持有者无法获得防御和联防效果
```

变更：在 DEFENSE 和 GUARD 之后新增 `FRAGILE = "fragile"`。

### 1.2 PlayerStatus

不变。`PlayerStatus(status_type, lifetime)` 已支持任意 StatusType。`to_dict()` 序列化为 `{"type": "fragile", "lifetime": 2}`。

---

## 2. `game/player.py` — Player.grant_status()

### 2.1 签名变更

```python
class Player:
    def grant_status(self, status_type: StatusType, lifetime: int = 1) -> None:
        """Grant a status with specified lifetime, appending to the statuses list.

        Args:
            status_type: The status type to grant.
            lifetime: Number of rounds the status lasts before expiry.
                      Default 1 (expires at end of current round after tick).
        """
```

变更：新增 `lifetime: int = 1` 参数。现有调用点无需修改。

### 2.2 has_status()

不变。`has_status(StatusType.FRAGILE)` 复用现有逻辑，通过 `has_active_status()` 检查 lifetime > 0。

---

## 3. `game/card_def.py` — CardDef 映射表

### 3.1 _CARD_DEFS 变更行

```python
_CARD_DEFS: dict[CardType, CardDef] = {
    # ... 不变的行省略 ...
    CardType.HEAVY: CardDef(
        card_type=CardType.HEAVY,
        build_cost=2,           # 1 → 2
        display_name="重击",
        resolvers=(HeavyDamageResolver, GrantFragileResolver),  # 新增 GrantFragileResolver
    ),
    CardType.COUNTER: CardDef(
        card_type=CardType.COUNTER,
        build_cost=2,
        display_name="防守反击",  # "反击" → "防守反击"
        resolvers=(GrantDefenseResolver, ReflectResolver),
    ),
    # ... 其余不变 ...
}
```

变更点：
- HEAVY: `build_cost` 1→2；`resolvers` 新增 `GrantFragileResolver`
- COUNTER: `display_name` "反击"→"防守反击"

---

## 4. `game/resolver/grant_fragile_resolver.py` — 新建

```python
class GrantFragileResolver(Resolver):
    """Grants FRAGILE status to opponent if not blocked by GUARD (priority 10).

    Per card.md: 重击赋予对方持续2回合的脆弱状态。
    Blocked by opponent GUARD status (shared condition with HeavyDamageResolver).
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant FRAGILE (lifetime=2) to opponent unless GUARD blocks it.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
```

逻辑：
1. 检查 `ctx.opponent.has_status(StatusType.GUARD)`
2. 若有 GUARD：记录 "Fragile blocked by opponent GUARD"，返回
3. 若无 GUARD：调用 `ctx.opponent.grant_status(StatusType.FRAGILE, lifetime=2)`，记录 "gains FRAGILE (2 rounds)"

---

## 5. `game/resolver/grant_defense_resolver.py` — 修改

### 5.1 resolve() 变更

```python
class GrantDefenseResolver(Resolver):
    """Grants DEFENSE status to the card_owner (priority 11).

    Per card.md: 拥有脆弱状态的玩家无法获得防御效果。
    """

    def get_priority(self) -> int:
        return 11

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant DEFENSE to card_owner unless FRAGILE blocks it."""
```

逻辑变更：
1. 新增检查 `ctx.card_owner.has_status(StatusType.FRAGILE)`
2. 若有 FRAGILE：记录 "DEFENSE blocked by FRAGILE"，返回
3. 若无 FRAGILE：执行原有逻辑（`ctx.card_owner.grant_status(StatusType.DEFENSE)`）

---

## 6. `game/resolver/grant_guard_resolver.py` — 修改

### 6.1 resolve() 变更

```python
class GrantGuardResolver(Resolver):
    """Grants GUARD if card_owner had DEFENSE pre-round (priority 12).

    Per card.md: 拥有脆弱状态的玩家无法获得联防效果。
    """

    def get_priority(self) -> int:
        return 12

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant GUARD to card_owner if pre-round DEFENSE and not FRAGILE."""
```

逻辑变更：
1. 新增检查 `ctx.card_owner.has_status(StatusType.FRAGILE)`
2. 若有 FRAGILE：记录 "GUARD blocked by FRAGILE"，返回
3. 若无 FRAGILE：执行原有逻辑（检查 pre-round DEFENSE → 授予 GUARD）

---

## 7. `game/resolver/__init__.py` — 导出变更

```python
# 新增一行：
from src.backend.game.resolver.grant_fragile_resolver import GrantFragileResolver
```

其余导出不变。
