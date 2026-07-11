# Interface Design — 卡牌名字效果变更 + 描述/标签字段 (intention #36)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `game/card_def.py` — CardDef 及映射表

### 1.1 CardDef 新增字段

```python
@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata.

    Per intention #36: description and tags fields added.

    Attributes:
        card_type: CardType enum member.
        build_cost: Energy cost to construct.
        display_name: Chinese name for UI display.
        description: Text describing the card's effect, for frontend display.
        tags: Tuple of tag strings (e.g. ("攻",), ("防",), ("攻","防")).
              Empty tuple means no tags.
        resolvers: Resolver classes in registration order.
    """

    card_type: CardType
    build_cost: int
    display_name: str
    description: str            # NEW
    tags: tuple[str, ...]       # NEW
    resolvers: tuple[type, ...]
```

变更：新增 `description: str` 和 `tags: tuple[str, ...]` 两个字段。

### 1.2 _CARD_DEFS 完整定义

```python
_CARD_DEFS: dict[CardType, CardDef] = {
    CardType.ATTACK: CardDef(
        card_type=CardType.ATTACK,
        build_cost=1,
        display_name="连续击",
        description="快速地连续出击。如果对方没有防御状态，则对对方造成x点伤害。x等于我方连续出连续击的回合数（包括本回合，但最高不超过3）",
        tags=("攻",),
        resolvers=(AttackDamageResolver,),
    ),
    CardType.DEFENSE: CardDef(
        card_type=CardType.DEFENSE,
        build_cost=1,
        display_name="格挡",
        description="专注防守。获得一回合防御状态。连续防御则可获得联防状态",
        tags=("防",),
        resolvers=(GrantDefenseResolver, GrantGuardResolver),
    ),
    CardType.HEAVY: CardDef(
        card_type=CardType.HEAVY,
        build_cost=2,
        display_name="破防重击",
        description="以巨大蛮力破开对方的防守，造成1点伤害并给对方施加2回合脆弱状态。但该卡无法破开联防。",
        tags=("攻",),
        resolvers=(HeavyDamageResolver, RemoveDefenseResolver, GrantFragileResolver),
    ),
    CardType.COUNTER: CardDef(
        card_type=CardType.COUNTER,
        build_cost=2,
        display_name="防守反击",
        description="在招架对手进攻的同时进行反击，反击成功则造成1点伤害，连续防御则可获得联防状态",
        tags=("防",),
        resolvers=(GrantDefenseResolver, ReflectResolver, GrantGuardResolver),
    ),
    CardType.ENERGY: CardDef(
        card_type=CardType.ENERGY,
        build_cost=1,
        display_name="聚能",
        description="获得1格能量。如果本轮及上轮有打出过聚能，则额外增加一点能量",
        tags=(),
        resolvers=(GainEnergyResolver,),
    ),
    CardType.BUILD: CardDef(
        card_type=CardType.BUILD,
        build_cost=1,
        display_name="构建",
        description="构建一张新的卡牌",
        tags=(),
        resolvers=(BuildResolver,),
    ),
    CardType.FLEX: CardDef(
        card_type=CardType.FLEX,
        build_cost=2,
        display_name="攻守兼备",
        description="一边进攻一边防守，只有高手才能做到如此平衡。如果本轮及上轮有打出过'防'卡，则额外增加一回合防御状态。但本卡无法获得联防状态",
        tags=("攻", "防"),
        resolvers=(GrantDefenseResolver, BonusDefenseResolver, FlexDamageResolver),
    ),
}
```

---

## 2. `game/player.py` — Player 新增方法

### 2.1 extend_status()

```python
class Player:
    def extend_status(self, status_type: StatusType, extra: int) -> None:
        """Extend the lifetime of an existing active status.

        Finds the first active status matching the given type and adds
        `extra` to its lifetime. If no matching active status exists,
        grants the status with initial lifetime = extra.

        Args:
            status_type: Status type to extend (e.g. StatusType.FRAGILE).
            extra: Number of rounds to add to the current lifetime.
        """
```

### 2.2 remove_status()

```python
class Player:
    def remove_status(self, status_type: StatusType) -> None:
        """Remove all active instances of the given status type.

        Args:
            status_type: Status type to remove (e.g. StatusType.DEFENSE).
        """
```

---

## 3. `game/resolver/remove_defense_resolver.py` — 新建

```python
class RemoveDefenseResolver(Resolver):
    """Removes DEFENSE status from opponent (priority 10).

    Used exclusively by 破防重击 (HEAVY) per card.md:
    "立刻移除对方的防御状态".

    Same guard condition as HeavyDamageResolver:
    the resolution is still dispatched; if opponent has GUARD,
    the damage is blocked but this resolver still runs
    (GUARD only blocks damage + fragile, not defense removal).
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Remove opponent's DEFENSE status.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
```

逻辑：
1. 检查 `ctx.opponent.has_status(StatusType.DEFENSE)`
2. 若有 DEFENSE：调用 `ctx.opponent.remove_status(StatusType.DEFENSE)`，记录 "removes opponent DEFENSE"
3. 若无 DEFENSE：不操作

注意：此 Resolver 的 GUARD 不阻挡防御移除，仅阻挡伤害和脆弱。

---

## 4. `game/resolver/grant_fragile_resolver.py` — 修改

```python
class GrantFragileResolver(Resolver):
    """Extends opponent's FRAGILE status by 2 rounds (priority 10).

    Per intention #36: fragile changed from "grant(lifetime=2)"
    to additive "+2 rounds". Blocked by opponent GUARD.
    """

    FRAGILE_EXTRA: int = 2

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Extend FRAGILE by 2 rounds unless GUARD blocks it.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
```

逻辑：
1. 检查 `ctx.opponent.has_status(StatusType.GUARD)`
2. 若有 GUARD：记录 "Fragile blocked by opponent GUARD"，返回
3. 若无 GUARD：调用 `ctx.opponent.extend_status(StatusType.FRAGILE, extra=2)`，记录 "extends FRAGILE by 2 rounds"

---

## 5. `game/resolver/reflect_resolver.py` — 修改

```python
class ReflectResolver(Resolver):
    """Settles COUNTER reflect damage (priority 10).

    Per intention #36: condition changed from "opponent played
    ATTACK or HEAVY" to "opponent's card has '攻' tag AND
    opponent has no DEFENSE status".
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Deal 1 reflect damage if opponent's card has '攻' tag
        and opponent has no DEFENSE.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
```

逻辑：
1. 若 `ctx.card_played_by_opponent is None`（对方 pass）：返回
2. 通过 `get_card_def(ctx.card_played_by_opponent)` 获取对方卡牌的 tags
3. 若 `"攻" not in tags`：返回
4. 若 `ctx.opponent.has_status(StatusType.DEFENSE)`：记录 "Counter blocked by opponent DEFENSE"，返回
5. 调用 `ctx.opponent.take_damage(1)`，记录 "Counter reflects, deals 1 to opponent"

---

## 6. `game/resolver/bonus_defense_resolver.py` — 新建

```python
class BonusDefenseResolver(Resolver):
    """Grants +1 extra DEFENSE if last round played a '防'-tagged card
    (priority 11).

    Used exclusively by 攻守兼备 (FLEX) per card.md:
    "如果本轮及上轮有打出过'防'卡，则额外增加一回合防御状态".

    Per D2: "本轮" excludes the current card, so only checks
    card_owner.history[-1] (last round).
    """

    def get_priority(self) -> int:
        return 11

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant +1 DEFENSE if last round played a '防' tagged card.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
```

逻辑：
1. 若 `len(ctx.card_owner.history) == 0`（首轮）：返回
2. 通过 `get_card_def(ctx.card_owner.history[-1].card)` 获取上轮卡牌的 tags
3. 若 `"防" not in tags`：返回
4. 若 `ctx.card_owner.has_status(StatusType.FRAGILE)`：记录 "Bonus DEFENSE blocked by FRAGILE"，返回
5. 调用 `ctx.card_owner.grant_status(StatusType.DEFENSE)`，记录 "gains +1 bonus DEFENSE"

---

## 7. `game/resolver/gain_energy_resolver.py` — 修改

```python
class GainEnergyResolver(Resolver):
    """Grants energy to card_owner (priority 10).

    Per intention #36: baseline 1 energy + bonus 1 if ENERGY was
    played last round (consecutive bonus).

    Per D2: "本轮" excludes the current card.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant 1 energy, or 2 if ENERGY was played last round.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
```

逻辑：
1. 基础：`ctx.card_owner.add_energy(1)`
2. 若 `len(ctx.card_owner.history) > 0` 且 `ctx.card_owner.history[-1].card == CardType.ENERGY`：
   `ctx.card_owner.add_energy(1)`（额外+1），记录 "gains 2 energy (consecutive)"
3. 否则：记录 "gains 1 energy"

---

## 8. `controller/api.py` — GET /api/cards 响应变更

```python
@app.route("/api/cards", methods=["GET"])
def list_cards():
    """Return metadata for all card types."""
    cards = []
    for ct in CardType:
        cd = get_card_def(ct)
        cards.append({
            "type": ct.value,
            "display_name": cd.display_name,
            "build_cost": cd.build_cost,
            "description": cd.description,   # NEW
            "tags": list(cd.tags),            # NEW: tuple→list for JSON
        })
    return jsonify({"cards": cards}), 200
```

---

## 9. `game/resolver/__init__.py` — 导出变更

新增两行导出：
```python
from src.backend.game.resolver.remove_defense_resolver import RemoveDefenseResolver
from src.backend.game.resolver.bonus_defense_resolver import BonusDefenseResolver
```
