"""CardDef dataclass and static card metadata table (design/22, design/37).

Per design/22: CardDef gains a `resolvers` field that stores Resolver
subclass references. CardResolverRegistry has been deleted — the mapping
is now defined inline in _CARD_DEFS.

Per intention #36: description and tags fields added to CardDef.
Per design/37: BUILD updated to 构建/拆解 with DismantleResolver;
IRON_DEFENSE card added.

CardType was extracted to card_type.py to break the circular dependency
with resolver/.
"""

from dataclasses import dataclass

from src.backend.game.card_type import CardType
from src.backend.game.resolver.attack_damage_resolver import AttackDamageResolver
from src.backend.game.resolver.bonus_defense_resolver import BonusDefenseResolver
from src.backend.game.resolver.build_resolver import BuildResolver
from src.backend.game.resolver.dismantle_resolver import DismantleResolver
from src.backend.game.resolver.flex_damage_resolver import FlexDamageResolver
from src.backend.game.resolver.gain_energy_resolver import GainEnergyResolver
from src.backend.game.resolver.grant_defense_resolver import GrantDefenseResolver
from src.backend.game.resolver.grant_guard_resolver import GrantGuardResolver
from src.backend.game.resolver.grant_fragile_resolver import GrantFragileResolver
from src.backend.game.resolver.heavy_damage_resolver import HeavyDamageResolver
from src.backend.game.resolver.reflect_resolver import ReflectResolver
from src.backend.game.resolver.remove_defense_resolver import RemoveDefenseResolver
from src.backend.game.resolver.strong_defense_resolver import StrongDefenseResolver


@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata (design/22, design/23, intention #36).

    Per design/21: effects and priority fields removed — effect logic
    encapsulated in Resolver subclasses.
    Per design/22: resolvers field added — Resolver subclass references
    replace the deleted CardResolverRegistry.
    Per design/23: icon field removed — display logic moved to frontend.
    Per intention #36: description and tags fields added.

    Attributes:
        card_type: The CardType enum member this definition belongs to.
        build_cost: Energy cost (1 or 2) to construct this card via BUILD.
        display_name: Human-readable Chinese name for UI display.
        description: Text describing the card's effect, for frontend display.
        tags: Tuple of tag strings (e.g. ("攻",), ("防",), ("攻","防")).
              Empty tuple means no tags.
        resolvers: Resolver classes for this card, in registration order.
    """

    card_type: CardType
    build_cost: int
    display_name: str
    description: str
    tags: tuple[str, ...]
    resolvers: tuple[type, ...]


# ---------------------------------------------------------------------------
# Static card metadata table — one CardDef per CardType.
# resolvers field replaces CardResolverRegistry (design/22).
# ---------------------------------------------------------------------------

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
        display_name="构建/拆解",
        description="(上半部分)构建一张新的卡牌；(下半部分)从弃牌堆中移除一张卡牌",
        tags=(),
        resolvers=(BuildResolver, DismantleResolver),
    ),
    CardType.FLEX: CardDef(
        card_type=CardType.FLEX,
        build_cost=2,
        display_name="攻守兼备",
        description="一边进攻一边防守，只有高手才能做到如此平衡。如果本轮及上轮有打出过'防'卡，则额外增加一回合防御状态。但本卡无法获得联防状态",
        tags=("攻", "防"),
        resolvers=(GrantDefenseResolver, BonusDefenseResolver, FlexDamageResolver),
    ),
    CardType.IRON_DEFENSE: CardDef(
        card_type=CardType.IRON_DEFENSE,
        build_cost=2,
        display_name="严防死守",
        description="调动全身力量进行防守",
        tags=("防",),
        resolvers=(StrongDefenseResolver, GrantGuardResolver),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_card_def(card_type: CardType) -> CardDef:
    """Return the CardDef for a given CardType.

    Per design/22: the returned CardDef includes resolver class references
    in the `resolvers` field, replacing the deleted CardResolverRegistry.

    Args:
        card_type: Any of the 8 CardType enum members.

    Returns:
        Immutable CardDef with card_type, build_cost, display_name,
        resolvers.
    """
    return _CARD_DEFS[card_type]


def get_initial_draw_pile() -> list[CardType]:
    """Return the standard 7-card initial draw pile (fresh copy).

    Composition (design/38): 3×ATTACK, 2×DEFENSE, 1×ENERGY, 1×BUILD.
    HEAVY and COUNTER are no longer in the initial draw pile.

    Returns a fresh list on each call to prevent shared-mutation bugs.
    """
    return [
        CardType.ATTACK,
        CardType.ATTACK,
        CardType.ATTACK,
        CardType.DEFENSE,
        CardType.DEFENSE,
        CardType.ENERGY,
        CardType.BUILD,
    ]
