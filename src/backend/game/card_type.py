"""CardType enum — pure identifier for the custom card game v2 (design/22).

Per design/22: CardType extracted from card.py into a standalone module
to break the circular dependency between card.py and resolver/.
This module has zero dependencies on other game/ modules.
"""

from enum import Enum


class CardType(Enum):
    """Eight card types available in the v2 game.

    Per design/37: IRON_DEFENSE added as 8th card type.
    Values are short string identifiers used for serialization and display.
    """

    ATTACK = "attack"          # 连续击
    DEFENSE = "defense"        # 格挡
    HEAVY = "heavy"            # 破防重击
    COUNTER = "counter"        # 防守反击
    ENERGY = "energy"          # 聚能
    BUILD = "build"            # 构建/拆解
    FLEX = "flex"              # 攻守兼备
    IRON_DEFENSE = "iron_defense"  # 严防死守
