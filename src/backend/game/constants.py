"""Game-level constants shared across game/ modules.

Extracted here to avoid circular imports between game_core, card_def,
and resolver modules. These have zero dependencies on other game/ modules.
"""

# Design/26 D1: 3 rounds per cycle.
ROUNDS_PER_CYCLE: int = 3
# Design/26 D2: draw 5 cards per cycle start.
CARDS_PER_CYCLE: int = 5
# Design/38: Player initial HP.
INITIAL_HP: int = 30
# Design/38: Player initial energy.
INITIAL_ENERGY: int = 2
