"""Game-level constants shared across game/ modules.

Extracted here to avoid circular imports between game_core, card_def,
and resolver modules. These have zero dependencies on other game/ modules.
"""

# Design/26 D1: 3 rounds per cycle.
ROUNDS_PER_CYCLE: int = 3
