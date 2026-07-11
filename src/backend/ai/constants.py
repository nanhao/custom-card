"""Shared AI model constants — single source of truth for input/output dimensions (design/31, design/37).

All rnn_v1 and rnn_v2 models share these values (design/31 D4).
SampleLoader and other shared modules import from here instead of coupling to a specific model version.

Per design/37: INPUT_SIZE 56→62 (8 card types), POLICY_OUTPUT_SIZE 14→24 (adds dismantle), NUM_CARD_TYPES 7→8.
"""

INPUT_SIZE = 62       # 62-d feature vector (design/37, per intention/input.md)
HIDDEN_SIZE = 32      # GRU hidden state dimension
NUM_CARD_TYPES = 8    # BUILD, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
POLICY_OUTPUT_SIZE = 24  # 24-d: pass(0) + 8×(play, build, dismantle) triplets (design/37)
