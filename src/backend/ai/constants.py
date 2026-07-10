"""Shared AI model constants — single source of truth for input/output dimensions (design/31).

All rnn_v1 and rnn_v2 models share these values (design/31 D4).
SampleLoader and other shared modules import from here instead of coupling to a specific model version.
"""

INPUT_SIZE = 42       # 42-d feature vector (design/27)
HIDDEN_SIZE = 32      # GRU hidden state dimension
NUM_CARD_TYPES = 7    # attack, defense, heavy, counter, energy, build, flex
POLICY_OUTPUT_SIZE = 14  # 14-d: pass(0) + interleaved build/play pairs (design/28)
