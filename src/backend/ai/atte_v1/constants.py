"""Shared AI model constants for atte_v1 — single source of truth (design/40).

All atte_v1 model components import from here instead of hardcoding dimensions.
"""

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

ACTION_VOCAB_SIZE = 25      # 24 action IDs + 1 PAD (for last_action=-1)
ACTION_EMBEDDING_DIM = 32   # e1 dimension
STATUS_VOCAB_SIZE = 3       # DEFENSE=0, GUARD=1, FRAGILE=2
STATUS_EMBEDDING_DIM = 8    # e2 dimension
ORDINAL_VOCAB_SIZE = 20     # max ordinal 19 (min(ordinal, 19))
ORDINAL_EMBEDDING_DIM = 2   # e3 dimension

# ---------------------------------------------------------------------------
# Derived dimensions
# ---------------------------------------------------------------------------

EMB2_DIM = ACTION_EMBEDDING_DIM + ORDINAL_EMBEDDING_DIM  # 34
Q_DIM = EMB2_DIM                                          # 34, Q1/Q2/Q3

# ---------------------------------------------------------------------------
# Network dimensions
# ---------------------------------------------------------------------------

GRU_INPUT_SIZE = 80         # last_action_emb x 2(64) + SS1(8) + SS2(8)
HIDDEN_SIZE = 32            # GRU hidden state dimension
FEATURE_CONCAT_INPUT_SIZE = 169  # HPx2+energyx2+round+last_act_embx2+HAZ+Z3+h_out
OUTPUT_FEATURE_DIM = 64     # out vector dimension
POLICY_HIDDEN_DIM = 32      # Policy Head hidden layer
VALUE_HIDDEN_DIM = 32       # Value Head hidden layer
POLICY_OUTPUT_SIZE = 24     # Policy output dimension

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

MAX_HP = 30                 # HP normalization divisor
MAX_ENERGY = 5              # Energy normalization divisor

# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

PAD_ACTION_ID = 24          # last_action=-1 maps to this e1 token index
