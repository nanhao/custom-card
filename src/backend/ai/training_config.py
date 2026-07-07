"""Training configuration dataclass for split PPO training runs.

Per design/9 interfaces.md §3 and design/10: immutable frozen dataclass
with validation in __post_init__. Supports split policy/value networks
with separate learning rates and unified epoch count.

Attributes:
    sample_path: File path or directory. If directory, all .json files
                 inside are loaded as training samples.
    output_dir: Directory to save policy.pth and value.pth.
    model: AI registry name to train (e.g. "rnn_v2.1").
    epoch: Number of training epochs for both value (Phase 1)
           and policy (Phase 3) phases (>=1).
    policy_lr: Learning rate for policy optimizer (>0).
    value_lr: Learning rate for value optimizer (>0).
    weight_decay: Weight decay coefficient for AdamW (>=0).
    gamma: Discount factor for GAE (0 < gamma <= 1).
    gae_lambda: GAE lambda parameter (0 <= lambda <= 1).
    clip_eps: PPO clipping epsilon (>0).
    max_grad_norm: Maximum gradient norm for clipping (>0).
    value_coef: Value loss coefficient (>=0) — Phase 1 only.
    entropy_coef: Entropy bonus coefficient (>=0) — Phase 3 only.
    batch_size: Number of perspectives per accumulation group (>=1).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a three-phase split PPO training run."""

    sample_path: str
    output_dir: str
    model: str
    epoch: int
    policy_lr: float
    value_lr: float
    weight_decay: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    max_grad_norm: float
    value_coef: float
    entropy_coef: float
    batch_size: int

    def __post_init__(self) -> None:
        """Validate all field values after initialization.

        Raises:
            ValueError: If any field fails its validation constraint.
        """
        # Validate integer fields
        if self.epoch < 1:
            raise ValueError("epoch must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        # Validate output_dir is non-empty
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")

        # Validate gamma: (0, 1]
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")

        # Validate gae_lambda: [0, 1]
        if not (0 <= self.gae_lambda <= 1):
            raise ValueError("gae_lambda must be in [0, 1]")

        # Validate positive floats
        if self.clip_eps <= 0:
            raise ValueError("clip_eps must be > 0")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        if self.policy_lr <= 0:
            raise ValueError("policy_lr must be > 0")
        if self.value_lr <= 0:
            raise ValueError("value_lr must be > 0")

        # Validate non-negative floats
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.value_coef < 0:
            raise ValueError("value_coef must be >= 0")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be >= 0")

        # Validate sample_path exists
        if not os.path.exists(self.sample_path):
            raise ValueError(
                f"sample_path does not exist: {self.sample_path!r}"
            )
