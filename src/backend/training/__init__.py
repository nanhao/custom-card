"""Training package — self-play orchestration and PPO sample collection.

Contains:
    - TrainingManager: AI action sampling and self-play game loop
    - SelfPlayRunner: self-play game execution loop
    - SampleCollector: PPO training sample collection and persistence
    - SelfPlayConfig / SelfPlayResult: configuration and result dataclasses
    - BaseTrainingRunner: abstract PPO training runner (3-phase)
    - TrainingConfig: training configuration dataclass
    - SampleLoader: training sample JSON loading and validation
"""

from src.backend.training.training_manager import TrainingManager
from src.backend.training.selfplay_runner import SelfPlayRunner
from src.backend.training.sample_collector import SampleCollector
from src.backend.training.selfplay_config import SelfPlayConfig, SelfPlayResult
from src.backend.training.training_runner import BaseTrainingRunner
from src.backend.training.training_config import TrainingConfig
from src.backend.training.sample_loader import SampleLoader
from src.backend.training.training_utils import compute_grad_norm, save_split_training_artifacts
