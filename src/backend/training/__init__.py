"""Training package — self-play orchestration and PPO sample collection.

Contains:
    - TrainingManager: AI action sampling and self-play game loop
    - SelfPlayRunner: self-play game execution loop
    - SampleCollector: PPO training sample collection and persistence
    - SelfPlayConfig / SelfPlayResult: configuration and result dataclasses
"""

from src.backend.training.training_manager import TrainingManager
from src.backend.training.selfplay_runner import SelfPlayRunner
from src.backend.training.sample_collector import SampleCollector
from src.backend.training.selfplay_config import SelfPlayConfig, SelfPlayResult
