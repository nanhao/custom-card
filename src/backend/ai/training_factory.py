"""Training runner factory — creates V1/V2TrainingRunner based on model name (design/31 D7).

Factory class that resolves the correct TrainingRunner subclass
without if-else in client code (train.py).
"""

from src.backend.ai.registry import parse_ai_name
from src.backend.ai.training_runner import BaseTrainingRunner


class TrainingRunnerFactory:
    """Factory that returns the correct TrainingRunner subclass for a given model."""

    @staticmethod
    def create(model_name: str) -> BaseTrainingRunner:
        """Create a TrainingRunner instance based on the model name.

        Parses model_name (e.g. "rnn_v1.3" or "rnn_v2.1") to determine
        the architecture, then returns the corresponding subclass instance.

        Args:
            model_name: AI identifier in name.gen format.

        Returns:
            V1TrainingRunner or V2TrainingRunner instance.

        Raises:
            ValueError: If the architecture name is not recognized.
        """
        name, _gen = parse_ai_name(model_name)
        if name == "rnn_v1":
            from src.backend.ai.rnn_v1.v1_training_runner import V1TrainingRunner
            return V1TrainingRunner()
        elif name == "rnn_v2":
            from src.backend.ai.rnn_v2.v2_training_runner import V2TrainingRunner
            return V2TrainingRunner()
        raise ValueError(f"Unknown model: {name!r}")
