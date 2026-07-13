"""Training runner factory — creates V2TrainingRunner based on model name (design/31 D7).

Factory class that creates the correct TrainingRunner subclass.
V1 (rnn_v1) is deprecated — only V2TrainingRunner is supported.
"""

from src.backend.training.training_runner import BaseTrainingRunner


class TrainingRunnerFactory:
    """Factory that returns the correct TrainingRunner subclass for a given model."""

    @staticmethod
    def create(model_name: str) -> BaseTrainingRunner:
        """Create a TrainingRunner instance based on the model name.

        Only V2TrainingRunner is supported; rnn_v1 is deprecated.

        Args:
            model_name: AI identifier in name.gen format (e.g. "rnn_v2.1").

        Returns:
            V2TrainingRunner instance.

        Raises:
            ValueError: If the architecture name is not recognized.
        """
        from src.backend.ai.registry import parse_ai_name
        name, _gen = parse_ai_name(model_name)
        if name == "rnn_v2":
            from src.backend.training.rnn_v2.v2_training_runner import V2TrainingRunner
            return V2TrainingRunner()
        raise ValueError(
            f"Unknown or deprecated model: {name!r}. "
            f"Only rnn_v2 is supported."
        )
