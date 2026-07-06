"""PPO Training CLI — thin wrapper around TrainingRunner.run().

Per design/8 D3, this script contains NO business logic — only argument
parsing, config loading, and delegation. The training loop and all
PPO logic lives in ai/training_runner.py.

Usage:
    python src/backend/scripts/train.py ^
        --sample-path samples/rnn_v1_self_play.json ^
        --output-path src/backend/ai/rnn_v1/gen2/rnn_v1.pth ^
        --model rnn_v1.1 ^
        --epoch 10

    python src/backend/scripts/train.py ^
        --sample-path data/samples/ ^
        --output-path output.pth ^
        --model rnn_v1.1 ^
        --epoch 50
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for src.* imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backend.ai.training_config import TrainingConfig
from src.backend.ai.training_runner import TrainingRunner


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser with all training options."""
    parser = argparse.ArgumentParser(
        description="PPO Training — train an AI model on self-play samples."
    )
    parser.add_argument(
        "--sample-path", required=True,
        help="File path or directory of JSON sample files",
    )
    parser.add_argument(
        "--output-path", required=True,
        help="Output .pth file path for trained weights",
    )
    parser.add_argument(
        "--model", required=True,
        help="AI registry name to train (e.g. rnn_v1.1)",
    )
    parser.add_argument(
        "--epoch", type=int, required=True,
        help="Number of training epochs",
    )
    return parser


def _load_training_config() -> dict:
    """Load training hyperparameters from config/backend.json.

    Returns the "training" dict. Raises SystemExit if config is
    missing, malformed, or missing the training section.
    """
    config_path = _PROJECT_ROOT / "src" / "backend" / "config" / "backend.json"
    try:
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        raw = config_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Failed to read config: {exc}", file=sys.stderr)
        sys.exit(1)

    training_cfg = data.get("training")
    if not isinstance(training_cfg, dict):
        print("ERROR: Missing 'training' section in backend.json", file=sys.stderr)
        sys.exit(1)

    # Validate required keys with correct types
    required = {
        "lr": float, "gamma": float, "gae_lambda": float,
        "clip_eps": float, "max_grad_norm": float,
        "value_coef": float, "entropy_coef": float, "batch_size": int,
    }
    for key, expected_type in required.items():
        if key not in training_cfg:
            print(f"ERROR: Missing 'training.{key}' in backend.json", file=sys.stderr)
            sys.exit(1)
        if not isinstance(training_cfg[key], expected_type):
            print(
                f"ERROR: 'training.{key}' must be {expected_type.__name__}, "
                f"got {type(training_cfg[key]).__name__}",
                file=sys.stderr,
            )
            sys.exit(1)

    return training_cfg


def main() -> None:
    """Parse CLI args, load config, build TrainingConfig, delegate to runner."""
    parser = _build_parser()
    args = parser.parse_args()

    # Load hyperparameters from config
    training_cfg = _load_training_config()

    # Build immutable config
    config = TrainingConfig(
        sample_path=args.sample_path,
        output_path=args.output_path,
        model=args.model,
        epoch=args.epoch,
        lr=training_cfg["lr"],
        gamma=training_cfg["gamma"],
        gae_lambda=training_cfg["gae_lambda"],
        clip_eps=training_cfg["clip_eps"],
        max_grad_norm=training_cfg["max_grad_norm"],
        value_coef=training_cfg["value_coef"],
        entropy_coef=training_cfg["entropy_coef"],
        batch_size=training_cfg["batch_size"],
    )

    # Run training
    runner = TrainingRunner()
    runner.run(config)


if __name__ == "__main__":
    main()
