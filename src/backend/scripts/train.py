"""Split PPO Training CLI — thin wrapper around TrainingRunner.run().

Per design/9 D3, this script contains NO business logic — only argument
parsing, config loading, and delegation. Training logic in training_runner.py.

Usage:
    python src/backend/scripts/train.py ^
        --sample-path samples/rnn_v2_self_play.json ^
        --policy-output-path src/backend/ai/rnn_v1/gen2/policy.pth ^
        --value-output-path src/backend/ai/rnn_v1/gen2/value.pth ^
        --model rnn_v2.1 ^
        --value-epochs 20 ^
        --epoch 10
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
    """Build argument parser with split training options."""
    parser = argparse.ArgumentParser(
        description="Split PPO Training — train policy & value networks on self-play samples."
    )
    parser.add_argument(
        "--sample-path", required=True,
        help="File path or directory of JSON sample files",
    )
    parser.add_argument(
        "--policy-output-path", required=True,
        help="Output .pth for policy network weights",
    )
    parser.add_argument(
        "--value-output-path", required=True,
        help="Output .pth for value network weights",
    )
    parser.add_argument(
        "--model", required=True,
        help="AI registry name to train (e.g. rnn_v2.1)",
    )
    parser.add_argument(
        "--value-epochs", type=int, required=True,
        help="Number of value training epochs (Phase 1)",
    )
    parser.add_argument(
        "--epoch", type=int, required=True,
        help="Number of policy training epochs (Phase 3)",
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
        "policy_lr": float, "value_lr": float, "weight_decay": float,
        "gamma": float, "gae_lambda": float,
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

    training_cfg = _load_training_config()

    config = TrainingConfig(
        sample_path=args.sample_path,
        policy_output_path=args.policy_output_path,
        value_output_path=args.value_output_path,
        model=args.model,
        epoch=args.epoch,
        value_epochs=args.value_epochs,
        policy_lr=training_cfg["policy_lr"],
        value_lr=training_cfg["value_lr"],
        weight_decay=training_cfg["weight_decay"],
        gamma=training_cfg["gamma"],
        gae_lambda=training_cfg["gae_lambda"],
        clip_eps=training_cfg["clip_eps"],
        max_grad_norm=training_cfg["max_grad_norm"],
        value_coef=training_cfg["value_coef"],
        entropy_coef=training_cfg["entropy_coef"],
        batch_size=training_cfg["batch_size"],
    )

    runner = TrainingRunner()
    runner.run(config)


if __name__ == "__main__":
    main()
