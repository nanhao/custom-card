"""AI Self-Play CLI — thin wrapper around TrainingManager.run_selfplay().

Per design/5 D1, this script contains NO business logic — only argument
parsing and delegation. The game loop lives in app/selfplay_runner.py.

NEW (design/6): reads config/backend.json, calls LoggerBuilder.init().
NEW (design/10): adds --perspective argument for viewpoint filtering.

Usage:
    python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1
    python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 --game 200
    python src/backend/scripts/selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 \\
        --temperature 0.5 --temp_step 3 --temp_side both \\
        --perspective ai1 --sample_output data/samples.jsonl --temp_seed 42
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path for `src.*` imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backend.training.selfplay_config import SelfPlayConfig
from src.backend.training.rnn_v2.training_manager import TrainingManager
from src.backend.log.structured_logger import LoggerBuilder


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all self-play options.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="AI Self-Play — pit two AIs against each other."
    )
    parser.add_argument(
        "--ai1", required=True,
        help="AI name for player 1 (e.g. rnn_v1.1)",
    )
    parser.add_argument(
        "--ai2", required=True,
        help="AI name for player 2 (e.g. rnn_v1.1)",
    )
    parser.add_argument(
        "--game", type=int, default=100,
        help="Number of games to play (default: 100)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature. 0 = greedy (default: 1.0)",
    )
    parser.add_argument(
        "--temp_step", type=int, default=None,
        help="First N rounds use sampling, then greedy (default: no limit)",
    )
    parser.add_argument(
        "--temp_side", choices=["player1", "player2", "both"], default="both",
        help="Which side to sample (default: both)",
    )
    parser.add_argument(
        "--temp_seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--sample_output", type=str, default=None,
        help="Path for PPO sample JSONL output file",
    )
    parser.add_argument(
        "--perspective", choices=["ai1", "ai2", "both"], default="both",
        help="Perspectives to include in sample file (default: both)",
    )
    return parser


def _load_logging_config() -> tuple[bool, str]:
    """Load logging configuration from config/backend.json.

    Returns (enabled, log_dir) tuple. Defaults to (False, "logs")
    if the config file is missing, malformed, or has no logging section.

    Returns:
        (enabled, log_dir) where enabled is bool and log_dir is str.
    """
    config_path = _PROJECT_ROOT / "src" / "backend" / "config" / "backend.json"
    try:
        if not config_path.exists():
            return False, "logs"
        raw = config_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        # Malformed config or read error → default to disabled
        return False, "logs"

    logging_cfg = data.get("logging")
    if not isinstance(logging_cfg, dict):
        return False, "logs"

    # Explicit type check for enabled: only accept Python bool.
    # Avoid bool() coercion which would treat non-empty strings
    # (e.g., "false") and non-zero ints as True.
    raw_enabled = logging_cfg.get("enabled", False)
    if not isinstance(raw_enabled, bool):
        enabled = False
    else:
        enabled = raw_enabled

    log_dir = str(logging_cfg.get("log_dir", "logs"))
    return enabled, log_dir


def main() -> None:
    """Parse CLI arguments and delegate to AppManager.run_selfplay()."""
    parser = _build_parser()
    args = parser.parse_args()

    # Per design/10 TC-SP-05: warn if perspective is set but
    # sample_output is missing (perspective has no effect).
    if args.perspective != "both" and args.sample_output is None:
        print(
            "WARNING: --perspective is set to '{}' but --sample_output "
            "is not specified. Perspective filtering has no effect "
            "without sample output.".format(args.perspective),
            file=sys.stderr,
        )

    # Per design/6 D4: initialize the logger singleton from config.
    # Defaults to disabled if config is missing or logging section absent.
    logging_enabled, log_dir = _load_logging_config()
    LoggerBuilder.init(enabled=logging_enabled, log_dir=log_dir)

    # Build immutable config from CLI args
    config = SelfPlayConfig(
        ai1=args.ai1,
        ai2=args.ai2,
        game_count=args.game,
        temperature=args.temperature,
        temp_step=args.temp_step,
        temp_side=args.temp_side,
        temp_seed=args.temp_seed,
        sample_output=args.sample_output,
        perspective=args.perspective,
    )

    # Create TrainingManager and run
    training = TrainingManager()
    training.run_selfplay(config)


if __name__ == "__main__":
    main()
