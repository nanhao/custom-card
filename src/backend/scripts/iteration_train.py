"""Iterative PPO Training Pipeline — self-play → train → validate → repeat.

Per intention/40: iteratively trains AI generations, starting from --start-gen,
producing gen+1 weights each iteration, with multi-opponent validation.

Usage:
    python src/backend/scripts/iteration_train.py --ai atte_v1 --start-gen 1 ^
        --sample-dir samples/ --game 2000 --policy-epoch 5 --value-epoch 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for src.* imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from src.backend.training.selfplay_config import SelfPlayConfig
from src.backend.training.training_config import TrainingConfig
from src.backend.training.training_factory import TrainingRunnerFactory
from src.backend.training.rnn_v2.training_manager import TrainingManager
from src.backend.log.structured_logger import LoggerBuilder
from src.backend.ai.registry import parse_ai_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterative PPO Training — self-play + train + validate loop."
    )
    parser.add_argument("--ai", required=True, help="AI name (e.g. atte_v1)")
    parser.add_argument(
        "--start-gen", type=int, required=True,
        help="Starting generation number (e.g. 1)",
    )
    parser.add_argument(
        "--sample-dir", type=str, required=True,
        help="Root directory for sample output (e.g. samples/)",
    )
    parser.add_argument(
        "--game", type=int, required=True,
        help="Base game count for self-play per generation",
    )
    parser.add_argument(
        "--policy-epoch", type=int, default=5,
        help="Policy network training epochs (default: 5)",
    )
    parser.add_argument(
        "--value-epoch", type=int, default=10,
        help="Value network training epochs (default: 10)",
    )
    return parser


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_training_config() -> dict:
    config_path = _PROJECT_ROOT / "src" / "backend" / "config" / "backend.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return data["training"]


def _init_logger() -> None:
    """Initialize structured logger from config, defaults to disabled."""
    config_path = _PROJECT_ROOT / "src" / "backend" / "config" / "backend.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        logging_cfg = data.get("logging", {})
        enabled = logging_cfg.get("enabled", False)
        log_dir = logging_cfg.get("log_dir", "logs")
    except (json.JSONDecodeError, OSError):
        enabled, log_dir = False, "logs"
    LoggerBuilder.init(enabled=bool(enabled), log_dir=str(log_dir))


# ---------------------------------------------------------------------------
# Self-play runner
# ---------------------------------------------------------------------------


def _run_selfplay(
    ai1: str,
    ai2: str,
    game_count: int,
    sample_output: str | None,
    perspective: str = "both",
    temperature: float = 1.0,
    temp_step: int | None = None,
    temp_side: str = "both",
):
    """Run self-play games and return SelfPlayResult."""
    config = SelfPlayConfig(
        ai1=ai1,
        ai2=ai2,
        game_count=game_count,
        sample_output=sample_output,
        temperature=temperature,
        perspective=perspective,
        temp_step=temp_step,
        temp_side=temp_side,
    )
    manager = TrainingManager()
    return manager.run_selfplay(config)


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------


def _run_training(
    sample_dir: str,
    output_dir: str,
    model_name: str,
    policy_epoch: int,
    value_epoch: int,
    arch_name: str,
) -> None:
    """Run three-phase PPO training using TrainingRunnerFactory."""
    training_cfg = _load_training_config()
    config = TrainingConfig(
        sample_path=sample_dir,
        output_dir=output_dir,
        model=model_name,
        policy_epoch=policy_epoch,
        value_epoch=value_epoch,
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
        validation_split=training_cfg["validation_split"],
    )
    runner = TrainingRunnerFactory.create(model_name)
    runner.run(config)

    # For atte_v1: extract e1/e2 embeddings into separate weight files
    if arch_name == "atte_v1":
        _split_atte_v1_weights(output_dir)


# ---------------------------------------------------------------------------
# atte_v1 weight splitting
# ---------------------------------------------------------------------------


def _split_atte_v1_weights(output_dir: str) -> None:
    """Extract e1/e2 embeddings from full state_dicts into separate files.

    atte_v1 adapter expects 4 files per network:
      policy_e1_e2.pth + policy.pth  (for policy)
      value_e1_e2.pth  + value.pth   (for value)

    Training saves full state_dict as policy.pth / value.pth.
    We split e1/e2 keys into separate files, keeping main.pth with
    non-e1/e2 keys (strict=False loading is compatible either way).
    """
    for prefix in ("policy", "value"):
        full_path = os.path.join(output_dir, f"{prefix}.pth")
        if not os.path.exists(full_path):
            continue
        state = torch.load(full_path, map_location="cpu", weights_only=True)
        e1e2_state = {
            k: v for k, v in state.items()
            if k.startswith("body.e1.") or k.startswith("body.e2.")
        }
        main_state = {
            k: v for k, v in state.items()
            if not (k.startswith("body.e1.") or k.startswith("body.e2."))
        }
        e1e2_path = os.path.join(output_dir, f"{prefix}_e1_e2.pth")
        torch.save(e1e2_state, e1e2_path)
        torch.save(main_state, full_path)
        print(f"  Split {prefix}: e1e2={len(e1e2_state)} keys, main={len(main_state)} keys")


# ---------------------------------------------------------------------------
# Weight directory existence check
# ---------------------------------------------------------------------------


def _gen_weight_dir_exists(ai_name: str, gen: int) -> bool:
    """Check if weight directory for a generation exists."""
    arch_name, _ = parse_ai_name(f"{ai_name}.{gen}")
    weight_dir = os.path.join(
        _PROJECT_ROOT, "src", "backend", "ai", arch_name, f"gen{gen}",
    )
    return os.path.isdir(weight_dir) and os.listdir(weight_dir)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Initialize structured logger (required by TrainingManager)
    _init_logger()

    ai_name = args.ai
    arch_name, _ = parse_ai_name(f"{ai_name}.1")
    start_gen = args.start_gen
    sample_root = args.sample_dir.rstrip("/\\")
    base_game = args.game
    policy_epoch = args.policy_epoch
    value_epoch = args.value_epoch

    gen = start_gen
    prev_gens: list[int] = []  # track which historical gens exist

    # Verify start_gen weights exist
    start_model = f"{ai_name}.{start_gen}"
    if not _gen_weight_dir_exists(ai_name, start_gen):
        print(
            f"ERROR: start_gen={start_gen} weights not found for {start_model!r}. "
            f"Expected at src/backend/ai/{arch_name}/gen{start_gen}/",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Starting from {start_model}, weights confirmed.")

    while True:
        model_name = f"{ai_name}.{gen}"
        print(f"\n{'='*60}")
        print(f"=== Generation {gen} (model: {model_name}) ===")
        print(f"{'='*60}")

        # ----------------------------------------------------------------
        # Step 1: Generate training samples
        # ----------------------------------------------------------------
        sample_dir = os.path.join(sample_root, ai_name, f"gen{gen}")
        os.makedirs(sample_dir, exist_ok=True)

        # 1a. Self-play (gen vs gen)
        sp_path = os.path.join(sample_dir, "self_play.json")
        print(f"\n[Self-Play] {model_name} vs {model_name}, {base_game} games")
        sp_result = _run_selfplay(
            ai1=model_name, ai2=model_name,
            game_count=base_game,
            sample_output=sp_path,
            perspective="both",
        )
        print(f"  ai1_wins={sp_result.ai1_wins}, ai2_wins={sp_result.ai2_wins}, "
              f"draws={sp_result.draws}, avg_rounds={sp_result.avg_rounds:.1f}")

        # 1b. Cross-play against historical generations
        hist_gens: list[int] = []
        for k in range(1, 5):
            hist_gen = gen - k
            if hist_gen < 1:
                break
            hist_model = f"{ai_name}.{hist_gen}"
            if not _gen_weight_dir_exists(ai_name, hist_gen):
                print(f"\n[Cross-Play] gen {hist_gen} weights not found, skipping")
                break
            hist_gens.append(hist_gen)
            game_count = max(1, base_game // (2 ** k))
            cp_path = os.path.join(sample_dir, f"{hist_gen}_play.json")
            print(f"\n[Cross-Play] {model_name} vs {hist_model}, {game_count} games")
            cp_result = _run_selfplay(
                ai1=model_name, ai2=hist_model,
                game_count=game_count,
                sample_output=cp_path,
                perspective="ai1",
            )
            print(f"  ai1_wins={cp_result.ai1_wins}, ai2_wins={cp_result.ai2_wins}, "
                  f"draws={cp_result.draws}, avg_rounds={cp_result.avg_rounds:.1f}")

        # ----------------------------------------------------------------
        # Step 2: Train → gen+1 weights
        # ----------------------------------------------------------------
        next_gen = gen + 1
        output_dir = os.path.join(
            _PROJECT_ROOT, "src", "backend", "ai", arch_name, f"gen{next_gen}",
        )
        print(f"\n[Training] gen{gen} samples → gen{next_gen} weights")
        print(f"  sample_dir: {sample_dir}")
        print(f"  output_dir: {output_dir}")
        print(f"  policy_epoch: {policy_epoch}, value_epoch: {value_epoch}")

        _run_training(
            sample_dir=sample_dir,
            output_dir=output_dir,
            model_name=model_name,
            policy_epoch=policy_epoch,
            value_epoch=value_epoch,
            arch_name=arch_name,
        )
        print(f"  Training complete → {output_dir}")

        # ----------------------------------------------------------------
        # Step 3: Validation matches
        # ----------------------------------------------------------------
        next_model = f"{ai_name}.{next_gen}"
        val_opponents = [gen] + hist_gens  # gen, gen-1, ..., gen-4
        val_opponents = [g for g in val_opponents if g >= 1]

        all_passed = True
        print(f"\n[Validation] {next_model} vs historical gens, 500 games each")

        for opp_gen in val_opponents:
            opp_model = f"{ai_name}.{opp_gen}"
            if not _gen_weight_dir_exists(ai_name, opp_gen):
                print(f"  vs gen{opp_gen}: weights not found, skipping")
                continue

            print(f"  vs gen{opp_gen}...", end=" ", flush=True)
            val_result = _run_selfplay(
                ai1=next_model, ai2=opp_model,
                game_count=500,
                sample_output=None,
                temp_step=3,
                temp_side="both",
            )
            # win rate margin: (ai1_wins - ai2_wins) / total > 5%
            if val_result.total_games == 0:
                margin = 0.0
            else:
                margin = (val_result.ai1_wins - val_result.ai2_wins) / val_result.total_games
            passed = margin > 0.05
            status = "PASS" if passed else "FAIL"
            print(f"margin={margin:+.1%} ({val_result.ai1_wins}-{val_result.ai2_wins}-{val_result.draws}) [{status}]")

            if not passed:
                all_passed = False

        if not all_passed:
            print(f"\n*** STOP: gen{next_gen} failed validation. Exiting. ***")
            break

        print(f"\n*** gen{next_gen} passed all validations! Advancing to next iteration. ***")
        prev_gens.append(gen)
        gen = next_gen


if __name__ == "__main__":
    main()
