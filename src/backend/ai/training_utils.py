"""Utility functions for PPO training — artifact generation.

Per design/9 D7: generates loss/grad CSV files and PNG charts for
three-phase split PPO training with phase-aware columns.
Separated from training_runner.py to stay under 500-line limit.
"""

import csv
import math
import os


def save_split_training_artifacts(
    phase_records: list[dict],
    output_dir: str | None = None,
) -> None:
    """Generate loss/grad CSV files for three-phase split PPO training (design/9 D7).

    Each record dict has keys: phase, epoch, policy_loss, value_loss, entropy,
    policy_grad_norm, value_grad_norm. Phase is "value" or "policy".

    Writes to loss/loss.csv and loss/grad.csv with a 'phase' column.
    """
    loss_dir = output_dir if output_dir is not None else os.path.join(os.getcwd(), "loss")
    os.makedirs(loss_dir, exist_ok=True)

    # Write loss CSV: phase,epoch,policy_loss,value_loss,val_value_loss,entropy
    _write_csv(
        os.path.join(loss_dir, "loss.csv"),
        ["phase", "epoch", "policy_loss", "value_loss", "val_value_loss", "entropy"],
        [
            (r["phase"], r["epoch"], r["policy_loss"], r["value_loss"],
             r.get("val_value_loss", float("nan")), r["entropy"])
            for r in phase_records
        ],
    )

    # Write grad CSV: phase,epoch,policy_grad_norm,value_grad_norm
    _write_csv(
        os.path.join(loss_dir, "grad.csv"),
        ["phase", "epoch", "policy_grad_norm", "value_grad_norm"],
        [
            (r["phase"], r["epoch"], r["policy_grad_norm"], r["value_grad_norm"])
            for r in phase_records
        ],
    )

    # Generate PNG charts (if matplotlib available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Split data by phase for chart
        value_epochs = [r["epoch"] for r in phase_records if r["phase"] == "value"]
        policy_epochs = [r["epoch"] for r in phase_records if r["phase"] == "policy"]
        value_losses = [r["value_loss"] for r in phase_records if r["phase"] == "value"]
        val_value_losses = [
            r.get("val_value_loss", float("nan"))
            for r in phase_records if r["phase"] == "value"
        ]
        policy_p_losses = [r["policy_loss"] for r in phase_records if r["phase"] == "policy"]
        policy_entropies = [r["entropy"] for r in phase_records if r["phase"] == "policy"]

        fig, ax = plt.subplots(figsize=(8, 5))
        if value_losses:
            ax.plot(value_epochs, value_losses, "r--", label="Value Loss (Phase 1)")
        # Add validation loss curve if any valid values exist (design/33)
        valid_val = [
            v for v in val_value_losses
            if v is not None and not math.isnan(v)
        ]
        if valid_val:
            ax.plot(value_epochs, val_value_losses, "b--",
                    label="Val Value Loss (Phase 1)")
        if policy_p_losses:
            total_policy = [p + e * 0.01 for p, e in zip(policy_p_losses, policy_entropies)]
            ax.plot(policy_epochs, policy_p_losses, "g-", label="Policy Loss (Phase 3)")
            ax.plot(policy_epochs, policy_entropies, "m-", label="Entropy (Phase 3)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Split Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(loss_dir, "loss.png"), dpi=100)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        v_grad = [r["value_grad_norm"] for r in phase_records if r["phase"] == "value"]
        p_grad = [r["policy_grad_norm"] for r in phase_records if r["phase"] == "policy"]
        if v_grad:
            ax.plot(value_epochs, v_grad, "r--", label="Value Grad Norm (Phase 1)")
        if p_grad:
            ax.plot(policy_epochs, p_grad, "g-", label="Policy Grad Norm (Phase 3)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norms")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(loss_dir, "grad.png"), dpi=100)
        plt.close(fig)

    except ImportError:
        pass


def compute_grad_norm(model) -> float:
    """Compute total gradient L2 norm across all parameters.

    Returns 0.0 if no gradients exist.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def _write_csv(filepath: str, header: list[str], rows) -> None:
    """Write a CSV file with header and data rows."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
