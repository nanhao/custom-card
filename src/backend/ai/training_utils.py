"""Utility functions for PPO training — artifact generation.

Per design/8 D7: generates loss/grad PNG charts and CSV data files.
Separated from training_runner.py to stay under 500-line limit.
"""

import csv
import math
import os


def save_training_artifacts(
    epoch_losses: list[float],
    epoch_policy_losses: list[float],
    epoch_value_losses: list[float],
    epoch_entropies: list[float],
    epoch_grad_norms: list[float],
    output_dir: str | None = None,
) -> None:
    """Generate loss/grad PNG charts and CSV files (design D7).

    By default writes to fixed loss/ directory relative to CWD.
    If output_dir is provided, writes there instead.
    Auto-creates the directory if it doesn't exist.
    Previous run's artifacts are overwritten.
    """
    loss_dir = output_dir if output_dir is not None else os.path.join(os.getcwd(), "loss")
    os.makedirs(loss_dir, exist_ok=True)

    epochs = list(range(1, len(epoch_losses) + 1))

    # Write loss CSV: epoch,avg_loss,policy_loss,value_loss,entropy
    _write_csv(
        os.path.join(loss_dir, "loss.csv"),
        ["epoch", "avg_loss", "policy_loss", "value_loss", "entropy"],
        zip(epochs, epoch_losses, epoch_policy_losses, epoch_value_losses, epoch_entropies),
    )

    # Write grad CSV: epoch,grad_norm
    _write_csv(
        os.path.join(loss_dir, "grad.csv"),
        ["epoch", "grad_norm"],
        zip(epochs, epoch_grad_norms),
    )

    # Generate PNG charts (if matplotlib is available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Loss chart: total loss + components
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, epoch_losses, "b-", label="Total Loss")
        ax.plot(epochs, epoch_policy_losses, "g--", label="Policy Loss")
        ax.plot(epochs, epoch_value_losses, "r--", label="Value Loss")
        ax.plot(epochs, epoch_entropies, "m--", label="Entropy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(loss_dir, "loss.png"), dpi=100)
        plt.close(fig)

        # Gradient norm chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, epoch_grad_norms, "b-")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm")
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
