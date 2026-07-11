"""Integration tests for train.py CLI argument parsing (design/32).

Test cases per design/32 test_cases.md: TC-CLI-01 through TC-CLI-05.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_script_path() -> str:
    """Return absolute path to train.py CLI script."""
    return str(
        Path(__file__).resolve().parent.parent.parent
        / "src" / "backend" / "scripts" / "train.py"
    )


def _make_sample_file(tmp_dir: str) -> str:
    """Create a minimal valid training sample JSON file."""
    sample = [{
        "game_id": "test_game",
        "winner": "1",
        "rounds": 6,
        "perspectives": [
            {
                "player": "1",
                "steps": [
                    {
                        "input": [float(i) for i in range(42)],
                        "action": 2,
                        "reward": 0.1,
                        "value": 0.5,
                        "log_prob": -0.1,
                    }
                ],
            },
            {
                "player": "2",
                "steps": [
                    {
                        "input": [float(i) for i in range(42)],
                        "action": 2,
                        "reward": 0.1,
                        "value": 0.5,
                        "log_prob": -0.1,
                    }
                ],
            },
        ],
    }]
    path = Path(tmp_dir) / "sample.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    return str(path)


def _run_train(args: list[str]) -> subprocess.CompletedProcess:
    """Run train.py with given args and return completed process."""
    return subprocess.run(
        [sys.executable, _train_script_path()] + args,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )


# ---------------------------------------------------------------------------
# TC-CLI-01 ~ TC-CLI-05
# ---------------------------------------------------------------------------


class TestTrainCLIEpochArgs:
    """TC-CLI-01 through TC-CLI-05: train.py CLI argument parsing."""

    def test_policy_epoch_is_required(self):
        """TC-CLI-01: missing --policy-epoch → argparse error, non-zero exit."""
        result = _run_train([
            "--sample-path", "dummy.json",
            "--output-dir", "/tmp",
            "--model", "rnn_v2.1",
        ])
        assert result.returncode != 0
        assert "--policy-epoch" in result.stderr

    def test_only_policy_epoch_runs_normally(self):
        """TC-CLI-02: only --policy-epoch (no --value-epoch) → value_epoch defaults."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_sample_file(td)
            # This test verifies that the CLI accepts --policy-epoch alone.
            # Full training would require actual weight files, so we check
            # that argparse passes and the error comes from missing weights,
            # not from missing --epoch.
            result = _run_train([
                "--sample-path", sample,
                "--output-dir", td,
                "--model", "rnn_v2.1",
                "--policy-epoch", "1",
            ])
            # Should fail later (model weight not found), not at arg parse
            # argparse success means --policy-epoch was accepted
            stderr = result.stderr.lower()
            assert "unrecognized arguments" not in stderr
            assert "argparse" not in stderr.lower()

    def test_both_policy_and_value_epoch_accepted(self):
        """TC-CLI-03: --policy-epoch 10 --value-epoch 5 → accepted."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_sample_file(td)
            result = _run_train([
                "--sample-path", sample,
                "--output-dir", td,
                "--model", "rnn_v2.1",
                "--policy-epoch", "1",
                "--value-epoch", "5",
            ])
            stderr = result.stderr.lower()
            assert "unrecognized arguments" not in stderr
            assert "argparse" not in stderr.lower()

    def test_old_epoch_argument_rejected(self):
        """TC-CLI-04: --epoch is no longer a valid argument, command fails."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_sample_file(td)
            result = _run_train([
                "--sample-path", sample,
                "--output-dir", td,
                "--model", "rnn_v2.1",
                "--epoch", "10",
            ])
            # argparse silently ignores --epoch (no argument matches),
            # then fails because --policy-epoch is required
            assert result.returncode != 0

    def test_help_shows_new_params_not_old(self):
        """TC-CLI-05: --help shows --policy-epoch and --value-epoch, not --epoch."""
        result = _run_train(["--help"])
        assert result.returncode == 0
        stdout = result.stdout
        # New params present
        assert "--policy-epoch" in stdout
        assert "--value-epoch" in stdout
        # Old param absent (--epoch appears as substring of --policy-epoch,
        # so check for standalone --epoch with whitespace boundary)
        assert " --epoch " not in stdout


class TestTrainCLIValidationSplit:
    """TC-CLI-VS: train.py --validation-split argument (design/33)."""

    def test_validation_split_accepted(self):
        """TC-CLI-VS-01: --validation-split 0.2 is accepted."""
        with tempfile.TemporaryDirectory() as td:
            sample = _make_sample_file(td)
            result = _run_train([
                "--sample-path", sample,
                "--output-dir", td,
                "--model", "rnn_v2.1",
                "--policy-epoch", "1",
                "--validation-split", "0.2",
            ])
            stderr = result.stderr.lower()
            assert "unrecognized arguments" not in stderr
            assert "argparse" not in stderr.lower()

    def test_help_shows_validation_split(self):
        """TC-CLI-VS-02: --help shows --validation-split."""
        result = _run_train(["--help"])
        assert result.returncode == 0
        assert "--validation-split" in result.stdout
