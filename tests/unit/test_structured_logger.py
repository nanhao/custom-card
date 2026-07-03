"""Unit tests for StructuredLogger — generic structured logging facility.

Covers test cases from design/6 test_cases.md:
    §1 — Construction and Properties (TC-L1 ~ TC-L5)
    §2 — Log Envelope Format (TC-L6 ~ TC-L9)
    §3 — Message Format (TC-M1 ~ TC-M9)
    §4 — File Management (TC-F1 ~ TC-F4)

All file I/O uses temporary directories (test rule: never touch real data).
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.backend.log.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# §1 — Construction and Properties (TC-L1 ~ TC-L5)
# ---------------------------------------------------------------------------


class TestConstructionAndProperties:
    """TC-L1 ~ TC-L5: Logger construction and trace_id property."""

    def test_disabled_logger_writes_nothing(self, tmp_path):
        """TC-L1: disabled logger's log() is a no-op, no file created."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=False, log_dir=str(log_dir))
        logger.log("INFO", "test", key="value")
        assert not log_dir.exists()

    def test_enabled_logger_writes_a_file(self, tmp_path):
        """TC-L2: enabled logger creates a dated log file with one line."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", key="value")
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.log"
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

    def test_trace_id_is_8_char_hex(self, tmp_path):
        """TC-L3: trace_id is exactly 8 hexadecimal characters."""
        logger = StructuredLogger(enabled=True, log_dir=str(tmp_path / "logs"))
        tid = logger.trace_id
        assert len(tid) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", tid) is not None

    def test_trace_id_differs_between_instances(self, tmp_path):
        """TC-L4: two instances produce different trace_ids."""
        ld = str(tmp_path / "logs")
        a = StructuredLogger(enabled=True, log_dir=ld)
        b = StructuredLogger(enabled=True, log_dir=ld)
        assert a.trace_id != b.trace_id

    def test_disabled_logger_has_valid_trace_id(self, tmp_path):
        """TC-L5: disabled logger still has a valid non-empty trace_id."""
        logger = StructuredLogger(enabled=False, log_dir=str(tmp_path / "logs"))
        tid = logger.trace_id
        assert isinstance(tid, str)
        assert len(tid) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", tid) is not None


# ---------------------------------------------------------------------------
# §2 — Log Envelope Format (TC-L6 ~ TC-L9)
# ---------------------------------------------------------------------------


class TestLogEnvelopeFormat:
    """TC-L6 ~ TC-L9: pipe-delimited log envelope structure."""

    def _read_line(self, log_dir: Path) -> str:
        """Read the single line from today's log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        return (log_dir / f"{today}.log").read_text(encoding="utf-8").strip()

    def test_log_line_has_correct_pipe_envelope(self, tmp_path):
        """TC-L6: log line has 7 pipe-delimited fields in correct format."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "selfplay.game_start", game_idx="1", game_count="100")
        line = self._read_line(log_dir)
        fields = line.split(" | ")
        assert len(fields) == 7
        # Field 1: ISO 8601 timestamp with .SSS milliseconds
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}", fields[0])
        # Field 2: 8-char hex traceId matching logger.trace_id
        assert fields[1] == logger.trace_id
        # Field 3: positive integer (process ID)
        assert int(fields[2]) > 0
        assert fields[2] == str(os.getpid())
        # Field 4: positive integer (thread ID)
        assert int(fields[3]) > 0
        # Field 5: level string
        assert fields[4] == "INFO"
        # Field 6: event name
        assert fields[5] == "selfplay.game_start"
        # Field 7: message (content tested in §3)
        assert "game_idx=1" in fields[6]
        assert "game_count=100" in fields[6]

    def test_trace_id_consistent_across_lines(self, tmp_path):
        """TC-L7: same traceId across all lines from one logger."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        for i in range(10):
            logger.log("INFO", "test", idx=str(i))
        today = datetime.now().strftime("%Y-%m-%d")
        lines = (log_dir / f"{today}.log").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 10
        trace_ids = {line.split(" | ")[1] for line in lines}
        assert len(trace_ids) == 1
        assert next(iter(trace_ids)) == logger.trace_id

    def test_pid_and_tid_are_real_os_values(self, tmp_path):
        """TC-L8: pid equals os.getpid(); tid is a positive integer."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test")
        line = self._read_line(log_dir)
        fields = line.split(" | ")
        assert int(fields[2]) == os.getpid()
        assert int(fields[3]) > 0

    def test_timestamp_increases_monotonically(self, tmp_path):
        """TC-L9: timestamps are monotonic between successive writes."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "first")
        logger.log("INFO", "second")
        today = datetime.now().strftime("%Y-%m-%d")
        lines = (log_dir / f"{today}.log").read_text(encoding="utf-8").strip().split("\n")
        t1 = lines[0].split(" | ")[0]
        t2 = lines[1].split(" | ")[0]
        assert t2 >= t1


# ---------------------------------------------------------------------------
# §3 — Message Format (TC-M1 ~ TC-M9)
# ---------------------------------------------------------------------------


class TestMessageFormat:
    """TC-M1 ~ TC-M9: message field key=value formatting rules."""

    def _get_message(self, log_dir: Path) -> str:
        """Return the message field (7th field) from the first log line."""
        today = datetime.now().strftime("%Y-%m-%d")
        line = (log_dir / f"{today}.log").read_text(encoding="utf-8").strip()
        return line.split(" | ")[6]

    def test_simple_key_value_pairs_space_separated(self, tmp_path):
        """TC-M1: multiple key=value pairs, single-space separated."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", a="1", b="2", c="3")
        msg = self._get_message(log_dir)
        # All three pairs must be present
        assert "a=1" in msg
        assert "b=2" in msg
        assert "c=3" in msg

    def test_none_renders_as_literal_none(self, tmp_path):
        """TC-M2: None renders as the literal string 'None'."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", winner=None, rounds="7")
        msg = self._get_message(log_dir)
        assert "winner=None" in msg
        assert "rounds=7" in msg

    def test_bool_renders_as_literal_true_false(self, tmp_path):
        """TC-M3: True → 'True', False → 'False'."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", enabled=True, success=False)
        msg = self._get_message(log_dir)
        assert "enabled=True" in msg
        assert "success=False" in msg

    def test_integers_render_as_decimal(self, tmp_path):
        """TC-M4: ints render as decimal strings, including negative."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", game_idx=5, count=0, negative=-3)
        msg = self._get_message(log_dir)
        assert "game_idx=5" in msg
        assert "count=0" in msg
        assert "negative=-3" in msg

    def test_floats_render_with_reasonable_precision(self, tmp_path):
        """TC-M5: floats render with at most 6 significant digits."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", avg_rounds=5.3, temp=1.0)
        msg = self._get_message(log_dir)
        assert "avg_rounds=5.3" in msg
        assert "temp=1.0" in msg

    def test_value_with_spaces_is_double_quoted(self, tmp_path):
        """TC-M6: values containing spaces are double-quoted."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("ERROR", "error", msg="play_round failed")
        msg = self._get_message(log_dir)
        assert 'msg="play_round failed"' in msg

    def test_value_with_double_quotes_escapes_them(self, tmp_path):
        """TC-M7: embedded double quotes are escaped with backslash."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", path='a "quoted" value')
        msg = self._get_message(log_dir)
        # Should contain escaped double quotes
        assert 'path="a \\"quoted\\" value"' in msg

    def test_empty_string_renders_as_empty_quotes(self, tmp_path):
        """TC-M8: empty string renders as ''."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", name="")
        msg = self._get_message(log_dir)
        assert 'name=""' in msg

    def test_simple_string_without_spaces_is_unquoted(self, tmp_path):
        """TC-M9: simple strings without special chars are unquoted."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", card1="attack", card2="defense")
        msg = self._get_message(log_dir)
        # Check both key=value pairs are present and unquoted
        assert "card1=attack" in msg
        assert "card2=defense" in msg


# ---------------------------------------------------------------------------
# §4 — File Management (TC-F1 ~ TC-F4)
# ---------------------------------------------------------------------------


class TestFileManagement:
    """TC-F1 ~ TC-F4: log directory creation, append, date rollover."""

    def test_log_dir_created_automatically(self, tmp_path):
        """TC-F1: log_dir is created if it doesn't exist."""
        log_dir = tmp_path / "nonexistent_subdir"
        assert not log_dir.exists()
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        logger.log("INFO", "test", key="value")
        assert log_dir.exists()
        today = datetime.now().strftime("%Y-%m-%d")
        assert (log_dir / f"{today}.log").exists()

    def test_multiple_writes_append_same_file(self, tmp_path):
        """TC-F2: 5 writes produce one file with 5 lines."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        for i in range(5):
            logger.log("INFO", "test", idx=str(i))
        today = datetime.now().strftime("%Y-%m-%d")
        lines = (log_dir / f"{today}.log").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_date_rollover_different_days(self, tmp_path):
        """TC-F3: logging on different dates creates separate files.

        Mocks datetime.now() to simulate writing on two different
        dates, then verifies two distinct daily log files exist.
        """
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)

        with patch(
            "src.backend.log.structured_logger.datetime"
        ) as mock_dt_class:
            # First call to now() returns yesterday, second returns tomorrow
            mock_dt_class.now.side_effect = [yesterday, tomorrow]
            logger.log("INFO", "first", day="yesterday")
            logger.log("INFO", "second", day="tomorrow")

        # Verify two separate files exist
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        yesterday_file = log_dir / f"{yesterday_str}.log"
        tomorrow_file = log_dir / f"{tomorrow_str}.log"
        assert yesterday_file.exists(), f"Expected {yesterday_file}"
        assert tomorrow_file.exists(), f"Expected {tomorrow_file}"
        # Each file should have exactly one line
        assert len(yesterday_file.read_text(encoding="utf-8").strip().split("\n")) == 1
        assert len(tomorrow_file.read_text(encoding="utf-8").strip().split("\n")) == 1

    def test_same_date_writes_to_same_file(self, tmp_path):
        """TC-F4: 3 writes on the same date → 1 file with 3 lines."""
        log_dir = tmp_path / "logs"
        logger = StructuredLogger(enabled=True, log_dir=str(log_dir))
        for i in range(3):
            logger.log("INFO", "test", idx=str(i))
        today = datetime.now().strftime("%Y-%m-%d")
        log_files = list(log_dir.glob("*.log"))
        # Only one log file (for today)
        today_files = [f for f in log_files if today in f.name]
        assert len(today_files) == 1
        lines = today_files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
