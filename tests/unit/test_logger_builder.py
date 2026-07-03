"""Unit tests for LoggerBuilder — singleton management.

Covers test cases from design/6 test_cases.md §5 (TC-B1 ~ TC-B5).

All file I/O uses temporary directories (test rule: never touch real data).
"""

import re

import pytest

from src.backend.log.structured_logger import LoggerBuilder, StructuredLogger


# ---------------------------------------------------------------------------
# Reset helper — LoggerBuilder is singleton, must reset between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_logger_builder():
    """Reset LoggerBuilder state before each test.

    LoggerBuilder is a singleton; without resetting, test order matters
    and tests fail unpredictably. This fixture uses _reset() to clear
    the internal state between tests.
    """
    LoggerBuilder._reset()
    yield
    LoggerBuilder._reset()


# ---------------------------------------------------------------------------
# §5 — LoggerBuilder Singleton (TC-B1 ~ TC-B5)
# ---------------------------------------------------------------------------


class TestLoggerBuilder:
    """TC-B1 ~ TC-B5: LoggerBuilder singleton management."""

    def test_init_creates_accessible_singleton(self, tmp_path):
        """TC-B1: init() then get_logger() returns valid StructuredLogger."""
        log_dir = str(tmp_path / "logs")
        LoggerBuilder.init(enabled=True, log_dir=log_dir)
        logger = LoggerBuilder.get_logger()
        assert isinstance(logger, StructuredLogger)
        assert re.fullmatch(r"[0-9a-f]{8}", logger.trace_id) is not None

    def test_get_logger_returns_same_instance(self, tmp_path):
        """TC-B2: two get_logger() calls return the exact same object."""
        log_dir = str(tmp_path / "logs")
        LoggerBuilder.init(enabled=True, log_dir=log_dir)
        a = LoggerBuilder.get_logger()
        b = LoggerBuilder.get_logger()
        assert a is b

    def test_get_logger_before_init_raises_runtime_error(self):
        """TC-B3: get_logger() before init() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not been initialized"):
            LoggerBuilder.get_logger()

    def test_init_twice_raises_runtime_error(self, tmp_path):
        """TC-B4: calling init() twice raises RuntimeError."""
        log_dir = str(tmp_path / "logs")
        LoggerBuilder.init(enabled=True, log_dir=log_dir)
        with pytest.raises(RuntimeError, match="already been initialized"):
            LoggerBuilder.init(enabled=False, log_dir=log_dir)

    def test_disabled_singleton_log_is_noop(self, tmp_path):
        """TC-B5: disabled singleton's log() creates no file."""
        log_dir = tmp_path / "logs"
        LoggerBuilder.init(enabled=False, log_dir=str(log_dir))
        logger = LoggerBuilder.get_logger()
        logger.log("INFO", "test", key="value")
        assert not log_dir.exists()
