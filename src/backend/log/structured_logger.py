"""Generic structured logging facility — reusable across the entire backend.

Provides two classes:
    StructuredLogger — writes machine-parseable log lines to a daily file.
    LoggerBuilder    — global singleton manager for the StructuredLogger.

Design: design/6 §2, §4 (interfaces.md).
"""

import os
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4


# ---------------------------------------------------------------------------
# StructuredLogger — generic, low-level logging
# ---------------------------------------------------------------------------


class StructuredLogger:
    """Write structured, machine-parseable log lines to a daily file.

    This is a generic, low-level logging facility. It knows nothing about
    game logic, AI models, or any project-specific domain. Callers choose
    their own event names and message fields.

    Log envelope (pipe-delimited, fixed field order):
        {timestamp} | {traceId} | {pid} | {tid} | {level} | {event} | {message}

    Fields:
        timestamp — ISO 8601 with milliseconds (e.g. 2026-07-03T10:30:45.123)
        traceId   — 8-char hex, unique per logger instance
        pid       — OS process ID (int)
        tid       — OS thread ID (int)
        level     — DEBUG | INFO | WARNING | ERROR (caller-chosen)
        event     — Event name in snake_case (caller-chosen)
        message   — Key=value pairs per the message format spec (§3)

    When enabled=False, all log() calls are no-ops.
    Log file: {log_dir}/{yyyy}-{mm}-{dd}.log, opened/closed per write.
    """

    def __init__(self, enabled: bool, log_dir: str) -> None:
        """Create a structured logger.

        Generates a unique 8-char hex traceId from uuid4.
        When enabled, validates that log_dir is writable at construction
        time (fail-fast). Directory is created if absent.

        Args:
            enabled: If False, all log() calls are no-ops.
            log_dir: Directory path for log files (relative to project
                     root, or absolute). Created if absent.

        Raises:
            OSError: If enabled=True and log_dir cannot be created
                    or is not writable.
        """
        self._enabled = enabled
        self._log_dir = Path(log_dir)
        self._trace_id = uuid4().hex[:8]
        # Cache the process ID — it never changes during the process lifetime
        self._pid = str(os.getpid())

        # Fail-fast: validate directory is writable at construction time
        if self._enabled:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            # Verify write access by creating and removing a sentinel file
            sentinel = self._log_dir / ".write_test"
            try:
                sentinel.touch()
                sentinel.unlink()
            except OSError:
                raise OSError(
                    f"Log directory is not writable: {self._log_dir}"
                )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def trace_id(self) -> str:
        """Return this logger instance's unique 8-char hex traceId."""
        return self._trace_id

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log(self, level: str, event: str, **kwargs) -> None:
        """Write a single formatted log line.

        Constructs the message field from kwargs according to the
        message format specification. Timestamp, pid, and tid
        are captured at call time.

        If enabled=False, this is a no-op.

        Args:
            level: Log level (DEBUG | INFO | WARNING | ERROR).
            event: Event name identifier in snake_case.
            **kwargs: Key-value pairs for the message content.
                      Values are converted to str, then formatted
                      per the message spec.
        """
        if not self._enabled:
            return

        # Capture timestamp and thread ID at call time
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        tid = str(threading.get_ident())

        # Build the message field from kwargs
        message = self._format_message(**kwargs)

        # Assemble the full log line
        line = f"{timestamp} | {self._trace_id} | {self._pid} | {tid} | {level} | {event} | {message}"

        # Write to the daily file (create dir if needed, open/close per write)
        daily_path = self._get_daily_path(now)
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        with daily_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ------------------------------------------------------------------
    # Internal: daily file path resolution
    # ------------------------------------------------------------------

    def _get_daily_path(self, dt: datetime) -> Path:
        """Return the log file path for the given date.

        File name format: {yyyy}-{mm}-{dd}.log

        Args:
            dt: The datetime whose date determines the file name.

        Returns:
            Full Path to the daily log file.
        """
        date_str = dt.strftime("%Y-%m-%d")
        return self._log_dir / f"{date_str}.log"

    # ------------------------------------------------------------------
    # Internal: message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_value(value) -> str:
        """Format a single value for the message field.

        Per the message format specification (interfaces.md §3.2):

        | Value Type | Format | Example |
        |------------|--------|---------|
        | None       | Literal `None` | `winner=None` |
        | bool       | `True` or `False` | `enabled=True` |
        | int        | Decimal representation | `game_idx=5` |
        | float      | Decimal with at most 6 significant digits | `avg_rounds=5.3` |
        | str (no spaces, no `"`, no `\\`) | Unquoted literal | `card1=attack` |
        | str (contains spaces, `"`, or `\\`) | Double-quoted, escaped | `msg="play_round failed"` |
        | str (empty) | `""` | `name=""` |

        Args:
            value: The Python value to format.

        Returns:
            String representation ready for embedding in a log message.
        """
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            # At most 6 significant digits, but always include
            # at least one decimal place (e.g. 1.0 not 1)
            result = f"{value:.6g}"
            # If the formatted result has no decimal point and
            # no exponent, add .0 to distinguish from int
            if "." not in result and "e" not in result.lower():
                result = result + ".0"
            return result
        # String: check if quoting is needed
        s = str(value)
        if s == "":
            return '""'
        # Needs quoting if it contains spaces, double quotes, or backslashes
        needs_quote = any(c in s for c in (' ', '"', '\\'))
        if needs_quote:
            escaped = s.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return s

    @classmethod
    def _format_message(cls, **kwargs) -> str:
        """Format keyword arguments into a space-separated key=value message.

        Args:
            **kwargs: Key-value pairs (keys are str, values are any).

        Returns:
            Space-separated key=value string.
        """
        if not kwargs:
            return ""
        parts = []
        for key, value in kwargs.items():
            formatted_value = cls._format_value(value)
            parts.append(f"{key}={formatted_value}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# LoggerBuilder — global singleton manager
# ---------------------------------------------------------------------------


class LoggerBuilder:
    """Global singleton manager for the StructuredLogger instance.

    Thread-safe: init(), get_logger(), and _reset() are guarded by
    a class-level lock to prevent races during concurrent access.

    Usage:
        # At startup (once):
        LoggerBuilder.init(enabled=True, log_dir="logs")

        # Anywhere in the codebase:
        logger = LoggerBuilder.get_logger()
        logger.log("INFO", "event_name", key1="val1", key2="val2")
    """

    _instance: StructuredLogger | None = None
    _initialized: bool = False
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public static interface
    # ------------------------------------------------------------------

    @staticmethod
    def init(enabled: bool, log_dir: str) -> None:
        """Initialize the global singleton StructuredLogger.

        Must be called exactly once before any get_logger() call.
        Creates the singleton StructuredLogger instance.
        Thread-safe via class-level lock.

        Args:
            enabled: If False, the singleton logger is a no-op.
            log_dir: Directory for log output files.

        Raises:
            RuntimeError: If init() has already been called.
            OSError: If enabled=True and log_dir is not writable.
        """
        with LoggerBuilder._lock:
            if LoggerBuilder._initialized:
                raise RuntimeError(
                    "LoggerBuilder.init() has already been initialized. "
                    "The logger singleton can only be initialized once."
                )
            LoggerBuilder._instance = StructuredLogger(
                enabled=enabled, log_dir=log_dir
            )
            LoggerBuilder._initialized = True

    @staticmethod
    def get_logger() -> StructuredLogger:
        """Return the global singleton StructuredLogger instance.

        Returns the same instance every time. When init() was called
        with enabled=False, returns a no-op logger (safe to call
        log() without side effects).
        Thread-safe via class-level lock.

        Returns:
            The singleton StructuredLogger.

        Raises:
            RuntimeError: If init() has not been called yet.
        """
        with LoggerBuilder._lock:
            if not LoggerBuilder._initialized:
                raise RuntimeError(
                    "LoggerBuilder.init() has not been initialized. "
                    "Call LoggerBuilder.init(enabled, log_dir) first "
                    "before accessing the logger."
                )
            return LoggerBuilder._instance

    @staticmethod
    def _reset() -> None:
        """Reset the singleton state (for testing only).

        Clears the singleton instance and initialization flag so
        tests can start from a clean state. This method should
        never be called in production code.
        Thread-safe via class-level lock.
        """
        with LoggerBuilder._lock:
            LoggerBuilder._instance = None
            LoggerBuilder._initialized = False
