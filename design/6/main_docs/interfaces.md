# Interface Design — Structured Logging

Public interfaces only. Private methods omitted per design rule.

---

## 1. `config/backend.json` — Shared Backend Config (Logging Section)

```json
{
  "logging": {
    "enabled": false,
    "log_dir": "logs"
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `logging.enabled` | boolean | `false` | Master switch. When `false`, the singleton logger is a silent no-op. |
| `logging.log_dir` | string | `"logs"` | Directory for log output (relative to project root). Resolved to `{project_root}/logs/`. Created if absent. |

The `backend.json` file is a shared configuration for the entire backend. Future backend-wide settings will be added as top-level keys alongside `"logging"`.

---

## 2. `log/structured_logger.py` — StructuredLogger

```python
class StructuredLogger:
    """Write structured, machine-parseable log lines to a daily file.

    This is a generic, low-level logging facility. It knows nothing about
    game logic, AI models, or any project-specific domain. Callers choose
    their own event names and message fields.

    Log envelope (pipe-delimited, fixed field order):
        {timestamp} | {traceId} | {pid} | {tid} | {level} | {event} | {message}

    See §3 below for the formal message format specification.

    Fields:
        timestamp — ISO 8601 with milliseconds (e.g. 2026-07-03T10:30:45.123)
        traceId   — 8-char hex, unique per logger instance
        pid       — OS process ID (int)
        tid       — OS thread ID (int)
        level     — DEBUG | INFO | WARNING | ERROR (caller-chosen)
        event     — Event name in snake_case (caller-chosen)
        message   — Key=value pairs per the message format spec (§3)

    When enabled=False, all methods are no-ops.
    Log file: {log_dir}/{yyyy}-{mm}-{dd}.log, opened/closed per write.
    """

    def __init__(self, enabled: bool, log_dir: str) -> None:
        """Create a structured logger.

        Generates a unique 8-char hex traceId from uuid4.

        Args:
            enabled: If False, all log() calls are no-ops.
            log_dir: Directory path for log files (relative to project root,
                     or absolute). Created if absent at construction time.
                     Raises OSError if directory cannot be created
                     or is not writable (fail-fast).
        """

    @property
    def trace_id(self) -> str:
        """Return this logger instance's unique 8-char hex traceId."""

    def log(self, level: str, event: str, **kwargs) -> None:
        """Write a single formatted log line.

        Constructs the message field from kwargs according to the
        message format specification (§3). Timestamp, pid, and tid
        are captured at call time.

        If enabled=False, this is a no-op.

        Args:
            level: Log level (DEBUG | INFO | WARNING | ERROR).
            event: Event name identifier in snake_case.
            **kwargs: Key-value pairs for the message content.
                      Values are converted to str, then formatted
                      per the message spec.
        """
```

---

## 3. Message Format Specification

The `message` field (7th pipe-delimited field) carries structured event data as a sequence of `key=value` pairs.

### 3.1 Structure

```
key1=value1 key2=value2 key3="value with spaces"
```

Rules:
- Pairs are separated by a single space character (`0x20`).
- Each pair is exactly `key=value` (no spaces around `=`).
- Keys use snake_case: lowercase letters, digits, and underscores only (`[a-z][a-z0-9_]*`).
- Keys within a single message are unique (no duplicate keys).

### 3.2 Value Formatting

| Value Type | Format | Example |
|------------|--------|---------|
| `None` | Literal `None` | `winner=None` |
| `bool` | `True` or `False` | `enabled=True` |
| `int` | Decimal representation | `game_idx=5` |
| `float` | Decimal with at most 6 significant digits | `avg_rounds=5.3` |
| `str` (no spaces, no `"`, no `\`) | Unquoted literal | `card1=attack` |
| `str` (contains spaces, `"`, or `\`) | Double-quoted, `\"` for embedded quotes, `\\` for backslash | `msg="play_round failed"` |

Empty strings are written as `""`.

### 3.3 Parsing (for script consumers)

A parser must:
1. Split the message on spaces, respecting double-quoted regions.
2. For each segment, split on the first `=` to separate key from value.
3. Interpret value literals: `None` → null, `True`/`False` → boolean, digits → number, `"..."` → unquoted string, else → unquoted string.

### 3.4 Canonical Event Field Sets

For the self-play lifecycle, each event uses a `selfplay.` namespace prefix and these canonical message keys:

| Event | Level | Required Keys | Optional Keys |
|-------|-------|---------------|---------------|
| `selfplay.run_start` | INFO | ai1, ai2, game_count | temperature, temp_step, temp_side, temp_seed, sample_output |
| `selfplay.ai_validated` | INFO | ai1, ai2 | — |
| `selfplay.model_loaded` | INFO | ai_name | — |
| `selfplay.game_start` | INFO | game_idx, game_count | — |
| `selfplay.round_action` | DEBUG | round, card1, card2, temp1, temp2 | — |
| `selfplay.round_result` | DEBUG | round, dmg1, dmg2, hp1, hp2 | — |
| `selfplay.game_end` | INFO | game_idx, winner, rounds | — |
| `selfplay.sample_write` | DEBUG | game_idx, count | — |
| `selfplay.error` | ERROR | error_type, msg | game_idx, round |
| `selfplay.run_end` | INFO | total_games, ai1_wins, ai2_wins, draws, avg_rounds | — |

---

## 4. `log/structured_logger.py` — LoggerBuilder (Singleton)

```python
class LoggerBuilder:
    """Global singleton manager for the StructuredLogger instance.

    Usage:
        # At startup (once):
        LoggerBuilder.init(enabled=True, log_dir="logs")

        # Anywhere in the codebase:
        logger = LoggerBuilder.get_logger()
        logger.log("INFO", "event_name", key1="val1", key2="val2")
    """

    @staticmethod
    def init(enabled: bool, log_dir: str) -> None:
        """Initialize the global singleton StructuredLogger.

        Must be called exactly once before any get_logger() call.
        Creates the singleton StructuredLogger instance.

        Args:
            enabled: If False, the singleton logger is a no-op.
            log_dir: Directory for log output files.

        Raises:
            RuntimeError: If init() has already been called.
        """

    @staticmethod
    def get_logger() -> StructuredLogger:
        """Return the global singleton StructuredLogger instance.

        Returns the same instance every time. When init() was called
        with enabled=False, returns a no-op logger (safe to call
        log() without side effects).

        Returns:
            The singleton StructuredLogger.

        Raises:
            RuntimeError: If init() has not been called yet.
        """
```

---

## 5. `app/selfplay_runner.py` — SelfPlayRunner (Modified)

```python
class SelfPlayRunner:
    """(existing __init__ unchanged)"""

    def run(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games with structured logging.

        Retrieves the logger via LoggerBuilder.get_logger().
        When logging is disabled (init called with enabled=False),
        all log() calls are no-ops — no conditional checks needed.

        Events emitted (see architecture.md §6 for details):
            selfplay.run_start, selfplay.ai_validated, selfplay.model_loaded, selfplay.game_start,
            selfplay.round_action, selfplay.round_result, selfplay.game_end, selfplay.sample_write,
            selfplay.error, selfplay.run_end

        Args:
            config: SelfPlayConfig with all run parameters.

        Returns:
            SelfPlayResult with win/loss/draw counts and avg rounds.
        """
```

---

## 6. `app/app_manager.py` — AppManager (Unchanged)

No changes. `run_selfplay()` signature remains:

```python
class AppManager:
    def run_selfplay(self, config: SelfPlayConfig) -> SelfPlayResult:
        """(unchanged — delegates to SelfPlayRunner)"""
```

---

## 7. `scripts/selfplay.py` — CLI Entry Point (Modified)

```
"""AI Self-Play CLI — thin wrapper around AppManager.run_selfplay().

Per design/5 D1, this script contains NO business logic — only argument
parsing and delegation.

NEW (design/6): reads config/backend.json, calls LoggerBuilder.init().
"""

import argparse
import json
from pathlib import Path

from src.backend.app.app_manager import AppManager
from src.backend.app.selfplay_config import SelfPlayConfig
from src.backend.game.session_store import SessionStore
from src.backend.log.structured_logger import LoggerBuilder

# CLI argument parsing, then:
#   1. Read config/backend.json, extract logging section
#   2. LoggerBuilder.init(enabled=..., log_dir=...)
#   3. config = SelfPlayConfig(ai1=..., ai2=..., ...)
#   4. app = AppManager(SessionStore())
#   5. app.run_selfplay(config)
```

---

## 8. `log/__init__.py` — Package Init

Empty file. Marks `log/` as a Python package.

---

## 9. Log Output Example

```
2026-07-03T10:30:45.123 | a1b2c3d4 | 12345 | 67890 | INFO | selfplay.run_start | ai1=rnn_v1.1 ai2=rnn_v1.1 game_count=100 temperature=1.0
2026-07-03T10:30:45.125 | a1b2c3d4 | 12345 | 67890 | INFO | selfplay.ai_validated | ai1=rnn_v1.1 ai2=rnn_v1.1
2026-07-03T10:30:45.130 | a1b2c3d4 | 12345 | 67890 | INFO | selfplay.model_loaded | ai_name=rnn_v1.1
2026-07-03T10:30:45.131 | a1b2c3d4 | 12345 | 67890 | INFO | selfplay.game_start | game_idx=1 game_count=100
2026-07-03T10:30:45.135 | a1b2c3d4 | 12345 | 67890 | DEBUG | selfplay.round_action | round=1 card1=attack card2=defense temp1=1.0 temp2=0.0
2026-07-03T10:30:45.136 | a1b2c3d4 | 12345 | 67890 | DEBUG | selfplay.round_result | round=1 dmg1=0 dmg2=0 hp1=3 hp2=3
2026-07-03T10:30:45.138 | a1b2c3d4 | 12345 | 67890 | DEBUG | selfplay.round_action | round=2 card1=heavy card2=attack temp1=1.0 temp2=0.0
2026-07-03T10:30:45.139 | a1b2c3d4 | 12345 | 67890 | DEBUG | selfplay.round_result | round=2 dmg1=0 dmg2=1 hp1=3 hp2=2
2026-07-03T10:30:45.150 | a1b2c3d4 | 12345 | 67890 | ERROR | selfplay.error | error_type=ValueError msg="play_round failed" game_idx=1 round=3
2026-07-03T10:30:45.151 | a1b2c3d4 | 12345 | 67890 | INFO | selfplay.game_end | game_idx=1 winner=P2 rounds=6
2026-07-03T10:30:45.152 | a1b2c3d4 | 12345 | 67890 | INFO | selfplay.game_start | game_idx=2 game_count=100
...
2026-07-03T10:35:12.456 | a1b2c3d4 | 12345 | 67890 | INFO | selfplay.run_end | total_games=100 ai1_wins=48 ai2_wins=42 draws=10 avg_rounds=5.3
```

Note: `msg="play_round failed"` is double-quoted because it contains a space. `winner=P2` has no spaces and is unquoted.
