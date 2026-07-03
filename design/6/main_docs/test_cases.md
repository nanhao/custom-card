# Test Cases — Structured Logging for Self-Play

Test cases use text and symbols only. No code. Tests are organized by target module.

---

## 1. `StructuredLogger` — Construction and Properties

### TC-L1: disabled logger writes nothing
- **Given**: `StructuredLogger(enabled=False, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", key="value")`
- **Then**: no file is created under `<temp>`; no exception is raised

### TC-L2: enabled logger writes a file
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", key="value")`
- **Then**: a file `<temp>/{yyyy}-{mm}-{dd}.log` exists with one line

### TC-L3: trace_id is an 8-char hex string
- **Given**: any `StructuredLogger` instance
- **When**: access `logger.trace_id`
- **Then**: returns a string of exactly 8 hexadecimal characters

### TC-L4: trace_id differs between instances
- **Given**: two `StructuredLogger` instances A and B
- **When**: compare `A.trace_id` and `B.trace_id`
- **Then**: the two traceIds are different

### TC-L5: disabled logger still has valid trace_id
- **Given**: `StructuredLogger(enabled=False, log_dir=<temp>)`
- **When**: access `logger.trace_id`
- **Then**: returns a non-empty 8-char hex string

---

## 2. `StructuredLogger` — Log Envelope Format

### TC-L6: log line has correct pipe-delimited envelope
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "game_start", game_idx="1", game_count="100")`
- **Then**: the written line matches pattern:
  - 7 fields separated by ` | ` (space-pipe-space)
  - Field 1: ISO 8601 timestamp with `.SSS` milliseconds
  - Field 2: 8-char hex traceId matching `logger.trace_id`
  - Field 3: integer > 0 (process ID)
  - Field 4: integer > 0 (thread ID)
  - Field 5: `INFO`
  - Field 6: `selfplay.game_start`
  - Field 7: message field (tested in §3)

### TC-L7: traceId is consistent across all lines from one logger
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log(...)` 10 times
- **Then**: all 10 lines contain the same traceId in field 2

### TC-L8: pid and tid are real OS values
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log(...)` once
- **Then**: field 3 equals current process ID; field 4 is a positive integer

### TC-L9: timestamp increases monotonically
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log(...)` twice with a small delay between
- **Then**: the second line's timestamp ≥ the first line's timestamp

---

## 3. `StructuredLogger` — Message Format

### TC-M1: simple key=value pairs are space-separated
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", a="1", b="2", c="3")`
- **Then**: message field contains all three pairs separated by single spaces; values are `1`, `2`, `3`

### TC-M2: None renders as literal "None"
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", winner=None, rounds="7")`
- **Then**: message field contains `winner=None rounds=7`

### TC-M3: True/False render as literal "True"/"False"
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", enabled=True, success=False)`
- **Then**: message field contains `enabled=True success=False`

### TC-M4: integers render as decimal strings
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", game_idx=5, count=0, negative=-3)`
- **Then**: message field contains `game_idx=5 count=0 negative=-3`

### TC-M5: floats render with reasonable precision
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", avg_rounds=5.3, temp=1.0)`
- **Then**: message field contains `avg_rounds=5.3 temp=1.0`

### TC-M6: value with spaces is double-quoted
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("ERROR", "error", msg="play_round failed")`
- **Then**: message field contains `msg="play_round failed"` (double-quoted)

### TC-M7: value with double quotes escapes them
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", path="a \"quoted\" value")`
- **Then**: message field contains `path="a \"quoted\" value"` (escaped double quotes inside outer quotes)

### TC-M8: empty string renders as empty quotes
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", name="")`
- **Then**: message field contains `name=""`

### TC-M9: simple string without spaces is unquoted
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log("INFO", "test", card1="attack", card2="defense")`
- **Then**: message field contains `card1=attack card2=defense` (unquoted)

---

## 4. `StructuredLogger` — File Management

### TC-F1: log_dir is created automatically if absent
- **Given**: `log_dir=<temp>/non_existent_subdir`
- **When**: call `logger.log(...)` once
- **Then**: the directory is created; the log file exists inside it

### TC-F2: multiple writes append to the same file
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `logger.log(...)` 5 times
- **Then**: the log file contains exactly 5 lines

### TC-F3: date rollover for different days
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `log()` with date T, then call `log()` with date T+1 (simulated)
- **Then**: two separate files exist: `{date_T}.log` and `{date_T+1}.log`

### TC-F4: same date writes to same file
- **Given**: `StructuredLogger(enabled=True, log_dir=<temp>)`
- **When**: call `log()` three times, each with the same date
- **Then**: exactly one log file exists with 3 lines

---

## 5. `LoggerBuilder` — Singleton

### TC-B1: init creates accessible singleton
- **Given**: no prior init
- **When**: call `LoggerBuilder.init(enabled=True, log_dir=<temp>)`, then `LoggerBuilder.get_logger()`
- **Then**: returns a `StructuredLogger` instance; `trace_id` is valid

### TC-B2: get_logger returns same instance every time
- **Given**: `LoggerBuilder.init(enabled=True, log_dir=<temp>)`
- **When**: call `LoggerBuilder.get_logger()` twice
- **Then**: both calls return the exact same object

### TC-B3: get_logger before init raises RuntimeError
- **Given**: no prior init
- **When**: call `LoggerBuilder.get_logger()`
- **Then**: `RuntimeError` raised

### TC-B4: init twice raises RuntimeError
- **Given**: `LoggerBuilder.init(enabled=True, log_dir=<temp>)` already called
- **When**: call `LoggerBuilder.init(enabled=False, log_dir=<temp>)` again
- **Then**: `RuntimeError` raised

### TC-B5: disabled singleton's log() is a no-op
- **Given**: `LoggerBuilder.init(enabled=False, log_dir=<temp>)`
- **When**: `LoggerBuilder.get_logger().log("INFO", "test", key="value")`
- **Then**: no file is created; no exception raised

---

## 6. `scripts/selfplay.py` — Config Loading

### TC-C1: config with logging disabled produces no log file
- **Given**: `config/backend.json` has `"logging": {"enabled": false, "log_dir": "<temp>"}`
- **When**: run `selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 --game 2`
- **Then**: no log file is created

### TC-C2: config with logging enabled produces log file
- **Given**: `config/backend.json` has `"logging": {"enabled": true, "log_dir": "<temp>"}`
- **When**: run `selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 --game 2`
- **Then**: log file exists at `<temp>/{today}.log`

### TC-C3: config file missing defaults to disabled
- **Given**: `config/backend.json` does not exist
- **When**: run `selfplay.py` with any valid args
- **Then**: no log file is created; script runs normally

### TC-C4: malformed config JSON defaults to disabled
- **Given**: `config/backend.json` contains invalid JSON
- **When**: run `selfplay.py`
- **Then**: no log file is created; script runs normally

### TC-C5: missing log_dir key defaults to "logs"
- **Given**: `config/backend.json` has `"logging": {"enabled": true}` with no `log_dir`
- **When**: run `selfplay.py --game 1`
- **Then**: log file is written to `logs/{today}.log` (project root)

### TC-C6: missing logging section defaults to disabled
- **Given**: `config/backend.json` has no `"logging"` key
- **When**: run `selfplay.py`
- **Then**: no log file is created

---

## 7. `SelfPlayRunner` — Logging Integration

### TC-R1: disabled singleton produces no log file
- **Given**: `LoggerBuilder.init(enabled=False, log_dir=<temp>)`, `SelfPlayConfig(game_count=3)`
- **When**: `runner.run(config)`
- **Then**: no log file is created; existing `print()` output is unchanged

### TC-R2: enabled singleton writes lifecycle events
- **Given**: `LoggerBuilder.init(enabled=True, log_dir=<temp>)`, `SelfPlayConfig(game_count=3)`
- **When**: `runner.run(config)`
- **Then**: log file contains at minimum: `selfplay.run_start`, 3× `selfplay.game_start`, 3× `selfplay.game_end`, `selfplay.run_end`

### TC-R3: all log lines share the same traceId
- **Given**: same as TC-R2
- **When**: `runner.run(config)` completes
- **Then**: every line in the log file has the same traceId

### TC-R4: round_action and round_result logged per round
- **Given**: same as TC-R2, `game_count=1`
- **When**: `runner.run(config)` completes
- **Then**: for each round in the game, one `selfplay.round_action` line and one `selfplay.round_result` line exist in order; round numbers are sequential from 1
- **Given**: same as TC-R4
- **When**: inspect a `selfplay.round_action` log line
- **Then**: message field contains `round`, `card1`, `card2`, `temp1`, `temp2` key=value pairs; card values are valid card strings; temp values are floats

### TC-R6: round_result message has correct fields
- **Given**: same as TC-R4
- **When**: inspect a `selfplay.round_result` log line
- **Then**: message field contains `round`, `dmg1`, `dmg2`, `hp1`, `hp2`; damage values are non-negative integers; HP values are in [0, 3]

### TC-R7: error event logged when play_round raises ValueError
- **Given**: setup that triggers `play_round()` ValueError
- **When**: runner catches the exception
- **Then**: an `ERROR` level line with event `selfplay.error` is written; message contains `error_type=ValueError`, `msg` (double-quoted), `game_idx`, `round`

### TC-R8: error message value is double-quoted if it contains spaces
- **Given**: an error with message "play_round failed"
- **When**: the error log line is written
- **Then**: `msg` value is double-quoted: `msg="play_round failed"`

### TC-R9: run_start message has all CLI parameters
- **Given**: `SelfPlayConfig` with all fields set to non-default values
- **When**: inspect the `selfplay.run_start` log line
- **Then**: message field contains `ai1`, `ai2`, `game_count`, and optionally `temperature`, `temp_step`, `temp_side`, `temp_seed`, `sample_output`

### TC-R10: run_end message has aggregate statistics
- **Given**: `game_count=3`
- **When**: inspect the `selfplay.run_end` log line
- **Then**: message field contains `total_games=3`, `ai1_wins`, `ai2_wins`, `draws`, `avg_rounds`; `ai1_wins + ai2_wins + draws = 3`

### TC-R11: sample_write logged when sample_output is configured
- **Given**: `SelfPlayConfig(game_count=1, sample_output=<temp_file>)`
- **When**: `runner.run(config)` completes
- **Then**: one `selfplay.sample_write` log line exists; message contains `game_idx` and `count`

### TC-R12: no sample_write when sample_output is None
- **Given**: `SelfPlayConfig(game_count=1, sample_output=None)`
- **When**: `runner.run(config)` completes
- **Then**: no log line with event `selfplay.sample_write` exists

### TC-R13: model_loaded logged when AI model is first loaded
- **Given**: `SelfPlayConfig(game_count=1)`
- **When**: runner loads AI model for the first time
- **Then**: a log line with event `selfplay.model_loaded` exists; message contains `ai_name`

### TC-R14: ai_validated logged after successful registry validation
- **Given**: `SelfPlayConfig(game_count=1)`
- **When**: runner validates AI names
- **Then**: a log line with event `selfplay.ai_validated` exists; message contains `ai1` and `ai2`

---

## 8. End-to-End — CLI Integration

### TC-E1: full run with logging produces complete event sequence
- **Given**: `config/backend.json` has `"logging": {"enabled": true, "log_dir": "<temp>"}`
- **When**: run `selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 --game 2`
- **Then**: log file exists; events appear in order: `selfplay.run_start` → `selfplay.ai_validated` → `selfplay.model_loaded` → `selfplay.game_start` → `selfplay.round_action`/`selfplay.round_result` (repeating) → `selfplay.game_end` → `selfplay.game_start` → ... → `selfplay.run_end`

### TC-E2: concurrent runs have different traceIds
- **Given**: start two `selfplay.py` instances simultaneously with logging enabled (different log_dirs)
- **When**: both complete
- **Then**: each log file contains a different traceId; no collision

### TC-E3: console output is unchanged when logging is enabled
- **Given**: logging enabled
- **When**: run `selfplay.py --game 2`
- **Then**: stdout still shows per-game `Game N/M | ...` lines and the `=== Summary ===` block
