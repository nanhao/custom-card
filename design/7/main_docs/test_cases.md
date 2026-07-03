# Test Cases — Sample File Restructuring

Test cases use text and symbols only. No code. Tests are organized by target behavior.

---

## 1. Sample File Structure

### TC-F1: output file is valid JSON array
- **Given**: SelfPlayRunner, config with ai1="rnn_v1.1", ai2="rnn_v1.1", game_count=3, sample_output set
- **When**: `run(config)` completes
- **Then**: file exists; top-level structure is a JSON array with exactly 3 elements

### TC-F2: each game element has required top-level keys
- **Given**: same as TC-F1
- **When**: `run(config)` completes
- **Then**: each array element has keys `game_id`, `winner`, `rounds`, `perspectives`; no other top-level keys

### TC-F3: game_id values are sequential starting from 1
- **Given**: same as TC-F1
- **When**: `run(config)` completes
- **Then**: `game_id` values are [1, 2, 3]

### TC-F4: each game has exactly two perspectives
- **Given**: same as TC-F1
- **When**: `run(config)` completes
- **Then**: every game's `perspectives` array has length 2

### TC-F5: perspectives have correct player assignments
- **Given**: same as TC-F1
- **When**: `run(config)` completes
- **Then**: for each game, `perspectives[0].player == "1"` and `perspectives[1].player == "2"`

### TC-F6: each perspective contains ai name
- **Given**: self-play with ai1="rnn_v1.1", ai2="rnn_v1.1"
- **When**: `run(config)` completes
- **Then**: player "1" perspective has `ai: "rnn_v1.1"`; player "2" perspective has `ai: "rnn_v1.1"`

### TC-F7: each step has required keys
- **Given**: same as TC-F1
- **When**: `run(config)` completes
- **Then**: every step in every perspective has keys `state`, `action`, `reward`, `value`, `log_prob`; no other keys

### TC-F8: state vector length is 14 in all steps
- **Given**: same as TC-F1
- **When**: `run(config)` completes
- **Then**: every step's `state` array has exactly 14 elements

---

## 2. Reward Assignment

### TC-R1: winning player steps get reward=1.0
- **Given**: game_count=1, game ends with winner="1"
- **When**: `run(config)` completes
- **Then**: all steps in player "1" perspective have reward=1.0

### TC-R2: losing player steps get reward=-1.0
- **Given**: game_count=1, game ends with winner="1"
- **When**: `run(config)` completes
- **Then**: all steps in player "2" perspective have reward=-1.0

### TC-R3: draw assigns reward=0.0 to both sides
- **Given**: self-play with game_count=5, temperature=0 (deterministic), game ends in draw
- **When**: `run(config)` completes
- **Then**: all steps in both perspectives of the drawn game have reward=0.0

### TC-R4: reward is consistent within a perspective
- **Given**: any game
- **When**: `run(config)` completes
- **Then**: all steps within a single perspective have the same reward value

---

## 3. Steps Content

### TC-S1: total step count matches round count × 2
- **Given**: game_count=1, temperature=0
- **When**: `run(config)` completes; game played R rounds
- **Then**: total steps across both perspectives = R × 2

### TC-S2: player 1 actions are valid for player 1
- **Given**: any game
- **When**: `run(config)` completes
- **Then**: player "1" perspective steps use actions from {"attack", "defense", "heavy", "counter"} (valid card strings)

### TC-S3: player 2 actions are valid for player 2
- **Given**: any game
- **When**: `run(config)` completes
- **Then**: player "2" perspective steps use actions from {"attack", "defense", "heavy", "counter"}

### TC-S4: log_prob is null for greedy steps
- **Given**: game_count=1, temperature=0 (all greedy)
- **When**: `run(config)` completes
- **Then**: every step has `log_prob: null`

### TC-S5: log_prob is float for sampled steps
- **Given**: game_count=1, temperature=1.0, temp_step=None, temp_side="both"
- **When**: `run(config)` completes
- **Then**: every step has `log_prob` as a finite float ≤ 0

### TC-S6: value is in [-1.0, 1.0]
- **Given**: any game
- **When**: `run(config)` completes
- **Then**: every step's `value` satisfies -1.0 ≤ value ≤ 1.0

### TC-S7: round count is accurate
- **Given**: game_count=1, temperature=0 (deterministic)
- **When**: `run(config)` completes; game played R rounds
- **Then**: game element's `rounds` field equals R

---

## 4. File Output

### TC-O1: sample_output not specified creates no file
- **Given**: config with sample_output=None, game_count=1
- **When**: `run(config)` completes
- **Then**: no sample file is created

### TC-O2: sample_output creates parent directories
- **Given**: config with sample_output pointing to a path where parent directories do not exist, game_count=1
- **When**: `run(config)` completes
- **Then**: parent directories are created; sample file exists at the specified path

### TC-O3: file is valid JSON (no trailing commas, no JSONL line separators)
- **Given**: same as TC-F1
- **When**: `run(config)` completes
- **Then**: file content can be parsed as a single JSON value (array) with no syntax errors

### TC-O4: empty file when game_count=0
- **Given**: config with game_count=0, sample_output set
- **When**: `run(config)` completes
- **Then**: file contains `[]`

---

## 5. Sub-Unit Training Boundary

### TC-T1: steps within a player perspective are temporally ordered
- **Given**: game_count=1
- **When**: `run(config)` completes
- **Then**: within a perspective, step[i] corresponds to round i+1 (sequential from round 1 to last round)

### TC-T2: hidden state boundary exists between perspectives
- **Given**: game_count=2, both games played
- **When**: analyzing the sample file structure
- **Then**: perspectives are separate arrays; training code can independently iterate each perspective as a sub-unit

### TC-T3: different games have independent perspectives
- **Given**: game_count=2
- **When**: `run(config)` completes
- **Then**: game 1's perspectives and game 2's perspectives are separate arrays; training code can reset h between games if desired
