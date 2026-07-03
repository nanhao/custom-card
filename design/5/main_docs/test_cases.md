# Test Cases — AI Self-Play Script

Test cases use text and symbols only. No code. Tests are organized by target module.

---

## 1. `GRUCardModel.forward()` — Temperature

### TC-M1: temperature=1.0 produces same probs as before
- **Given**: model loaded with gen1 weights, an input tensor x, hidden state h, no mask
- **When**: call `forward(x, h, temperature=1.0)`
- **Then**: `probs` equals the output of the current `forward(x, h)` (before modification), and `log_probs ≈ ln(probs)`

### TC-M2: temperature=0 produces one-hot probs (greedy)
- **Given**: model loaded with gen1 weights, input x, hidden h
- **When**: call `forward(x, h, temperature=0)`
- **Then**: `probs` is a one-hot vector with 1.0 at the argmax position and 0.0 elsewhere; `log_probs` is -inf except 0.0 at argmax position

### TC-M3: temperature > 1.0 flattens distribution
- **Given**: model, input x, hidden h
- **When**: call `forward(x, h, temperature=10.0)`
- **Then**: all four probs are close to 0.25 (uniform)

### TC-M4: temperature < 1.0 sharpens distribution
- **Given**: model, input x, hidden h
- **When**: call `forward(x, h, temperature=0.1)`
- **Then**: max(probs) is larger than with temperature=1.0; distribution is more peaked

### TC-M5: mask_fn still works with temperature
- **Given**: model, input x, hidden h, mask_fn that zeros index 0 and 1
- **When**: call `forward(x, h, mask_fn=mask_fn, temperature=1.0)`
- **Then**: probs at indices 0 and 1 are 0.0

### TC-M6: value head unchanged by temperature
- **Given**: model, input x, hidden h
- **When**: call `forward(x, h, temperature=0.0)`, then call `forward(x, h, temperature=3.0)`
- **Then**: `value` is identical in both calls (temperature only affects policy head)

### TC-M7: new_hidden unchanged by temperature
- **Given**: same as TC-M6
- **Then**: `new_hidden` is identical in both calls

---

## 2. `Converter.build_input_vector()` — Player Perspective

### TC-C1: player_key="2" (default) produces same output as before
- **Given**: a game state dict from GameCore.get_state() after round 1
- **When**: call `build_input_vector(state)` and `build_input_vector(state, player_key="2")`
- **Then**: both return identical vectors matching the current implementation

### TC-C2: player_key="1" swaps perspectives
- **Given**: game state where player 1 has hand [A,A,D], player 2 has hand [H,C]
- **When**: call `build_input_vector(state, player_key="1")`
- **Then**: indices 0–3 reflect player 1's hand counts; indices 4–5 are (P1 HP, P2 HP); indices 6–9 are P1's last move; indices 10–13 are P2's last move

### TC-C3: invalid player_key raises error
- **Given**: any game state
- **When**: call `build_input_vector(state, player_key="3")`
- **Then**: ValueError raised

---

## 3. `AppManager.ai_sample_action()` — Temperature Sampling

### TC-A1: temperature=0 returns greedy action, log_prob=None
- **Given**: a game state, hidden state, valid ai_name
- **When**: call `ai_sample_action(state, hidden, ai_name, temperature=0)`
- **Then**: returned card_str equals the card from `ai_select_action()`; log_prob is None

### TC-A2: temperature > 0 returns log_prob
- **Given**: game state, hidden state, ai_name, temperature=1.0
- **When**: call `ai_sample_action(state, hidden, ai_name, temperature=1.0)`
- **Then**: returned card_str is a valid card in the player's hand; log_prob is a finite float ≤ 0

### TC-A3: sampling respects masked (unavailable) cards
- **Given**: game state where player's hand has only [attack, defense] (no heavy, no counter)
- **When**: call `ai_sample_action()` repeatedly (≥100 times)
- **Then**: never returns "heavy" or "counter"

### TC-A4: player_key="1" queries from player 1's perspective
- **Given**: game state, hidden state, ai_name
- **When**: call `ai_sample_action(state, hidden, ai_name, 1.0, player_key="1")`
- **Then**: returned card is in player 1's hand (not player 2's)

### TC-A5: same seed + same temperature produces same sequence
- **Given**: fixed game states sequence, ai_name, temperature=1.0
- **When**: run `ai_sample_action()` through the sequence twice with same temp_seed
- **Then**: both runs produce identical action sequences and log_probs

### TC-A6: different seeds produce different sequences (statistically)
- **Given**: same as TC-A5 but different seeds
- **When**: run twice with different temp_seed
- **Then**: sequences differ at least once (for a sequence of ≥10 calls)

---

## 4. `AppManager.run_selfplay()` — Game Loop

### TC-S1: single game produces valid result
- **Given**: AppManager instance, config with ai1="rnn_v1.1", ai2="rnn_v1.1", game_count=1
- **When**: `app.run_selfplay(config)`
- **Then**: result.total_games=1, result.ai1_wins+result.ai2_wins+result.draws=1, 1≤result.avg_rounds≤7

### TC-S2: game_count=0 returns empty result
- **Given**: AppManager instance, config with game_count=0
- **When**: `app.run_selfplay(config)`
- **Then**: result.total_games=0, all counts are 0, avg_rounds=0

### TC-S3: two identical AIs produce roughly balanced outcomes
- **Given**: AppManager instance, config with ai1="rnn_v1.1", ai2="rnn_v1.1", game_count=100
- **When**: `app.run_selfplay(config)`
- **Then**: |ai1_wins − ai2_wins| ≤ 20 (statistically balanced for random-weight model)

### TC-S4: invalid ai_name raises ValueError
- **Given**: AppManager instance, config with ai1="nonexistent_ai"
- **When**: `app.run_selfplay(config)`
- **Then**: ValueError raised before any game starts

### TC-S5: game with deterministic AI (temperature=0 both sides) produces same outcome every run
- **Given**: AppManager instance, config with ai1="rnn_v1.1", ai2="rnn_v1.1", temperature=0, game_count=3
- **When**: `app.run_selfplay(config)` twice
- **Then**: both runs produce identical winner sequences (3 games, same outcome each)

### TC-S6: temp_seed with temperature>0 produces reproducible games
- **Given**: AppManager instance, config with temperature=1.0, temp_seed=42, game_count=5
- **When**: `app.run_selfplay(config)` twice
- **Then**: both runs produce identical winner sequences

---

## 5. `AppManager.run_selfplay()` — PPO Sample Output

### TC-P1: sample_output produces valid JSONL file
- **Given**: AppManager instance, config with sample_output pointing to a temp file, game_count=5
- **When**: `app.run_selfplay(config)` completes
- **Then**: file exists; each line is valid JSON; each line has keys `state`, `action`, `reward`, `value`, `log_prob`

### TC-P2: each game step produces one sample line
- **Given**: AppManager instance, config with game_count=1, sample_output set
- **When**: `app.run_selfplay(config)` completes
- **Then**: number of lines in file equals the number of rounds played in the game

### TC-P3: reward is game outcome for all steps
- **Given**: AppManager instance, config with game_count=1, sample_output set
- **When**: `app.run_selfplay(config)` completes, game ends with P1 winning
- **Then**: all sample lines for P1 have reward=1.0; all lines for P2 have reward=-1.0

### TC-P4: draw game assigns reward=0.0 to both sides
- **Given**: AppManager instance, a draw result
- **When**: samples are written
- **Then**: all reward values are 0.0

### TC-P5: sample_output not specified creates no file
- **Given**: AppManager instance, config with sample_output=None
- **When**: `app.run_selfplay(config)` completes
- **Then**: no sample file is created in the working directory

### TC-P6: log_prob is None for greedy steps
- **Given**: AppManager instance, config with temperature=0, sample_output set
- **When**: `app.run_selfplay(config)` completes
- **Then**: all sample lines have `log_prob: null`

### TC-P7: state vector length is 14
- **Given**: AppManager instance, config with sample_output set
- **When**: `app.run_selfplay(config)` completes
- **Then**: every sample line's `state` array has exactly 14 elements

---

## 6. `AppManager.run_selfplay()` — temp_step / temp_side

### TC-T1: temp_step=2 switches to greedy after round 2
- **Given**: AppManager instance, config with temperature=1.0, temp_step=2, temp_side="both", game_count=10, sample_output set
- **When**: `app.run_selfplay(config)` completes
- **Then**: rounds 1–2 have `log_prob` not null (sampled); rounds 3+ have `log_prob: null` (greedy)

### TC-T2: temp_step=None samples all rounds
- **Given**: AppManager instance, config with temperature=1.0, temp_step=None, game_count=10, sample_output set
- **When**: `app.run_selfplay(config)` completes
- **Then**: all rounds have `log_prob` not null

### TC-T3: temp_side="player1" only samples P1
- **Given**: AppManager instance, config with temperature=1.0, temp_side="player1", game_count=10, sample_output set
- **When**: `app.run_selfplay(config)` completes
- **Then**: P1 sample lines have `log_prob` not null; P2 sample lines have `log_prob: null`

### TC-T4: temp_side="player2" only samples P2
- **Given**: AppManager instance, config with temperature=1.0, temp_side="player2", sample_output set
- **When**: `app.run_selfplay(config)` completes
- **Then**: P2 samples have `log_prob` not null; P1 samples have `log_prob: null`

### TC-T5: temp_side invalid value raises error
- **Given**: config with temp_side="invalid"
- **When**: `SelfPlayConfig` instantiation
- **Then**: ValueError raised

---

## 7. `AppManager.run_selfplay()` — Progress Logging

### TC-L1: progress printed per game
- **Given**: AppManager instance, config with game_count=5
- **When**: `app.run_selfplay(config)`
- **Then**: 5 lines printed matching `Game N/5 | {ai1} vs {ai2} | Winner: {P1/P2/Draw} | Rounds: {N}`

### TC-L2: summary printed at end
- **Given**: AppManager instance, config with game_count=5, no sample_output
- **When**: `app.run_selfplay(config)` completes
- **Then**: summary block printed with total, win counts, percentages, avg rounds, no "Samples saved" line

### TC-L3: sample path shown in summary when specified
- **Given**: AppManager instance, config with sample_output="/tmp/samples.jsonl"
- **When**: `app.run_selfplay(config)` completes
- **Then**: summary includes `Samples saved to: /tmp/samples.jsonl`
