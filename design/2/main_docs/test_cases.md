# Test Cases — Human vs AI Mode

Tests organized by module. ✓ = verifiable assertion.

---

## 1. Unit: `ai/rnn_v1/model.py`

### TC-AI1 — Model instantiation
- **Given**: model is imported
- **When**: a `GRUCardModel` instance is created
- **Then**:
  - ✓ Model has `gru` layer (input=14, hidden=8)
  - ✓ Model has `fc_policy` layer (8→4)
  - ✓ Model has `fc_value` layer (8→1)
  - ✓ All parameters exist (non-empty state_dict)

### TC-AI2 — init_hidden_state returns zeros
- **Given**: a model instance
- **When**: `init_hidden_state()` is called
- **Then**:
  - ✓ Returns a `list[float]` of length 8
  - ✓ All values are `0.0`

### TC-AI3 — forward pass produces valid shapes
- **Given**: a model instance, a dummy input tensor of shape `(1, 14)`, hidden state of shape `(1, 8)`
- **When**: forward pass is run
- **Then**:
  - ✓ Policy output shape is `(1, 4)` — logits over 4 card types
  - ✓ Value output shape is `(1, 1)` — scalar in [-1, 1]

### TC-AI4 — select_action returns a legal card
- **Given**: a model instance, a game state dict where AI (player 2) has cards `[attack, defense, heavy]`
- **When**: `select_action(state, init_hidden_state())` is called
- **Then**:
  - ✓ Return type is `(str, list[float])`
  - ✓ Card string is one of `"attack"`, `"defense"`, `"heavy"` (never `"counter"`, which is absent)
  - ✓ Hidden state is a `list[float]` of length 8

### TC-AI5 — select_action with all 4 cards available
- **Given**: a model instance, round 1 state where AI has full 7-card hand (all 4 types present)
- **When**: `select_action()` is called
- **Then**:
  - ✓ Returned card is one of `"attack"`, `"defense"`, `"heavy"`, `"counter"`

### TC-AI6 — select_action with only 1 card left
- **Given**: a model instance, state where AI hand = `["defense"]` only
- **When**: `select_action()` is called
- **Then**:
  - ✓ Returns `"defense"` (only legal card)

### TC-AI7 — hidden state updates across calls
- **Given**: a model instance, round 1 state
- **When**: `select_action(state, h0)` → `(card1, h1)`, then `select_action(state, h1)` → `(card2, h2)`
- **Then**:
  - ✓ `h1` is a different list from `h0` (state changes)
  - ✓ `h2` is a different list from `h1`
  - ✓ Both `card1` and `card2` are legal

### TC-AI8 — input vector: round 1 (no prior moves)
- **Given**: round 1 game state
- **When**: input vector is built internally
- **Then**:
  - ✓ Indices 6–9 (my last move) are all `0.0`
  - ✓ Indices 10–13 (opp last move) are all `0.0`

### TC-AI9 — input vector: round 2 (with history)
- **Given**: round 2 game state where AI played "attack" last round, human played "defense"
- **When**: input vector is built
- **Then**:
  - ✓ Indices 6–9 = `[1.0, 0.0, 0.0, 0.0]` (attack one-hot)
  - ✓ Indices 10–13 = `[0.0, 1.0, 0.0, 0.0]` (defense one-hot)

### TC-AI10 — input vector: card counts reflect remaining hand
- **Given**: game state where AI hand = `[attack, attack, defense]` (2 atk, 1 def, 0 heavy, 0 counter), AI HP = 2, human HP = 1
- **When**: input vector is built
- **Then**:
  - ✓ Index 0 (atk count) = `2.0`
  - ✓ Index 1 (def count) = `1.0`
  - ✓ Index 2 (heavy count) = `0.0`
  - ✓ Index 3 (counter count) = `0.0`
  - ✓ Index 4 (my hp) = `2.0`
  - ✓ Index 5 (opp hp) = `1.0`

### TC-AI11 — deterministic output (same state → same card)
- **Given**: a model instance with fixed random seed (`torch.manual_seed(42)`), a specific game state, a specific hidden state
- **When**: `select_action()` is called twice with identical inputs
- **Then**:
  - ✓ Both calls return the same card string

### TC-AI14 — model operates in eval mode during inference
- **Given**: a model instance, a game state
- **When**: `select_action()` is called
- **Then**:
  - ✓ Model is in `eval()` mode (not `train()` mode)
  - ✓ Forward pass runs under `torch.inference_mode()` (no gradient computation, no autograd tracking)

### TC-AI15 — deterministic without per-call manual_seed
- **Given**: a single model instance with loaded weights, a specific game state, a specific hidden state
- **When**: `select_action()` is called twice with identical inputs (without calling `torch.manual_seed()` between calls)
- **Then**:
  - ✓ Both calls return the same card string (determinism is guaranteed by fixed weights + `model.eval()` + `torch.inference_mode()`, no per-call seed needed)

### TC-AI12 — weight file loading
- **Given**: `src/backend/ai/rnn_v1/gen1/rnn_v1.pth` exists
- **When**: model loads weights via `torch.load(path, map_location='cpu', weights_only=True)`
- **Then**:
  - ✓ No exception raised
  - ✓ `select_action()` produces a legal card (model is usable)

### TC-AI16 — corrupted weight file produces clear error
- **Given**: a truncated or malformed `.pth` file (e.g., empty file, or state_dict with mismatched keys)
- **When**: model attempts to load weights
- **Then**:
  - ✓ Exception is caught and re-raised as a clear error (e.g., `RuntimeError` with context "Failed to load model weights")
  - ✓ The error message includes the file path

### TC-AI13 — different weight files produce different outputs
- **Given**: two model instances with different random seeds
- **When**: `select_action()` called on both with same state
- **Then**:
  - ✓ Outputs MAY differ (verify models are independent; not strictly required but validates randomization)

---

## 2. Unit: `ai/registry.py`

### TC-REG1 — list_ais returns known AIs
- **Given**: registry module is imported
- **When**: `list_ais()` is called
- **Then**:
  - ✓ Returns a `list[dict]` with at least 1 entry
  - ✓ Each entry has keys: `name` and `description`
  - ✓ Entry `"rnn_v1.1"` exists with a non-empty description

### TC-REG2 — load_ai returns callable module
- **Given**: `list_ais()` contains `"rnn_v1.1"`
- **When**: `load_ai("rnn_v1.1")` is called
- **Then**:
  - ✓ Returns an object
  - ✓ Object has callable `init_hidden_state`
  - ✓ Object has callable `select_action`

### TC-REG3 — load_ai raises on unknown name
- **Given**: registry is loaded
- **When**: `load_ai("nonexistent_ai")` is called
- **Then**:
  - ✓ Raises `ValueError` with message containing the name

### TC-REG4 — list_ais is stable across calls
- **Given**: registry module
- **When**: `list_ais()` is called twice
- **Then**:
  - ✓ Both calls return identical lists (same entries, same order)

---

## 3. Unit: `game/game_manager.py` — Mode

### TC-GM1 — default mode is "pvp"
- **Given**: a new `GameManager` instance
- **When**: accessed `gm.mode`
- **Then**:
  - ✓ `gm.mode == "pvp"`

### TC-GM2 — mode persists in get_state
- **Given**: `GameManager` with `mode = "pve"`
- **When**: `gm.get_state()` is called
- **Then**:
  - ✓ Returned dict has `"mode": "pve"`

### TC-GM3 — play_round works in pve mode identically
- **Given**: `GameManager` with `mode = "pve"`, `start_game()` called
- **When**: `gm.play_round([CardType.ATTACK], [CardType.DEFENSE])`
- **Then**:
  - ✓ Returns valid `RoundResult` (mode does not affect resolution)

### TC-GM4 — mode does not affect game rules
- **Given**: two `GameManager` instances, one pvp one pve, both with identical hands
- **When**: same `play_round()` calls on both
- **Then**:
  - ✓ Both produce identical `RoundResult` (same damage, same HP after)

---

## 4. Unit: `game/session_store.py` — Updated Serialization

### TC-SS1 — serialize/deserialize preserves mode (pvp)
- **Given**: `GameManager` with `mode = "pvp"`, started game
- **When**: save → load round-trip
- **Then**:
  - ✓ Loaded `gm.mode == "pvp"`

### TC-SS2 — serialize/deserialize preserves mode (pve)
- **Given**: `GameManager` with `mode = "pve"`, started game
- **When**: save → load round-trip
- **Then**:
  - ✓ Loaded `gm.mode == "pve"`

### TC-SS3 — legacy file (no mode field) defaults to "pvp"
- **Given**: a save file without `"mode"` key (pre-intention-2 format)
- **When**: `SessionStore.load()` is called
- **Then**:
  - ✓ Loaded `gm.mode == "pvp"` (backward compatible)
  - ✓ No error raised

### TC-SS4 — save file includes ai and ai_state for PVE
- **Given**: PVE game with `gm.mode = "pve"`, `ai_state = [0.0, ...]`
- **When**: game is saved to JSON
- **Then**:
  - ✓ JSON file contains `"mode": "pve"`
  - ✓ JSON file contains `"ai": "rnn_v1.1"` (injected by controller)

### TC-SS5 — PVP save file has no ai_state
- **Given**: PVP game
- **When**: game is saved to JSON
- **Then**:
  - ✓ JSON file contains `"mode": "pvp"`
  - ✓ JSON file does NOT contain `"ai_state"` key (or value is null)

### TC-SS6 — PVE save file missing ai_state defaults to zeros
- **Given**: a save file with `mode: "pve"`, `ai: "rnn_v1.1"` but no `ai_state` key
- **When**: `SessionStore.load()` is called
- **Then**:
  - ✓ Load succeeds without error
  - ✓ `metadata["ai_state"]` is `[0.0]*8` (default fresh hidden state)
  - ✓ Warning is logged

### TC-SS7 — PVE save file with corrupted ai_state falls back to zeros
- **Given**: a save file with `mode: "pve"` and `ai_state: ["abc", 1, 2, 3, 4, 5, 6, 7]` (non-numeric)
- **When**: `SessionStore.load()` is called
- **Then**:
  - ✓ Load succeeds without error (game state is valid)
  - ✓ `metadata["ai_state"]` falls back to `[0.0]*8`
  - ✓ Warning is logged about invalid ai_state format

### TC-SS8 — PVE save file with wrong-length ai_state falls back to zeros
- **Given**: a save file with `mode: "pve"` and `ai_state: [0.1, 0.2]` (length 2, not 8)
- **When**: `SessionStore.load()` is called
- **Then**:
  - ✓ `metadata["ai_state"]` falls back to `[0.0]*8`

### TC-SS9 — PVE save file missing ai field raises error
- **Given**: a save file with `mode: "pve"` but no `ai` key
- **When**: `SessionStore.load()` is called
- **Then**:
  - ✓ Raises `ValueError` or metadata indicates corrupted config
  - ✓ Controller returns `500` with `code == "AI_CONFIG_ERROR"`

### TC-SS10 — PVP save file with stray ai_state silently ignored
- **Given**: a PVP save file containing `ai_state` and `ai` fields (e.g., from a bug)
- **When**: `SessionStore.load()` is called
- **Then**:
  - ✓ `gm.mode == "pvp"`
  - ✓ `metadata["ai"]` is `None`
  - ✓ `metadata["ai_state"]` is `None`

---

## 5. Integration: `controller/api.py` — AI Endpoints

### TC-API1 — GET /api/ais returns AI list
- **Given**: Flask test client, server running
- **When**: `GET /api/ais`
- **Then**:
  - ✓ HTTP 200
  - ✓ Response has `"ais"` key with array
  - ✓ Array contains at least `{"name": "rnn_v1.1", "description": "..."}`

### TC-API2 — POST /api/games with mode=pve creates AI game
- **Given**: Flask test client
- **When**: `POST /api/games` with body `{"mode": "pve", "ai": "rnn_v1.1"}`
- **Then**:
  - ✓ HTTP 201
  - ✓ Response `state.mode == "pve"`
  - ✓ Response `state.ai == "rnn_v1.1"`

### TC-API3 — POST /api/games with mode=pvp (backward compat)
- **Given**: Flask test client
- **When**: `POST /api/games` with body `{}` (empty, default)
- **Then**:
  - ✓ HTTP 201
  - ✓ Response `state.mode == "pvp"`
  - ✓ Response `state.ai` is `null`

### TC-API4 — POST /api/games with unknown AI (name not in registry)
- **Given**: Flask test client
- **When**: `POST /api/games` with body `{"mode": "pve", "ai": "nonexistent"}`
- **Then**:
  - ✓ HTTP 400
  - ✓ `code == "UNKNOWN_AI"`
  - ✓ No AI model is loaded (validation is via `list_ais()` name check only)

### TC-API5 — POST /api/games with pve mode but no ai field
- **Given**: Flask test client
- **When**: `POST /api/games` with body `{"mode": "pve"}`
- **Then**:
  - ✓ HTTP 400
  - ✓ Error message mentions missing AI

### TC-API6 — POST /play in PVE mode: cards_2 auto-computed
- **Given**: AI game created, round 1
- **When**: `POST /api/games/{id}/play` with body `{"cards_1": ["attack"]}`
- **Then**:
  - ✓ HTTP 200
  - ✓ `round_result.cards_2` is a non-empty list of 1 card
  - ✓ The card is a valid card type string
  - ✓ The card is present in Player 2's hand (AI hand before round = 7 cards, after round = 6)
  - ✓ Player 2's HP may be reduced (depending on resolution)

### TC-API7 — POST /play in PVP mode: cards_2 required
- **Given**: PVP game created
- **When**: `POST /api/games/{id}/play` with body `{"cards_1": ["attack"]}` (missing cards_2)
- **Then**:
  - ✓ HTTP 400 (cards_2 is required for PVP)

### TC-API8 — POST /play in PVE mode: ignores cards_2 if sent
- **Given**: AI game created
- **When**: `POST /api/games/{id}/play` with body `{"cards_1": ["attack"], "cards_2": ["defense"]}`
- **Then**:
  - ✓ HTTP 200
  - ✓ `round_result.cards_2` is AI's computed card, NOT "defense"

### TC-API13 — PVE POST /play with missing cards_1
- **Given**: AI game created
- **When**: `POST /api/games/{id}/play` with body `{}` (empty) or `{"cards_2": ["defense"]}` (missing cards_1)
- **Then**:
  - ✓ HTTP 400 with clear error message mentioning missing `cards_1`
  - ✓ Game state is unchanged (no round was played)

### TC-API14 — AI returns invalid card string
- **Given**: AI game where the AI module (mocked) returns `"unknown"` as the card string
- **When**: `POST /play` is called
- **Then**:
  - ✓ HTTP 500 with `code == "AI_INFERENCE_ERROR"`
  - ✓ Error message indicates invalid card output from AI

### TC-API15 — AI model load failure mid-game
- **Given**: AI game created successfully, then the `.pth` weight file is deleted or corrupted
- **When**: `POST /api/games/{id}/play` is called
- **Then**:
  - ✓ HTTP 500 with `code == "AI_LOAD_ERROR"`
  - ✓ Error message includes the AI name and file path context

### TC-API9 — AI game save file contains valid ai_state after round
- **Given**: AI game created
- **When**: one round played (POST /play)
- **Then**:
  - ✓ Save file on disk has `"ai_state"` key
  - ✓ `ai_state` is a `list` of length 8
  - ✓ All elements are `float` or `int` (valid tensor-compatible values)

### TC-API10 — AI hidden state persists across rounds
- **Given**: AI game created, 2 rounds played
- **When**: examining ai_state before round 2 and after round 2
- **Then**:
  - ✓ ai_state after round 2 is different from ai_state after round 1 (hidden state evolved)

### TC-API11 — GET /state includes mode and ai fields
- **Given**: AI game or PVP game created
- **When**: `GET /api/games/{id}`
- **Then**:
  - ✓ Response `state.mode` is present
  - ✓ Response `state.ai` is present (AI name or null)

### TC-API12 — DELETE cleans up AI game same as PVP
- **Given**: AI game created
- **When**: `DELETE /api/games/{id}`
- **Then**:
  - ✓ HTTP 204
  - ✓ Subsequent GET returns 404

---

## 6. Integration: Full AI Game Flow

### TC-FLOW1 — complete AI game: 7 rounds, AI selects legal cards each round
- **Given**: AI game created against `rnn_v1.1`
- **When**: human plays `"attack"` every round for 7 rounds
- **Then**:
  - ✓ Each POST /play returns 200
  - ✓ `round_result.cards_2` is a legal card in each round (present in AI hand before that round)
  - ✓ After 7 rounds: `state.is_over == true`
  - ✓ `state.winner` is `"1"` | `"2"` | `"draw"`

### TC-FLOW2 — AI never runs out of cards early (selects from remaining hand)
- **Given**: AI game, AI hand composition tracked round by round
- **When**: all 7 rounds are played
- **Then**:
  - ✓ In each round, AI selects a card it still has
  - ✓ After each round, AI hand size decreases by exactly 1
  - ✓ After 7 rounds, AI hand is empty

### TC-FLOW3 — human wins if AI HP reaches 0
- **Given**: AI game where AI takes heavy damage (e.g., consecutive Attacks)
- **When**: game ends with AI HP = 0
- **Then**:
  - ✓ `state.winner == "1"` (human wins)

### TC-FLOW4 — AI wins if human HP reaches 0
- **Given**: AI game where human HP reaches 0
- **When**: game ends
- **Then**:
  - ✓ `state.winner == "2"` (AI wins)

### TC-FLOW5 — draw if both HP reach 0 in same round
- **Given**: AI game where both players at 1 HP and both take 1 damage
- **When**: round resolves
- **Then**:
  - ✓ `state.is_over == true`
  - ✓ `state.winner == "draw"`

### TC-FLOW6 — new game reuses same AI with fresh hidden state
- **Given**: first AI game completed, then new AI game created with same AI name
- **When**: first round of new game
- **Then**:
  - ✓ New game starts with `round == 1`, HP = 3 for both
  - ✓ Save file `ai_state` is all zeros (fresh hidden state, not carried over)

---

## 7. Unit: `game/game_manager.py` — get_state includes mode

### TC-GM5 — get_state dict includes mode key
- **Given**: `GameManager` with any mode
- **When**: `get_state()` called
- **Then**:
  - ✓ Returned dict has `"mode"` key with correct value

---

## 8. Edge Cases

### TC-EDGE1 — AI has 1 card left, must pick it
- **Given**: AI game where AI hand = `["counter"]` only
- **When**: human plays `"attack"`
- **Then**:
  - ✓ AI selects `"counter"` (TC-AI6 already covers this at unit level; integration verifies via API)
  - ✓ Resolution uses Counter vs Attack → AI takes 0, human takes 1

### TC-EDGE2 — Round 7 with both HP > 0: higher HP wins
- **Given**: AI game reaches round 7, human HP = 3, AI HP = 2
- **When**: round 7 is played
- **Then**:
  - ✓ `state.is_over == true`
  - ✓ `state.winner == "1"` (human wins by HP)

### TC-EDGE3 — Legacy PVP save file loaded and played correctly
- **Given**: save file from intention-1 (no mode/ai fields)
- **When**: loaded via SessionStore, then POST /play with both cards
- **Then**:
  - ✓ Game loads successfully (`mode` defaults to "pvp")
  - ✓ Round plays normally

### TC-EDGE4 — PVE game: POST /play after game over
- **Given**: AI game that has ended
- **When**: another POST /play
- **Then**:
  - ✓ HTTP 400 with `code == "GAME_OVER"` (same as PVP)
