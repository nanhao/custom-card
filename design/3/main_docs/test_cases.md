# Test Cases — App Bridge Layer Refactoring

Tests organized by module. ✓ = verifiable assertion. Tests marked [EXISTING] are carried over from design/2 with minimal updates.

---

## 1. Unit: `ai/rnn_v1/model.py` — GRUCardModel (Updated)

### TC-M1 — Model instantiation [EXISTING, updated from TC-AI1]
- **Given**: model module is imported
- **When**: a `GRUCardModel` instance is created
- **Then**:
  - ✓ Model has `gru` layer (input=14, hidden=8)
  - ✓ Model has `fc_policy` layer (8→4)
  - ✓ Model has `fc_value` layer (8→1)
  - ✓ All parameters exist (non-empty state_dict)

### TC-M2 — INITIAL_HIDDEN_STATE constant
- **Given**: model class is imported
- **When**: `GRUCardModel.INITIAL_HIDDEN_STATE` is accessed
- **Then**:
  - ✓ Returns `[0.0] * 8`
  - ✓ All elements are `float`

### TC-M3 — forward pass shapes (no mask_fn)
- **Given**: a model instance, dummy input `(1,1,14)`, hidden `(1,1,8)`
- **When**: `model.forward(x, hidden)` is called
- **Then**:
  - ✓ Returns `(probs, value, new_hidden)` tuple
  - ✓ `probs` shape is `(1, 4)`, values sum to ~1.0 (softmax)
  - ✓ `value` shape is `(1, 1)`, value in [-1, 1]
  - ✓ `new_hidden` shape is `(1, 1, 8)`

### TC-M4 — forward pass with mask_fn
- **Given**: a model instance, dummy input, hidden state, and a mask_fn that sets index 0 to -inf
- **When**: `model.forward(x, hidden, mask_fn=mask)` is called
- **Then**:
  - ✓ `probs[0, 0]` ≈ 0.0 (masked out)
  - ✓ `probs` at non-masked indices sum to ~1.0
  - ✓ `new_hidden` is different from input hidden

### TC-M5 — forward pass with mask_fn that sets all but one to -inf
- **Given**: a model instance, dummy input, hidden, mask_fn that only allows index 2
- **When**: `model.forward(x, hidden, mask_fn=mask)` is called
- **Then**:
  - ✓ `probs[0, 2]` ≈ 1.0 (only legal card)
  - ✓ All other probs ≈ 0.0

### TC-M6 — no mask_fn → no masking applied
- **Given**: a model instance, dummy input, hidden
- **When**: `model.forward(x, hidden)` is called (no mask_fn)
- **Then**:
  - ✓ All four probs > 0 (no -inf masking)

### TC-M7 — forward pass runs under inference_mode
- **Given**: a model instance, dummy input, hidden
- **When**: `model.forward()` is called
- **Then**:
  - ✓ No gradients tracked (verify via `torch.is_inference_mode_enabled()` or equivalent)

### TC-M8 — deterministic output with same inputs
- **Given**: a model instance in eval mode, fixed input and hidden tensors
- **When**: `model.forward(x, hidden)` is called twice
- **Then**:
  - ✓ Both calls return identical probs, value, new_hidden

### TC-M9 — weight file loading [EXISTING, updated from TC-AI12]
- **Given**: `src/backend/ai/rnn_v1/gen1/rnn_v1.pth` exists
- **When**: model is instantiated and weights loaded via `torch.load(path, map_location='cpu', weights_only=True)`
- **Then**:
  - ✓ No exception raised
  - ✓ Forward pass produces valid output

### TC-M10 — corrupted weight file [EXISTING, updated from TC-AI16]
- **Given**: a nonexistent weight file path
- **When**: model attempts to load weights
- **Then**:
  - ✓ Raises `RuntimeError` with clear message including the file path

### TC-M11 — different models produce different outputs [EXISTING, from TC-AI13]
- **Given**: two model instances with different random seeds
- **When**: forward pass on both with same input
- **Then**:
  - ✓ Outputs differ (different parameter initializations)

---

## 2. Unit: `ai/registry.py` — Registry (Updated)

### TC-REG1 — list_ais returns known AIs [EXISTING]
- **Given**: registry module
- **When**: `list_ais()` is called
- **Then**:
  - ✓ Returns `list[dict]` with at least 1 entry
  - ✓ Each entry has `name` and `description` keys
  - ✓ Entry `"rnn_v1.1"` exists

### TC-REG2 — list_ais entries don't expose internal data [EXISTING]
- **Given**: registry module
- **When**: `list_ais()` is called
- **Then**:
  - ✓ No entry contains `weight_path`, `module`, or `class` keys

### TC-REG3 — get_ai_info returns full entry
- **Given**: registry module
- **When**: `get_ai_info("rnn_v1.1")` is called
- **Then**:
  - ✓ Returns dict with keys: `name`, `description`, `weight_path`, `module`, `class`
  - ✓ `module` is a valid import path string
  - ✓ `class` is `"GRUCardModel"`
  - ✓ `weight_path` is a valid path string

### TC-REG4 — get_ai_info raises on unknown name
- **Given**: registry module
- **When**: `get_ai_info("nonexistent")` is called
- **Then**:
  - ✓ Raises `ValueError` with message containing the name
  - ✓ Message lists available AI names

### TC-REG5 — list_ais is stable [EXISTING, from TC-REG4]
- **Given**: registry module
- **When**: `list_ais()` is called twice
- **Then**:
  - ✓ Both calls return identical lists (same entries, same order)

### TC-REG6 — no module import at registry load time
- **Given**: registry module
- **When**: the `ai.registry` module is imported
- **Then**:
  - ✓ No PyTorch import (verify `torch` not in `sys.modules` from registry alone, or verify no model modules imported)

---

## 3. Unit: `app/converter.py` — Converter

### TC-CV1 — card_to_index valid cards
- **Given**: Converter class
- **When**: `Converter.card_to_index("attack")`, same for defense/heavy/counter
- **Then**:
  - ✓ attack→0, defense→1, heavy→2, counter→3

### TC-CV2 — card_to_index invalid card
- **Given**: Converter class
- **When**: `Converter.card_to_index("unknown")` is called
- **Then**:
  - ✓ Returns -1

### TC-CV3 — index_to_card valid indices
- **Given**: Converter class
- **When**: `Converter.index_to_card(0)`, same for 1/2/3
- **Then**:
  - ✓ 0→attack, 1→defense, 2→heavy, 3→counter

### TC-CV4 — round-trip: card → index → card
- **Given**: any valid card string
- **When**: `Converter.index_to_card(Converter.card_to_index(card_str))`
- **Then**:
  - ✓ Returns the original card string

### TC-CV5 — build_input_vector: round 1 (no history) [EXISTING, from TC-AI8]
- **Given**: game state with round=1, empty histories
- **When**: `Converter.build_input_vector(state)` is called
- **Then**:
  - ✓ Indices 6–9 (AI last move) are all 0.0
  - ✓ Indices 10–13 (human last move) are all 0.0

### TC-CV6 — build_input_vector: round 2 (with history) [EXISTING, from TC-AI9]
- **Given**: game state where AI played "attack", human played "defense" last round
- **When**: `Converter.build_input_vector(state)` is called
- **Then**:
  - ✓ Indices 6–9 = `[1.0, 0.0, 0.0, 0.0]`
  - ✓ Indices 10–13 = `[0.0, 1.0, 0.0, 0.0]`

### TC-CV7 — build_input_vector: card counts [EXISTING, from TC-AI10]
- **Given**: AI hand = `["attack", "attack", "defense"]`, HP_AI=2, HP_human=1
- **When**: `Converter.build_input_vector(state)` is called
- **Then**:
  - ✓ Index 0 (attack count) = `2.0`
  - ✓ Index 1 (defense count) = `1.0`
  - ✓ Index 2 (heavy count) = `0.0`
  - ✓ Index 3 (counter count) = `0.0`
  - ✓ Index 4 (AI HP) = `2.0`
  - ✓ Index 5 (human HP) = `1.0`

### TC-CV8 — build_input_vector: HP clamped to [0, 3]
- **Given**: game state with AI HP=5, human HP=-2
- **When**: `Converter.build_input_vector(state)` is called
- **Then**:
  - ✓ AI HP (index 4) = `3.0` (clamped from 5)
  - ✓ Human HP (index 5) = `0.0` (clamped from -2)

---

## 4. Unit: `app/app_manager.py` — AppManager

### 4.1 create_game

### TC-AM1 — create_game PVP
- **Given**: AppManager with a SessionStore
- **When**: `app_manager.create_game(mode="pvp")` is called
- **Then**:
  - ✓ Returns `(game_id, state)` tuple
  - ✓ `game_id` is a valid UUID string
  - ✓ `state.mode == "pvp"`
  - ✓ `state.ai` is `None`
  - ✓ `state.round == 1`, both HP = 3, hands = 7 cards each
  - ✓ Game is saved to SessionStore (loadable via game_id)

### TC-AM2 — create_game PVE
- **Given**: AppManager with a SessionStore
- **When**: `app_manager.create_game(mode="pve", ai_name="rnn_v1.1")` is called
- **Then**:
  - ✓ Returns `(game_id, state)` tuple
  - ✓ `state.mode == "pve"`
  - ✓ `state.ai == "rnn_v1.1"`
  - ✓ Save file contains `ai_state: [0.0]*8` and `ai: "rnn_v1.1"`
  - ✓ Player 2's hand is a standard 7-card starting hand

### TC-AM3 — create_game PVE unknown AI
- **Given**: AppManager with a SessionStore
- **When**: `app_manager.create_game(mode="pve", ai_name="nonexistent")` is called
- **Then**:
  - ✓ Raises `ValueError`
  - ✓ No save file created
  - ✓ No model loaded

### TC-AM4 — create_game PVE missing ai_name
- **Given**: AppManager with a SessionStore
- **When**: `app_manager.create_game(mode="pve")` is called (ai_name=None)
- **Then**:
  - ✓ Raises `ValueError`
  - ✓ Error message mentions missing AI

### TC-AM5 — create_game produces unique IDs
- **Given**: AppManager with a SessionStore
- **When**: `create_game()` is called twice
- **Then**:
  - ✓ Two different game_ids returned
  - ✓ Both games are independently loadable

### 4.2 ai_select_action

### TC-AM6 — ai_select_action returns legal card (full hand) [EXISTING, from TC-AI5]
- **Given**: AppManager, round 1 game state with full 7-card AI hand
- **When**: `app_manager.ai_select_action(state, GRUCardModel.INITIAL_HIDDEN_STATE, "rnn_v1.1")` is called
- **Then**:
  - ✓ Returns `(card_str, new_hidden)` tuple
  - ✓ `card_str` is one of "attack", "defense", "heavy", "counter"
  - ✓ `new_hidden` is `list[float]` of length 8
  - ✓ `new_hidden != [0.0]*8` (hidden state evolved)

### TC-AM7 — ai_select_action with limited hand [EXISTING, from TC-AI6]
- **Given**: AppManager, game state where AI hand = `["counter"]` only
- **When**: `ai_select_action()` is called
- **Then**:
  - ✓ Returns `"counter"` (only legal card)

### TC-AM8 — ai_select_action: hidden state evolves [EXISTING, from TC-AI7]
- **Given**: AppManager, same game state
- **When**: `ai_select_action()` called twice sequentially, passing h1 into second call
- **Then**:
  - ✓ h1 != h0, h2 != h1
  - ✓ Both cards are legal

### TC-AM9 — ai_select_action deterministic [EXISTING, from TC-AI15]
- **Given**: AppManager, same state and hidden_state
- **When**: `ai_select_action()` called twice with identical inputs
- **Then**:
  - ✓ Both calls return the same card string

### TC-AM10 — ai_select_action unknown AI
- **Given**: AppManager, any game state
- **When**: `ai_select_action(state, hidden, "nonexistent")` is called
- **Then**:
  - ✓ Raises `ValueError`

### TC-AM11 — ai_select_action model cached
- **Given**: AppManager
- **When**: `ai_select_action()` called twice for the same AI
- **Then**:
  - ✓ Model is loaded only once (verify via mock or import count)
  - ✓ Both calls succeed

### TC-AM12 — select_action with 1-card hand returns that card only
- **Given**: AI hand = `["defense"]` only
- **When**: `ai_select_action()` is called
- **Then**:
  - ✓ Returns `"defense"` (only legal option after masking)

---

## 5. Integration: `controller/api.py` — Updated Endpoints

### 5.1 POST /api/games

### TC-API1 — create PVP game (backward compat) [EXISTING, from TC-API3]
- **Given**: Flask test client
- **When**: `POST /api/games` with body `{}` or `{"mode": "pvp"}`
- **Then**:
  - ✓ HTTP 201
  - ✓ Response `state.mode == "pvp"`, `state.ai` is `None`

### TC-API2 — create PVE game [EXISTING, from TC-API2]
- **Given**: Flask test client
- **When**: `POST /api/games` with body `{"mode": "pve", "ai": "rnn_v1.1"}`
- **Then**:
  - ✓ HTTP 201
  - ✓ Response `state.mode == "pve"`, `state.ai == "rnn_v1.1"`
  - ✓ Save file contains `ai_state: [0.0]*8`

### TC-API3 — create PVE unknown AI [EXISTING, from TC-API4]
- **Given**: Flask test client
- **When**: `POST /api/games` with body `{"mode": "pve", "ai": "nonexistent"}`
- **Then**:
  - ✓ HTTP 400, `code == "UNKNOWN_AI"`

### TC-API4 — create PVE missing ai [EXISTING, from TC-API5]
- **Given**: Flask test client
- **When**: `POST /api/games` with body `{"mode": "pve"}`
- **Then**:
  - ✓ HTTP 400, `code == "MISSING_AI"`

### 5.2 POST /api/games/{id}/play

### TC-API5 — PVE play round: cards_2 auto-computed [EXISTING, from TC-API6]
- **Given**: PVE game created
- **When**: `POST /play` with body `{"cards_1": ["attack"]}`
- **Then**:
  - ✓ HTTP 200
  - ✓ `round_result.cards_2` contains 1 legal card string
  - ✓ AI hand decreases by 1 after round

### TC-API6 — PVP play round: cards_2 required [EXISTING, from TC-API7]
- **Given**: PVP game created
- **When**: `POST /play` with body `{"cards_1": ["attack"]}` (no cards_2)
- **Then**:
  - ✓ HTTP 400, `code == "MISSING_CARDS_2"`

### TC-API7 — PVE ignores cards_2 if sent [EXISTING, from TC-API8]
- **Given**: PVE game created
- **When**: `POST /play` with body `{"cards_1": ["attack"], "cards_2": ["defense"]}`
- **Then**:
  - ✓ HTTP 200
  - ✓ `round_result.cards_2` is AI's computed card (not necessarily "defense")

### TC-API8 — PVE missing cards_1 [EXISTING, from TC-API13]
- **Given**: PVE game created
- **When**: `POST /play` with body `{}`
- **Then**:
  - ✓ HTTP 400, `code == "MISSING_CARDS_1"`

### TC-API9 — AI inference error (invalid card output)
- **Given**: PVE game, AI model (mocked) returns invalid card string
- **When**: `POST /play` is called
- **Then**:
  - ✓ HTTP 500, `code == "AI_INFERENCE_ERROR"`

### TC-API10 — AI load error mid-game
- **Given**: PVE game, weight file deleted after creation
- **When**: `POST /play` is called
- **Then**:
  - ✓ HTTP 500, `code == "AI_LOAD_ERROR"`

### 5.3 Other Endpoints

### TC-API11 — GET /api/ais [EXISTING, from TC-API1]
- **Given**: Flask test client
- **When**: `GET /api/ais`
- **Then**:
  - ✓ HTTP 200, `"ais"` array present, contains `"rnn_v1.1"`

### TC-API12 — GET /api/games/{id} includes mode and ai [EXISTING, from TC-API11]
- **Given**: PVE game created
- **When**: `GET /api/games/{id}`
- **Then**:
  - ✓ `state.mode == "pve"`, `state.ai == "rnn_v1.1"`

### TC-API13 — DELETE /api/games/{id} [EXISTING, from TC-API12]
- **Given**: any game created
- **When**: `DELETE /api/games/{id}`
- **Then**:
  - ✓ HTTP 204, subsequent GET returns 404

---

## 6. Integration: Full AI Game Flow [EXISTING]

### TC-FLOW1 — complete 7-round AI game [EXISTING]
- **Given**: PVE game against `rnn_v1.1`
- **When**: play all rounds until game over
- **Then**:
  - ✓ Each POST /play returns 200
  - ✓ AI selects legal card each round
  - ✓ After last round: `state.is_over == true`, winner is "1"/"2"/"draw"

### TC-FLOW2 — AI hand decreases each round [EXISTING]
- **Given**: PVE game
- **When**: rounds are played sequentially
- **Then**:
  - ✓ After each round, AI hand size decreases by exactly 1
  - ✓ After 7 rounds (if reached), AI hand is empty

### TC-FLOW3 — new game uses fresh hidden state [EXISTING, from TC-FLOW6]
- **Given**: two PVE games created sequentially
- **When**: first round of second game
- **Then**:
  - ✓ `round == 1`, HP = 3 for both players
  - ✓ Save file ai_state = `[0.0]*8` (fresh, not carried over)

### TC-FLOW4 — AI hidden state persists across rounds [EXISTING, from TC-API10]
- **Given**: PVE game, 2 rounds played
- **When**: examining hidden state after each round
- **Then**:
  - ✓ Hidden state after round 2 is different from hidden state after round 1

### TC-FLOW5 — game_over prevents further play [EXISTING, from TC-EDGE4]
- **Given**: completed PVE game
- **When**: another POST /play
- **Then**:
  - ✓ HTTP 400, `code == "GAME_OVER"`

---

## 7. Edge Cases

### TC-EDGE1 — AI hand with 1 card: must select it [EXISTING]
- **Given**: PVE game where AI hand = `["counter"]` only
- **When**: human plays `"attack"`
- **Then**:
  - ✓ AI selects `"counter"`
  - ✓ Resolution: Counter vs Attack → AI 0 damage, human 1 damage

### TC-EDGE2 — legacy PVP save file still works [EXISTING, from TC-EDGE3]
- **Given**: save file from intention-1 (no mode/ai fields)
- **When**: loaded and played via PVP route
- **Then**:
  - ✓ Game loads (mode defaults to "pvp")
  - ✓ Round plays normally

### TC-EDGE3 — PVE save file with missing ai_state [EXISTING, from TC-SS6]
- **Given**: save file with mode="pve", ai="rnn_v1.1", no ai_state
- **When**: game is loaded via SessionStore
- **Then**:
  - ✓ Load succeeds
  - ✓ metadata["ai_state"] = `[0.0]*8` (fallback)

### TC-EDGE4 — AppManager model cache survives across requests
- **Given**: AppManager created at startup
- **When**: `ai_select_action()` called for first request (loads model), then called again
- **Then**:
  - ✓ Model is not reloaded on second call (cached)
