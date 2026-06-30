# AI Module Design — `src/backend/ai/`

## 1. AI Interface — `IGameAI`

All AI implementations satisfy this protocol. No abstract base class required; structural typing is sufficient.

```
def init_hidden_state() -> list[float]
    """Return initial hidden state (list of 8 zeros for GRU)."""

def select_action(state: dict, hidden_state: list[float]) -> tuple[str, list[float]]
    """
    Given game state and current hidden state, return (card_string, new_hidden_state).
    card_string is one of "attack" | "defense" | "heavy" | "counter".
    The AI MUST only select cards present in the AI player's hand.
    """
```

**Module location:** Each AI implementation is a module that exposes these two functions. The registry discovers them by name.

---

## 2. GRU Model — `src/backend/ai/rnn_v1/model.py`

### 2.1 Network Architecture

```
Input(14) → GRU(8) → ┬→ FC(8→4) → mask → softmax    (policy head)
                      └→ FC(8→1) → tanh               (value head)
```

- **GRU**: 1 layer, hidden size 8, batch-first=False
- **Policy head**: Linear(8→4), outputs logits over 4 card types
- **Value head**: Linear(8→1), outputs win-rate estimate in [-1, 1]

### 2.2 Input Vector (14 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | my_attack_count | Number of Attack cards remaining in AI's hand |
| 1 | my_defense_count | Number of Defense cards remaining |
| 2 | my_heavy_count | Number of Heavy Strike cards remaining |
| 3 | my_counter_count | Number of Counter cards remaining |
| 4 | my_hp | AI's current HP (0–3) |
| 5 | opp_hp | Human's current HP (0–3) |
| 6–9 | my_last_move | One-hot of AI's card last round: [attack, defense, heavy, counter] |
| 10–13 | opp_last_move | One-hot of human's card last round: [attack, defense, heavy, counter] |

**Round 1 edge case:** Both one-hot vectors are all zeros `[0, 0, 0, 0]` since no prior round exists.

**Perspective:** The model always evaluates from the AI's perspective. "my" = AI, "opp" = human.

**HP clamping:** HP values are clamped to `[0, 3]` when building the input vector to prevent out-of-distribution values from game logic bugs.

### 2.3 Policy Head — Action Selection

```
logits = fc_policy(gru_output)       # (1, 4)
logits = mask_illegal(logits, hand)  # set -inf for cards not in hand
probs = softmax(logits)              # probability distribution
action = argmax(probs)               # select card with highest probability
```

**Illegal action masking:** Cards not in the AI's hand have their logit set to `-inf` before softmax, ensuring probability 0.

### 2.4 Value Head

```
value = fc_value(gru_output)  # (1, 1)
value = tanh(value)           # range [-1, 1]
```
- `-1`: estimated loss
- `1`: estimated win
- `0`: estimated draw

Value head output is not used for action selection in v1 (only policy head drives decisions). It exists for future training purposes.

### 2.5 Public Interface

```
def init_hidden_state() -> list[float]
    """Return initial GRU hidden state: list of 8 zeros."""

def select_action(state: dict, hidden_state: list[float]) -> tuple[str, list[float]]
    """
    Run forward pass and return selected card.
    state is a dict with keys: round, players, is_over, winner, round_log
    (same format as GameManager.get_state()).

    The state dict has the following nested structure:
    {
        "round": int,
        "players": {
            "1": {"hp": int, "hand": ["attack", ...], "history": ["attack", ...]},
            "2": {"hp": int, "hand": ["attack", ...], "history": ["attack", ...]}
        },
        "is_over": bool,
        "winner": str | None,
        "mode": str,
        "round_log": [RoundResult, ...]
    }

    The AI selects for player "2". hand entries are card type strings:
    "attack", "defense", "heavy", "counter".
    history is a list of card strings from prior rounds (empty in round 1).

    Internally builds the 14-d input vector from the state dict.
    The model is loaded at module import time and cached.
    """
```

**Model loading strategy:** The model is loaded eagerly at module import time. `torch.load()` with `weights_only=True` and `map_location='cpu'` is called once; the model instance is cached as a module-level variable for all subsequent inferences. No user request ever pays the model-loading cost. Server startup includes a health check that the model loaded successfully. If eager loading is undesirable, the implementation may defer to first `select_action()` call (lazy), but the `.pth` file is loaded at most once.

**Device handling:** Weights are loaded with `map_location='cpu'` for portability. The model always runs on CPU — no GPU support is needed for this model size (single forward pass per round).

**Inference mode:** Before each forward pass, `model.eval()` ensures training-specific behavior (dropout) is disabled, and `torch.inference_mode()` disables autograd tracking entirely (faster than `torch.no_grad()`).

### 2.6 Weight Files

| Version | Path | Description |
|---------|------|-------------|
| gen1 | `src/backend/ai/rnn_v1/gen1/rnn_v1.pth` | Random initialization, no training |

Weights are PyTorch `state_dict` format. Loading is done at module import time via `torch.load(path, map_location='cpu', weights_only=True)` and the model instance is cached.

---

## 3. AI Registry — `src/backend/ai/registry.py`

A declarative list of all available AIs. Adding a new AI requires only adding an entry here — no code changes elsewhere.

### 3.1 Registry Data

```
_REGISTRY: list[dict] = [
    {
        "name": "rnn_v1.1",
        "description": "GRU-based model, random weights (gen1)",
        "weight_path": "src/backend/ai/rnn_v1/gen1/rnn_v1.pth",
        "module": "src.backend.ai.rnn_v1.model",
    },
]
```

### 3.2 Public Interface

```
def list_ais() -> list[dict]
    """Return list of available AIs. Each dict has keys: name, description."""

def load_ai(name: str) -> object
    """
    Load and return the AI module for the given name.
    Returns an object with init_hidden_state() and select_action(state, hidden_state).
    Raises ValueError if name not found.
    """
```

---

## 4. AI Invocation Flow

Refer to [ai_inference.puml](diagrams/ai_inference.puml).

1. Controller receives `POST /play` with `cards_1` only (PVE mode)
2. Controller loads game state + metadata from SessionStore
3. Controller passes raw state dict to AI: `ai.select_action(gm.get_state(), hidden_state)`
4. AI module internally: builds 14-d input vector → GRU forward → mask illegal → pick action → return (card_str, new_hidden_state)
5. Controller converts `card_str` → `CardType` via `str_to_card()`, calls `gm.play_round(cards_1, [ai_card])`
6. Controller saves updated game state + new hidden state to SessionStore
