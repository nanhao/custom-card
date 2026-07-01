# Interface Design — App Bridge Layer

Public interfaces only. Private methods omitted per design rule.

---

## 1. `ai/rnn_v1/model.py` — GRUCardModel

After refactoring, `model.py` contains only the model class. All helper functions and `select_action` are removed.

```
INPUT_SIZE = 14
HIDDEN_SIZE = 8
NUM_CARD_TYPES = 4
WEIGHT_PATH = "src/backend/ai/rnn_v1/gen1/rnn_v1.pth"
```

```
class GRUCardModel(nn.Module):
    """GRU-based neural network for card selection.

    Architecture:
        Input(14) → GRU(8, 1 layer) → policy_head(8→4) + value_head(8→1)

    The GRU is not batch-first — inputs are (seq_len=1, batch=1, features=14).
    """

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None
        """Initialize GRU with random weights.
        Layers: self.gru, self.fc_policy, self.fc_value.
        """

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one GRU step, apply optional mask, softmax, return results.

        Args:
            x: Input tensor of shape (1, 1, 14).
            hidden: Hidden state tensor of shape (1, 1, 8).
            mask_fn: Optional callable that takes policy logits (1,4) and
                     returns masked logits (1,4). If None, no masking.

        Returns:
            (probs, value, new_hidden) tuple:
                probs: (1, 4) softmax probabilities over 4 card types.
                       Already masked if mask_fn is provided.
                value: (1, 1) win-rate estimate in [-1, 1].
                new_hidden: (1, 1, 8) updated GRU hidden state.
        """

    # Attributes:
    #   self.gru: nn.GRU — GRU layer (input=14, hidden=8, 1 layer)
    #   self.fc_policy: nn.Linear — policy head (8→4)
    #   self.fc_value: nn.Linear — value head (8→1)
```

**Removed** (compared to current):
- `init_hidden_state()` function → replaced by `GRUCardModel.INITIAL_HIDDEN_STATE`
- `select_action()` function → moved to `AppManager.ai_select_action()`
- `_build_input_vector()` → moved to `Converter.build_input_vector()`
- `_mask_illegal_actions()` → moved to `AppManager` as a closure factory
- `_index_to_card()`, `_card_to_index()` → moved to `Converter`
- `_load_model()` → moved to `AppManager`
- `import CardType` → removed

---

## 2. `ai/registry.py` — AI Registry

After refactoring, `load_ai()` is removed. Registry is pure metadata.

Internal `_REGISTRY` adds a `class` field:
```python
_REGISTRY: list[dict] = [
    {
        "name": "rnn_v1.1",
        "description": "GRU-based model, random weights (gen1)",
        "weight_path": "src/backend/ai/rnn_v1/gen1/rnn_v1.pth",
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUCardModel",
    },
]
```

### Public Interface

```
def list_ais() -> list[dict]:
    """Return list of available AIs for frontend display.

    Returns:
        List of dicts, each with "name" and "description" keys only.
        Internal fields (module, weight_path, class) are excluded.
    """

def get_ai_info(name: str) -> dict:
    """Return the full registry entry for an AI, including internal fields.

    Args:
        name: AI name as listed in the registry (e.g. "rnn_v1.1").

    Returns:
        dict with keys: name, description, weight_path, module, class.

    Raises:
        ValueError: If the name is not found in the registry.
    """
```

**Removed** (compared to current):
- `load_ai()` → model loading/validation moves to `AppManager`
- `_loaded_ais` cache → model cache moves to `AppManager`
- Module import validation (`importlib.import_module`) → moves to `AppManager`

---

## 3. `app/converter.py` — Converter

Pure transformation class. No state, no PyTorch. Converts between card strings, one-hot indices, and model input vectors.

```
class Converter:
    """Bidirectional conversion between card strings and model tensors.

    All methods are stateless. The class serves as a namespace for
    conversion logic that bridges game domain types and AI model inputs.
    """

    # Constants
    NUM_CARD_TYPES: int = 4
    INPUT_SIZE: int = 14

    # --- Card ↔ Index ---

    @staticmethod
    def card_to_index(card_str: str) -> int:
        """Convert card string to policy head index.

        Mapping: attack→0, defense→1, heavy→2, counter→3.
        Returns -1 for unrecognized card strings.
        """

    @staticmethod
    def index_to_card(index: int) -> str:
        """Convert policy head index to card string.

        Mapping: 0→attack, 1→defense, 2→heavy, 3→counter.
        """

    # --- Input Vector Construction ---

    @staticmethod
    def build_input_vector(state: dict) -> list[float]:
        """Build the 14-d input vector from a game state dict.

        Args:
            state: Game state dict from GameManager.get_state(), the AI
                   always plays as Player "2".

        Returns:
            list[float] of length 14:
                indices 0–3:  AI card counts (attack, defense, heavy, counter)
                indices 4–5:  HP values (AI HP, human HP), clamped [0,3]
                indices 6–9:  One-hot of AI's last move (zeros in round 1)
                indices 10–13: One-hot of human's last move (zeros in round 1)
        """
```

---

## 4. `app/app_manager.py` — AppManager

The bridge between `controller` and the `game`+`ai` packages. Holds model cache and `SessionStore` reference. Created once at server startup.

```
class AppManager:
    """Bridge layer orchestrating game creation and AI action selection.

    Owns model lifecycle: imports model modules, instantiates model classes,
    loads weights, and caches loaded instances.

    Holds a SessionStore reference for game persistence.
    """

    def __init__(self, session_store: SessionStore) -> None:
        """Create AppManager with a SessionStore for game persistence.

        Args:
            session_store: SessionStore instance for save/load/delete.
        """

    # --- Game Lifecycle ---

    def create_game(self, mode: str, ai_name: str | None = None) -> tuple[str, dict]:
        """Create a new game session and return its ID and state.

        For PVE mode: validates ai_name exists in registry, initializes
        hidden state to GRUCardModel.INITIAL_HIDDEN_STATE, and saves
        AI metadata alongside game state.

        For PVP mode: creates a standard game with no AI metadata.

        Args:
            mode: Game mode — "pvp" or "pve".
            ai_name: AI name (required if mode=="pve", ignored if "pvp").

        Returns:
            (game_id, state) tuple:
                game_id: UUID string.
                state: dict from GameManager.get_state() with "ai" injected.

        Raises:
            ValueError: If mode=="pve" and ai_name is None or unknown.
        """

    # --- AI Action Selection ---

    def ai_select_action(
        self, state: dict, hidden_state: list[float], ai_name: str
    ) -> tuple[str, list[float]]:
        """Select the AI's card for the current round.

        Internally:
        1. Loads/retrieves the AI model via registry metadata
        2. Builds 14-d input vector via Converter.build_input_vector()
        3. Creates a mask_fn closure from the AI's hand state
        4. Runs model.forward(x, h, mask_fn) to get masked probs
        5. Selects card via argmax, converts index to string via Converter

        Args:
            state: Game state dict (same format as GameManager.get_state()).
            hidden_state: Current GRU hidden state as list of 8 floats.
            ai_name: AI name (e.g. "rnn_v1.1").

        Returns:
            (card_str, new_hidden_state) tuple:
                card_str: One of "attack", "defense", "heavy", "counter".
                new_hidden_state: Updated hidden state as list of 8 floats.

        Raises:
            ValueError: If ai_name is not found in registry.
            RuntimeError: If model loading or inference fails.
        """
```

---

## 5. `controller/api.py` — Controller (Modified)

Module-level changes compared to current:

- **Imports**: Removes `load_ai` from `ai.registry` import (keeps `list_ais`). Adds `from src.backend.app.app_manager import AppManager`.
- **`create_app()`**: Creates `AppManager(session_store)` alongside `SessionStore`. Passes it to route handlers.
- **`POST /api/games`**: Replaces inline game creation logic with `app_manager.create_game(mode, ai_name)`.
- **`POST /api/games/{id}/play` (PVE branch)**: Replaces `load_ai()` + `ai_module.select_action()` with `app_manager.ai_select_action(state, hidden_state, ai_name)`.
- **Other endpoints**: Unchanged (GET /state, DELETE, GET /ais, GET /).

---

## 6. `game/` — Unchanged

The `game/` package (card.py, player.py, resolver.py, game_manager.py, session_store.py) requires zero changes. All existing interfaces remain identical. The refactoring only affects how `ai/` and `controller/` interact with `game/`.
