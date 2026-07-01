"""App bridge layer — orchestrates game creation and AI card selection.

Per design/3 architecture D2, AppManager owns the bridge logic between
controller/ and the game/ + ai/ packages. It is created once at server
startup (D8) and holds:
    - A SessionStore reference for game persistence
    - A model cache for AI model instances (D7)

Public interface (per design/3 interfaces.md §4):
    create_game(mode, ai_name=None) → (game_id, state)
    ai_select_action(state, hidden_state, ai_name) → (card_str, new_hidden)
"""

import importlib
import uuid

import torch
import torch.nn as nn

from src.backend.ai.registry import get_ai_info
from src.backend.app.converter import Converter

# Lazy imports (imported when needed to avoid circular dependencies)
#   GameCore — game state machine
#   SessionStore — game persistence
from src.backend.game.game_core import GameCore
from src.backend.game.session_store import SessionStore


class AppManager:
    """Bridge layer orchestrating game creation and AI action selection.

    Owns model lifecycle: imports model modules, instantiates model classes,
    loads weights, and caches loaded instances (per design D7).

    Holds a SessionStore reference for game persistence.

    Attributes:
        _session_store: SessionStore instance for save/load/delete.
        _models: Dict[str, nn.Module] cache of loaded AI model instances.
    """

    def __init__(self, session_store: SessionStore) -> None:
        """Create AppManager with a SessionStore for game persistence.

        Args:
            session_store: SessionStore instance for save/load/delete.
        """
        self._session_store = session_store
        self._models: dict[str, nn.Module] = {}  # Model cache per design D7

    # ------------------------------------------------------------------
    # Game Lifecycle
    # ------------------------------------------------------------------

    def create_game(
        self, mode: str, ai_name: str | None = None
    ) -> tuple[str, dict]:
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
                state: dict from GameCore.get_state() with "ai" injected.

        Raises:
            ValueError: If mode=="pve" and ai_name is None or unknown.
        """
        # Validate mode
        if mode not in ("pvp", "pve"):
            raise ValueError(
                f"Invalid mode: {mode!r}. Must be 'pvp' or 'pve'."
            )

        # Validate PVE mode requirements
        if mode == "pve":
            if not ai_name:
                raise ValueError(
                    "Missing AI name: mode is 'pve' but no ai_name provided"
                )
            # Validate AI exists in registry (does NOT load the model)
            get_ai_info(ai_name)  # Raises ValueError if unknown

        # Create and initialize game
        game_id = str(uuid.uuid4())
        gm = GameCore(mode=mode)
        gm.start_game()

        # Prepare AI metadata for PVE
        ai_state = None
        if mode == "pve":
            # Hardcode initial hidden state to zeros (avoids loading model at
            # game creation time). Matches GRUCardModel.INITIAL_HIDDEN_STATE.
            ai_state = [0.0] * 8

        # Persist game state
        self._session_store.save(game_id, gm, ai_state=ai_state, ai=ai_name)

        # Build response state (inject ai from metadata, not from GameManager)
        state = gm.get_state()
        state["ai"] = ai_name

        return game_id, state

    # ------------------------------------------------------------------
    # AI Action Selection
    # ------------------------------------------------------------------

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
            state: Game state dict (same format as GameCore.get_state()).
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
        # --- Load or retrieve cached model ---
        model = self._get_model(ai_name)

        # --- Build input vector ---
        input_vector = Converter.build_input_vector(state)

        # --- Prepare tensors ---
        # x: (1, 1, 14) — seq_len=1, batch=1, features=14
        x = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, Converter.INPUT_SIZE)
        # h: (1, 1, 8) — num_layers=1, batch=1, hidden=8
        h = torch.tensor(hidden_state, dtype=torch.float32).view(1, 1, 8)

        # --- Create mask function closure ---
        mask_fn = self._make_mask_fn(state["players"]["2"]["hand"])

        # --- Forward pass ---
        with torch.inference_mode():
            probs, _value, new_hidden = model(x, h, mask_fn=mask_fn)
            # probs shape: (1, 4) — already softmaxed + masked
            action_idx = int(torch.argmax(probs, dim=-1).item())

        # --- Convert index to card string ---
        card_str = Converter.index_to_card(action_idx)

        # --- Extract new hidden state as flat list ---
        new_hidden_list = new_hidden.squeeze(1).squeeze(0).tolist()

        return card_str, new_hidden_list

    # ------------------------------------------------------------------
    # Internal: Model loading and caching
    # ------------------------------------------------------------------

    def _get_model(self, ai_name: str) -> nn.Module:
        """Load (or retrieve from cache) the AI model for the given name.

        Uses registry metadata to import the module, instantiate the
        model class, and load weights. Caches the result per design D7.

        Args:
            ai_name: AI name from the registry (e.g. "rnn_v1.1").

        Returns:
            The loaded model instance in eval mode.

        Raises:
            ValueError: If the AI name is not in the registry.
            RuntimeError: If module import or weight loading fails.
        """
        # Check cache first
        if ai_name in self._models:
            return self._models[ai_name]

        # Get registry metadata
        info = get_ai_info(ai_name)  # Raises ValueError if unknown

        # Import the model module
        module_name = info["module"]
        class_name = info["class"]
        weight_path = info["weight_path"]

        try:
            ai_module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import AI module {module_name!r} for AI {ai_name!r}: {exc}"
            ) from exc

        # Instantiate the model class
        model_class = getattr(ai_module, class_name, None)
        if model_class is None:
            raise RuntimeError(
                f"AI module {module_name!r} has no class {class_name!r}"
            )

        model = model_class()

        # Load weights
        try:
            state_dict = torch.load(
                weight_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise RuntimeError(
                f"Failed to load model weights: file not found at {weight_path!r}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model weights from {weight_path!r}: {exc}"
            ) from exc

        model.eval()
        self._models[ai_name] = model
        return model

    # ------------------------------------------------------------------
    # Internal: Mask function factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mask_fn(hand: list[str]):
        """Create a mask function closure for illegal-action masking.

        Per design D3, the mask function is defined as a closure in
        AppManager and passed to model.forward(). It sets logits to -inf
        for card types not in the AI's hand.

        Args:
            hand: List of card strings in the AI's hand.

        Returns:
            A callable that takes logits (1,4) tensor and returns
            masked logits (1,4) tensor.
        """
        legal_cards = set(hand)

        def mask_fn(logits: torch.Tensor) -> torch.Tensor:
            """Set logits to -inf for card types not in the AI's hand."""
            masked = logits.clone()
            for card_str in Converter.ALL_CARD_TYPES:
                if card_str not in legal_cards:
                    idx = Converter.card_to_index(card_str)
                    masked[0, idx] = float("-inf")
            return masked

        return mask_fn
