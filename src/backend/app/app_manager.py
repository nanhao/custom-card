"""App bridge layer — orchestrates game creation and AI card selection.

Per design/9, AppManager owns two model caches for split policy/value
networks. Uses registry split-format entries to load models independently.

Public interface (per design/9 interfaces.md §5):
    create_game(mode, ai_name=None) → (game_id, state)
    ai_select_action(state, hidden_state, ai_name) → (card_str, new_hidden)
    ai_sample_action(state, h_pol, h_val, ai_name, temp, player_key)
        → (card_str, new_h_pol, log_prob, value, new_h_val)
    run_selfplay(config) → SelfPlayResult
"""

import importlib
import uuid

import torch
import torch.nn as nn

from src.backend.ai.registry import get_ai_info
from src.backend.app.converter import Converter
from src.backend.app.selfplay_config import SelfPlayConfig, SelfPlayResult
from src.backend.app.selfplay_runner import SelfPlayRunner
from src.backend.game.game_core import GameCore
from src.backend.game.session_store import SessionStore


# ---------------------------------------------------------------------------
# AppManager
# ---------------------------------------------------------------------------


class AppManager:
    """Bridge layer orchestrating game creation and AI action selection.

    Owns model lifecycle: imports model modules, instantiates model classes,
    loads weights, and caches loaded instances in two caches — one for
    policy models, one for value models (per design/9).

    Attributes:
        _session_store: SessionStore instance for save/load/delete.
        _policy_models: Dict[str, nn.Module] cache of policy model instances.
        _value_models: Dict[str, nn.Module] cache of value model instances.
    """

    def __init__(self, session_store: SessionStore) -> None:
        """Create AppManager with a SessionStore for game persistence.

        Args:
            session_store: SessionStore instance for save/load/delete.
        """
        self._session_store = session_store
        self._policy_models: dict[str, nn.Module] = {}
        self._value_models: dict[str, nn.Module] = {}

    # ------------------------------------------------------------------
    # Game Lifecycle
    # ------------------------------------------------------------------

    def create_game(
        self, mode: str, ai_name: str | None = None
    ) -> tuple[str, dict]:
        """Create a new game session and return its ID and state.

        For PVE mode: validates ai_name exists in registry, initializes
        policy and value hidden states to zeros, and saves AI metadata
        alongside game state. Uses split-format registry.

        For PVP mode: creates a standard game with no AI metadata.

        Args:
            mode: Game mode — "pvp" or "pve".
            ai_name: AI name (required if mode=="pve", ignored if "pvp").

        Returns:
            (game_id, state) tuple with ai field injected.

        Raises:
            ValueError: If mode=="pve" and ai_name is None or unknown.
        """
        if mode not in ("pvp", "pve"):
            raise ValueError(
                f"Invalid mode: {mode!r}. Must be 'pvp' or 'pve'."
            )

        if mode == "pve":
            if not ai_name:
                raise ValueError(
                    "Missing AI name: mode is 'pve' but no ai_name provided"
                )
            get_ai_info(ai_name)

        game_id = str(uuid.uuid4())
        gm = GameCore(mode=mode)
        gm.start_game()

        # AI metadata for PVE — store both policy and value hidden states
        ai_state = None
        ai_value_state = None
        if mode == "pve":
            ai_state = [0.0] * 8  # policy hidden state
            ai_value_state = [0.0] * 8  # value hidden state

        self._session_store.save(
            game_id, gm,
            ai_state=ai_state,
            ai_value_state=ai_value_state,
            ai=ai_name,
        )

        state = gm.get_state()
        state["ai"] = ai_name
        return game_id, state

    # ------------------------------------------------------------------
    # AI Action Selection (greedy, PVE mode)
    # ------------------------------------------------------------------

    def ai_select_action(
        self, state: dict, hidden_state: list[float], ai_name: str
    ) -> tuple[str, list[float]]:
        """Select the AI's card greedily. Only uses policy network.

        Args:
            state: Game state dict from GameCore.get_state().
            hidden_state: Current policy GRU hidden state (list of 8 floats).
            ai_name: AI name (e.g. "rnn_v2.1").

        Returns:
            (card_str, new_hidden_state): Updated policy hidden state.

        Raises:
            ValueError: If ai_name is not found in registry.
            RuntimeError: If model loading or inference fails.
        """
        model = self._get_policy_model(ai_name)
        input_vector = Converter.build_input_vector(state)

        x = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, Converter.INPUT_SIZE)
        h = torch.tensor(hidden_state, dtype=torch.float32).view(1, 1, 8)
        mask_fn = self._make_mask_fn(state["players"]["2"]["hand"])

        with torch.inference_mode():
            probs, _log_probs, new_hidden = model(x, h, mask_fn=mask_fn)
            action_idx = int(torch.argmax(probs, dim=-1).item())

        card_str = Converter.index_to_card(action_idx)
        new_hidden_list = new_hidden.squeeze(1).squeeze(0).tolist()
        return card_str, new_hidden_list

    # ------------------------------------------------------------------
    # AI Temperature-Based Action Sampling (self-play mode)
    # ------------------------------------------------------------------

    def ai_sample_action(
        self,
        state: dict,
        hidden_state_policy: list[float],
        hidden_state_value: list[float],
        ai_name: str,
        temperature: float,
        player_key: str = "2",
    ) -> tuple[str, list[float], float | None, float, list[float]]:
        """Select AI card with temperature-based sampling.

        Uses policy and value networks independently. Each network
        maintains its own hidden state and produces its own output.

        Args:
            state: Game state dict from GameCore.get_state().
            hidden_state_policy: Current policy GRU hidden state (list of 8).
            hidden_state_value: Current value GRU hidden state (list of 8).
            ai_name: AI name from registry (e.g. "rnn_v2.1").
            temperature: Sampling temperature. 0 = greedy.
            player_key: Which player the AI controls — "1" or "2".

        Returns:
            (card_str, new_h_policy, log_prob, value, new_h_value):
                card_str: Selected card string.
                new_h_policy: Updated policy hidden state.
                log_prob: Log-probability of selected action.
                value: Model's win-rate estimate (unbounded).
                new_h_value: Updated value hidden state.

        Raises:
            ValueError: If ai_name not found in registry.
            RuntimeError: If model loading or inference fails.
        """
        policy_model = self._get_policy_model(ai_name)
        value_model = self._get_value_model(ai_name)

        input_vector = Converter.build_input_vector(state, player_key=player_key)

        x = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, Converter.INPUT_SIZE)
        h_pol = torch.tensor(hidden_state_policy, dtype=torch.float32).view(1, 1, 8)
        h_val = torch.tensor(hidden_state_value, dtype=torch.float32).view(1, 1, 8)
        mask_fn = self._make_mask_fn(state["players"][player_key]["hand"])

        with torch.inference_mode():
            # Policy forward
            probs, log_probs, new_h_pol = policy_model(
                x, h_pol, mask_fn=mask_fn, temperature=temperature,
            )
            # Value forward — separate network, separate hidden state
            value, new_h_val = value_model(x, h_val)

            if temperature == 0.0:
                action_idx = int(torch.argmax(probs, dim=-1).item())
                log_prob = None
            else:
                action_idx = int(torch.multinomial(probs, 1).item())
                log_prob = float(log_probs[0, action_idx].item())

        card_str = Converter.index_to_card(action_idx)
        new_h_pol_list = new_h_pol.squeeze(1).squeeze(0).tolist()
        new_h_val_list = new_h_val.squeeze(1).squeeze(0).tolist()
        value_float = float(value.item())

        return card_str, new_h_pol_list, log_prob, value_float, new_h_val_list

    # ------------------------------------------------------------------
    # Model loading and caching (split)
    # ------------------------------------------------------------------

    def _get_policy_model(self, ai_name: str) -> nn.Module:
        """Load or retrieve cached policy model for the given AI name."""
        if ai_name in self._policy_models:
            return self._policy_models[ai_name]

        info = get_ai_info(ai_name)
        policy_info = info["policy"]

        model = self._load_model(
            policy_info["module"],
            policy_info["class"],
            policy_info["weight_path"],
        )
        self._policy_models[ai_name] = model
        return model

    def _get_value_model(self, ai_name: str) -> nn.Module:
        """Load or retrieve cached value model for the given AI name."""
        if ai_name in self._value_models:
            return self._value_models[ai_name]

        info = get_ai_info(ai_name)
        value_info = info["value"]

        model = self._load_model(
            value_info["module"],
            value_info["class"],
            value_info["weight_path"],
        )
        self._value_models[ai_name] = model
        return model

    @staticmethod
    def _load_model(module_name: str, class_name: str, weight_path: str) -> nn.Module:
        """Import module, instantiate class, load weights, set eval mode."""
        try:
            ai_module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import module {module_name!r}: {exc}"
            ) from exc

        model_class = getattr(ai_module, class_name, None)
        if model_class is None:
            raise RuntimeError(
                f"Module {module_name!r} has no class {class_name!r}"
            )

        model = model_class()
        try:
            state_dict = torch.load(
                weight_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise RuntimeError(
                f"Failed to load weights: file not found at {weight_path!r}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load weights from {weight_path!r}: {exc}"
            ) from exc

        model.eval()
        return model

    # ------------------------------------------------------------------
    # Mask function factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mask_fn(hand: list[str]):
        """Create a mask function closure for illegal-action masking."""
        legal_cards = set(hand)

        def mask_fn(logits: torch.Tensor) -> torch.Tensor:
            masked = logits.clone()
            for card_str in Converter.ALL_CARD_TYPES:
                if card_str not in legal_cards:
                    idx = Converter.card_to_index(card_str)
                    masked[0, idx] = float("-inf")
            return masked

        return mask_fn

    # ------------------------------------------------------------------
    # Self-play game loop
    # ------------------------------------------------------------------

    def run_selfplay(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games and return aggregate results.

        Thin wrapper that delegates to SelfPlayRunner with the
        split-network action sampling function.

        Args:
            config: SelfPlayConfig with all run parameters.

        Returns:
            SelfPlayResult with win/loss/draw counts and avg rounds.

        Raises:
            ValueError: If ai1 or ai2 is not found in the registry.
        """
        runner = SelfPlayRunner(self.ai_sample_action)
        return runner.run(config)
