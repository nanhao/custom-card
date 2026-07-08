"""App bridge layer — orchestrates game creation and AI card selection (design/14).

Per design/14: policy output 13-d interleaved (build/play pairs),
mask_fn uses 13-d interleaved masking (hand cards + build energy constraints).

Public interface:
    create_game(mode, ai_name=None) -> (game_id, state)
    ai_select_action(state, hidden_state, ai_name) -> (card_str, build_target, new_hidden)
    ai_sample_action(state, h_pol, h_val, ai_name, temp, player_key)
        -> (card_str, build_target, new_h_pol, log_prob, value, new_h_val)
    run_selfplay(config) -> SelfPlayResult
"""

import importlib
import uuid

import torch
import torch.nn as nn

from src.backend.ai.registry import get_ai_info
from src.backend.ai.rnn_v1.model import HIDDEN_SIZE, POLICY_OUTPUT_SIZE
from src.backend.app.converter import Converter
from src.backend.app.selfplay_config import SelfPlayConfig, SelfPlayResult
from src.backend.app.selfplay_runner import SelfPlayRunner
from src.backend.game.game_core import GameCore
from src.backend.game.session_store import SessionStore


class AppManager:
    """Bridge layer orchestrating game creation and AI action selection.

    Owns model lifecycle: imports model modules, instantiates model classes,
    loads weights, and caches loaded instances in two caches — one for
    policy models, one for value models.

    Attributes:
        _session_store: SessionStore instance for save/load/delete.
        _policy_models: Dict[str, nn.Module] cache of policy model instances.
        _value_models: Dict[str, nn.Module] cache of value model instances.
    """

    def __init__(self, session_store: SessionStore) -> None:
        """Create AppManager with a SessionStore for game persistence."""
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

        For PVE mode: validates ai_name, initializes hidden states to zeros (32-d).
        For PVP mode: creates a standard game with no AI metadata.

        Args:
            mode: Game mode — "pvp" or "pve".
            ai_name: AI name (required if mode=="pve").

        Returns:
            (game_id, state) tuple with ai field injected.

        Raises:
            ValueError: If mode invalid or ai_name missing/unknown.
        """
        if mode not in ("pvp", "pve"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'pvp' or 'pve'.")

        if mode == "pve":
            if not ai_name:
                raise ValueError(
                    "Missing AI name: mode is 'pve' but no ai_name provided"
                )
            get_ai_info(ai_name)

        game_id = str(uuid.uuid4())
        gm = GameCore(mode=mode)
        gm.start_game()

        ai_state = None
        ai_value_state = None
        if mode == "pve":
            ai_state = [0.0] * HIDDEN_SIZE  # 32
            ai_value_state = [0.0] * HIDDEN_SIZE  # 32

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
    ) -> tuple[str, str | None, list[float]]:
        """Select the AI's card greedily using policy network.

        Returns (card_str, build_target_str_or_None, new_hidden_state).
        build_target is non-None only when card_str == "build".

        Args:
            state: Game state dict from GameCore.get_state().
            hidden_state: Current policy GRU hidden state (list of 32 floats).
            ai_name: AI name (e.g. "rnn_v2.1").

        Returns:
            (card_str, build_target, new_hidden_state).
        """
        model = self._get_policy_model(ai_name)
        input_vector = Converter.build_input_vector(state)
        energy = state["players"]["2"]["energy"]
        hand = state["players"]["2"]["hand"]

        x = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, Converter.INPUT_SIZE)
        h = torch.tensor(hidden_state, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)
        mask_fn = self._make_mask_fn(hand, energy)

        with torch.inference_mode():
            probs, _log_probs, new_hidden = model(x, h, mask_fn=mask_fn)
            action_idx = int(torch.argmax(probs, dim=-1).item())

        # Decode 13-d interleaved output (design/14)
        card_str, build_target = Converter.policy_index_to_action(action_idx)

        new_hidden_list = new_hidden.squeeze(1).squeeze(0).tolist()
        return card_str, build_target, new_hidden_list

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
    ) -> tuple[str, str | None, list[float], float | None, float, list[float]]:
        """Select AI card with temperature-based sampling.

        Uses policy and value networks independently.

        Returns (card_str, build_target, new_h_pol, log_prob, value, new_h_val).
        build_target is non-None only when card_str == "build".
        log_prob is None when temperature == 0.
        """
        policy_model = self._get_policy_model(ai_name)
        value_model = self._get_value_model(ai_name)

        input_vector = Converter.build_input_vector(state, player_key=player_key)
        energy = state["players"][player_key]["energy"]
        hand = state["players"][player_key]["hand"]

        x = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, Converter.INPUT_SIZE)
        h_pol = torch.tensor(hidden_state_policy, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)
        h_val = torch.tensor(hidden_state_value, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)
        mask_fn = self._make_mask_fn(hand, energy)

        with torch.inference_mode():
            probs, log_probs, new_h_pol = policy_model(
                x, h_pol, mask_fn=mask_fn, temperature=temperature,
            )
            value, new_h_val = value_model(x, h_val)

            if temperature == 0.0:
                action_idx = int(torch.argmax(probs, dim=-1).item())
                log_prob = None
            else:
                action_idx = int(torch.multinomial(probs, 1).item())
                log_prob = float(log_probs[0, action_idx].item())

        # Decode 13-d interleaved output (design/14)
        card_str, build_target = Converter.policy_index_to_action(action_idx)

        new_h_pol_list = new_h_pol.squeeze(1).squeeze(0).tolist()
        new_h_val_list = new_h_val.squeeze(1).squeeze(0).tolist()
        value_float = float(value.item())

        return card_str, build_target, new_h_pol_list, log_prob, value_float, new_h_val_list

    # ------------------------------------------------------------------
    # Model loading and caching
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
    # Mask function factory (13-d interleaved, design/14)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mask_fn(hand: list[str], energy: int):
        """Create a 13-d interleaved mask function closure (design/14).

        Odd indices (1,3,5,7,9,11): pure card plays.
            - Block cards not in hand.

        Even indices (0,2,4,6,8,10,12): build + build target.
            - Block all if BUILD not in hand or energy == 0.
            - Otherwise, block targets exceeding available energy.

        Args:
            hand: List of card strings in the player's hand.
            energy: Current energy of the player.

        Returns:
            A callable mask_fn(logits: Tensor) -> Tensor.
        """
        legal_cards = set(hand)
        can_build = "build" in legal_cards and energy > 0

        def mask_fn(logits: torch.Tensor) -> torch.Tensor:
            masked = logits.clone()
            # Odd indices (1,3,5,7,9,11): pure card plays
            pure_play_cards = [
                (1, "attack"), (3, "defense"), (5, "heavy"),
                (7, "counter"), (9, "energy"), (11, "flex"),
            ]
            for idx, card_str in pure_play_cards:
                if card_str not in legal_cards:
                    masked[0, idx] = float("-inf")

            # Even indices (0,2,4,6,8,10,12): build + target
            # Block all build actions if can't build
            build_indices = [0, 2, 4, 6, 8, 10, 12]
            build_targets = ["build", "attack", "defense", "heavy",
                             "counter", "energy", "flex"]
            if not can_build:
                for idx in build_indices:
                    masked[0, idx] = float("-inf")
            else:
                for idx, target_str in zip(build_indices, build_targets):
                    cost = Converter.build_cost(target_str)
                    if cost > energy:
                        masked[0, idx] = float("-inf")

            return masked

        return mask_fn

    # ------------------------------------------------------------------
    # Self-play game loop
    # ------------------------------------------------------------------

    def run_selfplay(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games and return aggregate results.

        Thin wrapper that delegates to SelfPlayRunner.
        """
        runner = SelfPlayRunner(self.ai_sample_action)
        return runner.run(config)
