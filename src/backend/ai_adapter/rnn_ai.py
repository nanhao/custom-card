"""Concrete RNN-based AI adapter for the custom card game.

Implements AiInterface using PyTorch RNN models and Converter for
input/output encoding. Model metadata is injected via constructor —
no dependency on AiRegistry.
"""

import importlib

import torch
import torch.nn as nn

from src.backend.ai.rnn_v2.constants import HIDDEN_SIZE, INPUT_SIZE
from src.backend.ai_adapter.ai_interface import AiInterface
from src.backend.ai_adapter.converter import Converter
from src.backend.game.card_type import CardType
from src.backend.game.game_state_snapshot import GameStateSnapshot


class RnnAiAdapter(AiInterface):
    """Concrete AI adapter using RNN-based policy models.

    Constructed with policy model metadata (module, class, weight_path).
    Lazily loads and caches the model on first use. Owns mask function
    construction and greedy action selection via forward pass.
    """

    def __init__(
        self,
        policy_module: str,
        policy_class: str,
        policy_weight_path: str,
        value_module: str,
        value_class: str,
        value_weight_path: str,
    ) -> None:
        """Create an RnnAiAdapter with injected model metadata.

        Args:
            policy_module: Fully-qualified Python module path
                (e.g. "src.backend.ai.rnn_v1.model").
            policy_class: Model class name within that module
                (e.g. "GRUPolicyModel").
            policy_weight_path: Relative path to the .pth weight file
                (e.g. "src/backend/ai/rnn_v1/gen3/policy.pth").
            value_module: Module path for the value model.
            value_class: Class name for the value model.
            value_weight_path: Weight file path for the value model.
        """
        self._policy_module = policy_module
        self._policy_class = policy_class
        self._policy_weight_path = policy_weight_path
        self._value_module = value_module
        self._value_class = value_class
        self._value_weight_path = value_weight_path
        self._policy_model: nn.Module | None = None
        self._value_model_cache: nn.Module | None = None

    # ------------------------------------------------------------------
    # AiInterface implementation
    # ------------------------------------------------------------------

    def ai_select_action(
        self, state: GameStateSnapshot, hidden_state: list[float],
    ) -> tuple[CardType | None, CardType | None, CardType | None, list[float]]:
        """Select the AI's card greedily using policy network.

        Per design/37: returns (card_type, build_target, dismantle_target, new_hidden_state).
        build_target/dismantle_target are non-None only when card_type=BUILD.
        """
        model = self._get_policy_model()
        input_vector = Converter.build_input_vector(state)
        energy = state.players["2"].energy
        hand = state.players["2"].hand

        x = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, INPUT_SIZE)
        h = torch.tensor(hidden_state, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)
        mask_fn = self._make_mask_fn(
            hand, energy, discard_pile=state.players["2"].discard_pile,
        )

        with torch.inference_mode():
            probs, _log_probs, new_hidden = model(x, h, mask_fn=mask_fn)
            action_idx = int(torch.argmax(probs, dim=-1).item())

        card_type, build_target, dismantle_target = (
            Converter.policy_index_to_action(action_idx)
        )

        new_hidden_list = new_hidden.squeeze(1).squeeze(0).tolist()
        return card_type, build_target, dismantle_target, new_hidden_list

    def ai_sample_action(
        self,
        state: GameStateSnapshot,
        hidden_state_policy: list[float],
        hidden_state_value: list[float],
        temperature: float,
        player_key: str = "2",
    ) -> tuple[
        CardType | None, CardType | None, CardType | None,
        list[float], float | None, float, list[float],
    ]:
        """Select AI card with temperature-based sampling.

        Per design/37: returns (card_type, build_target, dismantle_target,
        new_h_pol, log_prob, value, new_h_val).
        build_target/dismantle_target non-None only when card_type=BUILD.
        log_prob is None when temperature == 0.
        """
        policy_model = self._get_policy_model()
        value_model = self._get_value_model()

        input_vector = Converter.build_input_vector(state, player_key=player_key)
        energy = state.players[player_key].energy
        hand = state.players[player_key].hand

        x = torch.tensor(input_vector, dtype=torch.float32).view(1, 1, INPUT_SIZE)
        h_pol = torch.tensor(hidden_state_policy, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)
        h_val = torch.tensor(hidden_state_value, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)
        mask_fn = self._make_mask_fn(
            hand, energy, discard_pile=state.players[player_key].discard_pile,
        )

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

        card_type, build_target, dismantle_target = (
            Converter.policy_index_to_action(action_idx)
        )

        new_h_pol_list = new_h_pol.squeeze(1).squeeze(0).tolist()
        new_h_val_list = new_h_val.squeeze(1).squeeze(0).tolist()
        value_float = float(value.item())

        return card_type, build_target, dismantle_target, new_h_pol_list, log_prob, value_float, new_h_val_list

    # ------------------------------------------------------------------
    # Model loading (lazy, cached)
    # ------------------------------------------------------------------

    def _get_policy_model(self) -> nn.Module:
        """Load and cache the policy model on first call."""
        if self._policy_model is None:
            self._policy_model = self._load_model(
                self._policy_module,
                self._policy_class,
                self._policy_weight_path,
            )
        return self._policy_model

    def _get_value_model(self) -> nn.Module:
        """Load and cache the value model on first call."""
        if self._value_model_cache is None:
            self._value_model_cache = self._load_model(
                self._value_module,
                self._value_class,
                self._value_weight_path,
            )
        return self._value_model_cache

    @staticmethod
    def _load_model(
        module_name: str, class_name: str, weight_path: str,
    ) -> nn.Module:
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
                weight_path, map_location="cpu", weights_only=True,
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
    # Mask function factory (24-d, design/37)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mask_fn(
        hand: list[CardType],
        energy: int,
        discard_pile: list[CardType] | None = None,
    ):
        """Create a 24-d mask function closure (design/37).

        Per design/37 (24-d layout):
        - Index 0: pass — always legal, never masked.
        - Play indices (3,6,9,12,15,18,21): block cards not in hand.
        - Build indices (1,4,7,10,13,16,19,22): block if BUILD not in hand,
          energy=0, or energy < build_cost of target.
        - Dismantle indices (2,5,8,11,14,17,20,23): block if BUILD not in hand,
          or discard_pile is empty/doesn't contain target card.
        """
        legal_cards = set(hand)
        can_build = CardType.BUILD in legal_cards and energy > 0
        has_build_card = CardType.BUILD in legal_cards

        discard_set: set[CardType] = (
            set(discard_pile) if discard_pile else set()
        )

        def mask_fn(logits: torch.Tensor) -> torch.Tensor:
            masked = logits.clone()
            for card_type, idx in Converter._PLAY_TO_INDEX.items():
                if card_type not in legal_cards:
                    masked[0, idx] = float('-inf')
            for card_type, _, build_idx, _ in Converter._CARD_MAP:
                if not can_build or Converter.build_cost(card_type) > energy:
                    masked[0, build_idx] = float('-inf')
            for card_type, _, _, dismantle_idx in Converter._CARD_MAP:
                if not has_build_card or card_type not in discard_set:
                    masked[0, dismantle_idx] = float('-inf')
            return masked

        return mask_fn
