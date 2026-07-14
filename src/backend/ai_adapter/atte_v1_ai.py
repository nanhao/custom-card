"""AtteV1Adapter — concrete atte_v1 AI adapter implementing AiInterface (design/40).

Uses AtteV1Converter for game-state ↔ model-input conversion and
AtteV1PolicyModel/AtteV1ValueModel for inference. Model loading is lazy.
Hidden state is passed through by the caller.
"""

import importlib
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from src.backend.ai.atte_v1.constants import HIDDEN_SIZE
from src.backend.ai_adapter.ai_interface import AiInterface
from src.backend.ai_adapter.atte_v1_converter import AtteV1Converter
from src.backend.game.card_type import CardType
from src.backend.game.game_state_snapshot import GameStateSnapshot


class AtteV1Adapter(AiInterface):
    """Concrete atte_v1 AI adapter implementing AiInterface.

    Constructed with model metadata (8 params for 4 weight files).
    Lazily loads and caches models on first use.
    Hidden state is managed externally by the caller.
    """

    def __init__(
        self,
        policy_module: str,
        policy_class: str,
        policy_e1e2_weight_path: str,
        policy_weight_path: str,
        value_module: str,
        value_class: str,
        value_e1e2_weight_path: str,
        value_weight_path: str,
    ) -> None:
        """Create an AtteV1Adapter with injected model metadata.

        Weights are not loaded until first ai_select_action/ai_sample_action call.

        Args:
            policy_module: Python module path for policy model.
            policy_class: Class name for policy model.
            policy_e1e2_weight_path: Path to policy e1/e2 embedding weights.
            policy_weight_path: Path to policy body+head weights.
            value_module: Python module path for value model.
            value_class: Class name for value model.
            value_e1e2_weight_path: Path to value e1/e2 embedding weights.
            value_weight_path: Path to value body+head weights.
        """
        # Path validation
        for path_str in [
            policy_e1e2_weight_path, policy_weight_path,
            value_e1e2_weight_path, value_weight_path,
        ]:
            _validate_weight_path(path_str)

        self._policy_module = policy_module
        self._policy_class = policy_class
        self._policy_e1e2_path = policy_e1e2_weight_path
        self._policy_path = policy_weight_path
        self._value_module = value_module
        self._value_class = value_class
        self._value_e1e2_path = value_e1e2_weight_path
        self._value_path = value_weight_path
        self._policy_model: nn.Module | None = None
        self._value_model: nn.Module | None = None

    # ------------------------------------------------------------------
    # AiInterface implementation
    # ------------------------------------------------------------------

    def ai_select_action(
        self, state: GameStateSnapshot, hidden_state: list[float],
    ) -> tuple[CardType | None, CardType | None, CardType | None, list[float]]:
        """Greedy action selection using policy network.

        Returns (card, build_target, dismantle_target, new_hidden).
        """
        model = self._get_policy_model()
        atte_input = AtteV1Converter.build_input(state, "2")
        hand = state.players["2"].hand
        energy = state.players["2"].energy
        discard = state.players["2"].discard_pile
        mask_fn = self._make_mask_fn(hand, energy, discard)

        kwargs = _build_model_kwargs(atte_input, hidden_state)
        with torch.inference_mode():
            probs, _log_probs, new_hidden = model(
                **kwargs, mask_fn=mask_fn,
            )
            action_idx = int(torch.argmax(probs, dim=-1).item())

        card_type, build_target, dismantle_target = (
            AtteV1Converter.policy_index_to_action(action_idx)
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
        """Temperature-based action sampling.

        Returns (card, build_target, dismantle_target, new_h_pol, log_prob, value, new_h_val).
        log_prob is None when temperature == 0.
        """
        policy_model = self._get_policy_model()
        value_model = self._get_value_model()
        atte_input = AtteV1Converter.build_input(state, player_key)
        hand = state.players[player_key].hand
        energy = state.players[player_key].energy
        discard = state.players[player_key].discard_pile
        mask_fn = self._make_mask_fn(hand, energy, discard)

        pol_kwargs = _build_model_kwargs(atte_input, hidden_state_policy)
        val_kwargs = _build_model_kwargs(atte_input, hidden_state_value)

        with torch.inference_mode():
            probs, log_probs, new_h_pol = policy_model(
                **pol_kwargs, mask_fn=mask_fn, temperature=temperature,
            )
            value, new_h_val = value_model(**val_kwargs)

            if temperature == 0.0:
                action_idx = int(torch.argmax(probs, dim=-1).item())
                log_prob = None
            else:
                action_idx = int(torch.multinomial(probs, 1).item())
                log_prob = float(log_probs[0, action_idx].item())

        card_type, build_target, dismantle_target = (
            AtteV1Converter.policy_index_to_action(action_idx)
        )
        new_h_pol_list = new_h_pol.squeeze(1).squeeze(0).tolist()
        new_h_val_list = new_h_val.squeeze(1).squeeze(0).tolist()
        return (
            card_type, build_target, dismantle_target,
            new_h_pol_list, log_prob, float(value.item()), new_h_val_list,
        )

    # ------------------------------------------------------------------
    # Model loading (lazy, cached)
    # ------------------------------------------------------------------

    def _get_policy_model(self) -> nn.Module:
        """Load and cache the policy model on first call."""
        if self._policy_model is None:
            self._policy_model = self._load_model(
                self._policy_module, self._policy_class,
                self._policy_e1e2_path, self._policy_path,
            )
        return self._policy_model

    def _get_value_model(self) -> nn.Module:
        """Load and cache the value model on first call."""
        if self._value_model is None:
            self._value_model = self._load_model(
                self._value_module, self._value_class,
                self._value_e1e2_path, self._value_path,
            )
        return self._value_model

    @staticmethod
    def _load_model(
        module_name: str, class_name: str,
        e1e2_path: str, main_path: str,
    ) -> nn.Module:
        """Import module, instantiate, load both weight files, set eval mode."""
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
            e1e2_state = torch.load(e1e2_path, map_location="cpu", weights_only=True)
            main_state = torch.load(main_path, map_location="cpu", weights_only=True)
            combined = {**e1e2_state, **main_state}
            model.load_state_dict(combined, strict=False)
        except FileNotFoundError:
            raise RuntimeError(
                f"Failed to load weights: file not found at {e1e2_path!r} "
                f"or {main_path!r}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load weights: {exc}"
            ) from exc
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Mask function factory (24-d)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mask_fn(
        hand: list[CardType],
        energy: int,
        discard_pile: list[CardType],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Create a 24-d mask function closure.

        Same logic as RnnAiAdapter._make_mask_fn but uses AtteV1Converter.
        """
        legal_cards = set(hand)
        can_build = CardType.BUILD in legal_cards and energy > 0
        has_build_card = CardType.BUILD in legal_cards
        discard_set = set(discard_pile)

        def mask_fn(logits: torch.Tensor) -> torch.Tensor:
            masked = logits.clone()
            for card_type, idx in AtteV1Converter._PLAY_TO_INDEX.items():
                if card_type not in legal_cards:
                    masked[0, idx] = float("-inf")
            for card_type, _, build_idx, _ in AtteV1Converter._CARD_MAP:
                if not can_build or AtteV1Converter.build_cost(card_type) > energy:
                    masked[0, build_idx] = float("-inf")
            for card_type, _, _, dismantle_idx in AtteV1Converter._CARD_MAP:
                if not has_build_card or card_type not in discard_set:
                    masked[0, dismantle_idx] = float("-inf")
            return masked

        return mask_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_weight_path(path_str: str) -> None:
    """Validate that a weight path exists and is within the project directory.

    Args:
        path_str: Relative or absolute path to a weight file.

    Raises:
        ValueError: If the path does not appear to be valid.
    """
    path = Path(path_str)
    # Normalize and check it's not empty
    if not path_str or not path_str.strip():
        raise ValueError(f"Invalid empty weight path: {path_str!r}")


def _build_model_kwargs(
    atte_input, hidden_state: list[float],
) -> dict:
    """Build kwargs dict for model.forward() from AtteV1Input and hidden state.

    Converts Python lists to PyTorch tensors in the format expected by the model.
    """
    h = torch.tensor(hidden_state, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)

    def _tensor(ids: list[int]) -> torch.Tensor:
        if not ids:
            return torch.zeros(1, 0, dtype=torch.long)
        return torch.tensor([ids], dtype=torch.long)

    def _scalar(val: float) -> torch.Tensor:
        return torch.tensor([[val]], dtype=torch.float32)

    def _scalar_int(val: int) -> torch.Tensor:
        return torch.tensor([[val]], dtype=torch.float32)

    return {
        "own_hand_ids": _tensor(atte_input.own.hand_ids),
        "own_discard_ids": _tensor(atte_input.own.discard_ids),
        "own_draw_ids": _tensor(atte_input.own.draw_ids),
        "own_last_action": torch.tensor(
            [[atte_input.own.last_action]], dtype=torch.long,
        ),
        "own_status_ids": _tensor(atte_input.own.status_ids),
        "own_status_durations": torch.tensor(
            [atte_input.own.status_durations], dtype=torch.float32,
        ),
        "own_opp_hand_ids": _tensor(atte_input.own.opp_hand_ids),
        "own_opp_discard_ids": _tensor(atte_input.own.opp_discard_ids),
        "own_opp_draw_ids": _tensor(atte_input.own.opp_draw_ids),
        "opp_hand_ids": _tensor(atte_input.opp.hand_ids),
        "opp_discard_ids": _tensor(atte_input.opp.discard_ids),
        "opp_draw_ids": _tensor(atte_input.opp.draw_ids),
        "opp_last_action": torch.tensor(
            [[atte_input.opp.last_action]], dtype=torch.long,
        ),
        "opp_status_ids": _tensor(atte_input.opp.status_ids),
        "opp_status_durations": torch.tensor(
            [atte_input.opp.status_durations], dtype=torch.float32,
        ),
        "own_hp": _scalar(atte_input.own.hp),
        "own_energy": _scalar(atte_input.own.energy),
        "opp_hp": _scalar(atte_input.opp.hp),
        "opp_energy": _scalar(atte_input.opp.energy),
        "round_in_cycle": _scalar_int(atte_input.round_in_cycle),
        "hidden": h,
    }
