"""Unit tests for AiRegistry and RnnAiAdapter in the ai_adapter package."""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from src.backend.ai.constants import HIDDEN_SIZE, POLICY_OUTPUT_SIZE
from src.backend.ai.rnn_v1.model import GRUPolicyModel
from src.backend.ai_adapter.ai_interface import AiInterface
from src.backend.ai_adapter.ai_registry import AiRegistry
from src.backend.ai_adapter.rnn_ai import RnnAiAdapter
from src.backend.app.converter import Converter
from src.backend.game.card_type import CardType


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def temp_weight_dir():
    """Create a temp directory with valid GRUPolicyModel and GRUValueModel weight files."""
    policy_model = GRUPolicyModel()
    from src.backend.ai.rnn_v1.model import GRUValueModel
    value_model = GRUValueModel()
    with tempfile.TemporaryDirectory() as td:
        policy_path = os.path.join(td, "policy.pth")
        value_path = os.path.join(td, "value.pth")
        torch.save(policy_model.state_dict(), policy_path)
        torch.save(value_model.state_dict(), value_path)
        yield td


# ===========================================================================
# RnnAiAdapter
# ===========================================================================


class TestRnnAiAdapter:
    """Tests for RnnAiAdapter construction and behavior."""

    def test_constructor_stores_params(self, temp_weight_dir):
        """Adapter stores injected params and lazily loads model."""
        adapter = RnnAiAdapter(
            policy_module="m", policy_class="C", policy_weight_path="p",
            value_module="mv", value_class="Cv", value_weight_path="pv",
        )
        assert adapter._policy_module == "m"
        assert adapter._policy_model is None
        assert adapter._value_model_cache is None

    def test_get_model_loads_and_caches(self, temp_weight_dir):
        """_get_policy_model loads on first call, caches on subsequent."""
        policy_path = os.path.join(temp_weight_dir, "policy.pth")
        value_path = os.path.join(temp_weight_dir, "value.pth")
        adapter = RnnAiAdapter(
            policy_module="src.backend.ai.rnn_v1.model",
            policy_class="GRUPolicyModel",
            policy_weight_path=policy_path,
            value_module="src.backend.ai.rnn_v1.model",
            value_class="GRUValueModel",
            value_weight_path=value_path,
        )
        model1 = adapter._get_policy_model()
        assert isinstance(model1, GRUPolicyModel)
        model2 = adapter._get_policy_model()
        assert model2 is model1

    def test_ai_select_action_returns_expected_shape(self, temp_weight_dir):
        """Forward pass produces valid card_type and hidden state."""
        policy_path = os.path.join(temp_weight_dir, "policy.pth")
        value_path = os.path.join(temp_weight_dir, "value.pth")
        adapter = RnnAiAdapter(
            policy_module="src.backend.ai.rnn_v1.model",
            policy_class="GRUPolicyModel",
            policy_weight_path=policy_path,
            value_module="src.backend.ai.rnn_v1.model",
            value_class="GRUValueModel",
            value_weight_path=value_path,
        )
        from tests.helpers import make_game_state

        state = make_game_state(
            hand_2=[CardType.ATTACK],
            hp_1=20, hp_2=20,
        )
        hidden = [0.0] * HIDDEN_SIZE
        card, bt, dt, new_h = adapter.ai_select_action(state, hidden)

        # We don't know which card the random model will pick,
        # but the output types must be correct
        assert isinstance(new_h, list)
        assert len(new_h) == HIDDEN_SIZE


class TestRnnAiAdapterMaskFn:
    """Tests for _make_mask_fn static method."""

    def test_hand_card_not_masked(self):
        """Card in hand should NOT be masked (can be selected)."""
        mask = RnnAiAdapter._make_mask_fn([CardType.ATTACK], energy=0)
        logits = torch.zeros(1, POLICY_OUTPUT_SIZE)
        result = mask(logits)
        # ATTACK play index = 3 (from Converter._PLAY_TO_INDEX)
        assert result[0, 3] == 0.0

    def test_card_not_in_hand_masked(self):
        """Card not in hand should be masked to -inf."""
        mask = RnnAiAdapter._make_mask_fn([CardType.ATTACK], energy=0)
        logits = torch.zeros(1, POLICY_OUTPUT_SIZE)
        result = mask(logits)
        # DEFENSE play index = 6 — should be masked since not in hand
        assert result[0, 6] == float('-inf')

    def test_pass_never_masked(self):
        """Pass (index 0) is always allowed."""
        mask = RnnAiAdapter._make_mask_fn([], energy=0)
        logits = torch.zeros(1, POLICY_OUTPUT_SIZE)
        result = mask(logits)
        assert result[0, 0] == 0.0

    def test_build_blocked_when_energy_zero(self):
        """Build targets masked when energy is 0."""
        mask = RnnAiAdapter._make_mask_fn([CardType.BUILD], energy=0)
        logits = torch.zeros(1, POLICY_OUTPUT_SIZE)
        result = mask(logits)
        # Build ATTACK index = 4 — masked because energy=0
        assert result[0, 4] == float('-inf')

    def test_build_allowed_when_energy_sufficient(self):
        """Build targets allowed when energy >= build_cost."""
        mask = RnnAiAdapter._make_mask_fn([CardType.BUILD], energy=2)
        logits = torch.zeros(1, POLICY_OUTPUT_SIZE)
        result = mask(logits)
        # Build ATTACK index = 4, cost=1 — allowed (energy=2 >= 1)
        assert result[0, 4] == 0.0

    def test_dismantle_blocked_when_discard_empty(self):
        """Dismantle masked when discard pile is empty."""
        mask = RnnAiAdapter._make_mask_fn(
            [CardType.BUILD], energy=0, discard_pile=[],
        )
        logits = torch.zeros(1, POLICY_OUTPUT_SIZE)
        result = mask(logits)
        # Dismantle ATTACK index = 5 — masked because discard empty
        assert result[0, 5] == float('-inf')

    def test_dismantle_allowed_when_target_in_discard(self):
        """Dismantle allowed when target card is in discard pile."""
        mask = RnnAiAdapter._make_mask_fn(
            [CardType.BUILD], energy=0, discard_pile=[CardType.ATTACK],
        )
        logits = torch.zeros(1, POLICY_OUTPUT_SIZE)
        result = mask(logits)
        # Dismantle ATTACK index = 5 — allowed (ATTACK in discard)
        assert result[0, 5] == 0.0


# ===========================================================================
# AiRegistry
# ===========================================================================


class TestAiRegistry:
    """Tests for AiRegistry.get_ai_interface factory."""

    def test_returns_ai_interface_instance(self):
        """get_ai_interface returns an AiInterface instance."""
        adapter = AiRegistry.get_ai_interface("rnn_v1.1")
        assert isinstance(adapter, AiInterface)

    def test_caches_same_instance_for_same_name(self):
        """Same ai_name returns the exact same cached instance."""
        a1 = AiRegistry.get_ai_interface("rnn_v1.1")
        a2 = AiRegistry.get_ai_interface("rnn_v1.1")
        assert a1 is a2

    def test_different_names_return_different_instances(self):
        """Different ai_names return different instances."""
        a1 = AiRegistry.get_ai_interface("rnn_v1.1")
        a2 = AiRegistry.get_ai_interface("rnn_v2.1")
        assert a1 is not a2

    def test_unknown_name_raises_value_error(self):
        """Unknown ai_name raises ValueError."""
        with pytest.raises(ValueError, match="unknown|not found"):
            AiRegistry.get_ai_interface("nonexistent.1")
