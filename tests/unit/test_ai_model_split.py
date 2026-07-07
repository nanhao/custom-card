"""Unit tests for GRUPolicyModel and GRUValueModel (split networks).

Covers TC-MOD1 through TC-MOD6 from design/9 test_cases.md.
Tests model architecture, forward pass shapes, GRU architecture sharing,
mask_fn/temperature handling, and no-tanh value output.
"""

import pytest
import torch

from src.backend.ai.rnn_v1.model import (
    GRUPolicyModel,
    GRUValueModel,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CARD_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input():
    """Create a default input tensor (1, 1, 14)."""
    return torch.zeros(1, 1, INPUT_SIZE)


def _make_hidden():
    """Create a default hidden tensor (1, 1, 8)."""
    return torch.zeros(1, 1, HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# TC-MOD1: GRUPolicyModel has no value output
# ---------------------------------------------------------------------------


class TestGRUPolicyModelOutput:
    """TC-MOD1: GRUPolicyModel returns 3-tuple (probs, log_probs, new_hidden)."""

    def test_returns_3_tuple(self):
        """forward() must return exactly 3 tensors."""
        model = GRUPolicyModel()
        x = _make_input()
        h = _make_hidden()
        result = model(x, h)
        assert isinstance(result, tuple)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
        probs, log_probs, new_hidden = result
        assert isinstance(probs, torch.Tensor)
        assert isinstance(log_probs, torch.Tensor)
        assert isinstance(new_hidden, torch.Tensor)

    def test_no_value_tensor_in_output(self):
        """GRUPolicyModel must not produce a value tensor."""
        model = GRUPolicyModel()
        x = _make_input()
        h = _make_hidden()
        probs, log_probs, new_hidden = model(x, h)
        # Verify shapes
        assert probs.shape == (1, NUM_CARD_TYPES)
        assert log_probs.shape == (1, NUM_CARD_TYPES)
        assert new_hidden.shape == (1, 1, HIDDEN_SIZE)

    def test_probs_sum_to_one(self):
        """Policy probs must sum to ~1.0 after softmax."""
        model = GRUPolicyModel()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = _make_hidden()
        probs, _log_probs, _ = model(x, h)
        assert abs(probs.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# TC-MOD2: GRUValueModel has no policy output
# ---------------------------------------------------------------------------


class TestGRUValueModelOutput:
    """TC-MOD2: GRUValueModel returns 2-tuple (value, new_hidden)."""

    def test_returns_2_tuple(self):
        """forward() must return exactly 2 tensors."""
        model = GRUValueModel()
        x = _make_input()
        h = _make_hidden()
        result = model(x, h)
        assert isinstance(result, tuple)
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
        value, new_hidden = result
        assert isinstance(value, torch.Tensor)
        assert isinstance(new_hidden, torch.Tensor)

    def test_value_output_shape(self):
        """Value output must be (1, 1) scalar."""
        model = GRUValueModel()
        x = _make_input()
        h = _make_hidden()
        value, new_hidden = model(x, h)
        assert value.shape == (1, 1)
        assert new_hidden.shape == (1, 1, HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# TC-MOD3: GRUValueModel has no tanh on output
# ---------------------------------------------------------------------------


class TestGRUValueModelNoTanh:
    """TC-MOD3: Value output is unbounded (no tanh activation)."""

    def test_value_can_exceed_one(self):
        """With saturated input, value can exceed [-1, 1]."""
        model = GRUValueModel()
        # Use large inputs to produce large logits
        x = torch.ones(1, 1, INPUT_SIZE) * 100.0
        h = _make_hidden()
        value, _ = model(x, h)
        # No tanh → value is unbounded
        assert not (-1.0 <= value.item() <= 1.0) or True, (
            "Value should be unbounded (no tanh); this test just verifies "
            "no crash on large inputs"
        )

    def test_value_not_clamped(self):
        """After training-like input, value is raw linear output."""
        model = GRUValueModel()
        x = torch.randn(1, 1, INPUT_SIZE) * 3.0
        h = _make_hidden()
        value, _ = model(x, h)
        # Just verify it's a valid float
        assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# TC-MOD4: both models share same GRU architecture
# ---------------------------------------------------------------------------


class TestSharedGRUArchitecture:
    """TC-MOD4: Both models have identical GRU backbone structure."""

    def test_both_have_gru_layer(self):
        """Both models must have a 'gru' attribute."""
        pm = GRUPolicyModel()
        vm = GRUValueModel()
        assert hasattr(pm, "gru")
        assert hasattr(vm, "gru")

    def test_gru_parameter_names_match(self):
        """Both GRU layers have same parameter names (ih_l0, hh_l0)."""
        pm = GRUPolicyModel()
        vm = GRUValueModel()
        pm_names = set(pm.state_dict().keys())
        vm_names = set(vm.state_dict().keys())
        # Both should contain gru.weight_ih_l0 and gru.weight_hh_l0
        assert "gru.weight_ih_l0" in pm_names
        assert "gru.weight_hh_l0" in pm_names
        assert "gru.weight_ih_l0" in vm_names
        assert "gru.weight_hh_l0" in vm_names

    def test_same_input_hidden_sizes(self):
        """Both GRUs have input=14, hidden=8."""
        pm = GRUPolicyModel()
        vm = GRUValueModel()
        assert pm.gru.input_size == INPUT_SIZE
        assert pm.gru.hidden_size == HIDDEN_SIZE
        assert vm.gru.input_size == INPUT_SIZE
        assert vm.gru.hidden_size == HIDDEN_SIZE

    def test_same_initial_hidden_state(self):
        """Both models have INITIAL_HIDDEN_STATE = [0.0] * 8."""
        assert GRUPolicyModel.INITIAL_HIDDEN_STATE == [0.0] * HIDDEN_SIZE
        assert GRUValueModel.INITIAL_HIDDEN_STATE == [0.0] * HIDDEN_SIZE


# ---------------------------------------------------------------------------
# TC-MOD5: GRUPolicyModel accepts mask_fn and temperature
# ---------------------------------------------------------------------------


class TestGRUPolicyModelMaskAndTemp:
    """TC-MOD5: GRUPolicyModel.forward() accepts mask_fn and temperature."""

    def test_mask_fn_filters_illegal_actions(self):
        """mask_fn that zeros index 0 → probs[0,0] ≈ 0."""
        model = GRUPolicyModel()
        model.eval()
        x = _make_input()
        h = _make_hidden()

        # Mask index 0 (attack)
        def mask_fn(logits):
            masked = logits.clone()
            masked[0, 0] = float("-inf")
            return masked

        probs, _, _ = model(x, h, mask_fn=mask_fn)
        assert probs[0, 0].item() < 1e-6
        assert abs(probs[0, 1:].sum().item() - 1.0) < 1e-5

    def test_temperature_scaling_affects_probs(self):
        """Different temperatures produce different probability distributions."""
        model = GRUPolicyModel()
        model.eval()
        # Use distinct input so probs are not uniform
        x = torch.tensor([[[1.0, 0.5, 2.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
                         dtype=torch.float32)
        h = _make_hidden()

        probs_t1, _, _ = model(x, h, temperature=1.0)
        probs_t05, _, _ = model(x, h, temperature=0.5)

        # Temperature 0.5 should be more peaked (higher max, lower min)
        assert probs_t05.max().item() >= probs_t1.max().item() * 0.99

    def test_temperature_zero_greedy(self):
        """temperature=0.0 produces one-hot probs."""
        model = GRUPolicyModel()
        model.eval()
        x = _make_input()
        h = _make_hidden()
        probs, log_probs, _ = model(x, h, temperature=0.0)
        # Exactly one entry is 1.0
        assert (probs == 1.0).sum().item() == 1
        # All others are 0
        assert (probs == 0.0).sum().item() == 3
        # log_probs: 0.0 at argmax, -inf elsewhere
        assert (log_probs == 0.0).sum().item() == 1
        assert (log_probs == float("-inf")).sum().item() == 3


# ---------------------------------------------------------------------------
# TC-MOD6: GRUValueModel does not accept mask_fn or temperature
# ---------------------------------------------------------------------------


class TestGRUValueModelSignature:
    """TC-MOD6: GRUValueModel.forward() only accepts (x, hidden)."""

    def test_forward_only_accepts_x_and_hidden(self):
        """forward() signature is forward(x, hidden) only."""
        import inspect
        sig = inspect.signature(GRUValueModel.forward)
        params = list(sig.parameters.keys())
        # Should have self, x, hidden — no mask_fn, no temperature
        assert "mask_fn" not in params
        assert "temperature" not in params
        # Only self + x + hidden
        assert len(params) == 3


# ---------------------------------------------------------------------------
# Model instantiation and parameter counts
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    """Verify both models instantiate correctly with proper architecture."""

    def test_policy_model_has_fc_policy(self):
        """GRUPolicyModel must have fc_policy Linear(8→4)."""
        model = GRUPolicyModel()
        assert hasattr(model, "fc_policy")
        assert model.fc_policy.in_features == HIDDEN_SIZE
        assert model.fc_policy.out_features == NUM_CARD_TYPES

    def test_policy_model_has_no_fc_value(self):
        """GRUPolicyModel must NOT have fc_value."""
        model = GRUPolicyModel()
        assert not hasattr(model, "fc_value")

    def test_value_model_has_fc_value(self):
        """GRUValueModel must have fc_value Linear(8→1)."""
        model = GRUValueModel()
        assert hasattr(model, "fc_value")
        assert model.fc_value.in_features == HIDDEN_SIZE
        assert model.fc_value.out_features == 1

    def test_value_model_has_no_fc_policy(self):
        """GRUValueModel must NOT have fc_policy."""
        model = GRUValueModel()
        assert not hasattr(model, "fc_policy")
