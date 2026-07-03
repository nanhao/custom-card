"""Unit tests for ai/rnn_v1/model.py — GRUCardModel.

Covers TC-M1 through TC-M11 from design/3 test_cases.md §1, plus
new TC-M1 through TC-M7 from design/5 test_cases.md §1 (temperature).
Tests model instantiation, INITIAL_HIDDEN_STATE constant, forward pass
with and without mask_fn, temperature scaling, determinism, inference
mode, and weight file loading.

Per design/3 architecture D1, model.py contains only GRUCardModel —
all helper functions and game logic have moved to app/Converter and
app/AppManager.
"""

import pytest
import torch

from src.backend.ai.rnn_v1.model import (
    GRUCardModel,
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_CARD_TYPES,
    WEIGHT_PATH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_weight_path() -> str | None:
    """Resolve the model weight file path, returning None if not found."""
    import os

    weight_path = WEIGHT_PATH
    if os.path.exists(weight_path):
        return weight_path
    # Try resolving from the test's working directory
    weight_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "backend",
        "ai", "rnn_v1", "gen1", "rnn_v1.pth"
    )
    weight_path = os.path.abspath(weight_path)
    return weight_path if os.path.exists(weight_path) else None


# ---------------------------------------------------------------------------
# TC-M1: Model instantiation [EXISTING, updated]
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    """TC-M1: Model instantiation and architecture verification."""

    def test_model_has_gru_layer(self):
        """Model must have a GRU layer with input=14, hidden=8."""
        model = GRUCardModel()
        assert hasattr(model, "gru")
        assert model.gru.input_size == INPUT_SIZE  # 14
        assert model.gru.hidden_size == HIDDEN_SIZE  # 8

    def test_model_has_fc_policy_layer(self):
        """Model must have fc_policy Linear(8→4)."""
        model = GRUCardModel()
        assert hasattr(model, "fc_policy")
        assert model.fc_policy.in_features == HIDDEN_SIZE  # 8
        assert model.fc_policy.out_features == NUM_CARD_TYPES  # 4

    def test_model_has_fc_value_layer(self):
        """Model must have fc_value Linear(8→1)."""
        model = GRUCardModel()
        assert hasattr(model, "fc_value")
        assert model.fc_value.in_features == HIDDEN_SIZE  # 8
        assert model.fc_value.out_features == 1

    def test_model_has_parameters(self):
        """Model state_dict must be non-empty (all parameters exist)."""
        model = GRUCardModel()
        state_dict = model.state_dict()
        assert len(state_dict) > 0
        # Should have at least GRU weights + policy weights + value weights
        assert any("gru" in k for k in state_dict)
        assert any("fc_policy" in k for k in state_dict)
        assert any("fc_value" in k for k in state_dict)


# ---------------------------------------------------------------------------
# TC-M2: INITIAL_HIDDEN_STATE constant
# ---------------------------------------------------------------------------


class TestInitialHiddenState:
    """TC-M2: INITIAL_HIDDEN_STATE is a class constant returning [0.0]*8."""

    def test_is_list_of_8_zeros(self):
        """INITIAL_HIDDEN_STATE must be [0.0] * 8."""
        result = GRUCardModel.INITIAL_HIDDEN_STATE
        assert isinstance(result, list)
        assert len(result) == HIDDEN_SIZE  # 8
        assert result == [0.0] * HIDDEN_SIZE

    def test_all_elements_are_float(self):
        """All elements must be float type, not int."""
        result = GRUCardModel.INITIAL_HIDDEN_STATE
        for val in result:
            assert isinstance(val, float)

    def test_is_class_attribute_not_instance(self):
        """INITIAL_HIDDEN_STATE must be accessible without instantiation."""
        # Access as class attribute directly (no instance needed)
        assert GRUCardModel.INITIAL_HIDDEN_STATE == [0.0] * 8


# ---------------------------------------------------------------------------
# TC-M3: forward pass shapes (no mask_fn)
# ---------------------------------------------------------------------------


class TestForwardShapes:
    """TC-M3: Forward pass produces correct output shapes and value range."""

    def test_policy_output_shape(self):
        """Policy output must be (1, 4) — probabilities after softmax."""
        model = GRUCardModel()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, _log_probs, value, new_h = model(x, h)
        assert probs.shape == (1, NUM_CARD_TYPES)

    def test_value_output_shape(self):
        """Value output must be (1, 1)."""
        model = GRUCardModel()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, _log_probs, value, new_h = model(x, h)
        assert value.shape == (1, 1)

    def test_value_in_range(self):
        """Value output must be in [-1, 1] after tanh."""
        model = GRUCardModel()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, _log_probs, value, new_h = model(x, h)
        assert -1.0 <= value.item() <= 1.0

    def test_new_hidden_shape(self):
        """New hidden state must be (1, 1, 8)."""
        model = GRUCardModel()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, _log_probs, value, new_h = model(x, h)
        assert new_h.shape == (1, 1, HIDDEN_SIZE)

    def test_probs_sum_to_one(self):
        """Probabilities from softmax must sum to ~1.0."""
        model = GRUCardModel()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, _log_probs, _, _ = model(x, h)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_returns_tuple_of_four(self):
        """forward() must return a 4-tuple of tensors: (probs, log_probs, value, new_hidden)."""
        model = GRUCardModel()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        result = model(x, h)
        assert isinstance(result, tuple)
        assert len(result) == 4
        probs, log_probs, value, new_hidden = result
        assert probs.shape == (1, NUM_CARD_TYPES)
        assert log_probs.shape == (1, NUM_CARD_TYPES)
        assert value.shape == (1, 1)
        assert new_hidden.shape == (1, 1, HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# TC-M4, TC-M5, TC-M6: forward pass with mask_fn
# ---------------------------------------------------------------------------


class TestForwardWithMaskFn:
    """TC-M4–TC-M6: mask_fn parameter in forward()."""

    @staticmethod
    def _mask_index_0(logits: torch.Tensor) -> torch.Tensor:
        """Mask function that sets index 0 (attack) to -inf."""
        logits = logits.clone()
        logits[0, 0] = float("-inf")
        return logits

    @staticmethod
    def _mask_all_but_index_2(logits: torch.Tensor) -> torch.Tensor:
        """Mask function that only allows index 2 (heavy)."""
        result = torch.full_like(logits, float("-inf"))
        result[0, 2] = logits[0, 2]
        return result

    def test_mask_fn_sets_index_0_to_zero(self):
        """TC-M4: mask_fn that zeros index 0 → probs[0,0] ≈ 0."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, _log_probs, value, new_h = model(x, h, mask_fn=self._mask_index_0)
        # Index 0 should be ~0 after masking
        assert probs[0, 0].item() < 1e-6, f"Expected ~0, got {probs[0, 0].item()}"
        # Non-masked indices should sum to ~1
        assert abs(probs[0, 1:].sum().item() - 1.0) < 1e-5

    def test_mask_fn_returns_only_legal_card(self):
        """TC-M5: mask all but index 2 → only index 2 has prob ≈ 1."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, _log_probs, value, new_h = model(x, h, mask_fn=self._mask_all_but_index_2)
        assert abs(probs[0, 2].item() - 1.0) < 1e-5
        assert probs[0, 0].item() < 1e-6
        assert probs[0, 1].item() < 1e-6
        assert probs[0, 3].item() < 1e-6

    def test_no_mask_fn_all_probs_positive(self):
        """TC-M6: No mask_fn → all four probs > 0."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, _log_probs, _, _ = model(x, h)  # No mask_fn
        for i in range(4):
            assert probs[0, i].item() > 0, f"Prob at index {i} should be > 0"

    def test_mask_fn_receives_logits_of_correct_shape(self):
        """mask_fn must receive a (1, 4) tensor."""
        model = GRUCardModel()
        model.eval()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        received_shapes = []

        def tracking_mask_fn(logits: torch.Tensor) -> torch.Tensor:
            received_shapes.append(logits.shape)
            return logits  # pass-through

        model(x, h, mask_fn=tracking_mask_fn)
        assert len(received_shapes) == 1
        assert received_shapes[0] == (1, NUM_CARD_TYPES)


# ---------------------------------------------------------------------------
# TC-M7: forward pass under inference_mode
# ---------------------------------------------------------------------------


class TestInferenceMode:
    """TC-M7: Forward pass runs without gradient tracking."""

    def test_forward_runs_under_inference_mode(self):
        """Under torch.inference_mode(), no gradients are tracked."""
        model = GRUCardModel()
        model.eval()
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            probs, _log_probs, value, new_h = model(x, h)

        # Verify outputs exist and are valid
        assert probs is not None
        assert value is not None
        assert new_h is not None
        # Under inference_mode, tensors should not have grad_fn
        assert probs.grad_fn is None
        assert value.grad_fn is None


# ---------------------------------------------------------------------------
# TC-M8: Deterministic output with same inputs
# ---------------------------------------------------------------------------


class TestDeterminism:
    """TC-M8: Same model + same input = same output (eval mode)."""

    def test_deterministic_output(self):
        """Two forward passes with same input produce identical results."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            probs1, _, val1, h1 = model(x, h)
            probs2, _, val2, h2 = model(x, h)

        assert torch.allclose(probs1, probs2)
        assert torch.allclose(val1, val2)
        assert torch.allclose(h1, h2)


# ---------------------------------------------------------------------------
# TC-M9: Weight file loading
# ---------------------------------------------------------------------------


class TestWeightFileLoading:
    """TC-M9: Weight file loads successfully and produces usable model.

    Note: Actual weight loading is orchestrated by AppManager per design D7.
    This test verifies that the weight file is compatible with the model
    architecture.
    """

    def test_weights_load_into_model(self):
        """Loading the weight file into a GRUCardModel should succeed."""
        weight_path = _resolve_weight_path()
        if weight_path is None:
            pytest.skip("Weight file not found")

        model = GRUCardModel()
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Verify model produces valid output after loading
        x = torch.zeros(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        with torch.inference_mode():
            probs, _log_probs, value, _ = model(x, h)
        assert probs.shape == (1, NUM_CARD_TYPES)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_loaded_model_is_in_eval_mode(self):
        """After loading weights, model should be usable in eval mode."""
        weight_path = _resolve_weight_path()
        if weight_path is None:
            pytest.skip("Weight file not found")

        model = GRUCardModel()
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        assert not model.training, "Model should be in eval mode"


# ---------------------------------------------------------------------------
# TC-M10: Corrupted / nonexistent weight file
# ---------------------------------------------------------------------------


class TestCorruptedWeightFile:
    """TC-M10: Nonexistent weight files produce clear error messages."""

    def test_raises_on_nonexistent_file(self):
        """Loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            torch.load("/nonexistent/path/model.pth", map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# TC-M11: Different models produce different outputs
# ---------------------------------------------------------------------------


class TestDifferentModels:
    """TC-M11: Different random seeds produce different model parameters."""

    def test_different_seeds_produce_different_weights(self):
        """Two models with different seeds have different parameters."""
        torch.manual_seed(42)
        m1 = GRUCardModel()
        torch.manual_seed(99)
        m2 = GRUCardModel()

        # Compare a key parameter
        w1 = m1.fc_policy.weight.data.clone()
        w2 = m2.fc_policy.weight.data.clone()
        assert not torch.allclose(w1, w2), (
            "Different seeds should produce different weights"
        )


# ---------------------------------------------------------------------------
# Verify model.py has zero game/ imports
# ---------------------------------------------------------------------------


class TestModelPurity:
    """Per design/3 D1: model.py must have zero imports from game/, app/, controller/."""

    def test_no_game_imports(self):
        """model.py must not import from src.backend.game."""
        import src.backend.ai.rnn_v1.model as model_module
        import inspect

        source = inspect.getsource(model_module)
        assert "src.backend.game" not in source, (
            "model.py must not import from game/ (per design D1)"
        )

    def test_no_app_imports(self):
        """model.py must not import from src.backend.app."""
        import src.backend.ai.rnn_v1.model as model_module
        import inspect

        source = inspect.getsource(model_module)
        assert "src.backend.app" not in source, (
            "model.py must not import from app/ (per design D1)"
        )

    def test_no_controller_imports(self):
        """model.py must not import from src.backend.controller."""
        import src.backend.ai.rnn_v1.model as model_module
        import inspect

        source = inspect.getsource(model_module)
        assert "src.backend.controller" not in source, (
            "model.py must not import from controller/ (per design D1)"
        )

    def test_model_module_has_no_select_action(self):
        """model.py must NOT expose select_action (moved to AppManager)."""
        import src.backend.ai.rnn_v1.model as model_module
        assert not hasattr(model_module, "select_action"), (
            "select_action() moved to AppManager"
        )

    def test_model_module_has_no_init_hidden_state_function(self):
        """init_hidden_state() is removed, replaced by class constant."""
        import src.backend.ai.rnn_v1.model as model_module
        assert not hasattr(model_module, "init_hidden_state"), (
            "init_hidden_state() replaced by GRUCardModel.INITIAL_HIDDEN_STATE"
        )

    def test_model_module_has_no_build_input_vector(self):
        """_build_input_vector() moved to Converter."""
        import src.backend.ai.rnn_v1.model as model_module
        assert not hasattr(model_module, "_build_input_vector"), (
            "_build_input_vector() moved to Converter"
        )

    def test_model_module_has_no_mask_illegal_actions(self):
        """_mask_illegal_actions() replaced by mask_fn in AppManager."""
        import src.backend.ai.rnn_v1.model as model_module
        assert not hasattr(model_module, "_mask_illegal_actions"), (
            "_mask_illegal_actions() replaced by mask_fn closure"
        )

    def test_model_module_has_no_index_to_card(self):
        """_index_to_card() moved to Converter."""
        import src.backend.ai.rnn_v1.model as model_module
        assert not hasattr(model_module, "_index_to_card"), (
            "_index_to_card() moved to Converter"
        )

    def test_model_module_has_no_card_to_index(self):
        """_card_to_index() moved to Converter."""
        import src.backend.ai.rnn_v1.model as model_module
        assert not hasattr(model_module, "_card_to_index"), (
            "_card_to_index() moved to Converter"
        )


# ---------------------------------------------------------------------------
# New tests from design/5: Temperature in forward() (TC-M1 through TC-M7)
# ---------------------------------------------------------------------------


class TestTemperatureDefault:
    """TC-M1: temperature=1.0 produces same probs as default (no temperature)."""

    def test_temperature_1_equals_default(self):
        """forward(x, h, temperature=1.0) == forward(x, h)."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            probs1, log_probs1, val1, h1 = model(x, h)
            probs2, log_probs2, val2, h2 = model(x, h, temperature=1.0)

        assert torch.allclose(probs1, probs2)
        assert torch.allclose(log_probs1, log_probs2)
        assert torch.allclose(val1, val2)
        assert torch.allclose(h1, h2)

    def test_log_probs_equal_ln_of_probs_at_temp_1(self):
        """At temperature=1.0, log_probs should equal ln(probs)."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            probs, log_probs, _, _ = model(x, h, temperature=1.0)

        assert torch.allclose(log_probs, torch.log(probs))


class TestTemperatureGreedy:
    """TC-M2: temperature=0 produces one-hot probs (greedy)."""

    def test_temperature_0_produces_one_hot(self):
        """Probs should be one-hot at the argmax position."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            probs, log_probs, _, _ = model(x, h, temperature=0)

        # One value should be ~1.0, others ~0.0
        max_idx = int(torch.argmax(probs).item())
        for i in range(NUM_CARD_TYPES):
            if i == max_idx:
                assert abs(probs[0, i].item() - 1.0) < 1e-5
            else:
                assert abs(probs[0, i].item() - 0.0) < 1e-5

    def test_temperature_0_log_probs(self):
        """Log probs: 0.0 at argmax, -inf elsewhere."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            _, log_probs, _, _ = model(x, h, temperature=0)

        max_idx = int(torch.argmax(log_probs).item())
        assert abs(log_probs[0, max_idx].item() - 0.0) < 1e-5
        for i in range(NUM_CARD_TYPES):
            if i != max_idx:
                assert log_probs[0, i].item() == float("-inf")


class TestTemperatureFlatten:
    """TC-M3: temperature > 1.0 flattens distribution toward uniform."""

    def test_high_temperature_approaches_uniform(self):
        """At temperature=100.0, all probs should be close to 0.25."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            probs, _, _, _ = model(x, h, temperature=100.0)

        for i in range(NUM_CARD_TYPES):
            assert abs(probs[0, i].item() - 0.25) < 0.05


class TestTemperatureSharpen:
    """TC-M4: temperature < 1.0 sharpens distribution."""

    def test_low_temperature_peaks_higher(self):
        """max(probs) at temp=0.1 should be larger than at temp=1.0."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            probs1, _, _, _ = model(x, h, temperature=1.0)
            probs2, _, _, _ = model(x, h, temperature=0.1)

        assert probs2.max().item() > probs1.max().item()


class TestTemperatureWithMask:
    """TC-M5: mask_fn still works with temperature."""

    @staticmethod
    def _mask_index_0_and_1(logits: torch.Tensor) -> torch.Tensor:
        """Mask function that zeros indices 0 and 1."""
        logits = logits.clone()
        logits[0, 0] = float("-inf")
        logits[0, 1] = float("-inf")
        return logits

    def test_masked_indices_zero_with_temperature(self):
        """Indices 0 and 1 should have prob 0 even with temperature."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            probs, _, _, _ = model(
                x, h, mask_fn=self._mask_index_0_and_1, temperature=2.0
            )

        assert probs[0, 0].item() < 1e-6
        assert probs[0, 1].item() < 1e-6
        # Remaining indices should sum to ~1
        assert abs(probs[0, 2:].sum().item() - 1.0) < 1e-5


class TestTemperatureValueUnchanged:
    """TC-M6: value head is unchanged by temperature."""

    def test_value_same_for_different_temperatures(self):
        """Same input, different temperature → same value."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            _, _, val0, _ = model(x, h, temperature=0.0)
            _, _, val3, _ = model(x, h, temperature=3.0)

        assert torch.allclose(val0, val3)


class TestTemperatureHiddenUnchanged:
    """TC-M7: new_hidden is unchanged by temperature."""

    def test_hidden_same_for_different_temperatures(self):
        """Same input, different temperature → same new_hidden."""
        model = GRUCardModel()
        model.eval()
        x = torch.ones(1, 1, INPUT_SIZE) * 0.5
        h = torch.zeros(1, 1, HIDDEN_SIZE)

        with torch.inference_mode():
            _, _, _, h0 = model(x, h, temperature=0.0)
            _, _, _, h3 = model(x, h, temperature=3.0)

        assert torch.allclose(h0, h3)
