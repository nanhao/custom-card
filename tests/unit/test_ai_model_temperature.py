"""Unit tests for temperature scaling in GRUCardModel.forward().

Covers TC-M1 through TC-M7 from design/5 test_cases.md §1.
Tests temperature=1.0 equivalence, greedy one-hot, flatten,
sharpen, mask interaction, and value/hidden invariance.
"""

import pytest
import torch

from src.backend.ai.rnn_v1.model import (
    GRUCardModel,
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_CARD_TYPES,
)


# ---------------------------------------------------------------------------
# TC-M1: temperature=1.0 produces same probs as default
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


# ---------------------------------------------------------------------------
# TC-M2: temperature=0 produces one-hot probs (greedy)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TC-M3: temperature > 1.0 flattens distribution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TC-M4: temperature < 1.0 sharpens distribution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TC-M5: mask_fn still works with temperature
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TC-M6: value head unchanged by temperature
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TC-M7: new_hidden unchanged by temperature
# ---------------------------------------------------------------------------


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
