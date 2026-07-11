"""Unit tests for rnn_v2 skip-connection models (design/34 test_cases.md, design/37).

Per design/37: INPUT_SIZE 56→62, POLICY_OUTPUT_SIZE 14→24, NUM_CARD_TYPES 7→8.
"""

import torch
import pytest
from collections.abc import Callable

from src.backend.ai.rnn_v2.model import (
    GRUPolicyModelV2,
    GRUValueModelV2,
    INPUT_SIZE,
    HIDDEN_SIZE,
    POLICY_OUTPUT_SIZE,
    NUM_CARD_TYPES,
)


# ===========================================================================
# 1. GRUPolicyModelV2 — TC-PV2-01 ~ TC-PV2-07
# ===========================================================================


class TestPolicyModelV2:
    """TC-PV2: GRUPolicyModelV2 structure and forward propagation."""

    def test_constants(self):
        """TC-PV2-01: INPUT_SIZE=62, HIDDEN_SIZE=32, POLICY_OUTPUT_SIZE=24, NUM_CARD_TYPES=8 (design/37)."""
        assert INPUT_SIZE == 62
        assert HIDDEN_SIZE == 32
        assert POLICY_OUTPUT_SIZE == 24
        assert NUM_CARD_TYPES == 8

    def test_skip_input_size_and_fc1(self):
        """TC-IS-04/05: SKIP_INPUT_SIZE=94 (=62+32), fc1.in_features=94 (design/37)."""
        model = GRUPolicyModelV2()
        assert model.fc1.in_features == 94, f"Expected fc1.in_features=94, got {model.fc1.in_features}"
        assert model.fc2.in_features == 64
        assert model.fc_policy.out_features == POLICY_OUTPUT_SIZE
        value_model = GRUValueModelV2()
        assert value_model.fc1.in_features == 94

    def test_forward_output_shapes(self):
        """TC-PV2-02: probs=(1,24), log_probs=(1,24), new_hidden=(1,1,32), sums ≈ 1.0."""
        model = GRUPolicyModelV2()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.randn(1, 1, HIDDEN_SIZE)

        probs, log_probs, new_hidden = model.forward(x, hidden)

        assert probs.shape == (1, POLICY_OUTPUT_SIZE)
        assert log_probs.shape == (1, POLICY_OUTPUT_SIZE)
        assert new_hidden.shape == (1, 1, HIDDEN_SIZE)
        # Probability mass ≈ 1.0
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(torch.exp(log_probs).sum(), torch.tensor(1.0), atol=1e-5)

    def test_temperature_zero_greedy(self):
        """TC-PV2-03: temperature=0 → exactly one entry = 1.0, rest = 0.0."""
        model = GRUPolicyModelV2()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.zeros(1, 1, HIDDEN_SIZE)

        probs, log_probs, _ = model.forward(x, hidden, temperature=0.0)

        ones = (probs == 1.0).sum().item()
        zeros = (probs == 0.0).sum().item()
        assert ones == 1
        assert zeros == POLICY_OUTPUT_SIZE - 1
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        # Corresponding log_prob entry = 0.0, rest = -inf
        assert (log_probs == 0.0).sum().item() == 1
        assert (log_probs == float("-inf")).sum().item() == POLICY_OUTPUT_SIZE - 1

    def test_temperature_positive_sampling(self):
        """TC-PV2-04: temperature=1.0 → all probs > 0, sum=1, no NaN/inf in log_probs."""
        model = GRUPolicyModelV2()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.zeros(1, 1, HIDDEN_SIZE)

        probs, log_probs, _ = model.forward(x, hidden, temperature=1.0)

        assert (probs > 0).all()
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()

    def test_mask_fn_applied(self):
        """TC-PV2-05: mask_fn setting index 2 to -inf → probs[0,2] ≈ 0.0."""

        def mask_fn(logits: torch.Tensor) -> torch.Tensor:
            logits[:, 2] = float("-inf")
            return logits

        model = GRUPolicyModelV2()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.zeros(1, 1, HIDDEN_SIZE)

        probs, _, _ = model.forward(x, hidden, mask_fn=mask_fn, temperature=1.0)

        assert probs[0, 2].item() == pytest.approx(0.0, abs=1e-5)

    def test_initial_hidden_state(self):
        """TC-PV2-06: INITIAL_HIDDEN_STATE = [0.0]*32."""
        state = GRUPolicyModelV2.INITIAL_HIDDEN_STATE
        assert len(state) == HIDDEN_SIZE
        assert all(v == 0.0 for v in state)

    def test_hidden_state_propagates(self):
        """TC-PV2-07: consecutive forward calls produce different hidden states, no NaN."""
        model = GRUPolicyModelV2()
        x1 = torch.randn(1, 1, INPUT_SIZE)
        x2 = torch.randn(1, 1, INPUT_SIZE)
        h0 = torch.zeros(1, 1, HIDDEN_SIZE)

        _, _, h1 = model.forward(x1, h0)
        _, _, h2 = model.forward(x2, h1)

        assert not torch.equal(h1, h0)
        assert not torch.equal(h2, h1)
        assert not torch.isnan(h1).any()
        assert not torch.isnan(h2).any()


# ===========================================================================
# 2. GRUValueModelV2 — TC-VV2-01 ~ TC-VV2-04
# ===========================================================================


class TestValueModelV2:
    """TC-VV2: GRUValueModelV2 structure and forward propagation."""

    def test_forward_output_shapes(self):
        """TC-VV2-01: value=(1,1), new_hidden=(1,1,32), finite value."""
        model = GRUValueModelV2()
        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.randn(1, 1, HIDDEN_SIZE)

        value, new_hidden = model.forward(x, hidden)

        assert value.shape == (1, 1)
        assert new_hidden.shape == (1, 1, HIDDEN_SIZE)
        assert torch.isfinite(value).all()

    def test_initial_hidden_state(self):
        """TC-VV2-02: INITIAL_HIDDEN_STATE = [0.0]*32."""
        state = GRUValueModelV2.INITIAL_HIDDEN_STATE
        assert len(state) == HIDDEN_SIZE
        assert all(v == 0.0 for v in state)

    def test_value_no_tanh_unbounded(self):
        """TC-VV2-03: large input → value is finite, may be outside [-1,1]."""
        model = GRUValueModelV2()
        x = torch.ones(1, 1, INPUT_SIZE) * 100.0
        hidden = torch.zeros(1, 1, HIDDEN_SIZE)

        value, _ = model.forward(x, hidden)

        assert torch.isfinite(value).all()
        # No assertion on range — value is not constrained by tanh

    def test_hidden_state_propagates(self):
        """TC-VV2-04: consecutive forward calls produce different hidden states, no NaN."""
        model = GRUValueModelV2()
        x1 = torch.randn(1, 1, INPUT_SIZE)
        x2 = torch.randn(1, 1, INPUT_SIZE)
        h0 = torch.zeros(1, 1, HIDDEN_SIZE)

        _, h1 = model.forward(x1, h0)
        _, h2 = model.forward(x2, h1)

        assert not torch.equal(h1, h0)
        assert not torch.equal(h2, h1)
        assert not torch.isnan(h1).any()
        assert not torch.isnan(h2).any()


# ===========================================================================
# 3. Skip Connection — TC-SC-01 ~ TC-SC-02
# ===========================================================================


class TestSkipConnection:
    """TC-SC: Verify skip connection makes head sensitive to input variation."""

    def test_different_input_different_probs(self):
        """TC-SC-01: all-zero vs all-one input → different probs distributions."""
        model = GRUPolicyModelV2()
        x_a = torch.zeros(1, 1, INPUT_SIZE)
        x_b = torch.ones(1, 1, INPUT_SIZE)
        h0 = torch.zeros(1, 1, HIDDEN_SIZE)

        probs_a, _, _ = model.forward(x_a, h0)
        probs_b, _, _ = model.forward(x_b, h0)

        assert not torch.allclose(probs_a, probs_b)

    def test_hand_card_segment_affects_policy(self):
        """TC-SC-02: changing indices 6-13 (own hand, 8 types) → different probs distribution."""
        model = GRUPolicyModelV2()
        x_normal = torch.zeros(1, 1, INPUT_SIZE)
        x_normal[0, 0, 6] = 3.0  # 3 attack cards in hand
        x_normal[0, 0, 7] = 2.0  # 2 defense cards in hand

        x_no_hand = torch.zeros(1, 1, INPUT_SIZE)
        # indices 6-13 remain all zeros (empty hand)

        h0 = torch.zeros(1, 1, HIDDEN_SIZE)
        probs_normal, _, _ = model.forward(x_normal, h0)
        probs_no_hand, _, _ = model.forward(x_no_hand, h0)

        assert not torch.allclose(probs_normal, probs_no_hand)


# ===========================================================================
# 4. v1 Compatibility — TC-CP-01 ~ TC-CP-02
# ===========================================================================


class TestV1Compatibility:
    """TC-CP: v2 models accept same input/output shapes as v1."""

    def test_policy_v1_input_shape_works_on_v2(self):
        """TC-CP-01: same (1,1,62) input works on v2 policy, output shapes match v1."""
        from src.backend.ai.rnn_v1.model import GRUPolicyModel

        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.zeros(1, 1, HIDDEN_SIZE)

        model_v1 = GRUPolicyModel()
        model_v2 = GRUPolicyModelV2()

        probs_v1, log_probs_v1, h_v1 = model_v1.forward(x, hidden)
        probs_v2, log_probs_v2, h_v2 = model_v2.forward(x, hidden)

        assert probs_v1.shape == probs_v2.shape == (1, POLICY_OUTPUT_SIZE)
        assert log_probs_v1.shape == log_probs_v2.shape == (1, POLICY_OUTPUT_SIZE)
        assert h_v1.shape == h_v2.shape == (1, 1, HIDDEN_SIZE)

    def test_value_v1_input_shape_works_on_v2(self):
        """TC-CP-01: same (1,1,62) input works on v2 value, output shapes match v1."""
        from src.backend.ai.rnn_v1.model import GRUValueModel

        x = torch.randn(1, 1, INPUT_SIZE)
        hidden = torch.zeros(1, 1, HIDDEN_SIZE)

        model_v1 = GRUValueModel()
        model_v2 = GRUValueModelV2()

        value_v1, h_v1 = model_v1.forward(x, hidden)
        value_v2, h_v2 = model_v2.forward(x, hidden)

        assert value_v1.shape == value_v2.shape == (1, 1)
        assert h_v1.shape == h_v2.shape == (1, 1, HIDDEN_SIZE)

    def test_v2_importable_via_registry_dynamic_import(self):
        """TC-CP-02: rnn_v2 model can be imported dynamically like v1."""
        import importlib

        module = importlib.import_module("src.backend.ai.rnn_v2.model")
        policy_class = getattr(module, "GRUPolicyModelV2")
        value_class = getattr(module, "GRUValueModelV2")

        policy_model = policy_class()
        value_model = value_class()

        assert isinstance(policy_model, torch.nn.Module)
        assert isinstance(value_model, torch.nn.Module)

        # Smoke test: can call forward
        x = torch.randn(1, 1, INPUT_SIZE)
        h = torch.zeros(1, 1, HIDDEN_SIZE)
        probs, log_probs, _ = policy_model.forward(x, h)
        value, _ = value_model.forward(x, h)
        assert probs.shape == (1, POLICY_OUTPUT_SIZE)
        assert value.shape == (1, 1)


# ===========================================================================
# 5. Batch vs Step-by-Step Equivalence (TC-SEQ)
# ===========================================================================


class TestV2SequenceEquivalence:
    """Verify that batch internal layer access equals step-by-step forward().

    V2TrainingRunner uses direct internal layer access (model.gru + model.fc1 +
    model.fc2 + model.fc_*) for GPU-efficient batch training. This test ensures
    that approach produces the same output as calling model.forward() per timestep.
    """

    def test_value_batch_equals_stepwise(self):
        """V2 value: batch internal layers == step-by-step forward()."""
        torch.manual_seed(123)
        model = GRUValueModelV2()
        model.eval()

        T = 5
        x_seq = torch.randn(T, 1, INPUT_SIZE)

        # Step-by-step via model.forward()
        h_step = torch.zeros(1, 1, HIDDEN_SIZE)
        step_vals = []
        for t in range(T):
            x_t = x_seq[t:t + 1]
            val, h_step = model.forward(x_t, h_step)
            step_vals.append(val.item())

        # Batch via internal layers (same as V2TrainingRunner)
        h_seq = torch.zeros(1, 1, HIDDEN_SIZE)
        with torch.no_grad():
            gru_out, _ = model.gru(x_seq, h_seq)
            combined = torch.cat(
                [gru_out.squeeze(1), x_seq.squeeze(1)], dim=-1,
            )
            h1 = torch.relu(model.fc1(combined))
            h2 = torch.relu(model.fc2(h1))
            seq_vals = model.fc_value(h2).squeeze()

        for t in range(T):
            assert abs(step_vals[t] - seq_vals[t].item()) < 1e-5, (
                f"Step {t}: step={step_vals[t]:.6f}, batch={seq_vals[t].item():.6f}"
            )

    def test_policy_batch_equals_stepwise(self):
        """V2 policy: batch internal layers == step-by-step forward()."""
        torch.manual_seed(123)
        model = GRUPolicyModelV2()
        model.eval()

        T = 5
        x_seq = torch.randn(T, 1, INPUT_SIZE)

        # Step-by-step via model.forward()
        h_step = torch.zeros(1, 1, HIDDEN_SIZE)
        step_log_probs = []
        for t in range(T):
            x_t = x_seq[t:t + 1]
            probs, log_probs, h_step = model.forward(
                x_t, h_step, mask_fn=None, temperature=1.0,
            )
            step_log_probs.append(log_probs)

        # Batch via internal layers (same as V2TrainingRunner)
        h_seq = torch.zeros(1, 1, HIDDEN_SIZE)
        with torch.no_grad():
            gru_out, _ = model.gru(x_seq, h_seq)
            combined = torch.cat(
                [gru_out.squeeze(1), x_seq.squeeze(1)], dim=-1,
            )
            h1 = torch.relu(model.fc1(combined))
            h2 = torch.relu(model.fc2(h1))
            policy_logits = model.fc_policy(h2)
            seq_log_probs = torch.log_softmax(policy_logits, dim=-1)

        for t in range(T):
            diff = (
                (seq_log_probs[t] - step_log_probs[t].squeeze(0)).abs().max().item()
            )
            assert diff < 1e-5, (
                f"Step {t}: max log_prob diff={diff:.6f}"
            )
