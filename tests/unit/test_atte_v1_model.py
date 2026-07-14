"""Unit tests for AtteV1PolicyModel and AtteV1ValueModel — design/40 T5-T7."""

import torch

from src.backend.ai.atte_v1.constants import (
    ACTION_VOCAB_SIZE,
    HIDDEN_SIZE,
    POLICY_OUTPUT_SIZE,
)
from src.backend.ai.atte_v1.model import (
    AtteV1PolicyModel,
    AtteV1ValueModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestModelHelpers:
    """Shared helpers for building model inputs."""

    @staticmethod
    def _make_batch_inputs(
        batch_size: int = 1,
        own_hand_ids: list[int] | None = None,
        own_discard_ids: list[int] | None = None,
        own_draw_ids: list[int] | None = None,
        own_last_action: int = 3,
        own_status_ids: list[int] | None = None,
        own_status_durations: list[int] | None = None,
        own_opp_hand_ids: list[int] | None = None,
        own_opp_discard_ids: list[int] | None = None,
        own_opp_draw_ids: list[int] | None = None,
        opp_hand_ids: list[int] | None = None,
        opp_discard_ids: list[int] | None = None,
        opp_draw_ids: list[int] | None = None,
        opp_last_action: int = 6,
        opp_status_ids: list[int] | None = None,
        opp_status_durations: list[int] | None = None,
        own_hp: float = 1.0,
        own_energy: float = 1.0,
        opp_hp: float = 1.0,
        opp_energy: float = 1.0,
        round_in_cycle: int = 0,
        hidden: list[float] | None = None,
    ) -> dict:
        """Build kwargs dict for model.forward()."""
        if hidden is None:
            hidden = [0.0] * HIDDEN_SIZE

        def _tensor(ids: list[int] | None) -> torch.Tensor:
            if ids is None:
                ids = [3]  # default single ATTACK
            return torch.tensor([ids], dtype=torch.long)

        def _scalar(val: float) -> torch.Tensor:
            return torch.tensor([[val]], dtype=torch.float32)

        def _hidden(val: list[float]) -> torch.Tensor:
            return torch.tensor(val, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)

        return {
            "own_hand_ids": _tensor(own_hand_ids),
            "own_discard_ids": _tensor(own_discard_ids or []),
            "own_draw_ids": _tensor(own_draw_ids or []),
            "own_last_action": torch.tensor([[own_last_action]], dtype=torch.long),
            "own_status_ids": _tensor(own_status_ids or []),
            "own_status_durations": torch.tensor(
                [own_status_durations or []], dtype=torch.float32,
            ),
            "own_opp_hand_ids": _tensor(own_opp_hand_ids or []),
            "own_opp_discard_ids": _tensor(own_opp_discard_ids or []),
            "own_opp_draw_ids": _tensor(own_opp_draw_ids or []),
            "opp_hand_ids": _tensor(opp_hand_ids or []),
            "opp_discard_ids": _tensor(opp_discard_ids or []),
            "opp_draw_ids": _tensor(opp_draw_ids or []),
            "opp_last_action": torch.tensor([[opp_last_action]], dtype=torch.long),
            "opp_status_ids": _tensor(opp_status_ids or []),
            "opp_status_durations": torch.tensor(
                [opp_status_durations or []], dtype=torch.float32,
            ),
            "own_hp": _scalar(own_hp),
            "own_energy": _scalar(own_energy),
            "opp_hp": _scalar(opp_hp),
            "opp_energy": _scalar(opp_energy),
            "round_in_cycle": _scalar(round_in_cycle),
            "hidden": _hidden(hidden),
        }


# ---------------------------------------------------------------------------
# T5: AtteV1PolicyModel
# ---------------------------------------------------------------------------


class TestAtteV1PolicyModel:
    """T5: AtteV1PolicyModel initialization and forward pass."""

    def test_instantiation(self) -> None:
        """T5.1: Model instantiates without error, parameters exist."""
        model = AtteV1PolicyModel()
        params = list(model.parameters())
        assert len(params) > 0

    def test_initial_hidden_state(self) -> None:
        """T5.2: INITIAL_HIDDEN_STATE is 32 zeros."""
        h = AtteV1PolicyModel.INITIAL_HIDDEN_STATE
        assert len(h) == 32
        assert all(v == 0.0 for v in h)

    def test_forward_shape(self) -> None:
        """T5.3: forward returns correct shapes."""
        model = AtteV1PolicyModel()
        inputs = TestModelHelpers._make_batch_inputs()
        probs, log_probs, new_hidden = model(**inputs)
        assert probs.shape == (1, 24)
        assert log_probs.shape == (1, 24)
        assert new_hidden.shape == (1, 1, 32)

    def test_probs_sum_to_one(self) -> None:
        """T5.4: probs sum ≈ 1.0 (softmax normalized)."""
        model = AtteV1PolicyModel()
        inputs = TestModelHelpers._make_batch_inputs()
        probs, _, _ = model(**inputs)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_empty_hand(self) -> None:
        """T5.5: empty hand does not crash."""
        model = AtteV1PolicyModel()
        inputs = TestModelHelpers._make_batch_inputs(own_hand_ids=[])
        probs, _, _ = model(**inputs)
        assert probs.shape == (1, 24)

    def test_empty_status(self) -> None:
        """T5.6: empty status does not crash."""
        model = AtteV1PolicyModel()
        inputs = TestModelHelpers._make_batch_inputs(
            own_status_ids=[], own_status_durations=[],
        )
        probs, _, _ = model(**inputs)
        assert probs.shape == (1, 24)

    def test_pad_last_action(self) -> None:
        """T5.7: last_action=24 (PAD) does not crash."""
        model = AtteV1PolicyModel()
        inputs = TestModelHelpers._make_batch_inputs(own_last_action=24)
        probs, _, _ = model(**inputs)
        assert probs.shape == (1, 24)

    def test_mask_fn(self) -> None:
        """T5.8: mask_fn blocks all non-pass → probs[0] ≈ 1.0."""
        model = AtteV1PolicyModel()

        def mask_all(logits: torch.Tensor) -> torch.Tensor:
            masked = logits.clone()
            masked[0, 1:] = float("-inf")
            return masked

        inputs = TestModelHelpers._make_batch_inputs()
        probs, _, _ = model(**inputs, mask_fn=mask_all)
        assert abs(probs[0, 0].item() - 1.0) < 1e-5

    def test_temperature_zero(self) -> None:
        """T5.9: temperature=0 → deterministic output (argmax)."""
        model = AtteV1PolicyModel()
        inputs = TestModelHelpers._make_batch_inputs()
        probs1, _, _ = model(**inputs, temperature=0.0)
        probs2, _, _ = model(**inputs, temperature=0.0)
        assert torch.equal(probs1, probs2)

    def test_temperature_high(self) -> None:
        """T5.10: high temperature → near-uniform distribution."""
        model = AtteV1PolicyModel()
        inputs = TestModelHelpers._make_batch_inputs()
        probs, _, _ = model(**inputs, temperature=100.0)
        # Each prob should be close to 1/24
        expected = 1.0 / 24
        for i in range(24):
            assert abs(probs[0, i].item() - expected) < 0.04

    def test_e1_vocab_size(self) -> None:
        """T5.11: e1 embedding weight shape is (25, 32)."""
        model = AtteV1PolicyModel()
        assert model.e1.weight.shape == (ACTION_VOCAB_SIZE, 32)

    def test_all_empty_input(self) -> None:
        """T5.12: all sequences empty, only scalar features → no crash."""
        model = AtteV1PolicyModel()
        inputs = TestModelHelpers._make_batch_inputs(
            own_hand_ids=[],
            own_discard_ids=[],
            own_draw_ids=[],
            own_status_ids=[],
            own_status_durations=[],
            own_opp_hand_ids=[],
            own_opp_discard_ids=[],
            own_opp_draw_ids=[],
            opp_hand_ids=[],
            opp_discard_ids=[],
            opp_draw_ids=[],
            opp_status_ids=[],
            opp_status_durations=[],
        )
        probs, log_probs, new_hidden = model(**inputs)
        assert probs.shape == (1, 24)
        assert log_probs.shape == (1, 24)
        assert new_hidden.shape == (1, 1, 32)


# ---------------------------------------------------------------------------
# T6: AtteV1ValueModel
# ---------------------------------------------------------------------------


class TestAtteV1ValueModel:
    """T6: AtteV1ValueModel initialization and forward pass."""

    def test_instantiation(self) -> None:
        """T6.1: Model instantiates without error."""
        model = AtteV1ValueModel()
        assert len(list(model.parameters())) > 0

    def test_initial_hidden_state(self) -> None:
        """T6.2: INITIAL_HIDDEN_STATE is 32 zeros."""
        h = AtteV1ValueModel.INITIAL_HIDDEN_STATE
        assert len(h) == 32
        assert all(v == 0.0 for v in h)

    def test_forward_shape(self) -> None:
        """T6.3: forward returns (value(1,1), new_hidden(1,1,32))."""
        model = AtteV1ValueModel()
        inputs = TestModelHelpers._make_batch_inputs()
        # Remove mask_fn/temperature kwargs for value model
        value_inputs = {k: v for k, v in inputs.items()}
        value, new_hidden = model(**value_inputs)
        assert value.shape == (1, 1)
        assert new_hidden.shape == (1, 1, 32)

    def test_different_hidden_different_value(self) -> None:
        """T6.4: different hidden states → different values."""
        model = AtteV1ValueModel()
        inputs1 = TestModelHelpers._make_batch_inputs(
            hidden=[0.0] * HIDDEN_SIZE,
        )
        inputs2 = TestModelHelpers._make_batch_inputs(
            hidden=[1.0] * HIDDEN_SIZE,
        )
        v1, _ = model(**{k: v for k, v in inputs1.items()})
        v2, _ = model(**{k: v for k, v in inputs2.items()})
        assert not torch.equal(v1, v2)


# ---------------------------------------------------------------------------
# T7: Weight sharing
# ---------------------------------------------------------------------------


class TestWeightSharing:
    """T7: Policy Head FC shares weight with e1 embedding."""

    def test_e1_weight_shape(self) -> None:
        """T7.1: e1.weight shape is (25, 32)."""
        model = AtteV1PolicyModel()
        assert model.e1.weight.shape == (25, 32)

    def test_policy_fc_shape(self) -> None:
        """T7.2: Policy Head fc weight shape is (24, 32)."""
        model = AtteV1PolicyModel()
        assert model.fc_policy.weight.shape == (24, 32)
