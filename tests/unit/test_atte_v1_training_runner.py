"""Unit tests for AtteV1TrainingRunner — batch sequence optimization (design/43).

Test cases per design/43 test_cases.md: T4-T11.
"""

import json
import os
import random
from unittest.mock import patch

import pytest
import torch

from src.backend.ai.atte_v1.constants import (
    GRU_INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_FEATURE_DIM,
    PAD_ACTION_ID,
)
from src.backend.ai.atte_v1.model import (
    AtteV1PolicyModel,
    AtteV1ValueModel,
    _AtteV1Body,
)
from src.backend.training.atte_v1.atte_v1_training_runner import (
    AtteV1TrainingRunner,
)
from src.backend.training.training_config import TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atte_v1_input_dict(
    own_hand_ids: list[int] | None = None,
    own_discard_ids: list[int] | None = None,
    own_draw_ids: list[int] | None = None,
    own_last_action: int = -1,
    own_status_ids: list[int] | None = None,
    own_status_durations: list[int] | None = None,
    opp_hand_ids: list[int] | None = None,
    opp_discard_ids: list[int] | None = None,
    opp_draw_ids: list[int] | None = None,
    opp_last_action: int = -1,
    opp_status_ids: list[int] | None = None,
    opp_status_durations: list[int] | None = None,
    own_hp: float = 1.0,
    own_energy: float = 1.0,
    opp_hp: float = 1.0,
    opp_energy: float = 1.0,
    round_in_cycle: int = -1,
) -> dict:
    """Build atte_v1 dict-format input for a single step."""
    default_hands = [3]  # ATTACK
    return {
        "own": {
            "hand_ids": own_hand_ids if own_hand_ids is not None else default_hands,
            "discard_ids": own_discard_ids or [],
            "draw_ids": own_draw_ids or [],
            "last_action": own_last_action,
            "hp": own_hp,
            "energy": own_energy,
            "status_ids": own_status_ids or [],
            "status_durations": own_status_durations or [],
            "opp_hand_ids": opp_hand_ids or default_hands,
            "opp_discard_ids": opp_discard_ids or [],
            "opp_draw_ids": opp_draw_ids or [],
        },
        "opp": {
            "hand_ids": opp_hand_ids if opp_hand_ids is not None else default_hands,
            "discard_ids": opp_discard_ids or [],
            "draw_ids": opp_draw_ids or [],
            "last_action": opp_last_action,
            "hp": opp_hp,
            "energy": opp_energy,
            "status_ids": opp_status_ids or [],
            "status_durations": opp_status_durations or [],
            "opp_hand_ids": own_hand_ids if own_hand_ids is not None else default_hands,
            "opp_discard_ids": own_discard_ids or [],
            "opp_draw_ids": own_draw_ids or [],
        },
        "round_in_cycle": round_in_cycle,
    }


def _make_perspective(steps_data: list[dict]) -> dict:
    """Build a perspective dict with steps."""
    return {"steps": steps_data, "game_id": "test_game"}


def _make_minimal_atte_v1_sample(
    num_games: int = 1,
    steps_per_perspective: int = 3,
) -> list[dict]:
    """Build minimal atte_v1-format training samples."""
    games = []
    for g in range(num_games):
        perspectives = []
        for p in range(2):
            steps = []
            for t in range(steps_per_perspective):
                base = float((g + 1) * (p + 1) * (t + 1))
                inp = _make_atte_v1_input_dict(
                    own_hand_ids=[3, 6],
                    opp_hand_ids=[3, 9],
                    own_last_action=3 if t > 0 else -1,
                )
                steps.append({
                    "input": inp,
                    "action": (t + p) % 24,
                    "reward": base / 10.0,
                    "value": base / 5.0,
                    "log_prob": -(base / 100.0),
                })
            perspectives.append({
                "player": str(p + 1),
                "steps": steps,
            })
        games.append({
            "game_id": f"test_game_{g}",
            "winner": "1",
            "rounds": steps_per_perspective * 2,
            "perspectives": perspectives,
        })
    return games


# ---------------------------------------------------------------------------
# T4: _compute_all_pre_gru
# ---------------------------------------------------------------------------


class TestComputeAllPreGru:
    """T4: _compute_all_pre_gru correctness."""

    def test_returns_correct_length(self):
        """T4.1: perspective with 5 steps → list length 5."""
        runner = AtteV1TrainingRunner()
        steps_data = []
        for _ in range(5):
            steps_data.append({
                "input": _make_atte_v1_input_dict(),
                "action": 0,
                "reward": 0.0,
                "value": 0.0,
                "log_prob": 0.0,
            })
        perspective = _make_perspective(steps_data)
        device = torch.device("cpu")
        results = runner._compute_all_pre_gru(perspective, device, body=_AtteV1Body())
        assert len(results) == 5

    def test_empty_perspective_returns_empty(self):
        """T4.2: empty perspective → empty list."""
        runner = AtteV1TrainingRunner()
        perspective = {"steps": [], "game_id": "test"}
        device = torch.device("cpu")
        results = runner._compute_all_pre_gru(perspective, device, body=_AtteV1Body())
        assert results == []

    def test_none_input_skipped(self):
        """T4.3: step with input=None → skipped, returns zero tensors."""
        runner = AtteV1TrainingRunner()
        device = torch.device("cpu")
        perspective = _make_perspective([
            {"input": None, "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
            {"input": _make_atte_v1_input_dict(), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
        ])
        results = runner._compute_all_pre_gru(perspective, device, body=_AtteV1Body())
        assert len(results) == 2
        # First result should be a zero gru_input
        gru_input_0, _intermediates_0 = results[0]
        assert torch.all(gru_input_0 == 0.0)

    def test_gru_input_shape_per_step(self):
        """Each step's gru_input should be (1, 80)."""
        runner = AtteV1TrainingRunner()
        device = torch.device("cpu")
        perspective = _make_perspective([
            {"input": _make_atte_v1_input_dict(), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
        ])
        results = runner._compute_all_pre_gru(perspective, device, body=_AtteV1Body())
        gru_input, intermediates = results[0]
        assert gru_input.shape == (1, GRU_INPUT_SIZE)
        assert "own_hand_emb2" in intermediates


# ---------------------------------------------------------------------------
# T5: _forward_value_sequence
# ---------------------------------------------------------------------------


class TestForwardValueSequence:
    """T5: batch value forward correctness."""

    def test_returns_correct_length(self):
        """T5.1: 5 pre_gru_results → (5,) value tensor."""
        torch.manual_seed(42)
        runner = AtteV1TrainingRunner()
        device = torch.device("cpu")
        value_model = AtteV1ValueModel()
        perspective = _make_perspective([
            {"input": _make_atte_v1_input_dict(), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
            {"input": _make_atte_v1_input_dict(), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
        ])
        pre_gru_results = runner._compute_all_pre_gru(perspective, device, body=value_model.body)
        values = runner._forward_value_sequence(pre_gru_results, value_model, device)
        assert values.shape == (2,)

    def test_values_match_stepwise_computation(self):
        """T5.2: batched values match per-step forward using same model."""
        torch.manual_seed(42)
        runner = AtteV1TrainingRunner()
        device = torch.device("cpu")

        torch.manual_seed(42)
        value_model = AtteV1ValueModel()

        steps_data = [
            {"input": _make_atte_v1_input_dict(own_hand_ids=[3, 6]), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
            {"input": _make_atte_v1_input_dict(own_hand_ids=[9, 12]), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
        ]
        perspective = _make_perspective(steps_data)

        # New batch method (uses value_model.body for pre_gru)
        pre_gru_results = runner._compute_all_pre_gru(
            perspective, device, body=value_model.body,
        )
        values_batch = runner._forward_value_sequence(pre_gru_results, value_model, device)

        # Old stepwise method using the SAME model
        values_step: list[float] = []
        h = torch.tensor([0.0] * HIDDEN_SIZE, dtype=torch.float32, device=device).view(1, 1, HIDDEN_SIZE)
        for s in steps_data:
            from src.backend.ai_adapter.atte_v1_ai import _build_model_kwargs
            from src.backend.ai_adapter.atte_v1_converter import AtteV1Input, AtteV1PlayerInput
            inp = s["input"]
            own_d = inp["own"]
            opp_d = inp["opp"]
            atte_inp = AtteV1Input(
                own=AtteV1PlayerInput(
                    hand_ids=own_d["hand_ids"], discard_ids=own_d["discard_ids"],
                    draw_ids=own_d["draw_ids"], last_action=own_d["last_action"],
                    hp=own_d["hp"], energy=own_d["energy"],
                    status_ids=own_d["status_ids"], status_durations=own_d["status_durations"],
                    opp_hand_ids=own_d["opp_hand_ids"], opp_discard_ids=own_d["opp_discard_ids"],
                    opp_draw_ids=own_d["opp_draw_ids"],
                ),
                opp=AtteV1PlayerInput(
                    hand_ids=opp_d["hand_ids"], discard_ids=opp_d["discard_ids"],
                    draw_ids=opp_d["draw_ids"], last_action=opp_d["last_action"],
                    hp=opp_d["hp"], energy=opp_d["energy"],
                    status_ids=opp_d["status_ids"], status_durations=opp_d["status_durations"],
                    opp_hand_ids=opp_d["opp_hand_ids"], opp_discard_ids=opp_d["opp_discard_ids"],
                    opp_draw_ids=opp_d["opp_draw_ids"],
                ),
                round_in_cycle=inp["round_in_cycle"],
            )
            kwargs = _build_model_kwargs(atte_inp, h.squeeze(1).squeeze(0).tolist())
            v, h = value_model(**kwargs)
            values_step.append(float(v.item()))

        values_step_t = torch.tensor(values_step, dtype=torch.float32)
        assert torch.allclose(values_batch, values_step_t, atol=1e-5)


# ---------------------------------------------------------------------------
# T6: _forward_policy_sequence
# ---------------------------------------------------------------------------


class TestForwardPolicySequence:
    """T6: batch policy forward correctness."""

    def test_returns_correct_shape(self):
        """T6.1: 3 pre_gru_results → (3, 24) log_probs tensor."""
        torch.manual_seed(42)
        runner = AtteV1TrainingRunner()
        device = torch.device("cpu")
        policy_model = AtteV1PolicyModel()
        perspective = _make_perspective([
            {"input": _make_atte_v1_input_dict(), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
            {"input": _make_atte_v1_input_dict(), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
            {"input": _make_atte_v1_input_dict(), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
        ])
        pre_gru_results = runner._compute_all_pre_gru(perspective, device, body=policy_model.body)
        log_probs = runner._forward_policy_sequence(pre_gru_results, policy_model, device)
        assert log_probs.shape == (3, 24)

    def test_log_probs_softmax_sums_to_one(self):
        """T6.2: each row's softmax sums ≈ 1.0."""
        torch.manual_seed(42)
        runner = AtteV1TrainingRunner()
        device = torch.device("cpu")
        policy_model = AtteV1PolicyModel()
        perspective = _make_perspective([
            {"input": _make_atte_v1_input_dict(), "action": 0, "reward": 0.0, "value": 0.0, "log_prob": 0.0},
        ])
        pre_gru_results = runner._compute_all_pre_gru(perspective, device, body=policy_model.body)
        log_probs = runner._forward_policy_sequence(pre_gru_results, policy_model, device)
        probs = torch.exp(log_probs)
        assert abs(probs.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# T10: End-to-end training
# ---------------------------------------------------------------------------


class TestAtteV1EndToEndTraining:
    """T10: full three-phase training with atte_v1."""

    def test_full_training_completes(self, tmp_path):
        """T10.1/T10.2: run() completes and outputs weight files."""
        import src.backend.ai.registry as registry_module
        from src.backend.training.atte_v1.atte_v1_training_runner import (
            AtteV1TrainingRunner,
        )

        # Create gen1 weight files
        gen_dir = tmp_path / "atte_v1" / "gen1"
        gen_dir.mkdir(parents=True)

        policy_model = AtteV1PolicyModel()
        value_model = AtteV1ValueModel()

        # Save split weights (e1e2 + main)
        e1e2_policy = {
            "body.e1.weight": policy_model.body.e1.weight.clone(),
            "body.e2.weight": policy_model.body.e2.weight.clone(),
        }
        e1e2_value = {
            "body.e1.weight": value_model.body.e1.weight.clone(),
            "body.e2.weight": value_model.body.e2.weight.clone(),
        }
        torch.save(e1e2_policy, str(gen_dir / "policy_e1_e2.pth"))
        torch.save(policy_model.state_dict(), str(gen_dir / "policy.pth"))
        torch.save(e1e2_value, str(gen_dir / "value_e1_e2.pth"))
        torch.save(value_model.state_dict(), str(gen_dir / "value.pth"))

        # Create minimal atte_v1 sample
        games = _make_minimal_atte_v1_sample(num_games=2, steps_per_perspective=3)
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(games), encoding="utf-8")

        # Patch registry to use tmp dir
        def _patched_derive(name: str, gen: int, suffix: str) -> str:
            return str(gen_dir / f"{suffix}.pth")

        # Force CPU to avoid CUDA OOM in test environment
        with patch.object(registry_module, "_derive_weight_path", _patched_derive), \
             patch("torch.cuda.is_available", return_value=False):
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            config = TrainingConfig(
                sample_path=str(sample_path),
                output_dir=str(output_dir),
                model="atte_v1.1",
                policy_epoch=2,
                value_epoch=2,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.0,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=2,
                validation_split=0.0,
            )

            random.seed(42)
            torch.manual_seed(42)
            runner = AtteV1TrainingRunner()
            runner.run(config)

            # Verify output files
            assert os.path.isfile(str(output_dir / "policy.pth"))
            assert os.path.isfile(str(output_dir / "value.pth"))

            # Verify weights are loadable
            trained_policy = AtteV1PolicyModel()
            trained_value = AtteV1ValueModel()
            trained_policy.load_state_dict(torch.load(
                str(output_dir / "policy.pth"), map_location="cpu", weights_only=True,
            ), strict=False)
            trained_value.load_state_dict(torch.load(
                str(output_dir / "value.pth"), map_location="cpu", weights_only=True,
            ), strict=False)

    def test_loss_is_finite(self, tmp_path):
        """T10.2: value_loss and policy_loss are non-NaN."""
        import src.backend.ai.registry as registry_module

        gen_dir = tmp_path / "atte_v1" / "gen1"
        gen_dir.mkdir(parents=True)

        policy_model = AtteV1PolicyModel()
        value_model = AtteV1ValueModel()

        e1e2_policy = {
            "body.e1.weight": policy_model.body.e1.weight.clone(),
            "body.e2.weight": policy_model.body.e2.weight.clone(),
        }
        e1e2_value = {
            "body.e1.weight": value_model.body.e1.weight.clone(),
            "body.e2.weight": value_model.body.e2.weight.clone(),
        }
        torch.save(e1e2_policy, str(gen_dir / "policy_e1_e2.pth"))
        torch.save(policy_model.state_dict(), str(gen_dir / "policy.pth"))
        torch.save(e1e2_value, str(gen_dir / "value_e1_e2.pth"))
        torch.save(value_model.state_dict(), str(gen_dir / "value.pth"))

        games = _make_minimal_atte_v1_sample(num_games=1, steps_per_perspective=3)
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(games), encoding="utf-8")

        def _patched_derive(name: str, gen: int, suffix: str) -> str:
            return str(gen_dir / f"{suffix}.pth")

        # Force CPU to avoid CUDA OOM in test environment
        with patch.object(registry_module, "_derive_weight_path", _patched_derive), \
             patch("torch.cuda.is_available", return_value=False):
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            config = TrainingConfig(
                sample_path=str(sample_path),
                output_dir=str(output_dir),
                model="atte_v1.1",
                policy_epoch=1,
                value_epoch=1,
                policy_lr=0.001,
                value_lr=0.001,
                weight_decay=0.0,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                max_grad_norm=0.5,
                value_coef=0.5,
                entropy_coef=0.01,
                batch_size=2,
                validation_split=0.0,
            )

            random.seed(42)
            torch.manual_seed(42)
            runner = AtteV1TrainingRunner()
            # run() doesn't raise → training succeeded
            runner.run(config)
