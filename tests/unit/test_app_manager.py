"""Unit tests for AppManager mask_fn — two-segment hand + build target masking."""

import torch

from src.backend.app.app_manager import AppManager


class TestMaskFn:
    def test_no_build_in_hand_blocks_all_build_targets(self):
        """TC-AM-03b: hand=['attack','energy'] energy=2 — all back segment masked."""
        mask_fn = AppManager._make_mask_fn(["attack", "energy"], energy=2)
        masked = mask_fn(torch.zeros(1, 14))
        for i in range(7, 14):
            assert masked[0, i].item() == float("-inf"), f"idx {i} not masked"

    def test_energy_zero_blocks_build_and_back_segment(self):
        """TC-AM-03b: energy=0 blocks BUILD(idx=5) and ALL build targets (7-13)."""
        mask_fn = AppManager._make_mask_fn(["attack", "build"], energy=0)
        masked = mask_fn(torch.zeros(1, 14))
        assert masked[0, 5].item() == float("-inf"), "BUILD(idx=5) not masked"
        for i in range(7, 14):
            assert masked[0, i].item() == float("-inf"), f"idx {i} not masked"

    def test_build_targets_filtered_by_energy(self):
        """counter(cost=2) and flex(cost=2) available with energy=2."""
        mask_fn = AppManager._make_mask_fn(["build"], energy=2)
        masked = mask_fn(torch.zeros(1, 14))
        assert masked[0, 10].item() == 0.0  # counter build target
        assert masked[0, 13].item() == 0.0   # flex build target

    def test_energy_1_blocks_cost_2_targets(self):
        """counter(cost=2) and flex(cost=2) blocked with energy=1."""
        mask_fn = AppManager._make_mask_fn(["build"], energy=1)
        masked = mask_fn(torch.zeros(1, 14))
        assert masked[0, 10].item() == float("-inf")  # counter
        assert masked[0, 13].item() == float("-inf")  # flex

    def test_hand_cards_masked_in_front_segment(self):
        """Cards not in hand are masked in indices 0-6."""
        mask_fn = AppManager._make_mask_fn(["attack", "defense"], energy=2)
        masked = mask_fn(torch.zeros(1, 14))
        # attack(0) and defense(1) should be 0
        assert masked[0, 0].item() == 0.0
        assert masked[0, 1].item() == 0.0
        # heavy(2), counter(3), energy(4), build(5), flex(6) masked
        for i in [2, 3, 4, 5, 6]:
            assert masked[0, i].item() == float("-inf"), f"idx {i} not masked"
