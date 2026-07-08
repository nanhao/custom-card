"""Unit tests for AppManager mask_fn - 13-d interleaved masking (design/14)."""

import torch

from src.backend.app.app_manager import AppManager


class TestMaskFn:
    """TC-AM tests adapted for 13-d interleaved output (design/14).

    Policy output layout (13-d):
        even (0,2,4,6,8,10,12): build + build target
        odd  (1,3,5,7,9,11):    pure card play
    """

    def test_no_build_in_hand_blocks_all_even_indices(self):
        """hand=['attack','energy'] -- all even indices (build) masked."""
        mask_fn = AppManager._make_mask_fn(["attack", "energy"], energy=2)
        masked = mask_fn(torch.zeros(1, 13))
        for i in [0, 2, 4, 6, 8, 10, 12]:
            assert masked[0, i].item() == float("-inf"), f"even idx {i} not masked"

    def test_energy_zero_blocks_all_even_indices(self):
        """energy=0 blocks ALL even indices (build actions)."""
        mask_fn = AppManager._make_mask_fn(["attack", "build"], energy=0)
        masked = mask_fn(torch.zeros(1, 13))
        for i in [0, 2, 4, 6, 8, 10, 12]:
            assert masked[0, i].item() == float("-inf"), f"even idx {i} not masked"

    def test_build_targets_filtered_by_energy(self):
        """counter(cost=2) and flex(cost=2) available with energy=2."""
        mask_fn = AppManager._make_mask_fn(["build"], energy=2)
        masked = mask_fn(torch.zeros(1, 13))
        assert masked[0, 8].item() == 0.0   # build+counter (cost 2)
        assert masked[0, 12].item() == 0.0  # build+flex (cost 2)

    def test_energy_1_blocks_cost_2_targets(self):
        """counter(cost=2) and flex(cost=2) blocked with energy=1."""
        mask_fn = AppManager._make_mask_fn(["build"], energy=1)
        masked = mask_fn(torch.zeros(1, 13))
        assert masked[0, 8].item() == float("-inf")   # counter (cost 2)
        assert masked[0, 12].item() == float("-inf")  # flex (cost 2)

    def test_hand_cards_masked_at_odd_indices(self):
        """Cards not in hand are masked at odd indices."""
        mask_fn = AppManager._make_mask_fn(["attack", "defense"], energy=2)
        masked = mask_fn(torch.zeros(1, 13))
        assert masked[0, 1].item() == 0.0  # attack in hand
        assert masked[0, 3].item() == 0.0  # defense in hand
        for i in [5, 7, 9, 11]:  # heavy, counter, energy, flex not in hand
            assert masked[0, i].item() == float("-inf"), f"odd idx {i} not masked"
