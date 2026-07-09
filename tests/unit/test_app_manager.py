"""Unit tests for AppManager mask_fn - 14-d masking (design/28)."""

import torch

from src.backend.app.app_manager import AppManager
from src.backend.game.card_type import CardType


class TestMaskFn:
    """TC-AM tests adapted for 14-d output (design/28).

    Policy output layout (14-d, design/28):
        index 0:                     pass (always legal)
        odd indices (1,3,5,7,9,11,13): build + build target
        even indices (2,4,6,8,10,12):  pure card play
    """

    def test_pass_never_masked(self):
        """Index 0 (pass) is always legal, even with empty hand and 0 energy."""
        mask_fn = AppManager._make_mask_fn([], energy=0)
        masked = mask_fn(torch.zeros(1, 14))
        assert masked[0, 0].item() == 0.0, "pass at index 0 should not be masked"

    def test_no_build_in_hand_blocks_all_odd_indices(self):
        """hand=['attack','energy'] -- all odd indices (build) masked."""
        mask_fn = AppManager._make_mask_fn([CardType.ATTACK, CardType.ENERGY], energy=2)
        masked = mask_fn(torch.zeros(1, 14))
        for i in [1, 3, 5, 7, 9, 11, 13]:
            assert masked[0, i].item() == float("-inf"), f"odd idx {i} not masked"

    def test_energy_zero_blocks_all_odd_indices(self):
        """energy=0 blocks ALL odd indices (build actions)."""
        mask_fn = AppManager._make_mask_fn([CardType.ATTACK, CardType.BUILD], energy=0)
        masked = mask_fn(torch.zeros(1, 14))
        for i in [1, 3, 5, 7, 9, 11, 13]:
            assert masked[0, i].item() == float("-inf"), f"odd idx {i} not masked"

    def test_build_targets_filtered_by_energy(self):
        """counter(cost=2) and flex(cost=2) available with energy=2."""
        mask_fn = AppManager._make_mask_fn([CardType.BUILD], energy=2)
        masked = mask_fn(torch.zeros(1, 14))
        # Cost-1 targets should NOT be masked (odd indices)
        for idx in [1, 3, 5, 7, 11]:  # build, attack, defense, heavy, energy
            assert masked[0, idx].item() == 0.0, f"cost-1 idx {idx} should not be masked"
        # Cost-2 targets should NOT be masked with energy=2
        assert masked[0, 9].item() == 0.0   # build+counter (cost 2)
        assert masked[0, 13].item() == 0.0  # build+flex (cost 2)

    def test_energy_1_blocks_cost_2_targets(self):
        """counter(cost=2) and flex(cost=2) blocked with energy=1."""
        mask_fn = AppManager._make_mask_fn([CardType.BUILD], energy=1)
        masked = mask_fn(torch.zeros(1, 14))
        assert masked[0, 9].item() == float("-inf")   # counter (cost 2)
        assert masked[0, 13].item() == float("-inf")  # flex (cost 2)

    def test_hand_cards_masked_at_even_indices(self):
        """Cards not in hand are masked at even indices (2,4,6,8,10,12)."""
        mask_fn = AppManager._make_mask_fn([CardType.ATTACK, CardType.DEFENSE], energy=2)
        masked = mask_fn(torch.zeros(1, 14))
        assert masked[0, 0].item() == 0.0  # pass always legal
        assert masked[0, 2].item() == 0.0  # attack in hand
        assert masked[0, 4].item() == 0.0  # defense in hand
        for i in [6, 8, 10, 12]:  # heavy, counter, energy, flex not in hand
            assert masked[0, i].item() == float("-inf"), f"even idx {i} not masked"
