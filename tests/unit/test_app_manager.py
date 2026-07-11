"""Unit tests for AppManager mask_fn - 24-d masking (design/37)."""

import torch

from src.backend.app.app_manager import AppManager
from src.backend.ai.constants import POLICY_OUTPUT_SIZE
from src.backend.game.card_type import CardType


class TestMaskFn:
    """TC-AM tests adapted for 24-d output (design/37).

    Policy output layout (24-d, design/37):
        index 0:                        pass (always legal)
        play indices (3,6,9,12,15,18,21): pure card play
        build indices (1,4,7,10,13,16,19,22): build + target
        dismantle indices (2,5,8,11,14,17,20,23): dismantle + target
    """

    POLICY_DIM = POLICY_OUTPUT_SIZE  # 24

    def test_pass_never_masked(self):
        """Index 0 (pass) is always legal."""
        mask_fn = AppManager._make_mask_fn([], energy=0)
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        assert masked[0, 0].item() == 0.0

    def test_no_build_in_hand_blocks_build_and_dismantle(self):
        """hand=['attack','energy'] -- all build/dismantle indices masked."""
        mask_fn = AppManager._make_mask_fn([CardType.ATTACK, CardType.ENERGY], energy=2)
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        # All build indices masked
        for i in [1, 4, 7, 10, 13, 16, 19, 22]:
            assert masked[0, i].item() == float("-inf"), f"build idx {i} not masked"
        # All dismantle indices masked
        for i in [2, 5, 8, 11, 14, 17, 20, 23]:
            assert masked[0, i].item() == float("-inf"), f"dismantle idx {i} not masked"
        # Play indices for cards in hand should not be masked
        assert masked[0, 3].item() == 0.0  # play ATTACK
        assert masked[0, 15].item() == 0.0  # play ENERGY

    def test_energy_zero_blocks_build_but_not_dismantle(self):
        """energy=0 blocks build indices, dismantle still valid if BUILD in hand."""
        mask_fn = AppManager._make_mask_fn(
            [CardType.ATTACK, CardType.BUILD], energy=0,
            discard_pile=[CardType.ATTACK],
        )
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        # All build indices masked (energy=0)
        for i in [1, 4, 7, 10, 13, 16, 19, 22]:
            assert masked[0, i].item() == float("-inf"), f"build idx {i} not masked"
        # Dismantle ATTACK should be legal (in discard)
        assert masked[0, 5].item() == 0.0  # dismantle ATTACK
        # Dismantle other cards should be masked (not in discard)
        for i in [2, 8, 11, 14, 17, 20, 23]:
            assert masked[0, i].item() == float("-inf"), f"dismantle idx {i} not masked"

    def test_build_targets_filtered_by_energy(self):
        """cost-1 targets legal, cost-2 targets legal with energy=2."""
        mask_fn = AppManager._make_mask_fn([CardType.BUILD], energy=2)
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        # Cost-1 build targets should NOT be masked
        for idx in [1, 4, 7, 10, 16]:  # BUILD, ATTACK, DEFENSE, HEAVY, ENERGY
            assert masked[0, idx].item() == 0.0, f"cost-1 build idx {idx} should be legal"
        # Cost-2 build targets should NOT be masked with energy=2
        assert masked[0, 13].item() == 0.0  # build COUNTER (cost 2)
        assert masked[0, 19].item() == 0.0  # build FLEX (cost 2)
        assert masked[0, 22].item() == 0.0  # build IRON_DEFENSE (cost 2)

    def test_energy_1_blocks_cost_2_targets(self):
        """cost-2 targets blocked with energy=1."""
        mask_fn = AppManager._make_mask_fn([CardType.BUILD], energy=1)
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        assert masked[0, 13].item() == float("-inf")  # COUNTER (cost 2)
        assert masked[0, 19].item() == float("-inf")  # FLEX (cost 2)
        assert masked[0, 22].item() == float("-inf")  # IRON_DEFENSE (cost 2)
        # Cost-1 still legal
        assert masked[0, 4].item() == 0.0  # ATTACK (cost 1)

    def test_hand_cards_masked_at_play_indices(self):
        """Cards not in hand masked at play indices."""
        mask_fn = AppManager._make_mask_fn([CardType.ATTACK, CardType.DEFENSE], energy=2)
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        assert masked[0, 0].item() == 0.0  # pass always legal
        assert masked[0, 3].item() == 0.0  # play ATTACK (in hand)
        assert masked[0, 6].item() == 0.0  # play DEFENSE (in hand)
        # Cards not in hand masked
        for i in [9, 12, 15, 18, 21]:  # HEAVY, COUNTER, ENERGY, FLEX, IRON_DEFENSE
            assert masked[0, i].item() == float("-inf"), f"play idx {i} not masked"

    def test_dismantle_target_not_in_discard_masked(self):
        """Dismantle indices masked if card not in discard pile."""
        mask_fn = AppManager._make_mask_fn(
            [CardType.BUILD], energy=2,
            discard_pile=[CardType.ATTACK, CardType.DEFENSE],
        )
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        # ATTACK and DEFENSE in discard → legal
        assert masked[0, 5].item() == 0.0  # dismantle ATTACK
        assert masked[0, 8].item() == 0.0  # dismantle DEFENSE
        # Others not in discard → masked
        for i in [2, 11, 14, 17, 20, 23]:
            assert masked[0, i].item() == float("-inf"), f"dismantle idx {i} not masked"

    def test_dismantle_all_masked_when_discard_empty(self):
        """All dismantle indices masked when discard pile is empty."""
        mask_fn = AppManager._make_mask_fn([CardType.BUILD], energy=2, discard_pile=[])
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        for i in [2, 5, 8, 11, 14, 17, 20, 23]:
            assert masked[0, i].item() == float("-inf"), f"dismantle idx {i} not masked"

    def test_iron_defense_play_and_build(self):
        """IRON_DEFENSE play and build indices work correctly."""
        mask_fn = AppManager._make_mask_fn(
            [CardType.BUILD, CardType.IRON_DEFENSE], energy=2,
        )
        masked = mask_fn(torch.zeros(1, self.POLICY_DIM))
        assert masked[0, 21].item() == 0.0  # play IRON_DEFENSE
        assert masked[0, 22].item() == 0.0  # build IRON_DEFENSE (cost 2, energy 2)
