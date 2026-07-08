"""Bidirectional conversion between game state and AI model tensors (design/14).

Per design/14: 15-d input vector with metadata (HP, energy, round, cycle)
at the front, card counts in the middle, and policy-index-encoded last
actions at the tail. Policy output is 13-d interleaved (build/play pairs).

All methods are stateless. No PyTorch dependency — pure data transformation.
"""

import logging

from src.backend.game.card import CardType
from src.backend.game.player import PlayerHistoryItem

logger = logging.getLogger(__name__)


class Converter:
    """Bidirectional conversion between game state and model tensors.

    Per design/14:
    - Input: 15-d (HP×2, energy×2, round, cycle, card counts×7, last_action×2)
    - Policy output: 13-d interleaved (even=build+target, odd=pure play)
    - Last action encoded as policy index, -1 when no history.

    All methods are stateless. The class serves as a namespace for
    conversion logic bridging game domain types and AI model inputs.
    """

    NUM_CARD_TYPES: int = 7
    INPUT_SIZE: int = 15
    POLICY_OUTPUT_SIZE: int = 13

    # Public set of all valid card strings
    ALL_CARD_TYPES: frozenset[str] = frozenset({
        "attack", "defense", "heavy", "counter", "energy", "build", "flex",
    })

    # ------------------------------------------------------------------
    # Internal: card-string ↔ policy-index mapping tables
    # ------------------------------------------------------------------

    # Pure play actions: card_str → policy output index (odd numbers)
    _PLAY_TO_INDEX: dict[str, int] = {
        "attack": 1,
        "defense": 3,
        "heavy": 5,
        "counter": 7,
        "energy": 9,
        "flex": 11,
    }

    # Build actions: build_target → policy output index (even numbers)
    _BUILD_TO_INDEX: dict[str, int] = {
        "build": 0,
        "attack": 2,
        "defense": 4,
        "heavy": 6,
        "counter": 8,
        "energy": 10,
        "flex": 12,
    }

    # Reverse: policy index → (card_str, build_target | None)
    _INDEX_TO_ACTION: dict[int, tuple[str, str | None]] = {
        0: ("build", "build"),
        1: ("attack", None),
        2: ("build", "attack"),
        3: ("defense", None),
        4: ("build", "defense"),
        5: ("heavy", None),
        6: ("build", "heavy"),
        7: ("counter", None),
        8: ("build", "counter"),
        9: ("energy", None),
        10: ("build", "energy"),
        11: ("flex", None),
        12: ("build", "flex"),
    }

    # Build cost for mask_fn (by card string, used in AppManager)
    _BUILD_COST: dict[str, int] = {
        "build": 1,
        "attack": 1,
        "defense": 1,
        "heavy": 1,
        "counter": 2,
        "energy": 1,
        "flex": 2,
    }

    # ------------------------------------------------------------------
    # Card-string ↔ card-type-index (for card counting)
    # ------------------------------------------------------------------

    _CARD_TYPE_ORDER: dict[str, int] = {
        "attack": 0,
        "defense": 1,
        "heavy": 2,
        "counter": 3,
        "energy": 4,
        "build": 5,
        "flex": 6,
    }

    # ------------------------------------------------------------------
    # Input vector construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_input_vector(state: dict, player_key: str = "2") -> list[float]:
        """Build the 15-d input vector from a game state dict.

        Per design/14:
            index 0:     Own HP
            index 1:     Opponent HP
            index 2:     Own energy (0-5)
            index 3:     Opponent energy (0-5)
            index 4:     Current round_in_cycle (1-5)
            index 5:     Current cycle number
            indices 6-12:   This player's hand card counts (7 types)
            index 13:    Own last action policy index (-1 if no history)
            index 14:    Opponent last action policy index (-1 if no history)

        Args:
            state: Game state dict from GameCore.get_state().
            player_key: Which player's perspective — "1" or "2".

        Returns:
            list[float] of length 15.

        Raises:
            ValueError: If player_key is not "1" or "2".
        """
        if player_key not in ("1", "2"):
            raise ValueError(
                f"Invalid player_key: {player_key!r}. Must be '1' or '2'."
            )

        this_player = state["players"][player_key]
        opponent_key = "2" if player_key == "1" else "1"
        opp_player = state["players"][opponent_key]

        # Metadata (indices 0-5): HP, energy, round, cycle
        this_hp = float(this_player["hp"])
        opp_hp = float(opp_player["hp"])
        this_energy = float(this_player["energy"])
        opp_energy = float(opp_player["energy"])
        round_in_cycle = float(state.get("round_in_cycle", 1))
        cycle = float(state.get("cycle", 1))

        # Card counts (indices 6-12): count each type in hand
        counts = [0.0] * Converter.NUM_CARD_TYPES
        for card_str in this_player["hand"]:
            idx = Converter._CARD_TYPE_ORDER.get(card_str)
            if idx is not None:
                counts[idx] += 1.0

        # Last action indices (13-14): compute from PlayerHistoryItem in state
        this_last = -1.0
        opp_last = -1.0

        this_history: list = this_player.get("history", [])
        if this_history:
            last_item = this_history[-1]
            if isinstance(last_item, PlayerHistoryItem):
                this_last = float(
                    Converter._action_item_to_index(last_item)
                )
            else:
                logger.warning(
                    "history entry is not PlayerHistoryItem: %r; using -1",
                    type(last_item).__name__,
                )

        opp_history: list = opp_player.get("history", [])
        if opp_history:
            last_item = opp_history[-1]
            if isinstance(last_item, PlayerHistoryItem):
                opp_last = float(
                    Converter._action_item_to_index(last_item)
                )
            else:
                logger.warning(
                    "history entry is not PlayerHistoryItem: %r; using -1",
                    type(last_item).__name__,
                )

        # Build new-order vector per design/14:
        #   [HP_own, HP_opp, energy_own, energy_opp, round, cycle]
        #   + card_counts + [last_own, last_opp]
        return [
            this_hp, opp_hp,
            this_energy, opp_energy,
            round_in_cycle, cycle,
        ] + counts + [this_last, opp_last]

    # ------------------------------------------------------------------
    # Action encoding/decoding
    # ------------------------------------------------------------------

    @staticmethod
    def action_to_policy_index(
        card_str: str, build_target: str | None,
    ) -> int:
        """Convert game action (card + optional build target) to policy index.

        Mapping:
            Pure play → odd indices:  attack→1, defense→3, ...
            Build+target → even indices:  build+build→0, build+attack→2, ...

        Args:
            card_str: Card played, e.g. "attack", "build".
            build_target: Build target card, only meaningful when card_str="build".

        Returns:
            Policy output index (0-12).

        Raises:
            ValueError: If card_str is unknown, or build_target is invalid/absent
                        for a build action, or present for a non-build action.
        """
        if card_str == "build":
            if build_target is None:
                raise ValueError(
                    "build_target is required when card_str is 'build'."
                )
            idx = Converter._BUILD_TO_INDEX.get(build_target)
            if idx is None:
                raise ValueError(
                    f"Unknown build_target: {build_target!r}"
                )
            return idx

        if build_target is not None:
            raise ValueError(
                f"build_target must be None for non-build card {card_str!r}, "
                f"got {build_target!r}"
            )
        idx = Converter._PLAY_TO_INDEX.get(card_str)
        if idx is None:
            raise ValueError(f"Unknown card string: {card_str!r}")
        return idx

    @staticmethod
    def policy_index_to_action(index: int) -> tuple[str, str | None]:
        """Decode policy output index to (card_str, build_target).

        Odd indices → pure play (build_target=None).
        Even indices → build + target.

        Args:
            index: Policy output index (0-12).

        Returns:
            (card_str, build_target) tuple.

        Raises:
            IndexError: If index is outside [0, 12].
        """
        if index < 0 or index >= Converter.POLICY_OUTPUT_SIZE:
            raise IndexError(
                f"Policy index {index} out of range "
                f"[0, {Converter.POLICY_OUTPUT_SIZE - 1}]"
            )
        return Converter._INDEX_TO_ACTION[index]

    @staticmethod
    def build_cost(card_str: str) -> int:
        """Return the build (energy) cost for a card type.

        Used by AppManager._make_mask_fn for energy-based masking.

        Args:
            card_str: Card type string, e.g. "counter".

        Returns:
            Build cost (1 or 2). Returns 999 for unknown card types.
        """
        return Converter._BUILD_COST.get(card_str, 999)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_item_to_index(item: PlayerHistoryItem) -> int:
        """Convert a PlayerHistoryItem to policy index.

        Routes through action_to_policy_index using the .value of CardType
        enums to produce the string representation expected by the public API.
        """
        card_str = item.card.value
        build_target_str = item.build_target.value if item.build_target else None
        return Converter.action_to_policy_index(card_str, build_target_str)
