"""Bidirectional conversion between card strings and model tensors (design/13).

Per design/13, Converter builds 25-d input vectors with metadata (HP, round, cycle)
at the front and card-type-specific features at the tail for future extensibility.

All methods are stateless. No PyTorch dependency — pure data transformation.
"""


class Converter:
    """Bidirectional conversion between card strings and model tensors.

    Per design/13: input vector metadata (HP, round, cycle) at front;
    card-type-specific features (counts, one-hot) at tail.

    All methods are stateless. The class serves as a namespace for
    conversion logic bridging game domain types and AI model inputs.
    """

    NUM_CARD_TYPES: int = 7
    INPUT_SIZE: int = 25

    # Public set of all valid card strings
    ALL_CARD_TYPES: frozenset[str] = frozenset({
        "attack", "defense", "heavy", "counter", "energy", "build", "flex",
    })

    # Internal mapping tables
    _CARD_TO_INDEX: dict[str, int] = {
        "attack": 0,
        "defense": 1,
        "heavy": 2,
        "counter": 3,
        "energy": 4,
        "build": 5,
        "flex": 6,
    }

    _INDEX_TO_CARD: dict[int, str] = {
        0: "attack",
        1: "defense",
        2: "heavy",
        3: "counter",
        4: "energy",
        5: "build",
        6: "flex",
    }

    # Build cost lookup for mask_fn
    _BUILD_COST: dict[int, int] = {
        0: 1,  # attack
        1: 1,  # defense
        2: 1,  # heavy
        3: 2,  # counter
        4: 1,  # energy
        5: 1,  # build
        6: 2,  # flex
    }

    # ------------------------------------------------------------------
    # Card <-> Index conversion
    # ------------------------------------------------------------------

    @staticmethod
    def card_to_index(card_str: str) -> int:
        """Convert card string to policy head index.

        Mapping: attack->0, defense->1, heavy->2, counter->3,
                 energy->4, build->5, flex->6.
        Returns -1 for unrecognized card strings.

        This mapping is used for both the first 7-d (card action)
        and the second 7-d (build target) policy outputs.
        """
        return Converter._CARD_TO_INDEX.get(card_str, -1)

    @staticmethod
    def index_to_card(index: int) -> str:
        """Convert policy head index to card string.

        0->attack ... 6->flex.

        Raises:
            IndexError: If index is outside [0, 6].
        """
        if index < 0 or index >= Converter.NUM_CARD_TYPES:
            raise IndexError(
                f"Card index {index} out of range [0, {Converter.NUM_CARD_TYPES - 1}]"
            )
        return Converter._INDEX_TO_CARD[index]

    @staticmethod
    def build_cost_for_index(index: int) -> int:
        """Return the build cost for a card by its index (0-6)."""
        return Converter._BUILD_COST.get(index, 999)

    # ------------------------------------------------------------------
    # Input vector construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_input_vector(state: dict, player_key: str = "2") -> list[float]:
        """Build the 25-d input vector from a game state dict.

        Per design/13:
            index 0:     Own HP (this player), NOT clamped/normalized
            index 1:     Opponent HP, NOT clamped/normalized
            index 2:     Current round_in_cycle (1-5)
            index 3:     Current cycle number
            indices 4-10:   This player's hand card counts (7 types)
            indices 11-17:  One-hot of this player's last move (7 types)
            indices 18-24:  One-hot of opponent's last move (7 types)

        Metadata (HP, round, cycle) is placed at the front so that
        adding new card types in the future only extends the tail
        of the vector without shifting metadata indices.

        Args:
            state: Game state dict from GameCore.get_state().
            player_key: Which player's perspective — "1" or "2".

        Returns:
            list[float] of length 25.

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

        # Card counts (indices 4-10)
        counts = [0.0] * Converter.NUM_CARD_TYPES
        for card_str in this_player["hand"]:
            idx = Converter._CARD_TO_INDEX.get(card_str)
            if idx is not None:
                counts[idx] += 1.0

        # HP values (indices 0-1), NOT normalized per design D8
        this_hp = float(this_player["hp"])
        opp_hp = float(opp_player["hp"])

        # One-hot of last moves (indices 11-24)
        this_last = [0.0] * Converter.NUM_CARD_TYPES
        opp_last = [0.0] * Converter.NUM_CARD_TYPES

        if this_player["history"]:
            last_card = this_player["history"][-1]
            onehot_idx = Converter._CARD_TO_INDEX.get(last_card)
            if onehot_idx is not None:
                this_last[onehot_idx] = 1.0

        if opp_player["history"]:
            last_card = opp_player["history"][-1]
            onehot_idx = Converter._CARD_TO_INDEX.get(last_card)
            if onehot_idx is not None:
                opp_last[onehot_idx] = 1.0

        # Round metadata (indices 2-3)
        round_in_cycle = float(state.get("round_in_cycle", 1))
        cycle = float(state.get("cycle", 1))

        # Build new-order vector per design/13:
        #   [HP_own, HP_opp, round, cycle] + card_counts + onehot_own + onehot_opp
        return [this_hp, opp_hp, round_in_cycle, cycle] + counts + this_last + opp_last
