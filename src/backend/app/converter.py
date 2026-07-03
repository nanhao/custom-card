"""Bidirectional conversion between card strings and model tensors.

Per design/3 architecture D4, Converter separates transformations from models.
All methods are stateless. The class serves as a namespace for conversion logic
that bridges game domain types and AI model inputs.

No PyTorch dependency — pure data transformation.
"""


class Converter:
    """Bidirectional conversion between card strings and model tensors.

    All methods are stateless. The class serves as a namespace for
    conversion logic that bridges game domain types and AI model inputs.
    """

    # Constants (per design/3 interfaces.md §3)
    NUM_CARD_TYPES: int = 4
    INPUT_SIZE: int = 14

    # Public set of all valid card strings
    ALL_CARD_TYPES: frozenset[str] = frozenset({"attack", "defense", "heavy", "counter"})

    # Internal mapping tables (use card_to_index / index_to_card instead)
    _CARD_TO_INDEX: dict[str, int] = {
        "attack": 0,
        "defense": 1,
        "heavy": 2,
        "counter": 3,
    }

    _INDEX_TO_CARD: dict[int, str] = {
        0: "attack",
        1: "defense",
        2: "heavy",
        3: "counter",
    }

    # ------------------------------------------------------------------
    # Card ↔ Index conversion
    # ------------------------------------------------------------------

    @staticmethod
    def card_to_index(card_str: str) -> int:
        """Convert card string to policy head index.

        Mapping: attack→0, defense→1, heavy→2, counter→3.
        Returns -1 for unrecognized card strings.

        Args:
            card_str: Card string value (e.g. "attack", "defense").

        Returns:
            Integer index (0–3), or -1 if unrecognized.
        """
        return Converter._CARD_TO_INDEX.get(card_str, -1)

    @staticmethod
    def index_to_card(index: int) -> str:
        """Convert policy head index to card string.

        Mapping: 0→attack, 1→defense, 2→heavy, 3→counter.

        Args:
            index: Integer index (0–3).

        Returns:
            Corresponding card string.

        Raises:
            IndexError: If index is outside the valid range [0, 3].
        """
        if index < 0 or index >= Converter.NUM_CARD_TYPES:
            raise IndexError(
                f"Card index {index} out of range [0, {Converter.NUM_CARD_TYPES - 1}]"
            )
        return Converter._INDEX_TO_CARD[index]

    # ------------------------------------------------------------------
    # Input vector construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_input_vector(state: dict, player_key: str = "2") -> list[float]:
        """Build the 14-d input vector from a game state dict.

        Per design/3 interfaces.md §3 and design/5 D3:
            indices 0–3:   This player's card counts (attack, defense, heavy, counter)
            indices 4–5:   HP values (this player, opponent), clamped [0,3]
            indices 6–9:   One-hot of this player's last move (zeros in round 1)
            indices 10–13: One-hot of opponent's last move (zeros in round 1)

        Args:
            state: Game state dict from GameCore.get_state().
            player_key: Which player's perspective — "1" or "2".
                        Default "2" preserves backward compatibility.

        Returns:
            list[float] of length 14.

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

        # ---- Card counts (indices 0–3) ----
        counts = [0.0] * Converter.NUM_CARD_TYPES
        for card_str in this_player["hand"]:
            idx = Converter._CARD_TO_INDEX.get(card_str)
            if idx is not None:
                counts[idx] += 1.0

        # ---- HP values (indices 4–5), clamped to [0, 3] ----
        this_hp = max(0.0, min(3.0, float(this_player["hp"])))
        opp_hp = max(0.0, min(3.0, float(opp_player["hp"])))

        # ---- One-hot of last moves (indices 6–13) ----
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

        return counts + [this_hp, opp_hp] + this_last + opp_last
