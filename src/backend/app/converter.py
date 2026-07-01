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
    def build_input_vector(state: dict) -> list[float]:
        """Build the 14-d input vector from a game state dict.

        Per design/3 interfaces.md §3:
            indices 0–3:   AI card counts (attack, defense, heavy, counter)
            indices 4–5:   HP values (AI HP, human HP), clamped [0,3]
            indices 6–9:   One-hot of AI's last move (zeros in round 1)
            indices 10–13: One-hot of human's last move (zeros in round 1)

        Args:
            state: Game state dict from GameManager.get_state(). The AI
                   always plays as Player "2".

        Returns:
            list[float] of length 14.
        """
        ai_player = state["players"]["2"]
        human_player = state["players"]["1"]

        # ---- Card counts (indices 0–3) ----
        counts = [0.0] * Converter.NUM_CARD_TYPES
        for card_str in ai_player["hand"]:
            idx = Converter._CARD_TO_INDEX.get(card_str)
            if idx is not None:
                counts[idx] += 1.0

        # ---- HP values (indices 4–5), clamped to [0, 3] ----
        ai_hp = max(0.0, min(3.0, float(ai_player["hp"])))
        human_hp = max(0.0, min(3.0, float(human_player["hp"])))

        # ---- One-hot of last moves (indices 6–13) ----
        ai_last = [0.0] * Converter.NUM_CARD_TYPES
        human_last = [0.0] * Converter.NUM_CARD_TYPES

        if ai_player["history"]:
            last_card = ai_player["history"][-1]
            onehot_idx = Converter._CARD_TO_INDEX.get(last_card)
            if onehot_idx is not None:
                ai_last[onehot_idx] = 1.0

        if human_player["history"]:
            last_card = human_player["history"][-1]
            onehot_idx = Converter._CARD_TO_INDEX.get(last_card)
            if onehot_idx is not None:
                human_last[onehot_idx] = 1.0

        return counts + [ai_hp, human_hp] + ai_last + human_last
