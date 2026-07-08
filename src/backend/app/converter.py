"""Bidirectional conversion between game state and AI model tensors (design/14).

Per design/14: 15-d input vector with metadata (HP, energy, round, cycle)
at the front, card counts in the middle, and policy-index-encoded last
actions at the tail. Policy output is 13-d interleaved (build/play pairs).

Per design/20: _CARD_MAP uses CardType enum instead of string literals;
INPUT_SIZE/POLICY_OUTPUT_SIZE imported from ai.rnn_v1.model;
build_cost derived from get_card_def() instead of duplicated in _CARD_MAP.

All methods are stateless. No PyTorch dependency — pure data transformation.
"""

from src.backend.ai.rnn_v1.model import INPUT_SIZE, POLICY_OUTPUT_SIZE
from src.backend.game.card import CardType, get_card_def
from src.backend.game.game_state import GameState
from src.backend.game.player import PlayerHistoryItem


class Converter:
    """Bidirectional conversion between game state and model tensors.

    Per design/14:
    - Input: 15-d (HP×2, energy×2, round, cycle, card counts×7, last_action×2)
    - Policy output: 13-d interleaved (even=build+target, odd=pure play)
    - Last action encoded as policy index, -1 when no history.

    Per design/20: all card mappings use CardType enum; INPUT_SIZE,
    POLICY_OUTPUT_SIZE imported from ai.rnn_v1.model; build_cost derived
    from get_card_def().

    All methods are stateless. The class serves as a namespace for
    conversion logic bridging game domain types and AI model inputs.
    """

    NUM_CARD_TYPES: int = 7

    # ------------------------------------------------------------------
    # Master card mapping table — single source of truth for all
    # card↔index mappings (design/20).
    #
    # Format: (card_type, play_index, build_index).
    #   card_type: CardType enum member.
    #   play_index: odd-numbered policy output index (None if not directly playable).
    #   build_index: even-numbered policy output index for build+target.
    #
    # Per design/20 D3: build_cost is NOT stored here; it is derived
    # from get_card_def(card_type).build_cost to avoid duplication.
    #
    # All other mapping dicts (ALL_CARD_TYPES, _PLAY_TO_INDEX, etc.) are
    # derived from this single table.
    # ------------------------------------------------------------------

    _CARD_MAP: list[tuple[CardType, int | None, int]] = [
        (CardType.ATTACK,  1, 2),
        (CardType.DEFENSE, 3, 4),
        (CardType.HEAVY,   5, 6),
        (CardType.COUNTER, 7, 8),
        (CardType.ENERGY,  9, 10),
        (CardType.BUILD,   None, 0),
        (CardType.FLEX,   11, 12),
    ]

    # Public set of all valid CardType values
    ALL_CARD_TYPES: frozenset[CardType] = frozenset(c[0] for c in _CARD_MAP)

    # ------------------------------------------------------------------
    # Internal: CardType ↔ policy-index mapping tables
    # (all derived from _CARD_MAP)
    # ------------------------------------------------------------------

    _CARD_TYPE_ORDER: dict[CardType, int] = {c[0]: i for i, c in enumerate(_CARD_MAP)}

    # Pure play actions: CardType → policy output index (odd numbers)
    _PLAY_TO_INDEX: dict[CardType, int] = {
        c[0]: c[1] for c in _CARD_MAP if c[1] is not None
    }

    # Build actions: build_target (CardType) → policy output index (even numbers)
    _BUILD_TO_INDEX: dict[CardType, int] = {c[0]: c[2] for c in _CARD_MAP}

    # Reverse: policy index → (card, build_target)
    _INDEX_TO_ACTION: dict[int, tuple[CardType, CardType | None]] = {
        **{c[2]: (CardType.BUILD, c[0]) for c in _CARD_MAP},
        **{c[1]: (c[0], None) for c in _CARD_MAP if c[1] is not None},
    }

    # Build cost — derived from CardDef.build_cost (design/20 D3)
    _BUILD_COST: dict[CardType, int] = {
        c[0]: get_card_def(c[0]).build_cost for c in _CARD_MAP
    }

    # ------------------------------------------------------------------
    # Input vector construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_input_vector(state: GameState, player_key: str = "2") -> list[float]:
        """Build the 15-d input vector from a GameState.

        Per design/19 D6: accepts GameState directly instead of dict.
        Per design/20: _CARD_TYPE_ORDER uses CardType keys, so hand
        CardType values are used directly without .value conversion.

        Args:
            state: GameState from GameCore.get_state().
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

        this_player = state.players[player_key]
        opponent_key = "2" if player_key == "1" else "1"
        opp_player = state.players[opponent_key]

        # Metadata (indices 0-5): HP, energy, round, cycle
        this_hp = float(this_player.hp)
        opp_hp = float(opp_player.hp)
        this_energy = float(this_player.energy)
        opp_energy = float(opp_player.energy)
        round_in_cycle = float(state.round_in_cycle)
        cycle = float(state.cycle)

        # Card counts (indices 6-12): count each type in hand.
        # Per design/20: _CARD_TYPE_ORDER uses CardType keys, so we use
        # card directly (hand is list[CardType]).
        counts = [0.0] * Converter.NUM_CARD_TYPES
        for card in this_player.hand:
            idx = Converter._CARD_TYPE_ORDER.get(card)
            if idx is not None:
                counts[idx] += 1.0

        # Last action indices (13-14): compute from PlayerHistoryItem in state.
        this_last = -1.0
        opp_last = -1.0

        if this_player.history:
            last_item = this_player.history[-1]
            this_last = float(
                Converter._action_item_to_index(last_item)
            )

        if opp_player.history:
            last_item = opp_player.history[-1]
            opp_last = float(
                Converter._action_item_to_index(last_item)
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
        card: CardType, build_target: CardType | None,
    ) -> int:
        """Convert game action (card + optional build target) to policy index.

        Per design/20: parameters are CardType enum, not strings.

        Mapping:
            Pure play → odd indices:  ATTACK→1, DEFENSE→3, ...
            Build+target → even indices:  BUILD+ATTACK→2, ...

        Args:
            card: CardType played, e.g. CardType.ATTACK, CardType.BUILD.
            build_target: Build target CardType, only meaningful when card=BUILD.

        Returns:
            Policy output index (0-12).

        Raises:
            ValueError: If card is unknown, or build_target is invalid/absent
                        for a build action, or present for a non-build action.
        """
        if card == CardType.BUILD:
            if build_target is None:
                raise ValueError(
                    "build_target is required when card is BUILD."
                )
            idx = Converter._BUILD_TO_INDEX.get(build_target)
            if idx is None:
                raise ValueError(
                    f"Unknown build_target: {build_target!r}"
                )
            return idx

        if build_target is not None:
            raise ValueError(
                f"build_target must be None for non-build card {card!r}, "
                f"got {build_target!r}"
            )
        idx = Converter._PLAY_TO_INDEX.get(card)
        if idx is None:
            raise ValueError(f"Unknown card type: {card!r}")
        return idx

    @staticmethod
    def policy_index_to_action(index: int) -> tuple[CardType, CardType | None]:
        """Decode policy output index to (card, build_target).

        Per design/20: returns CardType enum, not strings.

        Odd indices → pure play (build_target=None).
        Even indices → build + target.

        Args:
            index: Policy output index (0-12).

        Returns:
            (card, build_target) tuple of CardType.

        Raises:
            IndexError: If index is outside [0, 12].
        """
        if index < 0 or index >= POLICY_OUTPUT_SIZE:
            raise IndexError(
                f"Policy index {index} out of range "
                f"[0, {POLICY_OUTPUT_SIZE - 1}]"
            )
        return Converter._INDEX_TO_ACTION[index]

    @staticmethod
    def build_cost(card_type: CardType) -> int:
        """Return the build (energy) cost for a CardType.

        Per design/20: parameter is CardType enum; cost derived from
        get_card_def() via _BUILD_COST dict.

        Used by AppManager._make_mask_fn for energy-based masking.

        Args:
            card_type: CardType enum member, e.g. CardType.COUNTER.

        Returns:
            Build cost (1 or 2). Returns 999 for unknown CardType values.
        """
        return Converter._BUILD_COST.get(card_type, 999)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_item_to_index(item: PlayerHistoryItem) -> int:
        """Convert a PlayerHistoryItem to policy index.

        Per design/20: passes item.card and item.build_target directly
        as CardType to action_to_policy_index, no .value conversion needed.
        """
        return Converter.action_to_policy_index(
            item.card, item.build_target
        )
