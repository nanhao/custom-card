"""Bidirectional conversion between game state and AI model tensors (design/14, design/27).

Per design/14: 15-d input vector with metadata (HP, energy, round, cycle)
at the front, card counts in the middle, and policy-index-encoded last
actions at the tail. Policy output is 13-d interleaved (build/play pairs).

Per design/27: INPUT_SIZE expanded to 42-d — adds opponent hand counts (7 dims),
status lifetimes for both players (6 dims), own discard pile counts (7 dims),
and own draw pile counts (7 dims).

Per design/20: _CARD_MAP uses CardType enum instead of string literals;
INPUT_SIZE/POLICY_OUTPUT_SIZE imported from ai.rnn_v1.model;
build_cost derived from get_card_def() instead of duplicated in _CARD_MAP.

All methods are stateless. No PyTorch dependency — pure data transformation.
"""

from src.backend.ai.rnn_v1.model import INPUT_SIZE, POLICY_OUTPUT_SIZE
from src.backend.game.card_type import CardType
from src.backend.game.card_def import get_card_def
from src.backend.game.game_state import GameState
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus


class Converter:
    """Bidirectional conversion between game state and model tensors.

    Per design/14, design/27:
    - Input: 42-d (HP×2, energy×2, round, cycle,
              own_hand×7, opp_hand×7,
              own_status×3, opp_status×3,
              own_discard×7, own_draw×7,
              last_action×2)
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
        """Build the 42-d input vector from a GameState.

        Per design/19 D6: accepts GameState directly instead of dict.
        Per design/27: 42-d format — own/opponent hand, status lifetimes,
        own discard/draw pile counts, last actions.

        Args:
            state: GameState from GameCore.get_state().
            player_key: Which player's perspective — "1" or "2".

        Returns:
            list[float] of length 42.

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
        metadata = [
            float(this_player.hp),
            float(opp_player.hp),
            float(this_player.energy),
            float(opp_player.energy),
            float(state.round_in_cycle),
            float(state.cycle),
        ]

        # Own hand counts (indices 6-12)
        own_hand_counts = Converter._count_by_card_type(this_player.hand)

        # Opponent hand counts (indices 13-19)
        opp_hand_counts = Converter._count_by_card_type(opp_player.hand)

        # Own status lifetimes (indices 20-22): DEFENSE, GUARD, FRAGILE
        own_status = [
            Converter._extract_status_lifetime(this_player.statuses, StatusType.DEFENSE),
            Converter._extract_status_lifetime(this_player.statuses, StatusType.GUARD),
            Converter._extract_status_lifetime(this_player.statuses, StatusType.FRAGILE),
        ]

        # Opponent status lifetimes (indices 23-25)
        opp_status = [
            Converter._extract_status_lifetime(opp_player.statuses, StatusType.DEFENSE),
            Converter._extract_status_lifetime(opp_player.statuses, StatusType.GUARD),
            Converter._extract_status_lifetime(opp_player.statuses, StatusType.FRAGILE),
        ]

        # Own discard pile counts (indices 26-32)
        # Per design/27 D4: draw_pile/discard_pile must not be None;
        # caller must pass state with matching player_perspective.
        if this_player.discard_pile is None:
            raise ValueError(
                f"discard_pile is None for player {player_key!r}. "
                f"Call get_state() with player_perspective={player_key!r}."
            )
        own_discard = Converter._count_by_card_type(this_player.discard_pile)

        # Own draw pile counts (indices 33-39)
        if this_player.draw_pile is None:
            raise ValueError(
                f"draw_pile is None for player {player_key!r}. "
                f"Call get_state() with player_perspective={player_key!r}."
            )
        own_draw = Converter._count_by_card_type(this_player.draw_pile)

        # Last action indices (40-41): compute from PlayerHistoryItem
        this_last = -1.0
        opp_last = -1.0

        if this_player.history:
            this_last = float(
                Converter._action_item_to_index(this_player.history[-1])
            )

        if opp_player.history:
            opp_last = float(
                Converter._action_item_to_index(opp_player.history[-1])
            )

        return (
            metadata
            + own_hand_counts
            + opp_hand_counts
            + own_status
            + opp_status
            + own_discard
            + own_draw
            + [this_last, opp_last]
        )

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
    def _count_by_card_type(cards: list[CardType]) -> list[float]:
        """Count occurrences of each CardType in a card list (design/27).

        Returns a 7-element float list in _CARD_TYPE_ORDER sequence
        (attack, defense, heavy, counter, energy, build, flex).

        Args:
            cards: List of CardType values to count.

        Returns:
            list[float] of length 7 — count per card type.
        """
        counts = [0.0] * Converter.NUM_CARD_TYPES
        for card in cards:
            idx = Converter._CARD_TYPE_ORDER[card]
            counts[idx] += 1.0
        return counts

    @staticmethod
    def _extract_status_lifetime(
        statuses: list[PlayerStatus], status_type: StatusType,
    ) -> float:
        """Extract remaining lifetime for a specific status type (design/27).

        Searches the statuses list for a PlayerStatus matching status_type
        and returns its lifetime. If the status type is not present, returns 0.0.

        Args:
            statuses: List of PlayerStatus instances to search.
            status_type: The status type to look for (DEFENSE, GUARD, FRAGILE).

        Returns:
            Lifetime as float, or 0.0 if the status type is not found.
        """
        for ps in statuses:
            if ps.status_type == status_type:
                return float(ps.lifetime)
        return 0.0

    @staticmethod
    def _action_item_to_index(item: PlayerHistoryItem) -> int:
        """Convert a PlayerHistoryItem to policy index.

        Per design/20: passes item.card and item.build_target directly
        as CardType to action_to_policy_index, no .value conversion needed.
        """
        return Converter.action_to_policy_index(
            item.card, item.build_target
        )
