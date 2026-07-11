"""Bidirectional conversion between game state and AI model tensors (design/14, design/34, design/28).

Per design/14: 15-d input vector with metadata (HP, energy, round, cycle)
at the front, card counts in the middle, and policy-index-encoded last
actions at the tail. Policy output is 13-d interleaved (build/play pairs).

Per design/34: INPUT_SIZE expanded to 56-d — adds opponent discard pile (7 dims),
opponent draw pile (7 dims); own draw pile and last actions shifted accordingly.
Input layout per intention/input.md.

Per design/28: POLICY_OUTPUT_SIZE expanded to 14-d — adds pass at index 0;
all non-pass indices shift +1; even indices now pure play, odd indices build.

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

    Per design/14, design/34, design/28:
    - Input: 56-d (HP×2, energy×2, round, cycle,
              own_hand×7, opp_hand×7,
              own_status×3, opp_status×3,
              own_discard×7, opp_discard×7,
              own_draw×7, opp_draw×7,
              last_action×2)
    - Policy output: 14-d (0=pass, even 2/4/6/8/10/12=pure play, odd 1/3/5/7/9/11/13=build)
    - Last action encoded as policy index, -1 when no history.

    Per design/20: all card mappings use CardType enum; INPUT_SIZE,
    POLICY_OUTPUT_SIZE imported from ai.rnn_v1.model; build_cost derived
    from get_card_def().

    Per design/34 D4: piles always non-None — no ValueError checks needed.

    All methods are stateless. The class serves as a namespace for
    conversion logic bridging game domain types and AI model inputs.
    """

    NUM_CARD_TYPES: int = 7

    # ------------------------------------------------------------------
    # Master card mapping table — single source of truth for all
    # card↔index mappings (design/20, design/28).
    #
    # Per design/28: all indices shifted +1 from original 13-d layout.
    # Index 0 is reserved for pass (not in _CARD_MAP).
    # Even indices (2,4,6,8,10,12) = pure play; odd indices (1,3,5,7,9,11,13) = build.
    #
    # Format: (card_type, play_index, build_index).
    #   card_type: CardType enum member.
    #   play_index: even-numbered policy output index (None if not directly playable).
    #   build_index: odd-numbered policy output index for build+target.
    #
    # Per design/20 D3: build_cost is NOT stored here; it is derived
    # from get_card_def(card_type).build_cost to avoid duplication.
    #
    # All other mapping dicts (ALL_CARD_TYPES, _PLAY_TO_INDEX, etc.) are
    # derived from this single table.
    # ------------------------------------------------------------------

    _CARD_MAP: list[tuple[CardType, int | None, int]] = [
        (CardType.BUILD,   None, 1),   # build_index 1 (design/28: +1)
        (CardType.ATTACK,  2, 3),      # play_index 2, build_index 3
        (CardType.DEFENSE, 4, 5),
        (CardType.HEAVY,   6, 7),
        (CardType.COUNTER, 8, 9),
        (CardType.ENERGY,  10, 11),
        (CardType.FLEX,    12, 13),
    ]

    # Public set of all valid CardType values
    ALL_CARD_TYPES: frozenset[CardType] = frozenset(c[0] for c in _CARD_MAP)

    # ------------------------------------------------------------------
    # Internal: CardType ↔ policy-index mapping tables
    # (all derived from _CARD_MAP)
    # ------------------------------------------------------------------

    _CARD_TYPE_ORDER: dict[CardType, int] = {c[0]: i for i, c in enumerate(_CARD_MAP)}

    # Pure play actions: CardType → policy output index (even numbers, design/28)
    _PLAY_TO_INDEX: dict[CardType, int] = {
        c[0]: c[1] for c in _CARD_MAP if c[1] is not None
    }

    # Build actions: build_target (CardType) → policy output index (odd numbers, design/28)
    _BUILD_TO_INDEX: dict[CardType, int] = {c[0]: c[2] for c in _CARD_MAP}

    # Reverse: policy index → (card, build_target), includes pass at index 0 (design/28)
    _INDEX_TO_ACTION: dict[int, tuple[CardType | None, CardType | None]] = {
        0: (None, None),  # pass (design/28)
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
        """Build the 56-d input vector from a GameState.

        Per design/19 D6: accepts GameState directly instead of dict.
        Per design/34: 56-d format — own/opponent hand, status lifetimes,
        own/opponent discard/draw pile counts, last actions.
        Input layout per intention/input.md.

        Args:
            state: GameState from GameCore.get_state().
            player_key: Which player's perspective — "1" or "2".

        Returns:
            list[float] of length 56.

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
        # Per design/34 D4: piles always non-None, no check needed.
        own_discard = Converter._count_by_card_type(this_player.discard_pile)

        # Opponent discard pile counts (indices 33-39) — NEW in design/34
        opp_discard = Converter._count_by_card_type(opp_player.discard_pile)

        # Own draw pile counts (indices 40-46) — shifted from 33-39 in design/27
        own_draw = Converter._count_by_card_type(this_player.draw_pile)

        # Opponent draw pile counts (indices 47-53) — NEW in design/34
        opp_draw = Converter._count_by_card_type(opp_player.draw_pile)

        # Last action indices (54-55): compute from PlayerHistoryItem — shifted from 40-41
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
            + opp_discard
            + own_draw
            + opp_draw
            + [this_last, opp_last]
        )

    # ------------------------------------------------------------------
    # Action encoding/decoding
    # ------------------------------------------------------------------

    @staticmethod
    def action_to_policy_index(
        card: CardType | None, build_target: CardType | None,
    ) -> int:
        """Convert game action (card + optional build target) to policy index.

        Per design/20: parameters are CardType enum, not strings.
        Per design/28: card=None, build_target=None → 0 (pass).

        Mapping (14-d, design/28):
            pass → 0
            Pure play → even indices:  ATTACK→2, DEFENSE→4, ...
            Build+target → odd indices:  BUILD+ATTACK→3, ...

        Args:
            card: CardType played, or None for pass.
            build_target: Build target CardType, only meaningful when card=BUILD.

        Returns:
            Policy output index (0-13).

        Raises:
            ValueError: If card is unknown, or build_target is invalid/absent
                        for a build action, or present for a non-build action.
        """
        # Pass action: no card and no build target (design/28)
        if card is None:
            if build_target is not None:
                raise ValueError(
                    f"build_target must be None when card is None (pass), "
                    f"got {build_target!r}"
                )
            return 0

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
    def policy_index_to_action(index: int) -> tuple[CardType | None, CardType | None]:
        """Decode policy output index to (card, build_target).

        Per design/20: returns CardType enum, not strings.
        Per design/28: index 0 returns (None, None) for pass.

        Even indices (2,4,6,8,10,12) → pure play (build_target=None).
        Odd indices (1,3,5,7,9,11,13) → build + target.

        Args:
            index: Policy output index (0-13).

        Returns:
            (card, build_target) tuple; both None if pass.

        Raises:
            IndexError: If index is outside [0, 13].
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
        as CardType to action_to_policy_index.
        Per design/28: item.card may be None (pass), returns 0.
        """
        return Converter.action_to_policy_index(
            item.card, item.build_target
        )
