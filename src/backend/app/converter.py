"""Bidirectional conversion between game state and AI model tensors (design/37).

Per design/37: INPUT_SIZE 56→62 (8 card types), POLICY_OUTPUT_SIZE 14→24
(adds dismantle dimension). _CARD_MAP is now a 4-tuple (play, build, dismantle).

Input layout per intention/input.md (62-d):
    indices 0-5:   metadata (HP×2, energy×2, round_in_cycle, cycle)
    indices 6-13:  own hand counts (8 card types)
    indices 14-21: opponent hand counts (8 card types)
    indices 22-24: own status lifetimes (DEFENSE, GUARD, FRAGILE)
    indices 25-27: opponent status lifetimes
    indices 28-35: own discard pile counts (8 types)
    indices 36-43: opponent discard pile counts (8 types)
    indices 44-51: own draw pile counts (8 types)
    indices 52-59: opponent draw pile counts (8 types)
    index 60:      own last action policy index (-1 if first round)
    index 61:      opponent last action policy index

Policy output per intention/output.md (24-d):
    index 0:                       pass
    index 1,4,7,...,22:            build mode + build target
    index 2,5,8,...,23:            dismantle mode + dismantle target
    index 3,6,9,...,21:            pure card play

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

    Per design/37:
    - Input: 62-d (HP×2, energy×2, round, cycle,
              own_hand×8, opp_hand×8,
              own_status×3, opp_status×3,
              own_discard×8, opp_discard×8,
              own_draw×8, opp_draw×8,
              last_action×2)
    - Policy output: 24-d (0=pass, triplets: play/build/dismantle per card type)
    - Last action encoded as policy index, -1 when no history.

    Per design/20: all card mappings use CardType enum; INPUT_SIZE,
    POLICY_OUTPUT_SIZE imported from ai.rnn_v1.model; build_cost derived
    from get_card_def().

    Per design/34 D4: piles always non-None — no ValueError checks needed.

    All methods are stateless. The class serves as a namespace for
    conversion logic bridging game domain types and AI model inputs.
    """

    NUM_CARD_TYPES: int = 8

    # ------------------------------------------------------------------
    # Master card mapping table — single source of truth for all
    # card↔index mappings (design/37).
    #
    # Format: (card_type, play_index, build_index, dismantle_index).
    #   card_type: CardType enum member.
    #   play_index: policy output index for pure play (None for BUILD).
    #   build_index: policy output index for build + target.
    #   dismantle_index: policy output index for dismantle + target.
    #
    # Per design/37: 24-d output — pass(0) + 23 action indices.
    # BUILD has no play_index; all other cards have play/build/dismantle.
    #
    # All other mapping dicts are derived from this single table.
    # ------------------------------------------------------------------

    _CARD_MAP: list[tuple[CardType, int | None, int, int]] = [
        (CardType.BUILD,         None,  1,  2),
        (CardType.ATTACK,           3,  4,  5),
        (CardType.DEFENSE,          6,  7,  8),
        (CardType.HEAVY,            9, 10, 11),
        (CardType.COUNTER,         12, 13, 14),
        (CardType.ENERGY,          15, 16, 17),
        (CardType.FLEX,            18, 19, 20),
        (CardType.IRON_DEFENSE,    21, 22, 23),
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

    # Build actions: build_target (CardType) → policy output index
    _BUILD_TO_INDEX: dict[CardType, int] = {c[0]: c[2] for c in _CARD_MAP}

    # Dismantle actions: dismantle_target (CardType) → policy output index (design/37)
    _DISMANTLE_TO_INDEX: dict[CardType, int] = {c[0]: c[3] for c in _CARD_MAP}

    # Reverse: policy index → (card, build_target, dismantle_target) — design/37 3-tuple
    _INDEX_TO_ACTION: dict[int, tuple[CardType | None, CardType | None, CardType | None]] = {
        0: (None, None, None),  # pass
        **{c[2]: (CardType.BUILD, c[0], None) for c in _CARD_MAP},     # build mode
        **{c[3]: (CardType.BUILD, None, c[0]) for c in _CARD_MAP},     # dismantle mode
        **{c[1]: (c[0], None, None) for c in _CARD_MAP if c[1] is not None},  # pure play
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
        """Build the 62-d input vector from a GameState (design/37).

        Per design/19 D6: accepts GameState directly instead of dict.
        Per design/34: 56-d format — own/opponent hand, status lifetimes,
        own/opponent discard/draw pile counts, last actions.
        Input layout per intention/input.md.

        Args:
            state: GameState from GameCore.get_state().
            player_key: Which player's perspective — "1" or "2".

        Returns:
            list[float] of length 62.

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
        card: CardType | None,
        build_target: CardType | None = None,
        dismantle_target: CardType | None = None,
    ) -> int:
        """Convert game action to policy index (design/37: 24-d).

        Args:
            card: CardType played, or None for pass.
            build_target: Build target (only when card=BUILD, build mode).
            dismantle_target: Dismantle target (only when card=BUILD, dismantle mode).

        Returns:
            Policy output index (0-23).

        Raises:
            ValueError: If card is unknown, build/dismantle targets are misconfigured.
        """
        # Pass action
        if card is None:
            if build_target is not None or dismantle_target is not None:
                raise ValueError(
                    "build_target/dismantle_target must be None for pass."
                )
            return 0

        if card == CardType.BUILD:
            # Must have exactly one target (build XOR dismantle)
            if build_target is not None and dismantle_target is not None:
                raise ValueError(
                    "build_target and dismantle_target are mutually exclusive."
                )
            if build_target is not None:
                idx = Converter._BUILD_TO_INDEX.get(build_target)
                if idx is None:
                    raise ValueError(f"Unknown build_target: {build_target!r}")
                return idx
            if dismantle_target is not None:
                idx = Converter._DISMANTLE_TO_INDEX.get(dismantle_target)
                if idx is None:
                    raise ValueError(f"Unknown dismantle_target: {dismantle_target!r}")
                return idx
            raise ValueError(
                "BUILD card requires build_target or dismantle_target."
            )

        # Non-BUILD card: pure play
        if build_target is not None or dismantle_target is not None:
            raise ValueError(
                f"build_target/dismantle_target must be None for non-BUILD card {card!r}."
            )
        idx = Converter._PLAY_TO_INDEX.get(card)
        if idx is None:
            raise ValueError(f"Unknown card type: {card!r}")
        return idx

    @staticmethod
    def policy_index_to_action(
        index: int,
    ) -> tuple[CardType | None, CardType | None, CardType | None]:
        """Decode policy output index to (card, build_target, dismantle_target).

        Per design/37: returns 3-tuple. Index 0 returns (None, None, None) for pass.

        Args:
            index: Policy output index (0-23).

        Returns:
            (card, build_target, dismantle_target) tuple.

        Raises:
            IndexError: If index is outside [0, 23].
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
        """Convert a PlayerHistoryItem to policy index (design/37).

        Passes dismantle_target for BUILD dismantle mode encoding.
        """
        return Converter.action_to_policy_index(
            item.card, item.build_target, item.dismantle_target,
        )
