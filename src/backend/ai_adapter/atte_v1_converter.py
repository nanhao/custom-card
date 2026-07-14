"""AtteV1Converter — bidirectional conversion between game state and atte_v1 model input.

Per design/40: handles variable-length card sequences instead of fixed 62-d vectors.
All methods are stateless. No PyTorch dependency — pure data transformation.
"""

from src.backend.ai.atte_v1.constants import MAX_ENERGY, MAX_HP, POLICY_OUTPUT_SIZE
from src.backend.game.card_type import CardType
from src.backend.game.card_def import get_card_def
from src.backend.game.game_state_snapshot import GameStateSnapshot
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import PlayerStatus, StatusType
from src.backend.game.player_state_snapshot import PlayerStateSnapshot


# ---------------------------------------------------------------------------
# Input/output dataclasses (per design/40 interfaces §2)
# ---------------------------------------------------------------------------


class AtteV1PlayerInput:
    """Single player's input data for one timestep.

    Includes own hand/discard/draw sequences plus opponent's hand/discard/draw.
    Per design/40 §2.1.
    """

    def __init__(
        self,
        hand_ids: list[int],
        discard_ids: list[int],
        draw_ids: list[int],
        last_action: int,
        hp: float,
        energy: float,
        status_ids: list[int],
        status_durations: list[int],
        opp_hand_ids: list[int],
        opp_discard_ids: list[int],
        opp_draw_ids: list[int],
    ) -> None:
        """Initialize AtteV1PlayerInput.

        Args:
            hand_ids: Action ID sequence for cards in hand.
            discard_ids: Action ID sequence for cards in discard pile.
            draw_ids: Action ID sequence for cards in draw pile.
            last_action: Previous action ID, -1 if first round.
            hp: Normalized HP in [0, 1].
            energy: Normalized energy in [0, 1].
            status_ids: Status type ID sequence.
            status_durations: Status duration sequence (same length as status_ids).
            opp_hand_ids: Opponent's hand action ID sequence.
            opp_discard_ids: Opponent's discard pile action ID sequence.
            opp_draw_ids: Opponent's draw pile action ID sequence.
        """
        self.hand_ids = hand_ids
        self.discard_ids = discard_ids
        self.draw_ids = draw_ids
        self.last_action = last_action
        self.hp = hp
        self.energy = energy
        self.status_ids = status_ids
        self.status_durations = status_durations
        self.opp_hand_ids = opp_hand_ids
        self.opp_discard_ids = opp_discard_ids
        self.opp_draw_ids = opp_draw_ids

    def __eq__(self, other: object) -> bool:
        """Equality check for testing convenience."""
        if not isinstance(other, AtteV1PlayerInput):
            return NotImplemented
        return (
            self.hand_ids == other.hand_ids
            and self.discard_ids == other.discard_ids
            and self.draw_ids == other.draw_ids
            and self.last_action == other.last_action
            and self.hp == other.hp
            and self.energy == other.energy
            and self.status_ids == other.status_ids
            and self.status_durations == other.status_durations
            and self.opp_hand_ids == other.opp_hand_ids
            and self.opp_discard_ids == other.opp_discard_ids
            and self.opp_draw_ids == other.opp_draw_ids
        )


class AtteV1Input:
    """Complete input for one timestep of atte_v1 model.

    Per design/40 §2.2: round_in_cycle uses -1/0/1 semantics,
    different from GameStateSnapshot.round_in_cycle (1-3).
    Converter handles the mapping: 1→-1, 2→0, 3→1.
    """

    def __init__(
        self,
        own: AtteV1PlayerInput,
        opp: AtteV1PlayerInput,
        round_in_cycle: int,
    ) -> None:
        """Initialize AtteV1Input.

        Args:
            own: Own player's input data.
            opp: Opponent player's input data.
            round_in_cycle: -1 (round 1), 0 (round 2), or 1 (round 3).
        """
        self.own = own
        self.opp = opp
        self.round_in_cycle = round_in_cycle


# ---------------------------------------------------------------------------
# AtteV1Converter
# ---------------------------------------------------------------------------


class AtteV1Converter:
    """Bidirectional conversion between game state and atte_v1 model input.

    Per design/40 §3: handles variable-length card sequences.
    All methods are stateless. The class serves as a namespace for
    conversion logic bridging game domain types and AI model inputs.

    Card → action ID mapping (per intention/new_network.md):
        ATTACK → 3, DEFENSE → 6, HEAVY → 9, COUNTER → 12,
        ENERGY → 15, FLEX → 18, IRON_DEFENSE → 21.
        BUILD maps to 16 action IDs (1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23)
        and is NOT directly mapped by card_to_action_id.

    Policy output (24-d, per intention/new_network.md):
        0=pass, then triplets of (play, build, dismantle) per card type
        (BUILD has no play index).
    """

    # ------------------------------------------------------------------
    # Card → action ID mapping table
    # ------------------------------------------------------------------

    _CARD_TO_ACTION: dict[CardType, int] = {
        CardType.ATTACK: 3,
        CardType.DEFENSE: 6,
        CardType.HEAVY: 9,
        CardType.COUNTER: 12,
        CardType.ENERGY: 15,
        CardType.FLEX: 18,
        CardType.IRON_DEFENSE: 21,
    }

    # ------------------------------------------------------------------
    # Status → ID mapping
    # ------------------------------------------------------------------

    _STATUS_TO_ID: dict[StatusType, int] = {
        StatusType.DEFENSE: 0,
        StatusType.GUARD: 1,
        StatusType.FRAGILE: 2,
    }

    # ------------------------------------------------------------------
    # Card → policy index mapping (24-d)
    # Format: (card_type, play_index, build_index, dismantle_index)
    # BUILD has no play_index (must use build or dismantle mode).
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

    # Derived lookup tables
    _PLAY_TO_INDEX: dict[CardType, int] = {
        c[0]: c[1] for c in _CARD_MAP if c[1] is not None
    }

    _BUILD_TO_INDEX: dict[CardType, int] = {c[0]: c[2] for c in _CARD_MAP}

    _DISMANTLE_TO_INDEX: dict[CardType, int] = {c[0]: c[3] for c in _CARD_MAP}

    _INDEX_TO_ACTION: dict[
        int, tuple[CardType | None, CardType | None, CardType | None]
    ] = {
        0: (None, None, None),
        **{c[2]: (CardType.BUILD, c[0], None) for c in _CARD_MAP},
        **{c[3]: (CardType.BUILD, None, c[0]) for c in _CARD_MAP},
        **{c[1]: (c[0], None, None) for c in _CARD_MAP if c[1] is not None},
    }

    _BUILD_COST: dict[CardType, int] = {
        c[0]: get_card_def(c[0]).build_cost for c in _CARD_MAP
    }

    # ------------------------------------------------------------------
    # Public API — Card / Status / Action mapping
    # ------------------------------------------------------------------

    @staticmethod
    def card_to_action_id(card: CardType) -> int:
        """Map CardType to play action ID. BUILD raises ValueError.

        Per intention/new_network.md: non-BUILD cards have a single play
        action ID. BUILD maps to 16 distinct action IDs depending on
        build/dismantle target — use action_to_policy_index for BUILD.

        Args:
            card: CardType to map.

        Returns:
            Action ID (int) for the play action.

        Raises:
            ValueError: If card is BUILD (cannot be mapped directly).
        """
        if card == CardType.BUILD:
            raise ValueError(
                "BUILD card cannot be mapped to a single action ID. "
                "Use action_to_policy_index with build_target or dismantle_target."
            )
        return AtteV1Converter._CARD_TO_ACTION[card]

    @staticmethod
    def status_to_id(status_type: StatusType) -> int:
        """Map StatusType to status ID.

        Args:
            status_type: StatusType to map.

        Returns:
            Status ID: DEFENSE→0, GUARD→1, FRAGILE→2.
        """
        return AtteV1Converter._STATUS_TO_ID[status_type]

    @staticmethod
    def action_to_policy_index(
        card: CardType | None,
        build_target: CardType | None = None,
        dismantle_target: CardType | None = None,
    ) -> int:
        """Convert game action to policy output index (24-d).

        Per design/40: 24-d output. Index 0 = pass.

        Args:
            card: CardType played, or None for pass.
            build_target: Build target (only when card=BUILD, build mode).
            dismantle_target: Dismantle target (only when card=BUILD, dismantle mode).

        Returns:
            Policy output index (0-23).

        Raises:
            ValueError: If card is unknown, or build/dismantle targets are
                mutually exclusive or missing.
        """
        if card is None:
            if build_target is not None or dismantle_target is not None:
                raise ValueError(
                    "build_target/dismantle_target must be None for pass."
                )
            return 0

        if card == CardType.BUILD:
            if build_target is not None and dismantle_target is not None:
                raise ValueError(
                    "build_target and dismantle_target are mutually exclusive."
                )
            if build_target is not None:
                idx = AtteV1Converter._BUILD_TO_INDEX.get(build_target)
                if idx is None:
                    raise ValueError(f"Unknown build_target: {build_target!r}")
                return idx
            if dismantle_target is not None:
                idx = AtteV1Converter._DISMANTLE_TO_INDEX.get(dismantle_target)
                if idx is None:
                    raise ValueError(f"Unknown dismantle_target: {dismantle_target!r}")
                return idx
            raise ValueError(
                "BUILD card requires build_target or dismantle_target."
            )

        if build_target is not None or dismantle_target is not None:
            raise ValueError(
                f"build_target/dismantle_target must be None "
                f"for non-BUILD card {card!r}."
            )
        idx = AtteV1Converter._PLAY_TO_INDEX.get(card)
        if idx is None:
            raise ValueError(f"Unknown card type: {card!r}")
        return idx

    @staticmethod
    def policy_index_to_action(
        index: int,
    ) -> tuple[CardType | None, CardType | None, CardType | None]:
        """Decode policy output index to (card, build_target, dismantle_target).

        Args:
            index: Policy output index (0-23).

        Returns:
            (card, build_target, dismantle_target) tuple.

        Raises:
            IndexError: If index is outside [0, 23].
        """
        if index < 0 or index >= POLICY_OUTPUT_SIZE:
            raise IndexError(
                f"Policy index {index} out of range [0, {POLICY_OUTPUT_SIZE - 1}]"
            )
        return AtteV1Converter._INDEX_TO_ACTION[index]

    @staticmethod
    def build_cost(card_type: CardType) -> int:
        """Return the build (energy) cost for a CardType.

        Args:
            card_type: CardType to query.

        Returns:
            Build cost (1 or 2). Returns 999 for unknown CardType values.
        """
        return AtteV1Converter._BUILD_COST.get(card_type, 999)

    # ------------------------------------------------------------------
    # Public API — Input construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_input(
        state: GameStateSnapshot, player_key: str,
    ) -> AtteV1Input:
        """Build AtteV1Input from a GameStateSnapshot.

        Per design/40 §3:
        - HP/30, energy/5 normalization.
        - round_in_cycle: GameStateSnapshot 1→-1, 2→0, 3→1.
        - First round last_action = -1 (no history).
        - BUILD cards are filtered from hand_ids (not directly mappable).
        - Opponent hand/discard/draw are included.

        Args:
            state: GameStateSnapshot from GameCore.get_state().
            player_key: "1" or "2" (required, no default).

        Returns:
            AtteV1Input with own/opp player data and round_in_cycle.

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

        # Build own player input
        own = AtteV1Converter._build_player_input(
            player=this_player,
            opponent=opp_player,
        )

        # Build opponent player input
        opp = AtteV1Converter._build_player_input(
            player=opp_player,
            opponent=this_player,
        )

        # Map round_in_cycle: 1→-1, 2→0, 3→1
        round_mapped = state.round_in_cycle - 2

        return AtteV1Input(own=own, opp=opp, round_in_cycle=round_mapped)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_player_input(
        player: PlayerStateSnapshot,
        opponent: PlayerStateSnapshot,
    ) -> AtteV1PlayerInput:
        """Build AtteV1PlayerInput for one player from snapshots.

        Args:
            player: This player's state snapshot.
            opponent: The opponent's state snapshot.

        Returns:
            AtteV1PlayerInput for the player.
        """
        # Map cards to action IDs, filtering out BUILD
        hand_ids = AtteV1Converter._cards_to_action_ids(player.hand)
        discard_ids = AtteV1Converter._cards_to_action_ids(player.discard_pile)
        draw_ids = AtteV1Converter._cards_to_action_ids(player.draw_pile)

        # Last action from history
        last_action = AtteV1Converter._extract_last_action(player.history)

        # Normalized scalars
        hp = float(player.hp) / MAX_HP
        energy = float(player.energy) / MAX_ENERGY

        # Status IDs and durations
        status_ids, status_durations = AtteV1Converter._extract_statuses(
            player.statuses,
        )

        # Opponent data (visible per design/34)
        opp_hand_ids = AtteV1Converter._cards_to_action_ids(opponent.hand)
        opp_discard_ids = AtteV1Converter._cards_to_action_ids(
            opponent.discard_pile,
        )
        opp_draw_ids = AtteV1Converter._cards_to_action_ids(opponent.draw_pile)

        return AtteV1PlayerInput(
            hand_ids=hand_ids,
            discard_ids=discard_ids,
            draw_ids=draw_ids,
            last_action=last_action,
            hp=hp,
            energy=energy,
            status_ids=status_ids,
            status_durations=status_durations,
            opp_hand_ids=opp_hand_ids,
            opp_discard_ids=opp_discard_ids,
            opp_draw_ids=opp_draw_ids,
        )

    @staticmethod
    def _cards_to_action_ids(cards: list[CardType]) -> list[int]:
        """Convert a list of CardType to action IDs, filtering out BUILD.

        Per design/40: BUILD cards are not directly mappable to a single
        action ID. They are filtered from the sequence — the caller uses
        action_to_policy_index to select the specific build/dismantle target.

        Args:
            cards: List of CardType values.

        Returns:
            List of action IDs for non-BUILD cards.
        """
        result: list[int] = []
        for card in cards:
            if card == CardType.BUILD:
                continue  # BUILD is filtered per design/40
            result.append(AtteV1Converter._CARD_TO_ACTION[card])
        return result

    @staticmethod
    def _extract_last_action(history: list[PlayerHistoryItem]) -> int:
        """Extract the last action as a policy index from history.

        Per design/40: returns -1 if history is empty (first round).
        The -1 is later mapped to PAD_ACTION_ID(24) by the model's
        embedding layer.

        Args:
            history: Player's action history.

        Returns:
            Policy index of last action, or -1 if no history.
        """
        if not history:
            return -1
        last = history[-1]
        return AtteV1Converter.action_to_policy_index(
            last.card, last.build_target, last.dismantle_target,
        )

    @staticmethod
    def _extract_statuses(
        statuses: list[PlayerStatus],
    ) -> tuple[list[int], list[int]]:
        """Extract status IDs and durations from a status list.

        Both returned lists have the same length and are aligned by index.

        Args:
            statuses: List of PlayerStatus instances.

        Returns:
            (status_ids, status_durations) tuple of aligned lists.
        """
        ids: list[int] = []
        durations: list[int] = []
        for ps in statuses:
            ids.append(AtteV1Converter._STATUS_TO_ID[ps.status_type])
            durations.append(ps.lifetime)
        return ids, durations
