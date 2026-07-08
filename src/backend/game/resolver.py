"""Priority-layered card resolution engine for the v2 game (design/11).

Per design/11 D3, cards are decomposed into Effect units and resolved
in priority order (11 first, then 10). Statuses are applied immediately
within each priority group; damage is accumulated and returned.

Per architecture D9, this module is pure logic — no UI, no I/O.
"""

from dataclasses import dataclass, field

from src.backend.game.card import CardType, EffectType, get_card_def
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus, has_active_status

# Maximum damage from consecutive attack streak
MAX_ATTACK_STREAK_DAMAGE = 3


@dataclass
class Effect:
    """Single card effect unit used internally by CardResolver.

    Attributes:
        source_player: "1" or "2" — which player produced this effect.
        priority: 10 or 11 — resolution priority.
        source_card: The CardType that produced this effect.
        effect_type: EffectType enum member identifying the effect logic.
    """

    source_player: str
    priority: int
    source_card: CardType
    effect_type: EffectType


@dataclass
class ResolutionResult:
    """Value object returned by CardResolver.resolve().

    Attributes:
        damage_1: Damage dealt to Player 1 this round.
        damage_2: Damage dealt to Player 2 this round.
        energy_delta_1: Energy gained by Player 1 this round (before clamping).
        energy_delta_2: Energy gained by Player 2 this round (before clamping).
        statuses_granted_1: New statuses granted to Player 1 this round.
        statuses_granted_2: New statuses granted to Player 2 this round.
        details: Human-readable resolution steps.
    """

    damage_1: int
    damage_2: int
    energy_delta_1: int = 0
    energy_delta_2: int = 0
    statuses_granted_1: list[StatusType] = field(default_factory=list)
    statuses_granted_2: list[StatusType] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


class CardResolver:
    """Stateless priority-layered card resolution engine.

    All methods are static. The resolve() method decomposes played cards
    into Effect units, groups by priority, and resolves statuses and damage.
    """

    @staticmethod
    def get_consecutive_attack(history: list[PlayerHistoryItem]) -> int:
        """Count consecutive ATTACK cards from the end of history.

        Only ATTACK cards count toward the streak. COUNTER, DEFENSE, and
        all other card types break it.

        Per design/16 D5, accepts list[PlayerHistoryItem] instead of
        list[CardType]; internally reads item.card for the card type.

        Used externally by GameCore (cross-round retention).

        Args:
            history: Chronological list of PlayerHistoryItem entries.

        Returns:
            Number of consecutive ATTACK cards at the end of history.
        """
        count = 0
        for item in reversed(history):
            if item.card == CardType.ATTACK:
                count += 1
            else:
                break
        return count

    @staticmethod
    def resolve(
        cards_1: list[CardType],
        cards_2: list[CardType],
        player_1_statuses: list[PlayerStatus],
        player_2_statuses: list[PlayerStatus],
        p1_attack_streak: int,
        p2_attack_streak: int,
    ) -> ResolutionResult:
        """Compute all effects for one turn via priority-layered resolution.

        Algorithm:
        1. Decompose each card into Effect units from CardDef.effects.
        2. Group effects by priority (11, 10).
        3. Process priority 11 first: status effects applied immediately.
        4. Process priority 10: damage accumulated, energy counted.
        5. Return aggregated ResolutionResult.

        Args:
            cards_1: Cards played by Player 1 (expects exactly 1).
            cards_2: Cards played by Player 2 (expects exactly 1).
            player_1_statuses: P1's statuses before this round.
            player_2_statuses: P2's statuses before this round.
            p1_attack_streak: P1's consecutive attack count before this round.
            p2_attack_streak: P2's consecutive attack count before this round.

        Returns:
            ResolutionResult with damage, energy deltas, and granted statuses.
        """
        # Track whether defensive status was pre-existing (for grant_guard logic).
        p1_had_defense_pre = has_active_status(player_1_statuses, StatusType.DEFENSE)
        p2_had_defense_pre = has_active_status(player_2_statuses, StatusType.DEFENSE)

        # Working state — start empty; all statuses must be re-granted by effects each round.
        # player_1_statuses/player_2_statuses are only used for had_defense_pre checks.
        statuses_1: list[StatusType] = []
        statuses_2: list[StatusType] = []
        dmg_1 = 0
        dmg_2 = 0
        energy_1 = 0
        energy_2 = 0
        details: list[str] = []

        # Decompose both cards into Effect units
        cd1 = get_card_def(cards_1[0])
        cd2 = get_card_def(cards_2[0])
        all_effects: list[Effect] = []
        for (etype, pri) in cd1.effects:
            all_effects.append(Effect("1", pri, cards_1[0], etype))
        for (etype, pri) in cd2.effects:
            all_effects.append(Effect("2", pri, cards_2[0], etype))

        # Process by priority: 11 first, then 10
        for priority in [11, 10]:
            group = [e for e in all_effects if e.priority == priority]
            for effect in group:
                opp = cards_2[0] if effect.source_player == "1" else cards_1[0]
                result = _process_effect(
                    effect, statuses_1, statuses_2,
                    p1_had_defense_pre, p2_had_defense_pre,
                    p1_attack_streak, p2_attack_streak, opp,
                    dmg_1, dmg_2, energy_1, energy_2,
                )
                dmg_1, dmg_2, energy_1, energy_2, statuses_1, statuses_2, detail = result
                if detail:
                    details.append(detail)

        # Return all statuses computed this round (no pre-round carry-over)

        return ResolutionResult(
            damage_1=dmg_1,
            damage_2=dmg_2,
            energy_delta_1=energy_1,
            energy_delta_2=energy_2,
            statuses_granted_1=list(statuses_1),
            statuses_granted_2=list(statuses_2),
            details=details,
        )


# ---------------------------------------------------------------------------
# Pure effect processor — no side effects, returns new state tuple
# ---------------------------------------------------------------------------


def _process_effect(
    effect: Effect,
    statuses_1: list[StatusType],
    statuses_2: list[StatusType],
    p1_had_defense_pre: bool,
    p2_had_defense_pre: bool,
    p1_attack_streak: int,
    p2_attack_streak: int,
    opp_card: CardType,
    dmg_1: int,
    dmg_2: int,
    energy_1: int,
    energy_2: int,
) -> tuple[int, int, int, int, list[StatusType], list[StatusType], str]:
    """Process a single Effect and return updated accumulators + detail string.

    Returns a tuple of (dmg_1, dmg_2, energy_1, energy_2, statuses_1, statuses_2, detail).
    Status lists are copied on modification to maintain immutability of the return.
    """
    detail = ""
    src = effect.source_player

    if effect.effect_type == EffectType.GRANT_DEFENSE:
        if src == "1":
            statuses_1 = list(statuses_1) + [StatusType.DEFENSE]
            detail = "P1 gains defense"
        elif src == "2":
            statuses_2 = list(statuses_2) + [StatusType.DEFENSE]
            detail = "P2 gains defense"

    elif effect.effect_type == EffectType.GRANT_GUARD:
        if src == "1" and p1_had_defense_pre:
            if StatusType.GUARD not in statuses_1:
                statuses_1 = list(statuses_1) + [StatusType.GUARD]
                detail = "P1 gains GUARD (consecutive defense)"
        elif src == "2" and p2_had_defense_pre:
            if StatusType.GUARD not in statuses_2:
                statuses_2 = list(statuses_2) + [StatusType.GUARD]
                detail = "P2 gains GUARD (consecutive defense)"

    elif effect.effect_type == EffectType.DAMAGE:
        if effect.source_card == CardType.ATTACK:
            streak = p1_attack_streak if src == "1" else p2_attack_streak
            raw = min(streak + 1, MAX_ATTACK_STREAK_DAMAGE)
            if src == "1":
                if StatusType.DEFENSE not in statuses_2:
                    dmg_2 += raw
                    detail = f"P1 Attack(streak={streak}) → P2 takes {raw}"
                else:
                    detail = "P1 Attack blocked by P2 defense"
            else:
                if StatusType.DEFENSE not in statuses_1:
                    dmg_1 += raw
                    detail = f"P2 Attack(streak={streak}) → P1 takes {raw}"
                else:
                    detail = "P2 Attack blocked by P1 defense"
        elif effect.source_card == CardType.HEAVY:
            if src == "1":
                if StatusType.GUARD not in statuses_2:
                    dmg_2 += 1
                    detail = "P1 Heavy Strike → P2 takes 1"
                else:
                    detail = "P1 Heavy Strike blocked by P2 GUARD"
            else:
                if StatusType.GUARD not in statuses_1:
                    dmg_1 += 1
                    detail = "P2 Heavy Strike → P1 takes 1"
                else:
                    detail = "P2 Heavy Strike blocked by P1 GUARD"

    elif effect.effect_type == EffectType.REFLECT:
        if opp_card in (CardType.ATTACK, CardType.HEAVY):
            if src == "1":
                dmg_2 += 1
                detail = f"P1 Counter reflects {opp_card.value} → P2 takes 1"
            else:
                dmg_1 += 1
                detail = f"P2 Counter reflects {opp_card.value} → P1 takes 1"

    elif effect.effect_type == EffectType.GAIN_ENERGY:
        if src == "1":
            energy_1 += 1
            detail = "P1 gains 1 energy"
        else:
            energy_2 += 1
            detail = "P2 gains 1 energy"

    elif effect.effect_type == EffectType.FLEX_DAMAGE:
        if src == "1":
            if StatusType.DEFENSE not in statuses_2:
                dmg_2 += 1
                detail = "P1 FLEX damage → P2 takes 1"
            else:
                detail = "P1 FLEX damage blocked by P2 defense"
        else:
            if StatusType.DEFENSE not in statuses_1:
                dmg_1 += 1
                detail = "P2 FLEX damage → P1 takes 1"
            else:
                detail = "P2 FLEX damage blocked by P1 defense"

    elif effect.effect_type == EffectType.BUILD:
        detail = f"P{src} BUILD — no immediate effect"

    return dmg_1, dmg_2, energy_1, energy_2, statuses_1, statuses_2, detail
