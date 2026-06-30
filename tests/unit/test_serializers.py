"""Unit tests for serializers — pure transformation functions.

Tests cover CardType ↔ string, RoundResult → dict, error_response,
and invalid-input handling per controller_api.md.

These are pure functions — no I/O, no HTTP, no side effects.
"""

import pytest

from src.backend.game.card import CardType
from src.backend.game.game_manager import RoundResult

# All imports are deferred until serializers.py is implemented:
# from src.backend.controller.serializers import (
#     card_to_str,
#     str_to_card,
#     round_result_to_dict,
#     error_response,
# )

A = CardType.ATTACK
D = CardType.DEFENSE
H = CardType.HEAVY_STRIKE
C = CardType.COUNTER


# ---------------------------------------------------------------------------
# CardType ↔ string
# ---------------------------------------------------------------------------


class TestCardToString:
    """Tests for card_to_str()."""

    def test_attack_to_str(self):
        from src.backend.controller.serializers import card_to_str
        assert card_to_str(A) == "attack"

    def test_defense_to_str(self):
        from src.backend.controller.serializers import card_to_str
        assert card_to_str(D) == "defense"

    def test_heavy_strike_to_str(self):
        from src.backend.controller.serializers import card_to_str
        assert card_to_str(H) == "heavy"

    def test_counter_to_str(self):
        from src.backend.controller.serializers import card_to_str
        assert card_to_str(C) == "counter"


class TestStringToCard:
    """Tests for str_to_card()."""

    def test_str_to_attack(self):
        from src.backend.controller.serializers import str_to_card
        assert str_to_card("attack") == A

    def test_str_to_defense(self):
        from src.backend.controller.serializers import str_to_card
        assert str_to_card("defense") == D

    def test_str_to_heavy(self):
        from src.backend.controller.serializers import str_to_card
        assert str_to_card("heavy") == H

    def test_str_to_counter(self):
        from src.backend.controller.serializers import str_to_card
        assert str_to_card("counter") == C

    def test_str_to_card_invalid_raises(self):
        from src.backend.controller.serializers import str_to_card
        with pytest.raises(ValueError, match="Unknown card type"):
            str_to_card("unknown_card")


class TestCardRoundTrip:
    """Verify that str_to_card(card_to_str(c)) == c for all CardType members."""

    @pytest.mark.parametrize("card", list(CardType))
    def test_round_trip(self, card):
        from src.backend.controller.serializers import card_to_str, str_to_card
        assert str_to_card(card_to_str(card)) == card


# ---------------------------------------------------------------------------
# RoundResult → dict
# ---------------------------------------------------------------------------


class TestRoundResultToDict:
    """Tests for round_result_to_dict()."""

    def test_basic_round_result(self):
        """A simple round result is serialized with all fields."""
        from src.backend.controller.serializers import round_result_to_dict

        rr = RoundResult(
            round=1,
            cards_1=[A],
            cards_2=[D],
            damage_1=0,
            damage_2=0,
            hp_1_after=3,
            hp_2_after=3,
            details=["P1 Attack blocked by P2 Defense"],
            is_game_over=False,
            winner=None,
        )
        result = round_result_to_dict(rr)
        assert result["round"] == 1
        assert result["cards_1"] == ["attack"]
        assert result["cards_2"] == ["defense"]
        assert result["damage_1"] == 0
        assert result["damage_2"] == 0
        assert result["hp_1_after"] == 3
        assert result["hp_2_after"] == 3
        assert result["details"] == ["P1 Attack blocked by P2 Defense"]
        assert result["is_game_over"] is False
        assert result["winner"] is None

    def test_game_over_result(self):
        """A game-over RoundResult includes winner."""
        from src.backend.controller.serializers import round_result_to_dict

        rr = RoundResult(
            round=3,
            cards_1=[A],
            cards_2=[A],
            damage_1=2,
            damage_2=1,
            hp_1_after=0,
            hp_2_after=1,
            details=["..."],
            is_game_over=True,
            winner="2",
        )
        result = round_result_to_dict(rr)
        assert result["is_game_over"] is True
        assert result["winner"] == "2"

    def test_multi_card_result(self):
        """Multiple cards per player are serialized as list of strings."""
        from src.backend.controller.serializers import round_result_to_dict

        rr = RoundResult(
            round=1,
            cards_1=[A, A],
            cards_2=[D, H],
            damage_1=1,
            damage_2=0,
            hp_1_after=2,
            hp_2_after=3,
            details=["..."],
            is_game_over=False,
            winner=None,
        )
        result = round_result_to_dict(rr)
        assert result["cards_1"] == ["attack", "attack"]
        assert result["cards_2"] == ["defense", "heavy"]

    def test_draw_winner(self):
        """Winner is 'draw' for a drawn game."""
        from src.backend.controller.serializers import round_result_to_dict

        rr = RoundResult(
            round=2,
            cards_1=[A],
            cards_2=[A],
            damage_1=2,
            damage_2=2,
            hp_1_after=0,
            hp_2_after=0,
            details=["..."],
            is_game_over=True,
            winner="draw",
        )
        result = round_result_to_dict(rr)
        assert result["winner"] == "draw"


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class TestErrorResponse:
    """Tests for error_response()."""

    def test_basic_error(self):
        from src.backend.controller.serializers import error_response

        result = error_response("Something went wrong", "ERR_CODE", 400)
        assert result == {
            "error": "Something went wrong",
            "code": "ERR_CODE",
        }

    def test_error_with_different_status(self):
        from src.backend.controller.serializers import error_response

        result = error_response("Not found", "GAME_NOT_FOUND", 404)
        assert result["code"] == "GAME_NOT_FOUND"
        assert result["error"] == "Not found"
