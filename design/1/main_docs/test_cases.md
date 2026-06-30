# Test Cases

All tests call `CardResolver.resolve(cards_1, cards_2, history_1, history_2)` and verify `damage_1` and `damage_2`. `cards_1` and `cards_2` are `list[CardType]` (current version: exactly 1 card each). `history_1` and `history_2` represent cards played in previous rounds (chronological order).
Initial state: HP=3 for both players.

---

## TC-GROUP-1: PRD Examples (7 cases)

| ID | card_1 | card_2 | hist_1 | hist_2 | dmg_1 | dmg_2 | Description |
|----|--------|--------|--------|--------|-------|-------|-------------|
| TC1 | ATTACK | ATTACK | [] | [] | 1 | 1 | Both Attack (×1), both take 1 |
| TC2 | DEFENSE | DEFENSE | [] | [] | 0 | 0 | Both Defense, no damage |
| TC3 | ATTACK | DEFENSE | [] | [] | 0 | 0 | Attack blocked by Defense |
| TC4 | ATTACK | COUNTER | [] | [] | 1 | 0 | Counter vs Attack, A takes 1 |
| TC5 | HEAVY | COUNTER | [] | [] | 1 | 1 | Both deal 1, both take 1 |
| TC6 | ATTACK | ATTACK | [ATK] | [] | 1 | 2 | A Attack ×2, B Attack ×1 |
| TC7 | DEFENSE | HEAVY | [DEF] | [] | 0 | 0 | Consecutive Defense blocks Heavy Strike |

---

## TC-GROUP-2: Full Resolution Matrix (16 cases)

Round 1 (empty history for both players).

| ID | card_1 | card_2 | dmg_1 | dmg_2 | Reason |
|----|--------|--------|-------|-------|--------|
| TC-M1 | ATTACK | ATTACK | 1 | 1 | Both ×1 Attack |
| TC-M2 | ATTACK | DEFENSE | 0 | 0 | Attack blocked by Defense |
| TC-M3 | ATTACK | HEAVY | 1 | 1 | Attack ×1 vs HS ×1 |
| TC-M4 | ATTACK | COUNTER | 1 | 0 | Counter blocks Attack, deals 1 |
| TC-M5 | DEFENSE | ATTACK | 0 | 0 | Defense blocks Attack |
| TC-M6 | DEFENSE | DEFENSE | 0 | 0 | Both Defense |
| TC-M7 | DEFENSE | HEAVY | 1 | 0 | No consecutive Defense → A takes 1 |
| TC-M8 | DEFENSE | COUNTER | 0 | 0 | No effect |
| TC-M9 | HEAVY | ATTACK | 1 | 1 | HS vs Attack, both deal 1 |
| TC-M10 | HEAVY | DEFENSE | 1 | 0 | No consecutive Defense → B takes 1 |
| TC-M11 | HEAVY | HEAVY | 1 | 1 | Both HS, both take 1 |
| TC-M12 | HEAVY | COUNTER | 1 | 1 | Both deal 1 |
| TC-M13 | COUNTER | ATTACK | 0 | 1 | Counter deals 1 vs Attack |
| TC-M14 | COUNTER | DEFENSE | 0 | 0 | No effect |
| TC-M15 | COUNTER | HEAVY | 1 | 1 | Counter vs HS, both take 1 |
| TC-M16 | COUNTER | COUNTER | 0 | 0 | Neither triggers |

---

## TC-GROUP-3: Consecutive Attack Tracking

| ID | card_1 | card_2 | hist_1 | hist_2 | dmg_1 | dmg_2 | Description |
|----|--------|--------|--------|--------|-------|-------|-------------|
| TC-A1 | ATTACK | ATTACK | [ATK] | [] | 1 | 2 | A ×2 consecutive |
| TC-A2 | ATTACK | ATTACK | [ATK, ATK] | [] | 1 | 3 | A ×3 consecutive |
| TC-A3 | ATTACK | ATTACK | [DEF] | [ATK] | 2 | 1 | A ×1, B ×2 (interrupted by Defense) |
| TC-A4 | ATTACK | ATTACK | [ATK, DEF, ATK] | [] | 1 | 2 | A streak broken by Defense, now ×2 |
| TC-A5 | ATTACK | DEFENSE | [ATK, ATK] | [] | 0 | 0 | Attack ×3 blocked by Defense |

---

## TC-GROUP-4: Consecutive Defense vs Heavy Strike

| ID | card_1 | card_2 | hist_1 | hist_2 | dmg_1 | dmg_2 | Description |
|----|--------|--------|--------|--------|-------|-------|-------------|
| TC-D1 | DEFENSE | HEAVY | [DEF] | [] | 0 | 0 | Consecutive Defense blocks HS |
| TC-D2 | DEFENSE | HEAVY | [ATK] | [] | 1 | 0 | No consecutive Defense, A takes 1 |
| TC-D3 | DEFENSE | HEAVY | [DEF, ATK, DEF] | [] | 0 | 0 | Last round was Defense, blocks |
| TC-D4 | HEAVY | DEFENSE | [] | [DEF] | 0 | 0 | B consecutive Defense blocks A HS |
| TC-D5 | HEAVY | DEFENSE | [] | [] | 0 | 1 | B no prev Defense, R1 → B takes 1 |
| TC-D6 | DEFENSE | HEAVY | [] | [] | 1 | 0 | R1, no prev → A takes 1 |
| TC-D7 | DEFENSE | HEAVY | [ATK, DEF, ATK] | [] | 1 | 0 | Last card ATK, not DEF → A takes 1 |

---

## TC-GROUP-5: Game End Detection (via GameManager)

These test `GameManager.play_round()` end-to-end.
**Note:** For scenarios requiring more cards of a type than the standard hand provides (e.g., 7 DEF plays), tests use pre-configured custom hands via direct `Player.hand` assignment after `start_game()`. Otherwise, the standard 7-card hand is used.

| ID | Setup | Round Actions | Expected is_over | Expected winner |
|----|-------|---------------|------------------|-----------------|
| TC-E1 | Fresh game | R1: ATK vs ATK, R2: ATK vs ATK | True (after R2) | "draw" (both 0: 3→2→0) |
| TC-E2 | Fresh game | R1: ATK vs DEF, R2: DEF vs ATK, R3: DEF vs DEF | False after R3 | None |
| TC-E3 | 1 1HP, 2 3HP | R1: ATK vs ATK | True | "2" (1→0, 2→2) |
| TC-E4 | 1 3HP, 2 1HP | R1: ATK vs ATK | True | "1" (1→2, 2→0) |
| TC-E5 | Custom: 7 DEF each | 7 rounds, all DEF vs DEF | True (after R7) | "draw" (3-3) |
| TC-E6 | Custom: A=ATK+6DEF, B=6DEF+ATK | R1: ATK vs ATK(A→2,B→2), R2-R7: all DEF vs DEF | True (after R7) | "draw" (2-2) |
| TC-E7 | Fresh game | R1: ATK vs DEF, R2: ATK vs ATK, R3: ATK vs HS | True (after R3) | "1" (1=1, 2=0: 1 consec 3→2 -3, 2 HS→1 -1) |

---

## TC-GROUP-6: Edge Cases

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| TC-X1 | HP drops below 0 (A at 1HP, takes 3 damage) | A HP = 0 (clamped), game ends |
| TC-X2 | play_round() called when game is over | Raises ValueError |
| TC-X3 | play_round() with card not in hand | Raises ValueError |
| TC-X4 | play_round() on round 7, game continues | If both HP > 0, game ends after R7 resolution |
| TC-X5 | Both play last card on R7, game not over | Game ends after R7 regardless |
| TC-X6 | Start game twice without reset | Second call reinitializes fresh state |
| TC-X7 | Fresh game, 7 rounds: R1 HS vs DEF (no prev) → 1=3,2=2; R2-R7 all DEF vs DEF | Game ends after R7, winner "1" (3-2). Uses custom hands for R2-R7. |
| TC-X8 | play_card(ATTACK) with [ATK, DEF, ATK, ATK] in hand | Exactly 2 ATTACK remain, hand length decreases by 1. Verifies removal of correct duplicate instance. |
| TC-X9 | Resolve HS vs DEF with empty history | dmg(DEF player)=1, not 2. Verifies no double-counting from DEF's perspective. |
