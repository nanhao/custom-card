# PRD: Custom Card Game

## 1. Game Overview

A 2-player turn-based card battle. Players choose one card per round face-down, reveal simultaneously, resolve effects, and repeat until a winner is determined or the card supply runs out.

## 2. Setup

- **Players**: 2
- **Initial HP**: 3 per player
- **Initial hand**: 7 cards — 3 Attack, 2 Defense, 1 Heavy Strike, 1 Counter
- **Max rounds**: 7 (equal to hand size)

## 3. Card Definitions

Each card resolves simultaneously — both effects apply at the same time.

### 3.1 Attack (×3 in hand)

- If the opponent's card is **not** Defense or Counter → opponent loses **X HP**, where X = the number of consecutive rounds the attacker has played Attack (including current round).
- If opponent plays Defense or Counter → no damage dealt.

### 3.2 Defense (×2 in hand)

- If opponent plays Attack, Defense, or Counter → no damage taken.
- If opponent plays Heavy Strike:
  - If **previous round** the defender also played Defense → no damage taken.
  - Otherwise (different previous card, or Round 1) → defender takes 1 damage.

### 3.3 Heavy Strike (×1 in hand)

- Unless the opponent played Defense in **both** the current round and the previous round → opponent takes **1 damage**.
- Round 1 edge case: no previous round, so opponent cannot satisfy the double-Defense condition → always deals 1 damage.

### 3.4 Counter (×1 in hand)

- If opponent plays Attack → opponent takes **1 damage**, the Counter player takes 0.
- If opponent plays Heavy Strike → **both players take 1 damage**.
- Otherwise (Defense or Counter) → neither player takes damage.

## 4. Round Flow

1. Both players select 1 card from their remaining hand (face-down).
   *Note: The API and save format support selecting multiple cards per round for future gameplay variants; the current version enforces exactly 1 card per player.*
2. Both reveal simultaneously.
3. Resolve both card effects at the same time.
4. Discard both played cards.
5. Check game-end conditions; if not met, proceed to next round.

## 5. Game End & Winner

Game ends immediately when:

- **One player's HP reaches 0** → the other player wins.
- **Both players' HP reach 0 simultaneously** → draw.
- **7 rounds completed**:
  - Higher remaining HP wins.
  - Equal HP → draw.

## 6. Edge Cases

| Scenario | Outcome |
|---|---|
| Both HP reach 0 in same round | Draw |
| Player HP drops below 0  | Treated as 0; game ends |
| Round 1 Defense vs Heavy Strike | Heavy Strike always deals 1 damage (no previous round for double-Defense check) |
| All 7 rounds played, both HP > 0 | Higher HP wins; equal HP is draw |


