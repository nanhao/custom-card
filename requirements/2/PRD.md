# PRD: Human vs AI Mode

## 1. Overview

Add a human-vs-AI battle mode to the existing custom card game. The player chooses cards face-down against an AI opponent. The game rules (cards, HP, win conditions, 7-round limit) remain identical to hot-seat PVP.

---

## 2. Main Menu

Before entering any game mode, the player sees a main menu with two options:

- **Hot-seat PVP** — Two human players on the same device (existing mode).
- **vs AI** — One human player against the computer.

Selecting "Hot-seat PVP" loads the existing game interface (unchanged).  
Selecting "vs AI" navigates to the **AI selection screen**.

The main menu is part of the same single-page app — no page reloads.

---

## 3. AI Selection Screen

When the player chooses "vs AI" from the main menu, a selection screen shows all available AI opponents.

Each AI is displayed with:
- **Name** — a unique human-readable identifier (e.g. `rnn_v1.1`)

The player picks one AI and proceeds to the game. The first available AI is **`rnn_v1.1`**. The player can return to the main menu from this screen without making a selection.

---

## 4. Human vs AI — Game Flow

The player is always **Player 1 (Human)**. The AI is always **Player 2**.

### 3.1 Round Flow

1. **Human selects a card** — The human sees their own hand and selects one card, same as current hot-seat mode. The AI's hand is shown as hidden card backs (same as before).
2. **AI auto-selects** — Immediately after the human picks, the AI chooses a card from its own hand.
3. **Reveal result** — Both cards are revealed simultaneously. Damage, HP changes, and round details are displayed exactly as in hot-seat mode.
4. **Next round** — The human taps "Next Round" to continue, or "New Game" if the game is over.

**No pass screen** — there is no second human to hand the device to.

### 3.2 Game End

Game end conditions are identical to hot-seat PVP: HP reaches 0 for either/both players, or 7 rounds completed. The winner banner and round log behave the same.

### 3.3 Back to Menu

After a game ends, the human can choose "New Game" (starts another human-vs-AI game) or navigate back to the main menu to switch modes.

---

## 5. AI Behavior

### 5.1 Decision Making

The AI selects a card each round based on the game state. It only selects cards it still has in its hand. Its decision is deterministic — given the same game state, it always picks the same card.

### 5.2 AI Versions

The system supports multiple AI versions identified by name. Each has a unique name and may produce different card selections for the same game state.

The first available version is **`rnn_v1.1`**. Additional versions may be added over time and will appear in the AI selection screen automatically.

### 5.3 Response Time

The AI's selection is near-instantaneous from the player's perspective. There is no artificial delay.

---

## 6. Acceptance Criteria

| # | Criterion |
|---|-----------|
| AC1 | Main menu displays two buttons: "Hot-seat PVP" and "vs AI" |
| AC2 | Clicking "Hot-seat PVP" loads the existing hot-seat game interface |
| AC3 | Clicking "vs AI" navigates to the AI selection screen |
| AC4 | The AI selection screen lists all available AIs by name |
| AC5 | The player can select an AI opponent and proceed to the game |
| AC6 | The player can return to the main menu from the AI selection screen |
| AC7 | In human-vs-AI mode, the human is always Player 1 and the AI is Player 2 |
| AC8 | The human sees their own hand and the AI's hand as card backs |
| AC9 | After the human selects a card, the AI's choice is revealed immediately along with the result — no pass screen |
| AC10 | The AI only selects cards it still has in its hand |
| AC11 | The AI selects exactly one card per round |
| AC12 | Round results display both players' cards, damage, HP changes, and resolution details |
| AC13 | The selected AI name is displayed during the game |
| AC14 | Game end banner and winner display work the same as hot-seat PVP |
| AC15 | "New Game" in human-vs-AI mode starts a fresh game against the same AI |
| AC16 | The player can return to the main menu from the game |
| AC17 | The AI responds instantly with no perceptible delay |
| AC18 | Round log accumulates and displays all rounds, same as hot-seat mode |
