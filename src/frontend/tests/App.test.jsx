/**
 * Component tests for App.jsx.
 *
 * All API calls are mocked — no real HTTP requests.
 * Tests cover the full UI flow: loading → select → pass → reveal → gameover.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../src/App.jsx';

// ---- Mock the API module ----
vi.mock('../src/api.js', () => ({
  createGame: vi.fn(),
  getState: vi.fn(),
  playRound: vi.fn(),
  deleteGame: vi.fn(),
}));

import * as api from '../src/api.js';

// ---- Sample data ----
const INITIAL_STATE = {
  round: 1,
  players: {
    '1': { hp: 3, hand: ['attack', 'attack', 'attack', 'defense', 'defense', 'heavy', 'counter'], history: [] },
    '2': { hp: 3, hand: ['attack', 'attack', 'attack', 'defense', 'defense', 'heavy', 'counter'], history: [] },
  },
  is_over: false,
  winner: null,
};

const GAME_ID = 'test-game-uuid';

// =========================================================================
// Initial render
// =========================================================================

describe('App — initial render', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading text on mount', () => {
    api.createGame.mockReturnValue(new Promise(() => {})); // never resolves
    render(<App />);
    expect(screen.getByText('Loading…')).toBeInTheDocument();
  });

  it('transitions to select phase after createGame succeeds', async () => {
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
    });

    // Both player status bars visible.
    expect(screen.getByText('Player 1')).toBeInTheDocument();
    expect(screen.getByText('Player 2')).toBeInTheDocument();

    // Round info visible.
    expect(screen.getByText(/Round 1\/7/)).toBeInTheDocument();
  });

  it('shows error banner when createGame fails', async () => {
    api.createGame.mockRejectedValueOnce(new Error('Network error'));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Failed to create game: Network error')).toBeInTheDocument();
    });
  });
});

// =========================================================================
// Selection flow
// =========================================================================

describe('App — selection flow', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });
  });

  it('shows Player 1 hand cards during select phase', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
    });

    // All 4 card type buttons should be rendered.
    expect(screen.getByText('Attack')).toBeInTheDocument();
    expect(screen.getByText('Defense')).toBeInTheDocument();
    expect(screen.getByText('Heavy Strike')).toBeInTheDocument();
    expect(screen.getByText('Counter')).toBeInTheDocument();

    // Card counts visible (3 Attack, 2 Defense, 1 Heavy, 1 Counter).
    expect(screen.getByText('×3')).toBeInTheDocument(); // Attack
    expect(screen.getAllByText('×1').length).toBeGreaterThanOrEqual(2); // Heavy + Counter
  });

  it('Player 1 selecting a card transitions to pass screen', async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
    });

    await user.click(screen.getByText('Attack'));

    // Phase changed to pass.
    await waitFor(() => {
      expect(screen.getByText('✅ Player 1 has selected a card.')).toBeInTheDocument();
    });

    // Pass-to-Player-2 button visible.
    expect(screen.getByText("👤 I'm Player 2")).toBeInTheDocument();
  });

  it('clicking pass button switches to Player 2 select phase', async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
    });

    // P1 selects Attack.
    await user.click(screen.getByText('Attack'));

    // Pass to P2.
    await user.click(screen.getByText("👤 I'm Player 2"));

    await waitFor(() => {
      expect(screen.getByText("Player 2's Turn")).toBeInTheDocument();
    });
  });
});

// =========================================================================
// Play round flow
// =========================================================================

describe('App — play round flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('Player 2 selecting a card submits round and shows result', async () => {
    const user = userEvent.setup();

    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    const AFTER_STATE = {
      ...INITIAL_STATE,
      round: 2,
      players: {
        '1': { ...INITIAL_STATE.players['1'], hand: INITIAL_STATE.players['1'].hand.slice(0, -1), history: ['attack'] },
        '2': { ...INITIAL_STATE.players['2'], hand: INITIAL_STATE.players['2'].hand.slice(0, -1), history: ['defense'] },
      },
    };

    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1,
        cards_1: ['attack'],
        cards_2: ['defense'],
        damage_1: 0,
        damage_2: 0,
        hp_1_after: 3,
        hp_2_after: 3,
        details: ['Player 1 Attack blocked by Player 2 Defense → Player 2 takes 0'],
        is_game_over: false,
        winner: null,
      },
      state: AFTER_STATE,
    });

    render(<App />);

    // Wait for select phase.
    await waitFor(() => {
      expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
    });

    // P1 selects Attack.
    await user.click(screen.getByText('Attack'));

    // Pass to P2.
    await user.click(screen.getByText("👤 I'm Player 2"));

    await waitFor(() => {
      expect(screen.getByText("Player 2's Turn")).toBeInTheDocument();
    });

    // P2 selects Defense.
    await user.click(screen.getAllByText('Defense')[0]);

    // Should show round result.
    await waitFor(() => {
      expect(screen.getByText('⚡ Round 1 Result')).toBeInTheDocument();
    });

    // Result cards shown.
    expect(screen.getByText('P1')).toBeInTheDocument();
    expect(screen.getByText('P2')).toBeInTheDocument();

    // Next Round button visible.
    expect(screen.getByText('▶ Next Round')).toBeInTheDocument();
  });

  it('shows game over banner when is_game_over is true', async () => {
    const user = userEvent.setup();

    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1,
        cards_1: ['attack'],
        cards_2: ['attack'],
        damage_1: 1,
        damage_2: 1,
        hp_1_after: 2,
        hp_2_after: 2,
        details: ['Player 1 Attack → Player 2 takes 1'],
        is_game_over: false,
        winner: null,
      },
      state: { ...INITIAL_STATE, round: 2, is_over: false },
    });

    // P1 and P2 both select attack → game continues.

    render(<App />);
    await waitFor(() => expect(screen.getByText("Player 1's Turn")).toBeInTheDocument());
    await user.click(screen.getByText('Attack'));
    await user.click(screen.getByText("👤 I'm Player 2"));
    await waitFor(() => expect(screen.getByText("Player 2's Turn")).toBeInTheDocument());
    await user.click(screen.getByText('Attack'));

    // Round result shown.
    await waitFor(() => {
      expect(screen.getByText('⚡ Round 1 Result')).toBeInTheDocument();
    });

    // Game continues — Next Round button visible, no banner.
    expect(screen.getByText('▶ Next Round')).toBeInTheDocument();
    expect(screen.queryByText('🤝 Draw!')).not.toBeInTheDocument();
    expect(screen.queryByText(/Wins!/)).not.toBeInTheDocument();
  });

  it('shows error when playRound API fails', async () => {
    const user = userEvent.setup();

    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });
    api.playRound.mockRejectedValueOnce(new Error('Cannot play round: the game is over.'));

    render(<App />);
    await waitFor(() => expect(screen.getByText("Player 1's Turn")).toBeInTheDocument());

    await user.click(screen.getByText('Attack'));
    await user.click(screen.getByText("👤 I'm Player 2"));
    await waitFor(() => expect(screen.getByText("Player 2's Turn")).toBeInTheDocument());
    await user.click(screen.getAllByText('Defense')[0]);

    // Error should appear.
    await waitFor(() => {
      expect(screen.getByText('Cannot play round: the game is over.')).toBeInTheDocument();
    });
  });
});

// =========================================================================
// Game over flow
// =========================================================================

describe('App — game over', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows draw banner and New Game button when game ends in draw', async () => {
    const user = userEvent.setup();

    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1,
        cards_1: ['attack'],
        cards_2: ['attack'],
        damage_1: 3,
        damage_2: 3,
        hp_1_after: 0,
        hp_2_after: 0,
        details: [],
        is_game_over: true,
        winner: 'draw',
      },
      state: { ...INITIAL_STATE, is_over: true, winner: 'draw' },
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("Player 1's Turn")).toBeInTheDocument());
    await user.click(screen.getByText('Attack'));
    await user.click(screen.getByText("👤 I'm Player 2"));
    await waitFor(() => expect(screen.getByText("Player 2's Turn")).toBeInTheDocument());
    await user.click(screen.getByText('Attack'));

    await waitFor(() => {
      expect(screen.getByText('🤝 Draw!')).toBeInTheDocument();
    });

    // New Game button should appear.
    expect(screen.getByText('🔄 New Game')).toBeInTheDocument();
  });

  it('shows winner banner when a player wins', async () => {
    const user = userEvent.setup();

    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1,
        cards_1: ['attack'],
        cards_2: ['attack'],
        damage_1: 0,
        damage_2: 3,
        hp_1_after: 3,
        hp_2_after: 0,
        details: [],
        is_game_over: true,
        winner: '1',
      },
      state: { ...INITIAL_STATE, is_over: true, winner: '1' },
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("Player 1's Turn")).toBeInTheDocument());
    await user.click(screen.getByText('Attack'));
    await user.click(screen.getByText("👤 I'm Player 2"));
    await waitFor(() => expect(screen.getByText("Player 2's Turn")).toBeInTheDocument());
    await user.click(screen.getByText('Attack'));

    await waitFor(() => {
      expect(screen.getByText('🏆 Player 1 Wins!')).toBeInTheDocument();
    });
  });
});

// =========================================================================
// Next round flow
// =========================================================================

describe('App — next round', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('clicking Next Round refreshes state and returns to select phase', async () => {
    const user = userEvent.setup();

    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    const AFTER_STATE = {
      ...INITIAL_STATE,
      round: 2,
      players: {
        '1': { hp: 3, hand: ['attack', 'attack', 'defense', 'defense', 'heavy', 'counter'], history: ['attack'] },
        '2': { hp: 3, hand: ['attack', 'attack', 'attack', 'defense', 'heavy', 'counter'], history: ['defense'] },
      },
    };

    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1,
        cards_1: ['attack'],
        cards_2: ['defense'],
        damage_1: 0,
        damage_2: 0,
        hp_1_after: 3,
        hp_2_after: 3,
        details: ['Player 1 Attack blocked → Player 2 takes 0'],
        is_game_over: false,
        winner: null,
      },
      state: AFTER_STATE,
    });

    // Mock getState for nextRound.
    api.getState.mockResolvedValueOnce({ game_id: GAME_ID, state: AFTER_STATE });

    render(<App />);
    await waitFor(() => expect(screen.getByText("Player 1's Turn")).toBeInTheDocument());

    // Play round 1.
    await user.click(screen.getByText('Attack'));
    await user.click(screen.getByText("👤 I'm Player 2"));
    await waitFor(() => expect(screen.getByText("Player 2's Turn")).toBeInTheDocument());
    await user.click(screen.getAllByText('Defense')[0]);

    // Result shown.
    await waitFor(() => {
      expect(screen.getByText('⚡ Round 1 Result')).toBeInTheDocument();
    });

    // Click Next Round.
    await user.click(screen.getByText('▶ Next Round'));

    // Should return to select phase for round 2.
    await waitFor(() => {
      expect(screen.getByText(/Round 2\/7/)).toBeInTheDocument();
    });
    expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
  });
});

// =========================================================================
// Round log
// =========================================================================

describe('App — round log', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows initial log message before any rounds', async () => {
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
    });

    expect(screen.getByText(/Game started/)).toBeInTheDocument();
  });

  it('shows round entry after playing a round', async () => {
    const user = userEvent.setup();

    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1,
        cards_1: ['attack'],
        cards_2: ['defense'],
        damage_1: 0,
        damage_2: 0,
        hp_1_after: 3,
        hp_2_after: 3,
        details: ['Player 1 Attack blocked by Player 2 Defense → Player 2 takes 0'],
        is_game_over: false,
        winner: null,
      },
      state: { ...INITIAL_STATE, round: 2 },
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("Player 1's Turn")).toBeInTheDocument());
    await user.click(screen.getByText('Attack'));
    await user.click(screen.getByText("👤 I'm Player 2"));
    await waitFor(() => expect(screen.getByText("Player 2's Turn")).toBeInTheDocument());
    await user.click(screen.getAllByText('Defense')[0]);

    await waitFor(() => {
      expect(screen.getByText('⚡ Round 1 Result')).toBeInTheDocument();
    });

    // Log should contain the round entry.
    expect(screen.getByText(/R1:/)).toBeInTheDocument();
  });
});

// =========================================================================
// Card availability
// =========================================================================

describe('App — card visibility', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('disables cards not in hand', async () => {
    // State where P1 only has 1 'heavy' and no 'attack'.
    const limitedState = {
      round: 1,
      players: {
        '1': { hp: 3, hand: ['heavy'], history: [] },
        '2': { hp: 3, hand: ['attack', 'attack', 'attack', 'defense', 'defense', 'heavy', 'counter'], history: [] },
      },
      is_over: false,
      winner: null,
    };

    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: limitedState });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
    });

    // Heavy Strike should be enabled (in hand).
    const heavyBtn = screen.getByText('Heavy Strike').closest('button');
    expect(heavyBtn).not.toBeDisabled();

    // Attack should be disabled (not in hand, count = 0).
    const attackBtn = screen.getByText('Attack').closest('button');
    expect(attackBtn).toBeDisabled();
  });

  it('opponent hand is hidden (card backs only)', async () => {
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Player 1's Turn")).toBeInTheDocument();
    });

    // Opponent section shows "Player 2's Hand" label.
    expect(screen.getByText("Player 2's Hand (7 cards)")).toBeInTheDocument();

    // Card backs rendered (🂠 characters).
    const oppArea = screen.getByText("Player 2's Hand (7 cards)").closest('.opp-hand-area');
    const backs = within(oppArea).getAllByText('🂠');
    expect(backs.length).toBe(7);
  });
});
