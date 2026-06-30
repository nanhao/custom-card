/**
 * Component tests for App.jsx.
 *
 * All API calls are mocked — no real HTTP requests.
 * Tests cover: menu → PVP flow, menu → AI select → PVE flow, game over.
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
  listAis: vi.fn(),
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
  mode: 'pvp',
  ai: null,
};

const PVE_STATE = { ...INITIAL_STATE, mode: 'pve', ai: 'rnn_v1.1' };

const GAME_ID = 'test-game-uuid';

/** Helper to go from menu to PVP game. */
async function setupPVPGame(user) {
  await user.click(screen.getByText('👥 Player vs Player'));
  await waitFor(() => {
    expect(screen.getByText(/Select a card/)).toBeInTheDocument();
  });
}

/** Helper to go from menu → AI select → PVE game. */
async function setupPVEGame(user) {
  await user.click(screen.getByText('🤖 Player vs AI'));
  await waitFor(() => {
    expect(screen.getByText('Choose Your Opponent')).toBeInTheDocument();
  });
  await user.click(screen.getByText(/rnn_v1\.1/));
  // rnn_v1.1 text appears inside the button: "🤖 rnn_v1.1\nGRU-based model..."
  // Regex needed — React splits text nodes: "🤖 " then "rnn_v1.1" then <span>
  await waitFor(() => {
    expect(screen.getByText(/Select a card/)).toBeInTheDocument();
  });
}

// =========================================================================
// Menu screen
// =========================================================================

describe('App — menu screen', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows menu with PVP and PVE buttons on mount', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText('Select Game Mode')).toBeInTheDocument();
    });
    expect(screen.getByText('👥 Player vs Player')).toBeInTheDocument();
    expect(screen.getByText('🤖 Player vs AI')).toBeInTheDocument();
  });
});

// =========================================================================
// PVP flow (updated for menu)
// =========================================================================

describe('App — PVP selection flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });
  });

  it('transitions to select phase after clicking PVP', async () => {
    const user = userEvent.setup();
    render(<App />);

    await setupPVPGame(user);

    // Both player status bars visible.
    expect(screen.getByText('Player 1')).toBeInTheDocument();
    expect(screen.getByText('Player 2')).toBeInTheDocument();

    // Round info visible.
    expect(screen.getByText(/Round 1\/7/)).toBeInTheDocument();

    // Card buttons visible.
    expect(screen.getByText('Attack')).toBeInTheDocument();
    expect(screen.getByText('Defense')).toBeInTheDocument();
  });

  it('Player 1 selecting a card shows pass screen in PVP', async () => {
    const user = userEvent.setup();
    render(<App />);

    await setupPVPGame(user);

    await user.click(screen.getByText('Attack'));

    await waitFor(() => {
      expect(screen.getByText('Player 1 has selected a card.')).toBeInTheDocument();
    });
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
        round: 1, cards_1: ['attack'], cards_2: ['defense'],
        damage_1: 0, damage_2: 0, hp_1_after: 3, hp_2_after: 3,
        details: ['Player 1 Attack blocked by Player 2 Defense → Player 2 takes 0'],
        is_game_over: false, winner: null,
      },
      state: AFTER_STATE,
    });

    render(<App />);
    await setupPVPGame(user);

    // P1 selects Attack, then passes the device.
    await user.click(screen.getByText('Attack'));
    // Wait for P2's hand to appear (may need extra render cycle).
    // P2 hand title uses regex because React splits JSX text across DOM nodes.
    await waitFor(() => {
      expect(screen.getByText(/🎮 Player 2's Hand/)).toBeInTheDocument();
    });
    // Find the Defense button inside P2's hand area.
    const p2HandArea = screen.getByText(/🎮 Player 2's Hand/).closest('.hand-area');
    const p2Defense = within(p2HandArea).getByText('Defense');
    await user.click(p2Defense);

    await waitFor(() => {
      expect(screen.getByText('⚡ Round 1 Result')).toBeInTheDocument();
    });
    expect(screen.getByText('▶ Next Round')).toBeInTheDocument();
  });
});

// =========================================================================
// PVE flow
// =========================================================================

describe('App — PVE flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows AI selection after clicking PVE', async () => {
    const user = userEvent.setup();
    api.listAis.mockResolvedValueOnce({
      ais: [{ name: 'rnn_v1.1', description: 'GRU-based model, random weights (gen1)' }],
    });

    render(<App />);
    await user.click(screen.getByText('🤖 Player vs AI'));

    await waitFor(() => {
      expect(screen.getByText('Choose Your Opponent')).toBeInTheDocument();
    });
    // Regex needed — React splits "🤖 " and "rnn_v1.1" into separate text nodes.
    expect(screen.getByText(/rnn_v1\.1/)).toBeInTheDocument();
  });

  it('starts PVE game after selecting an AI', async () => {
    const user = userEvent.setup();
    api.listAis.mockResolvedValueOnce({
      ais: [{ name: 'rnn_v1.1', description: 'GRU-based model, random weights (gen1)' }],
    });
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: PVE_STATE });

    render(<App />);
    await setupPVEGame(user);

    // Should show "vs AI" subtitle.
    expect(screen.getByText(/vs AI — rnn_v1.1/)).toBeInTheDocument();
    // Status shows "AI (rnn_v1.1)" for Player 2.
    expect(screen.getByText('AI (rnn_v1.1)')).toBeInTheDocument();
  });

  it('Player 1 selecting a card in PVE auto-plays the round', async () => {
    const user = userEvent.setup();
    api.listAis.mockResolvedValueOnce({
      ais: [{ name: 'rnn_v1.1', description: 'GRU-based model, random weights (gen1)' }],
    });
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: PVE_STATE });

    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1, cards_1: ['attack'], cards_2: ['heavy'],
        damage_1: 1, damage_2: 1, hp_1_after: 2, hp_2_after: 2,
        details: ['1 Attack (×1) → 2 takes 1', '2 Heavy Strike → 1 takes 1'],
        is_game_over: false, winner: null,
      },
      state: { ...PVE_STATE, round: 2, players: {
        '1': { ...PVE_STATE.players['1'], hand: PVE_STATE.players['1'].hand.slice(0, -1), history: ['attack'] },
        '2': { ...PVE_STATE.players['2'], hand: PVE_STATE.players['2'].hand.slice(0, -1), history: ['heavy'] },
      }},
    });

    render(<App />);
    await setupPVEGame(user);

    // In PVE, the "Your Turn" label is shown (not "Player 1's Turn").
    expect(screen.getByText(/Your Turn/)).toBeInTheDocument();

    // Select a card.
    await user.click(screen.getByText('Attack'));

    // Should immediately show round result (no pass screen in PVE).
    await waitFor(() => {
      expect(screen.getByText('⚡ Round 1 Result')).toBeInTheDocument();
    });

    // AI card shown in result.
    expect(screen.getByText('AI')).toBeInTheDocument();
    // Next Round button.
    expect(screen.getByText('▶ Next Round')).toBeInTheDocument();
  });

  it('PVE: playRound is called with only cards_1', async () => {
    const user = userEvent.setup();
    api.listAis.mockResolvedValueOnce({
      ais: [{ name: 'rnn_v1.1', description: 'GRU-based model, random weights (gen1)' }],
    });
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: PVE_STATE });
    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1, cards_1: ['attack'], cards_2: ['defense'],
        damage_1: 0, damage_2: 0, hp_1_after: 3, hp_2_after: 3,
        details: [], is_game_over: false, winner: null,
      },
      state: { ...PVE_STATE, round: 2 },
    });

    render(<App />);
    await setupPVEGame(user);
    await user.click(screen.getByText('Attack'));

    // Verify playRound was called with card1 only (no card2).
    expect(api.playRound).toHaveBeenCalledWith(GAME_ID, 'attack');
  });
});

// =========================================================================
// Game over
// =========================================================================

describe('App — game over', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows winner banner with AI name in PVE', async () => {
    const user = userEvent.setup();
    api.listAis.mockResolvedValueOnce({
      ais: [{ name: 'rnn_v1.1', description: 'GRU-based model, random weights (gen1)' }],
    });
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: PVE_STATE });
    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1, cards_1: ['attack'], cards_2: ['defense'],
        damage_1: 0, damage_2: 3, hp_1_after: 3, hp_2_after: 0,
        details: [], is_game_over: true, winner: '1',
      },
      state: { ...PVE_STATE, is_over: true, winner: '1' },
    });

    render(<App />);
    await setupPVEGame(user);
    await user.click(screen.getByText('Attack'));

    await waitFor(() => {
      expect(screen.getByText('🏆 Player 1 Wins!')).toBeInTheDocument();
    });

    // New Game and Menu buttons visible (two Menu buttons: result panel + bottom bar).
    expect(screen.getByText('🔄 New Game')).toBeInTheDocument();
    expect(screen.getAllByText('🏠 Menu').length).toBeGreaterThanOrEqual(1);
  });

  it('shows AI win banner in PVE', async () => {
    const user = userEvent.setup();
    api.listAis.mockResolvedValueOnce({
      ais: [{ name: 'rnn_v1.1', description: 'GRU-based model, random weights (gen1)' }],
    });
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: PVE_STATE });
    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1, cards_1: ['defense'], cards_2: ['attack'],
        damage_1: 3, damage_2: 0, hp_1_after: 0, hp_2_after: 3,
        details: [], is_game_over: true, winner: '2',
      },
      state: { ...PVE_STATE, is_over: true, winner: '2' },
    });

    render(<App />);
    await setupPVEGame(user);
    await user.click(screen.getByText('Defense'));

    await waitFor(() => {
      expect(screen.getByText('🏆 AI (rnn_v1.1) Wins!')).toBeInTheDocument();
    });
  });
});

// =========================================================================
// Next round
// =========================================================================

describe('App — next round', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('clicking Next Round refreshes state and returns to select', async () => {
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
        round: 1, cards_1: ['attack'], cards_2: ['defense'],
        damage_1: 0, damage_2: 0, hp_1_after: 3, hp_2_after: 3,
        details: ['Player 1 Attack blocked → Player 2 takes 0'],
        is_game_over: false, winner: null,
      },
      state: AFTER_STATE,
    });
    api.getState.mockResolvedValueOnce({ game_id: GAME_ID, state: AFTER_STATE });

    render(<App />);
    await setupPVPGame(user);

    // P1 selects, P2 selects.
    await user.click(screen.getByText('Attack'));
    await waitFor(() => expect(screen.getByText('Player 1 has selected a card.')).toBeInTheDocument());
    // Regex needed — React splits text nodes for interpolated values.
    const p2Hand = screen.getByText(/🎮 Player 2's Hand/).closest('.hand-area');
    await user.click(within(p2Hand).getByText('Defense'));

    await waitFor(() => {
      expect(screen.getByText('⚡ Round 1 Result')).toBeInTheDocument();
    });

    // Click Next Round.
    await user.click(screen.getByText('▶ Next Round'));

    await waitFor(() => {
      expect(screen.getByText(/Round 2\/7/)).toBeInTheDocument();
    });
  });
});

// =========================================================================
// Back to menu
// =========================================================================

describe('App — back to menu', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('Menu button in game returns to main menu', async () => {
    const user = userEvent.setup();
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: INITIAL_STATE });
    api.deleteGame.mockResolvedValueOnce(undefined);

    render(<App />);
    await setupPVPGame(user);

    // Click Menu button.
    await user.click(screen.getByText('🏠 Menu'));

    await waitFor(() => {
      expect(screen.getByText('Select Game Mode')).toBeInTheDocument();
    });
  });
});

// =========================================================================
// Card availability
// =========================================================================

describe('App — card visibility', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset mock state to clear any leftover mockResolvedValueOnce values
    // from previous tests (clearAllMocks only clears call history, not the queue).
    api.createGame.mockReset();
  });

  it('disables cards not in hand', async () => {
    const user = userEvent.setup();
    api.createGame.mockResolvedValueOnce({ game_id: GAME_ID, state: {
      round: 1,
      players: {
        '1': { hp: 3, hand: ['heavy'], history: [] },
        '2': { hp: 3, hand: ['attack', 'attack', 'attack', 'defense', 'defense', 'heavy', 'counter'], history: [] },
      },
      is_over: false, winner: null, mode: 'pvp', ai: null,
    } });

    render(<App />);
    // Inline setup to avoid helper mock interference.
    await user.click(screen.getByText('👥 Player vs Player'));
    await waitFor(() => {
      expect(screen.getByText(/Select a card/)).toBeInTheDocument();
    });

    // P1's hand area — only 1 card ('heavy') in hand.
    const p1HandArea = screen.getByText(/🎯 Your Hand/).closest('.hand-area');

    // Heavy Strike should be enabled (P1 has 1 heavy).
    const heavyBtn = within(p1HandArea).getByRole('button', { name: /Heavy Strike/ });
    expect(heavyBtn).not.toBeDisabled();

    // Attack should be disabled (P1 has 0 attack cards).
    // Use querySelector for direct DOM access to avoid text-matching ambiguity.
    const attackBtn = p1HandArea.querySelector('.card-attack');
    expect(attackBtn).toBeTruthy();
    // Check the DOM disabled property directly.
    expect(attackBtn.disabled).toBe(true);
  });
});
