/** Component tests for App.jsx — v2. All API calls mocked. */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../src/App.jsx';

vi.mock('../src/api.js', () => ({
  createGame: vi.fn(),
  getState: vi.fn(),
  playRound: vi.fn(),
  deleteGame: vi.fn(),
  listAis: vi.fn(),
}));
import * as api from '../src/api.js';

const STATE = {
  round: 1, cycle: 1, round_in_cycle: 1,
  players: {
    '1': { hp: 20, energy: 0, statuses: [],
      hand: ['attack', 'attack', 'defense', 'heavy', 'counter', 'energy', 'build'], history: [],
      draw_pile_count: 0, discard_pile_count: 0 },
    '2': { hp: 20, energy: 0, statuses: [],
      hand: ['attack', 'attack', 'defense', 'heavy', 'counter', 'energy', 'build'], history: [],
      draw_pile_count: 0, discard_pile_count: 0 },
  },
  is_over: false, winner: null, mode: 'pvp', ai: null, round_log: [],
};
const PVE_STATE = { ...STATE, mode: 'pve', ai: 'rnn_v1.1' };
const GID = 'test-uuid';

async function goPVP(user) {
  await user.click(screen.getByText(/Player vs Player/));
  await waitFor(() => { expect(screen.getByText(/Select a card/)).toBeInTheDocument(); });
}
async function goPVE(user) {
  await user.click(screen.getByText(/Player vs AI/));
  await waitFor(() => { expect(screen.getByText('Choose Your Opponent')).toBeInTheDocument(); });
  await user.click(screen.getByText(/rnn_v1\.1/));
  await waitFor(() => { expect(screen.getByText(/Select a card/)).toBeInTheDocument(); });
}

describe('Menu', () => {
  beforeEach(() => { vi.clearAllMocks(); });
  it('shows PVP and PVE buttons', async () => {
    render(<App />);
    await waitFor(() => { expect(screen.getByText('Select Game Mode')).toBeInTheDocument(); });
    expect(screen.getByText(/Player vs Player/)).toBeInTheDocument();
    expect(screen.getByText(/Player vs AI/)).toBeInTheDocument();
  });
});

describe('PVP flow', () => {
  beforeEach(() => { vi.clearAllMocks(); });
  it('shows 7 card types and cycle info', async () => {
    api.createGame.mockResolvedValueOnce({ game_id: GID, state: STATE });
    const user = userEvent.setup();
    render(<App />);
    await goPVP(user);
    expect(screen.getByText('Energy')).toBeInTheDocument();
    expect(screen.getByText('Build')).toBeInTheDocument();
    expect(screen.getByText('Flex')).toBeInTheDocument();
  });
});

describe('PVE flow', () => {
  beforeEach(() => { vi.clearAllMocks(); });
  it('Player selecting card auto-plays round', async () => {
    api.listAis.mockResolvedValueOnce({ ais: [{ name: 'rnn_v1.1', description: 'v2' }] });
    api.createGame.mockResolvedValueOnce({ game_id: GID, state: PVE_STATE });
    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1, cycle: 1, round_in_cycle: 1,
        cards_1: ['attack'], cards_2: ['defense'],
        damage_1: 0, damage_2: 0, hp_1_after: 20, hp_2_after: 20,
        energy_1_after: 0, energy_2_after: 0,
        statuses_1: [], statuses_2: [{"type": "defense", "lifetime": 1}],
        details: ['blocked'], is_game_over: false, winner: null, cycle_ended: false,
      },
      state: { ...PVE_STATE, round: 2, round_in_cycle: 2 },
    });
    const user = userEvent.setup();
    render(<App />);
    await goPVE(user);
    await user.click(screen.getByText('Attack'));
    await waitFor(() => { expect(screen.getByText(/Next Round/)).toBeInTheDocument(); });
  });
});

describe('Game over', () => {
  beforeEach(() => { vi.clearAllMocks(); });
  it('shows winner banner', async () => {
    api.listAis.mockResolvedValueOnce({ ais: [{ name: 'rnn_v1.1' }] });
    api.createGame.mockResolvedValueOnce({ game_id: GID, state: PVE_STATE });
    api.playRound.mockResolvedValueOnce({
      round_result: {
        round: 1, cycle: 1, round_in_cycle: 1,
        cards_1: ['attack'], cards_2: ['defense'],
        damage_1: 0, damage_2: 20, hp_1_after: 20, hp_2_after: 0,
        energy_1_after: 0, energy_2_after: 0,
        statuses_1: [], statuses_2: [],
        details: [], is_game_over: true, winner: '1', cycle_ended: false,
      },
      state: { ...PVE_STATE, is_over: true, winner: '1' },
    });
    const user = userEvent.setup();
    render(<App />);
    await goPVE(user);
    await user.click(screen.getByText('Attack'));
    await waitFor(() => { expect(screen.getByText(/Player 1 Wins!/)).toBeInTheDocument(); });
  });
});

describe('Next round', () => {
  beforeEach(() => { vi.clearAllMocks(); });
  it('returns to select phase', async () => {
    api.createGame.mockResolvedValueOnce({ game_id: GID, state: STATE });
    const after = { ...STATE, round: 2, round_in_cycle: 2,
      players: { '1': { ...STATE.players['1'], hand: STATE.players['1'].hand.slice(0,-1), history: ['attack'] },
                 '2': { ...STATE.players['2'], hand: STATE.players['2'].hand.slice(0,-1), history: ['defense'] } } };
    api.playRound.mockResolvedValueOnce({
      round_result: { round: 1, cycle: 1, round_in_cycle: 1, cards_1: ['attack'], cards_2: ['defense'],
        damage_1: 0, damage_2: 0, hp_1_after: 20, hp_2_after: 20,
        energy_1_after: 0, energy_2_after: 0, statuses_1: [], statuses_2: ['defense'],
        details: [], is_game_over: false, winner: null, cycle_ended: false },
      state: after,
    });
    api.getState.mockResolvedValueOnce({ game_id: GID, state: after });
    const user = userEvent.setup();
    render(<App />);
    await goPVP(user);
    await user.click(screen.getByText('Attack'));
    await waitFor(() => expect(screen.getByText('Player 1 has selected a card.')).toBeInTheDocument());
    const p2area = screen.getByText(/Player 2's Hand/).closest('.hand-area');
    await user.click(within(p2area).getByText('Defense'));
    await waitFor(() => { expect(screen.getByText(/Next Round/)).toBeInTheDocument(); });
    await user.click(screen.getByText(/Next Round/));
    await waitFor(() => { expect(screen.getByText(/Select a card/)).toBeInTheDocument(); });
  });
});
