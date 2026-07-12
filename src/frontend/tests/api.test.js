/** Unit tests for api.js — v2. All fetch calls mocked. */
import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;
const api = await import('../src/api.js');

function mockResponse(body, ok = true, status = 200) {
  return { ok, status, json: async () => body };
}
function mockErrorResponse(error, code, status = 400) {
  return mockResponse({ error, code }, false, status);
}

const STATE = {
  round: 1, cycle: 1, round_in_cycle: 1,
  players: {
    '1': { hp: 30, energy: 2, statuses: [], hand: ['attack'], history: [], draw_pile_count: 0, discard_pile_count: 0 },
    '2': { hp: 30, energy: 2, statuses: [], hand: ['attack'], history: [], draw_pile_count: 0, discard_pile_count: 0 },
  },
  is_over: false, winner: null, mode: 'pvp', ai: null, round_log: [],
};

describe('createGame', () => {
  beforeEach(() => { mockFetch.mockReset(); });
  it('returns game_id and state', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ game_id: 'abc', state: STATE }, true, 201));
    const r = await api.createGame();
    expect(r.game_id).toBe('abc');
    expect(r.state.players['1'].hp).toBe(30);
  });
  it('PVE sends mode and ai', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ game_id: 'pve', state: { ...STATE, mode: 'pve' } }, true, 201));
    const r = await api.createGame('pve', 'rnn_v1.1');
    expect(r.state.mode).toBe('pve');
  });
});

describe('playRound', () => {
  beforeEach(() => { mockFetch.mockReset(); });
  it('sends build_target when provided', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ round_result: {}, state: STATE }));
    await api.playRound('abc', 'build', 'defense', 'attack');
    expect(mockFetch).toHaveBeenCalledWith('/api/games/abc/play', expect.objectContaining({
      body: JSON.stringify({ cards_1: ['build'], cards_2: ['defense'], build_target_1: 'attack' }),
    }));
  });
  it('PVE omits cards_2', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ round_result: {}, state: STATE }));
    await api.playRound('abc', 'attack');
    expect(mockFetch).toHaveBeenCalledWith('/api/games/abc/play', expect.objectContaining({
      body: JSON.stringify({ cards_1: ['attack'] }),
    }));
  });
});

describe('listAis', () => {
  beforeEach(() => { mockFetch.mockReset(); });
  it('returns AI list', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ ais: [{ name: 'rnn_v1.1' }] }));
    const r = await api.listAis();
    expect(r.ais[0].name).toBe('rnn_v1.1');
  });
});

describe('deleteGame', () => {
  beforeEach(() => { mockFetch.mockReset(); });
  it('resolves on 204', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });
    await expect(api.deleteGame('abc')).resolves.toBeUndefined();
  });
});
