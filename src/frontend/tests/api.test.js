/**
 * Unit tests for the API client (api.js).
 *
 * All fetch calls are mocked — no real HTTP requests.
 * Tests cover: success responses, error responses, and edge cases.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// We must import after mocking fetch so the module uses our mock.
// vi.mock hoists, so it works even before the import.
vi.mock('../src/api.js', async (importOriginal) => {
  // We're testing the real module; just need to control global fetch.
  // So we DON'T mock the module — we mock global fetch instead.
  return importOriginal();
});

// But actually, we want to test the real functions with a mocked fetch.
// Let's restructure: mock global fetch, then import the real module.

// Clear approach: mock globalThis.fetch, import real api functions.
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

// Now import the real module (it will use our mocked globalThis.fetch).
const api = await import('../src/api.js');

// ---- Helpers ----
function mockResponse(body, ok = true, status = 200) {
  return {
    ok,
    status,
    json: async () => body,
  };
}

function mockErrorResponse(error, code, status = 400) {
  return mockResponse({ error, code }, false, status);
}

const SAMPLE_STATE = {
  round: 1,
  players: {
    '1': { hp: 3, hand: ['attack', 'attack', 'attack', 'defense', 'defense', 'heavy', 'counter'], history: [] },
    '2': { hp: 3, hand: ['attack', 'attack', 'attack', 'defense', 'defense', 'heavy', 'counter'], history: [] },
  },
  is_over: false,
  winner: null,
};

const SAMPLE_ROUND_RESULT = {
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
};

// =========================================================================
// createGame
// =========================================================================

describe('createGame', () => {
  beforeEach(() => { mockFetch.mockReset(); });

  it('returns game_id and state on success (201)', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({
      game_id: 'abc-123',
      state: SAMPLE_STATE,
    }, true, 201));

    const result = await api.createGame();
    expect(result.game_id).toBe('abc-123');
    expect(result.state.round).toBe(1);
    expect(mockFetch).toHaveBeenCalledWith('/api/games', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
  });

  it('throws on server error', async () => {
    mockFetch.mockResolvedValueOnce(mockErrorResponse('Server error', 'INTERNAL', 500));

    await expect(api.createGame()).rejects.toThrow('Server error');
  });

  it('throws generic message if response has no error field', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({}, false, 500));

    await expect(api.createGame()).rejects.toThrow('HTTP 500');
  });
});

// =========================================================================
// getState
// =========================================================================

describe('getState', () => {
  beforeEach(() => { mockFetch.mockReset(); });

  it('returns game_id and state on success (200)', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({
      game_id: 'abc-123',
      state: SAMPLE_STATE,
    }));

    const result = await api.getState('abc-123');
    expect(result.game_id).toBe('abc-123');
    expect(result.state.players['1'].hp).toBe(3);
    expect(mockFetch).toHaveBeenCalledWith('/api/games/abc-123');
  });

  it('throws on 404 (game not found)', async () => {
    mockFetch.mockResolvedValueOnce(mockErrorResponse('Game not found', 'GAME_NOT_FOUND', 404));

    await expect(api.getState('nonexistent')).rejects.toThrow('Game not found');
  });
});

// =========================================================================
// playRound
// =========================================================================

describe('playRound', () => {
  beforeEach(() => { mockFetch.mockReset(); });

  it('returns round_result and updated state on success', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({
      round_result: SAMPLE_ROUND_RESULT,
      state: { ...SAMPLE_STATE, round: 2 },
    }));

    const result = await api.playRound('abc-123', 'attack', 'defense');
    expect(result.round_result.round).toBe(1);
    expect(result.round_result.cards_1).toEqual(['attack']);
    expect(result.round_result.cards_2).toEqual(['defense']);
    expect(result.state.round).toBe(2);

    expect(mockFetch).toHaveBeenCalledWith('/api/games/abc-123/play', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cards_1: ['attack'], cards_2: ['defense'] }),
    });
  });

  it('throws on invalid card (400)', async () => {
    mockFetch.mockResolvedValueOnce(mockErrorResponse('Unknown card type', 'INVALID_CARD', 400));

    await expect(api.playRound('abc-123', 'unknown', 'defense')).rejects.toThrow('Unknown card type');
  });

  it('throws on game over (400)', async () => {
    mockFetch.mockResolvedValueOnce(mockErrorResponse('Cannot play round: the game is over.', 'GAME_OVER', 400));

    await expect(api.playRound('abc-123', 'attack', 'attack')).rejects.toThrow('Cannot play round');
  });

  it('throws on card not in hand (400)', async () => {
    mockFetch.mockResolvedValueOnce(mockErrorResponse(
      'Not all cards in [HEAVY_STRIKE] are in Player 1\'s hand.',
      'CARD_NOT_IN_HAND',
      400,
    ));

    await expect(api.playRound('abc-123', 'heavy', 'defense')).rejects.toThrow('Not all cards');
  });
});

// =========================================================================
// deleteGame
// =========================================================================

describe('deleteGame', () => {
  beforeEach(() => { mockFetch.mockReset(); });

  it('resolves silently on success (204)', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });

    await expect(api.deleteGame('abc-123')).resolves.toBeUndefined();
    expect(mockFetch).toHaveBeenCalledWith('/api/games/abc-123', { method: 'DELETE' });
  });

  it('throws on 404 (game not found)', async () => {
    mockFetch.mockResolvedValueOnce(mockErrorResponse('Game not found', 'GAME_NOT_FOUND', 404));

    await expect(api.deleteGame('nonexistent')).rejects.toThrow('Game not found');
  });
});
