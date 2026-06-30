/**
 * REST API client for the custom card game.
 *
 * All game state is managed server-side. This module only handles
 * HTTP communication — zero game logic lives here.
 */

const API_BASE = '/api';

/** Create a new game session. Returns { game_id, state }. */
export async function createGame() {
  const resp = await fetch(`${API_BASE}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

/** Get current game state. Returns { game_id, state }. */
export async function getState(gameId) {
  const resp = await fetch(`${API_BASE}/games/${gameId}`);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

/** Play a round with both players' cards. Returns { round_result, state }. */
export async function playRound(gameId, card1, card2) {
  const resp = await fetch(`${API_BASE}/games/${gameId}/play`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cards_1: [card1], cards_2: [card2] }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

/** Delete a game session. */
export async function deleteGame(gameId) {
  const resp = await fetch(`${API_BASE}/games/${gameId}`, {
    method: 'DELETE',
  });
  if (!resp.ok && resp.status !== 204) {
    const data = await resp.json();
    throw new Error(data.error || `HTTP ${resp.status}`);
  }
}
