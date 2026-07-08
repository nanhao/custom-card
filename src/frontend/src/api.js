/**
 * REST API client for the custom card game.
 *
 * All game state is managed server-side. This module only handles
 * HTTP communication — zero game logic lives here.
 */

const API_BASE = '/api';

/**
 * Create a new game session.
 *
 * @param {string} [mode='pvp'] - 'pvp' (default) or 'pve'.
 * @param {string} [ai=null] - AI name (required for PVE).
 * @returns {{ game_id: string, state: object }}
 */
export async function createGame(mode = 'pvp', ai = null) {
  const body = mode === 'pve' ? { mode, ai } : {};
  const resp = await fetch(`${API_BASE}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
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

/** Get game state with pile inspection for a player. */
export async function getMyPiles(gameId) {
  const resp = await fetch(`${API_BASE}/games/${gameId}?perspective=1`);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

/**
 * Play a round.
 *
 * @param {string} gameId
 * @param {string} card1 - Player 1's card.
 * @param {string|null} [card2=null] - Player 2's card (PVP only).
 * @param {string|null} [buildTarget=null] - Build target card (only when card1=build).
 * @returns {{ round_result: object, state: object }}
 */
export async function playRound(gameId, card1, card2 = null, buildTarget = null) {
  const body = { cards_1: [card1] };
  if (card2 !== null) body.cards_2 = [card2];
  if (buildTarget !== null) body.build_target_1 = buildTarget;
  const resp = await fetch(`${API_BASE}/games/${gameId}/play`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

/**
 * List available AI opponents.
 * @returns {{ ais: Array<{name: string, description: string}> }}
 */
export async function listAis() {
  const resp = await fetch(`${API_BASE}/ais`);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

/**
 * Fetch card type metadata (names, icons, costs, etc.).
 * @returns {{ cards: Array<{type: string, display_name: string, icon: string, build_cost: number}> }}
 */
export async function fetchCards() {
  const resp = await fetch(`${API_BASE}/cards`);
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
