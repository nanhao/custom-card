import { useState, useEffect, useCallback } from 'react';
import { createGame, getState, playRound, deleteGame } from './api.js';
import './App.css';

// =========================================================================
// Display constants (presentation only — no game logic)
// =========================================================================
const CARD_NAMES = { attack: 'Attack', defense: 'Defense', heavy: 'Heavy Strike', counter: 'Counter' };
const CARD_ICONS = { attack: '⚔️', defense: '🛡️', heavy: '💥', counter: '↩️' };
const CARD_ORDER = ['attack', 'defense', 'heavy', 'counter'];

/** Count occurrences of a card in a hand array. */
function countInHand(hand, card) {
  return hand.filter(c => c === card).length;
}

// =========================================================================
// App Component
// =========================================================================

export default function App() {
  const [gameId, setGameId] = useState(null);
  const [state, setState] = useState(null);       // latest server state
  const [phase, setPhase] = useState('loading');   // loading | select | pass | reveal | gameover
  const [currentPlayer, setCurrentPlayer] = useState('1');
  const [p1Card, setP1Card] = useState(null);      // buffered Player 1 card
  const [roundLog, setRoundLog] = useState([]);    // accumulated round results
  const [error, setError] = useState(null);

  // ---- Lifecycle: create game on mount, cleanup on unmount ----
  useEffect(() => {
    initGame();
    return () => { if (gameId) deleteGame(gameId).catch(() => {}); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function initGame() {
    setError(null);
    setPhase('loading');
    try {
      const data = await createGame();
      setGameId(data.game_id);
      setState(data.state);
      setRoundLog([]);
      setP1Card(null);
      setCurrentPlayer('1');
      setPhase('select');
    } catch (err) {
      setError('Failed to create game: ' + err.message);
    }
  }

  // ---- Card selection ----
  async function handleSelectCard(card) {
    if (phase !== 'select') return;
    const hand = state.players[currentPlayer].hand;
    if (countInHand(hand, card) === 0) return;

    if (currentPlayer === '1') {
      setP1Card(card);
      setPhase('pass');
    } else {
      // Player 2 chose → submit both cards to server.
      setPhase('loading');
      try {
        const data = await playRound(gameId, p1Card, card);
        setState(data.state);
        setRoundLog(prev => [...prev, data.round_result]);
        setP1Card(null);
        setCurrentPlayer('1');
        setPhase(data.round_result.is_game_over ? 'gameover' : 'reveal');
      } catch (err) {
        setError(err.message);
        setPhase('select');
      }
    }
  }

  // ---- Pass to Player 2 ----
  function handlePass() {
    if (phase !== 'pass') return;
    setCurrentPlayer('2');
    setPhase('select');
  }

  // ---- Next round ----
  async function handleNextRound() {
    setPhase('loading');
    try {
      const data = await getState(gameId);
      setState(data.state);
      setP1Card(null);
      setCurrentPlayer('1');
      setPhase('select');
    } catch (err) {
      setError(err.message);
    }
  }

  // ---- Clear error after timeout ----
  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 5000);
    return () => clearTimeout(t);
  }, [error]);

  // =========================================================================
  // Render helpers
  // =========================================================================

  const renderBanner = () => {
    if (phase !== 'gameover' || !state) return null;
    const w = state.winner;
    if (w === 'draw') {
      return (
        <div className="game-over-banner banner-draw">
          🤝 Draw!
          <div className="banner-sub">Both players end with equal HP</div>
        </div>
      );
    }
    return (
      <div className="game-over-banner banner-win">
        🏆 Player {w} Wins!
        <div className="banner-sub">
          Opponent HP reached 0 or {w} leads after 7 rounds
        </div>
      </div>
    );
  };

  const renderStatusBar = () => {
    if (!state) return null;
    return (
      <div className="status-bar">
        {['1', '2'].map(pid => {
          const p = state.players[pid];
          let cls = 'status-player';
          if (!state.is_over && phase === 'select' && currentPlayer === pid) cls += ' current';
          if (state.is_over && state.winner === pid) cls += ' winner';
          if (state.is_over && state.winner !== 'draw' && state.winner !== pid) cls += ' loser';

          const hearts = '❤️'.repeat(p.hp) + '🖤'.repeat(Math.max(0, 3 - p.hp));
          const chips = (p.history || []).map((c, i) => (
            <span key={i} className={`hist-chip chip-${c}`}>{CARD_ICONS[c]}</span>
          ));

          return (
            <div key={pid} className={cls}>
              <div className="status-name">Player {pid}</div>
              <div className="status-hp">
                <span className="hearts">{hearts}</span> <strong>{p.hp}/3</strong>
              </div>
              {chips.length > 0 && <div className="status-history">{chips}</div>}
            </div>
          );
        })}
      </div>
    );
  };

  const renderHandArea = () => {
    if (!state || phase === 'gameover') return null;
    const cur = currentPlayer;
    const opp = cur === '1' ? '2' : '1';
    const pH = state.players[cur];
    const pO = state.players[opp];

    // Current player's cards
    let cardElements;
    if (phase === 'select') {
      cardElements = CARD_ORDER.map(card => {
        const cnt = countInHand(pH.hand, card);
        return (
          <button
            key={card}
            className={`card-btn card-${card}`}
            disabled={cnt === 0}
            onClick={() => handleSelectCard(card)}
          >
            <span className="icon">{CARD_ICONS[card]}</span>
            <span>{CARD_NAMES[card]}</span>
            <span className="count">×{cnt}</span>
          </button>
        );
      });
    } else {
      // Show selected card during pass/reveal (non-interactive).
      const sel = phase === 'pass' ? p1Card : null;
      cardElements = sel ? (
        <div className={`card-btn card-${sel}`} style={{ opacity: 0.8, cursor: 'default' }}>
          <span className="icon">{CARD_ICONS[sel]}</span>
          <span>{CARD_NAMES[sel]}</span>
          <span className="count">✓</span>
        </div>
      ) : null;
    }

    // Opponent's hidden cards
    const oppBacks = Array.from({ length: pO.hand.length }, (_, i) => (
      <div key={i} className="opp-card-back">🂠</div>
    ));

    return (
      <>
        <div className="hand-area">
          <div className="hand-title">
            🎯 Your Hand — Player {cur} ({pH.hand.length} cards)
          </div>
          <div className="card-grid">{cardElements}</div>
        </div>
        <div className="opp-hand-area">
          <div className="opp-hand-title">Player {opp}'s Hand ({pO.hand.length} cards)</div>
          <div className="opp-card-grid">
            {oppBacks.length > 0 ? oppBacks : <span style={{ color: 'var(--muted)' }}>No cards left</span>}
          </div>
        </div>
      </>
    );
  };

  const renderAction = () => {
    if (!state) return null;

    // Selecting phase — show round info.
    if (phase === 'select') {
      return (
        <div className="round-info">
          Round {state.round}/7 —{' '}
          <span className="highlight">Player {currentPlayer}'s Turn</span>{' '}
          — Select a card
        </div>
      );
    }

    // Pass screen — Player 1 has chosen, hand to Player 2.
    if (phase === 'pass') {
      const opp = '2';
      return (
        <div className="pass-screen">
          <div className="pass-text">✅ Player 1 has selected a card.</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--muted)', marginBottom: 16 }}>
            Pass the device to <strong>Player {opp}</strong>
          </div>
          <button className="btn btn-primary btn-big" onClick={handlePass}>
            👤 I'm Player {opp}
          </button>
          <div className="pass-hint">
            Player 1's selection is hidden until both players have chosen.
          </div>
        </div>
      );
    }

    // Round result.
    if (phase === 'reveal' || phase === 'gameover') {
      const rr = roundLog[roundLog.length - 1];
      if (!rr) return null;

      const details = rr.details.map((line, i) => {
        const cls = line.includes('takes 0') ? 'res-safe' : 'res-dmg';
        return <div key={i} className={cls}>{line}</div>;
      });

      return (
        <div className="result-panel">
          <div className="result-title">⚡ Round {rr.round} Result</div>
          <div className="result-cards">
            <div className={`result-card card-${rr.cards_1[0]}`}>
              <span style={{ fontSize: '2rem' }}>{CARD_ICONS[rr.cards_1[0]]}</span>
              <span>P1</span>
            </div>
            <div className="result-vs">VS</div>
            <div className={`result-card card-${rr.cards_2[0]}`}>
              <span style={{ fontSize: '2rem' }}>{CARD_ICONS[rr.cards_2[0]]}</span>
              <span>P2</span>
            </div>
          </div>
          <div className="result-details">{details}</div>
          <div className="res-hp">
            After: P1 <strong>{rr.hp_1_after} HP</strong> | P2 <strong>{rr.hp_2_after} HP</strong>
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            {phase === 'reveal' ? (
              <button className="btn btn-primary btn-big" onClick={handleNextRound}>
                ▶ Next Round
              </button>
            ) : (
              <button className="btn btn-primary btn-big" onClick={initGame}>
                🔄 New Game
              </button>
            )}
          </div>
        </div>
      );
    }

    return null;
  };

  const renderLog = () => {
    const entries = [...roundLog].reverse();
    if (entries.length === 0) {
      return (
        <div className="log-area">
          <h3>📜 Round Log</h3>
          <span style={{ color: 'var(--muted)' }}>
            Game started. Each player has 3 HP and 7 cards.
          </span>
        </div>
      );
    }
    return (
      <div className="log-area">
        <h3>📜 Round Log</h3>
        {entries.map(e => {
          const lines = e.details.map((l, i) => {
            const cls = l.includes('takes 0') ? 'log-safe' : 'log-dmg';
            return <span key={i} className={cls}>{l}</span>;
          });
          // Interleave lines with <br /> separators, safely handling empty arrays.
          const interleaved = lines.length === 0
            ? null
            : lines.reduce((a, b) => [a, <br key="br" />, '  ', b]);
          return (
            <div key={e.round} className="log-entry">
              <strong>R{e.round}:</strong> P1 {CARD_ICONS[e.cards_1[0]]} vs P2 {CARD_ICONS[e.cards_2[0]]}
              {interleaved && <><br />&nbsp;&nbsp;{interleaved}</>}
              <br /><span style={{ color: 'var(--muted)' }}>
                → P1 {e.hp_1_after}HP | P2 {e.hp_2_after}HP
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  // ---- Main render ----
  if (phase === 'loading' && !state) {
    return (
      <>
        <h1 className="app-title">⚔️ Custom Card Game</h1>
        <div className="app-subtitle">Hot-seat — pass the device between players</div>
        {error && <div className="error-banner">{error}</div>}
        <div className="loading-text">Loading…</div>
      </>
    );
  }

  return (
    <>
      <h1 className="app-title">⚔️ Custom Card Game</h1>
      <div className="app-subtitle">Hot-seat — pass the device between players</div>

      {error && <div className="error-banner">{error}</div>}
      {renderBanner()}
      {renderStatusBar()}
      {renderHandArea()}
      {renderAction()}
      {renderLog()}
    </>
  );
}
