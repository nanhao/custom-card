/**
 * GameBoard — renders all in-game UI: status bar, hand area,
 * action area, result panel, round log, and game-over banner.
 *
 * Pure presentation component; all game logic lives in App.jsx.
 */

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
// GameBoard Component
// =========================================================================

export default function GameBoard({
  state,
  phase,
  mode,
  aiName,
  p1Card,
  roundLog,
  onSelectCard,
  onPlayer2Card,
  onNextRound,
  onNewGame,
  onBackToMenu,
}) {
  if (!state) return null;

  // =========================================================================
  // Banner
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
    const label = w === '2' && mode === 'pve' ? `AI (${aiName})` : `Player ${w}`;
    return (
      <div className="game-over-banner banner-win">
        🏆 {label} Wins!
        <div className="banner-sub">
          Opponent HP reached 0 or {w} leads after 7 rounds
        </div>
      </div>
    );
  };

  // =========================================================================
  // Status Bar
  // =========================================================================
  const renderStatusBar = () => {
    if (!state) return null;
    return (
      <div className="status-bar">
        {['1', '2'].map(pid => {
          const p = state.players[pid];
          let cls = 'status-player';
          const isCurrent = phase === 'select' || (mode === 'pvp' && phase === 'pvp_pass');
          if (!state.is_over && isCurrent && pid === '1') cls += ' current';
          if (state.is_over && state.winner === pid) cls += ' winner';
          if (state.is_over && state.winner !== 'draw' && state.winner !== pid) cls += ' loser';

          const hearts = '❤️'.repeat(p.hp) + '🖤'.repeat(Math.max(0, 3 - p.hp));
          const chips = (p.history || []).map((c, i) => (
            <span key={i} className={`hist-chip chip-${c}`}>{CARD_ICONS[c]}</span>
          ));

          // Display name: "Player 2" or "AI (rnn_v1.1)"
          const displayName = (pid === '2' && mode === 'pve')
            ? `AI (${aiName})`
            : `Player ${pid}`;

          return (
            <div key={pid} className={cls}>
              <div className="status-name">{displayName}</div>
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

  // =========================================================================
  // Hand Area
  // =========================================================================
  const renderHandArea = () => {
    if (!state || phase === 'gameover') return null;
    const pH = state.players['1'];
    const pO = state.players['2'];

    // P1 cards: clickable in select, selected-card-only otherwise.
    const p1Cards = (phase === 'select')
      ? CARD_ORDER.map(card => {
          const cnt = countInHand(pH.hand, card);
          return (
            <button key={card} className={`card-btn card-${card}`}
              disabled={cnt === 0} onClick={() => onSelectCard(card)}>
              <span className="icon">{CARD_ICONS[card]}</span>
              <span>{CARD_NAMES[card]}</span>
              <span className="count">×{cnt}</span>
            </button>
          );
        })
      : (p1Card ? (
          <div className={`card-btn card-${p1Card}`} style={{ opacity: 0.8, cursor: 'default' }}>
            <span className="icon">{CARD_ICONS[p1Card]}</span>
            <span>{CARD_NAMES[p1Card]}</span>
            <span className="count">✓</span>
          </div>
        ) : null);

    // P2 select shown only in PVP pass screen.
    const showP2Select = mode === 'pvp' && phase === 'pvp_pass' && !state.is_over;
    // Opponent hidden cards shown during select phase.
    const showOppHidden = mode === 'pve' || (mode === 'pvp' && phase === 'select');
    const oppLabel = mode === 'pve' ? `AI (${aiName})'s Hand` : `Player 2's Hand`;

    return (
      <>
        <div className="hand-area">
          <div className="hand-title">
            🎯 Your Hand — Player 1 ({pH.hand.length} cards)
          </div>
          <div className="card-grid">{p1Cards}</div>
        </div>
        {showP2Select && (
          <div className="hand-area" style={{ marginTop: 12 }}>
            <div className="hand-title" style={{ color: 'var(--defense)' }}>
              🎮 Player 2's Hand ({pO.hand.length} cards)
            </div>
            <div className="card-grid">
              {CARD_ORDER.map(card => {
                const cnt = countInHand(pO.hand, card);
                return (
                  <button key={card} className={`card-btn card-${card}`}
                    disabled={cnt === 0} onClick={() => onPlayer2Card(card)}>
                    <span className="icon">{CARD_ICONS[card]}</span>
                    <span>{CARD_NAMES[card]}</span>
                    <span className="count">×{cnt}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
        {showOppHidden && (
          <div className="opp-hand-area">
            <div className="opp-hand-title">{oppLabel} ({pO.hand.length} cards)</div>
            <div className="opp-card-grid">
              {pO.hand.length > 0
                ? Array.from({ length: pO.hand.length }, (_, i) => (
                    <div key={i} className="opp-card-back">🂠</div>
                  ))
                : <span style={{ color: 'var(--muted)' }}>No cards left</span>}
            </div>
          </div>
        )}
      </>
    );
  };

  // =========================================================================
  // Action / Result Area
  // =========================================================================
  const renderAction = () => {
    if (!state) return null;

    // Selecting phase — show round info.
    if (phase === 'select') {
      const turnLabel = mode === 'pve' ? 'Your Turn' : "Player 1's Turn";
      return (
        <div className="round-info">
          Round {state.round}/7 —{' '}
          <span className="highlight">{turnLabel}</span>{' '}
          — Select a card
        </div>
      );
    }

    // PVP pass screen — Player 1 has chosen, hand to Player 2.
    if (mode === 'pvp' && phase === 'pvp_pass' && !state.is_over) {
      return (
        <div className="pass-screen">
          <div className="pass-text">Player 1 has selected a card.</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--muted)', marginBottom: 12 }}>
            Player 2 — pick your card above
          </div>
          <div className="pass-hint">
            Both selections are hidden until both players have chosen.
          </div>
        </div>
      );
    }

    // Round result / game over.
    if (phase === 'reveal' && state.is_over !== true && roundLog.length === 0) return null;
    if (phase === 'reveal' || phase === 'gameover') {
      const rr = roundLog[roundLog.length - 1];
      if (!rr) return null;

      const details = rr.details.map((line, i) => {
        const cls = line.includes('takes 0') ? 'res-safe' : 'res-dmg';
        return <div key={i} className={cls}>{line}</div>;
      });

      const p2Label = mode === 'pve' ? `AI` : 'P2';

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
              <span>{p2Label}</span>
            </div>
          </div>
          <div className="result-details">{details}</div>
          <div className="res-hp">
            After: P1 <strong>{rr.hp_1_after} HP</strong> | {p2Label} <strong>{rr.hp_2_after} HP</strong>
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            {phase === 'reveal' ? (
              <button className="btn btn-primary btn-big" onClick={onNextRound}>
                ▶ Next Round
              </button>
            ) : (
              <>
                <button className="btn btn-primary btn-big" onClick={onNewGame}>
                  🔄 New Game
                </button>
                <button className="btn btn-secondary btn-big" onClick={onBackToMenu}>
                  🏠 Menu
                </button>
              </>
            )}
          </div>
        </div>
      );
    }

    return null;
  };

  // =========================================================================
  // Round Log
  // =========================================================================
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
          const interleaved = lines.length === 0
            ? null
            : lines.reduce((a, b) => [a, <br key="br" />, '  ', b]);
          return (
            <div key={e.round} className="log-entry">
              <strong>R{e.round}:</strong> P1 {CARD_ICONS[e.cards_1[0]]} vs{' '}
              {mode === 'pve' ? 'AI' : 'P2'} {CARD_ICONS[e.cards_2[0]]}
              {interleaved && <><br />&nbsp;&nbsp;{interleaved}</>}
              <br /><span style={{ color: 'var(--muted)' }}>
                → P1 {e.hp_1_after}HP |{' '}
                {mode === 'pve' ? 'AI' : 'P2'} {e.hp_2_after}HP
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  // ---- Main render ----
  return (
    <>
      {renderBanner()}
      {renderStatusBar()}
      {renderHandArea()}
      {renderAction()}
      {renderLog()}
      <div className="btn-row" style={{ marginTop: 8 }}>
        <button className="btn btn-secondary btn-small" onClick={onBackToMenu}>
          🏠 Menu
        </button>
      </div>
    </>
  );
}
