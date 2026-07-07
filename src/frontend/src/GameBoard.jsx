/**
 * GameBoard v2 — renders all in-game UI: status bar, hand area,
 * action area, result panel, round log, and game-over banner.
 *
 * Supports 7 card types, energy bar, status tags, cycle info,
 * and BUILD target selection (design/11).
 *
 * Pure presentation component; all game logic lives in App.jsx.
 */

import './App.css';

// =========================================================================
// Display constants — 7 card types (v2)
// =========================================================================

const CARD_NAMES = {
  attack: 'Attack', defense: 'Defense', heavy: 'Heavy',
  counter: 'Counter', energy: 'Energy', build: 'Build', flex: 'Flex',
};
const CARD_ICONS = {
  attack: '\u2694\uFE0F', defense: '\uD83D\uDEE1\uFE0F', heavy: '\uD83D\uDCA5',
  counter: '\u21A9\uFE0F', energy: '\u26A1', build: '\uD83D\uDD28', flex: '\u2696\uFE0F',
};
const CARD_ORDER = ['attack', 'defense', 'heavy', 'counter', 'energy', 'build', 'flex'];

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
  onSelectBuildTarget,
  buildPhase,
  pendingBuild,
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
          <div className="banner-sub">Both players reach 0 HP</div>
        </div>
      );
    }
    const label = w === '2' && mode === 'pve' ? `AI (${aiName})` : `Player ${w}`;
    return (
      <div className="game-over-banner banner-win">
        🏆 {label} Wins!
        <div className="banner-sub">Opponent HP reached 0</div>
      </div>
    );
  };

  // =========================================================================
  // Status Bar — v2: HP number, energy, status tags, cycle display
  // =========================================================================
  const renderStatusBar = () => {
    if (!state) return null;
    return (
      <div className="status-bar">
        {['1', '2'].map(pid => {
          const p = state.players[pid];
          let cls = 'status-player';
          const isCurrent = phase === 'select'
            || (mode === 'pvp' && phase === 'pvp_pass')
            || buildPhase !== null;
          if (!state.is_over && isCurrent && pid === '1') cls += ' current';
          if (state.is_over && state.winner === pid) cls += ' winner';
          if (state.is_over && state.winner !== 'draw' && state.winner !== pid) cls += ' loser';

          // History chips
          const chips = (p.history || []).map((c, i) => (
            <span key={i} className={`hist-chip chip-${c}`}>{CARD_ICONS[c]}</span>
          ));

          // Status tags
          const statusTags = (p.statuses || []).map((s, i) => {
            const tagCls = s === '联防' ? 'tag-guard' : 'tag-defense';
            return <span key={i} className={`status-tag ${tagCls}`}>{s}</span>;
          });

          // Display name
          const displayName = (pid === '2' && mode === 'pve')
            ? `AI (${aiName})`
            : `Player ${pid}`;

          // Energy display
          const energyBar = '⚡'.repeat(p.energy) + '\u2606'.repeat(Math.max(0, 5 - p.energy));

          return (
            <div key={pid} className={cls}>
              <div className="status-name">{displayName}</div>
              <div className="status-hp">
                ❤️ <strong>{p.hp}</strong>
              </div>
              <div className="status-energy">{energyBar} {p.energy}/5</div>
              {statusTags.length > 0 && <div className="status-tags">{statusTags}</div>}
              {chips.length > 0 && <div className="status-history">{chips}</div>}
              <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginTop: 2 }}>
                Hand: {p.hand.length} | Deck: {p.draw_pile_count ?? 0} | Disc: {p.discard_pile_count ?? 0}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // =========================================================================
  // Hand Area — v2: 7 card types, BUILD target selector
  // =========================================================================
  const renderHandArea = () => {
    if (!state || phase === 'gameover') return null;
    const pH = state.players['1'];
    const pO = state.players['2'];

    // P1 cards: clickable in select phase; during BUILD target selection show different UI.
    const renderP1Cards = () => {
      // BUILD target selection sub-phase
      if (buildPhase === 'select_target') {
        const energy = pH.energy;
        const allTargets = ['attack', 'defense', 'heavy', 'counter', 'energy', 'build', 'flex'];
        return (
          <div style={{ marginTop: 8 }}>
            <div style={{ textAlign: 'center', color: 'var(--gold)', marginBottom: 8, fontSize: '0.85rem' }}>
              🔨 Select a card to build (Energy: {energy})
            </div>
            <div className="card-grid">
              {allTargets.map(card => {
                // build_cost: counter=2, flex=2, others=1
                const cost = (card === 'counter' || card === 'flex') ? 2 : 1;
                const canAfford = cost <= energy;
                return (
                  <button key={card} className={`card-btn card-${card}`}
                    disabled={!canAfford}
                    onClick={() => onSelectBuildTarget?.(card)}
                    title={canAfford ? `Cost: ${cost} energy` : `Need ${cost} energy (have ${energy})`}
                  >
                    <span className="icon">{CARD_ICONS[card]}</span>
                    <span>{CARD_NAMES[card]}</span>
                    <span className="count">{canAfford ? `${cost}⚡` : '✘'}</span>
                  </button>
                );
              })}
            </div>
            <div style={{ textAlign: 'center', marginTop: 8 }}>
              <button className="btn btn-secondary btn-small" onClick={() => onSelectBuildTarget?.(null)}>
                Cancel
              </button>
            </div>
          </div>
        );
      }

      // Normal select phase
      if (phase === 'select') {
        return (
          <div className="card-grid">
            {CARD_ORDER.map(card => {
              const cnt = countInHand(pH.hand, card);
              return (
                <button key={card} className={`card-btn card-${card}`}
                  disabled={cnt === 0} onClick={() => onSelectCard(card)}>
                  <span className="icon">{CARD_ICONS[card]}</span>
                  <span>{CARD_NAMES[card]}</span>
                  <span className="count">×{cnt}</span>
                </button>
              );
            })}
          </div>
        );
      }

      // PVP pass / reveal: show selected card only
      return p1Card ? (
        <div className={`card-btn card-${p1Card}`} style={{ opacity: 0.8, cursor: 'default', margin: '0 auto' }}>
          <span className="icon">{CARD_ICONS[p1Card]}</span>
          <span>{CARD_NAMES[p1Card]}</span>
          <span className="count">✓</span>
        </div>
      ) : null;
    };

    // P2 select shown only in PVP pass screen.
    const showP2Select = mode === 'pvp' && phase === 'pvp_pass' && !state.is_over;
    // Opponent hidden cards shown during select phase.
    const showOppHidden = mode === 'pve' || (mode === 'pvp' && phase === 'select' && buildPhase === null);
    const oppLabel = mode === 'pve' ? `AI (${aiName})'s Hand` : `Player 2's Hand`;

    return (
      <>
        <div className="hand-area">
          <div className="hand-title">
            🎯 Your Hand — Player 1 ({pH.hand.length} cards)
          </div>
          {renderP1Cards()}
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
  // Action / Result Area — v2: cycle display, BUILD, cycle_ended
  // =========================================================================
  const renderAction = () => {
    if (!state) return null;

    // Selecting phase — show cycle + round info
    if (phase === 'select' && buildPhase === null) {
      const turnLabel = mode === 'pve' ? 'Your Turn' : "Player 1's Turn";
      return (
        <div className="round-info">
          Cycle {state.cycle}, Round {state.round_in_cycle}/5{' '}
          — <span className="highlight">{turnLabel}</span>{' '}
          — Select a card
        </div>
      );
    }

    // BUILD target selection
    if (buildPhase === 'select_target') {
      return (
        <div className="round-info">
          🔨 Select which card to build with your BUILD card
        </div>
      );
    }

    // PVP pass screen
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
    if (phase === 'reveal' || phase === 'gameover') {
      const rr = roundLog[roundLog.length - 1];
      if (!rr) return null;

      const details = rr.details.map((line, i) => {
        const cls = line.includes('takes 0') ? 'res-safe' : 'res-dmg';
        return <div key={i} className={cls}>{line}</div>;
      });

      const p2Label = mode === 'pve' ? 'AI' : 'P2';

      // Cycle ended indicator
      const cycleBleedNote = rr.cycle_ended ? (
        <div className="cycle-bleed" style={{ marginTop: 8 }}>
          ⚠️ Cycle {rr.cycle} ended — both players lose 1 HP. New cycle started.
        </div>
      ) : null;

      return (
        <div className="result-panel">
          <div className="result-title">
            ⚡ Round {rr.round} — Cycle {rr.cycle}, Round {rr.round_in_cycle}/5
          </div>
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
          {cycleBleedNote}
          <div className="res-hp">
            After: P1 <strong>{rr.hp_1_after} HP</strong>⚡{rr.energy_1_after}{' '}
            | {p2Label} <strong>{rr.hp_2_after} HP</strong>\u26A1{rr.energy_2_after}
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
  // Round Log — v2: cycle info in log entries
  // =========================================================================
  const renderLog = () => {
    const entries = [...roundLog].reverse();
    if (entries.length === 0) {
      return (
        <div className="log-area">
          <h3>📜 Round Log</h3>
          <span style={{ color: 'var(--muted)' }}>
            Game started. Each player has 20 HP. 7 cards drawn from shuffled deck.
          </span>
        </div>
      );
    }
    return (
      <div className="log-area">
        <h3>\uD83D\uDCDC Round Log</h3>
        {entries.map(e => {
          const lines = e.details.map((l, i) => {
            const cls = l.includes('takes 0') ? 'log-safe' : 'log-dmg';
            return <span key={i} className={cls}>{l}</span>;
          });
          const interleaved = lines.length === 0
            ? null
            : lines.reduce((a, b) => [a, <br key="br" />, '  ', b]);
          const bleedTag = e.cycle_ended ? <span className="cycle-bleed"> [Cycle End]</span> : null;
          return (
            <div key={e.round} className="log-entry">
              <strong>C{e.cycle}R{e.round_in_cycle}</strong> (R{e.round}):{' '}
              P1 {CARD_ICONS[e.cards_1[0]]} vs{' '}
              {mode === 'pve' ? 'AI' : 'P2'} {CARD_ICONS[e.cards_2[0]]}
              {bleedTag}
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
      <div className="cycle-info">
        Cycle {state.cycle}, Round {state.round_in_cycle}/5{' '}
        {state.round_log?.length > 0 && state.round_log[state.round_log.length - 1]?.cycle_ended
          ? <span className="cycle-bleed">[Cycle ended — HP bleed applied]</span>
          : null}
      </div>
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
