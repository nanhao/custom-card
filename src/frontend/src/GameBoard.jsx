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
// Icon mapping per card type — maintained on frontend (design/23).
// =========================================================================

const CARD_ICONS = {
  attack: '\u2694\ufe0f',
  defense: '\ud83d\udee1\ufe0f',
  heavy: '\ud83d\udca5',
  counter: '\u21a9\ufe0f',
  energy: '\u26a1',
  build: '\ud83d\udd28',
  flex: '\u2696\ufe0f',
};

// =========================================================================
// Helper — build derived lookup tables from cardMeta (from API)
// =========================================================================

function buildLookups(cardMeta) {
  if (!cardMeta) return { names: {}, order: [], costs: {} };
  return {
    names: Object.fromEntries(cardMeta.map(c => [c.type, c.display_name])),
    order: cardMeta.map(c => c.type),
    costs: Object.fromEntries(cardMeta.map(c => [c.type, c.build_cost])),
  };
}

// =========================================================================
// Display constants (fallback for before API loads)
// =========================================================================

const FALLBACK_ORDER = ['attack', 'defense', 'heavy', 'counter', 'energy', 'build', 'flex'];

/** Count occurrences of a card in a hand array. */
function countInHand(hand, card) {
  return hand.filter(c => c === card).length;
}

// =========================================================================
// GameBoard Component
// =========================================================================

export default function GameBoard({
  state,
  cardMeta,
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
  pileViewer,
  onViewPiles,
  onClosePiles,
}) {
  if (!state) return null;

  // Derive lookup tables from API metadata (or use fallback order)
  const { names: CARD_NAMES, order: apiOrder, costs: CARD_COSTS } = buildLookups(cardMeta);
  const cardOrder = apiOrder.length > 0 ? apiOrder : FALLBACK_ORDER;

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

          // History chips — design/14: c is {card, build_target}
          const chips = (p.history || []).map((c, i) => {
            const cardStr = typeof c === 'string' ? c : c.card;
            const label = c.build_target
              ? `${CARD_ICONS[cardStr]}→${CARD_ICONS[c.build_target]}`
              : CARD_ICONS[cardStr];
            return <span key={i} className={`hist-chip chip-${cardStr}`} title={label}>{label}</span>;
          });

          // Status tags — design/15: statuses are now [{"type": "defense", "lifetime": 1}]
          const statusTags = (p.statuses || []).map((s, i) => {
            const tagCls = s.type === 'guard' ? 'tag-guard' : 'tag-defense';
            return <span key={i} className={`status-tag ${tagCls}`}>{s.type}</span>;
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
              <div className="status-piles" onClick={pid === '1' ? onViewPiles : undefined} title={pid === '1' ? 'Click to view draw & discard piles' : undefined}>
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
  // Per design/27: opponent hand is always visible (no more face-down cards).
  // =========================================================================
  const renderHandArea = () => {
    if (!state || phase === 'gameover') return null;
    const pH = state.players['1'];
    const pO = state.players['2'];

    // P1 cards: clickable in select phase; during BUILD target selection show different UI.
    const renderP1Cards = () => {
      // Normal select phase
      if (phase === 'select') {
        return (
          <div className="card-grid">
            {cardOrder.map(card => {
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
    // Per design/27: opponent hand always visible — show real card grid.
    const showOppHand = (mode === 'pve' || mode === 'pvp') && phase !== 'pvp_pass' && buildPhase === null;
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
              {cardOrder.map(card => {
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
        {showOppHand && (
          <div className="hand-area" style={{ marginTop: 12 }}>
            <div className="hand-title" style={{ color: 'var(--defense)' }}>
              {oppLabel} ({pO.hand.length} cards)
            </div>
            <div className="card-grid">
              {cardOrder.map(card => {
                const cnt = countInHand(pO.hand, card);
                return (
                  <div key={card} className={`card-btn card-${card}`} style={{ opacity: 0.7, cursor: 'default' }}>
                    <span className="icon">{CARD_ICONS[card]}</span>
                    <span>{CARD_NAMES[card]}</span>
                    <span className="count">×{cnt}</span>
                  </div>
                );
              })}
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
          Cycle {state.cycle}, Round {state.round_in_cycle}/3{' '}
          — <span className="highlight">{turnLabel}</span>{' '}
          — Select a card
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
            Both players can see each other's hands.
          </div>
        </div>
      );
    }

    // Round result / game over.
    if (phase === 'reveal' || phase === 'gameover') {
      const rr = roundLog[roundLog.length - 1];
      if (!rr) return null;

      const details = rr.after_resolve.details.map((line, i) => {
        const cls = line.includes('takes 0') ? 'res-safe' : 'res-dmg';
        return <div key={i} className={cls}>{line}</div>;
      });

      const p2Label = mode === 'pve' ? 'AI' : 'P2';

      // Cycle ended indicator
      const cycleBleedNote = rr.end_actions.cycle_ended ? (
        <div className="cycle-bleed" style={{ marginTop: 8 }}>
          ⚠️ Cycle {rr.cycle} ended — both players lose 1 HP. New cycle started.
        </div>
      ) : null;

      return (
        <div className="result-panel">
          <div className="result-title">
            ⚡ Round {rr.round} — Cycle {rr.cycle}, Round {rr.round_in_cycle}/3
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
            After: P1 <strong>{rr.after_resolve.hp_1} HP</strong>⚡{rr.after_resolve.energy_1}{' '}
            | {p2Label} <strong>{rr.after_resolve.hp_2} HP</strong>\u26A1{rr.after_resolve.energy_2}
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
            Game started. Each player has 20 HP. 5 cards drawn from shuffled deck.
          </span>
        </div>
      );
    }
    return (
      <div className="log-area">
        <h3>📜 Round Log</h3>
        {entries.map(e => {
          const lines = e.after_resolve.details.map((l, i) => {
            const cls = l.includes('takes 0') ? 'log-safe' : 'log-dmg';
            return <span key={i} className={cls}>{l}</span>;
          });
          const interleaved = lines.length === 0
            ? null
            : lines.reduce((a, b) => [a, <br key="br" />, '  ', b]);
          const bleedTag = e.end_actions.cycle_ended ? <span className="cycle-bleed"> [Cycle End]</span> : null;
          return (
            <div key={e.round} className="log-entry">
              <strong>C{e.cycle}R{e.round_in_cycle}</strong> (R{e.round}):{' '}
              P1 {CARD_ICONS[e.cards_1[0]]} vs{' '}
              {mode === 'pve' ? 'AI' : 'P2'} {CARD_ICONS[e.cards_2[0]]}
              {bleedTag}
              {interleaved && <><br />&nbsp;&nbsp;{interleaved}</>}
              <br /><span style={{ color: 'var(--muted)' }}>
                → P1 {e.after_resolve.hp_1}HP |{' '}
                {mode === 'pve' ? 'AI' : 'P2'} {e.after_resolve.hp_2}HP
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
        Cycle {state.cycle}, Round {state.round_in_cycle}/3{' '}
        {state.round_log?.length > 0 && state.round_log[state.round_log.length - 1]?.end_actions.cycle_ended
          ? <span className="cycle-bleed">[Cycle ended — HP bleed applied]</span>
          : null}
      </div>
      {renderStatusBar()}
      {renderHandArea()}
      {renderAction()}
      {renderLog()}
      {pileViewer && (
        <div className="pile-overlay" onClick={onClosePiles}>
          <div className="pile-modal" onClick={e => e.stopPropagation()}>
            <div className="pile-modal-header">
              <h3>📦 Your Piles</h3>
              <button className="btn btn-secondary btn-small" onClick={onClosePiles}>✕</button>
            </div>
            <div className="pile-section">
              <h4>📥 Draw Pile ({pileViewer.draw_pile.length})</h4>
              <div className="pile-cards">
                {pileViewer.draw_pile.length === 0
                  ? <span style={{ color: 'var(--muted)' }}>Empty</span>
                  : pileViewer.draw_pile.map((c, i) => (
                      <span key={i} className={`pile-card-chip chip-${c}`}>{CARD_ICONS[c]}</span>
                    ))}
              </div>
            </div>
            <div className="pile-section">
              <h4>🗑️ Discard Pile ({pileViewer.discard_pile.length})</h4>
              <div className="pile-cards">
                {pileViewer.discard_pile.length === 0
                  ? <span style={{ color: 'var(--muted)' }}>Empty</span>
                  : pileViewer.discard_pile.map((c, i) => (
                      <span key={i} className={`pile-card-chip chip-${c}`}>{CARD_ICONS[c]}</span>
                    ))}
              </div>
            </div>
          </div>
        </div>
      )}
      {buildPhase === 'select_target' && (
        <div className="pile-overlay" onClick={() => onSelectBuildTarget?.(null)}>
          <div className="pile-modal" onClick={e => e.stopPropagation()}>
            <h3>🔨 选择要构建的卡牌</h3>
            <div className="energy-info" style={{ color: 'var(--muted)', fontSize: '0.85rem', margin: '8px 0 14px' }}>
              当前能量：<span style={{ color: 'var(--energy)', fontWeight: 700 }}>{state.players['1'].energy}</span>
            </div>
            <div className="build-grid">
              {cardOrder.map(card => {
                const cost = CARD_COSTS[card] || 1;
                const canAfford = cost <= state.players['1'].energy;
                return (
                  <button key={card} className={`build-card card-${card}`}
                    disabled={!canAfford}
                    onClick={() => onSelectBuildTarget?.(card)}>
                    <span className="build-icon">{CARD_ICONS[card]}</span>
                    <span>{CARD_NAMES[card]}</span>
                    <span className="build-cost">{cost}⚡</span>
                  </button>
                );
              })}
            </div>
            <div style={{ textAlign: 'center', marginTop: 12 }}>
              <button className="btn btn-secondary" onClick={() => onSelectBuildTarget?.(null)}>取消</button>
            </div>
          </div>
        </div>
      )}
      <div className="btn-row" style={{ marginTop: 8 }}>
        <button className="btn btn-secondary btn-small" onClick={onBackToMenu}>
          🏠 Menu
        </button>
      </div>
    </>
  );
}
