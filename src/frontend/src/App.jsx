import { useState, useEffect } from 'react';
import { createGame, getState, playRound, deleteGame, listAis, fetchCards } from './api.js';
import GameBoard from './GameBoard.jsx';
import './App.css';

/** Count occurrences of a card in a hand array (also used by GameBoard). */
function countInHand(hand, card) {
  return hand.filter(c => c === card).length;
}

// =========================================================================
// App Component
// =========================================================================

export default function App() {
  // ---- Top-level navigation ----
  const [screen, setScreen] = useState('menu'); // menu | ai_select | game

  // ---- Game mode ----
  const [mode, setMode] = useState(null);       // 'pvp' | 'pve' | null
  const [aiName, setAiName] = useState(null);   // AI name for PVE
  const [ais, setAis] = useState([]);           // available AI list

  // ---- Game state (active only when screen === 'game') ----
  const [gameId, setGameId] = useState(null);
  const [state, setState] = useState(null);
  const [cardMeta, setCardMeta] = useState(null);  // card metadata from API
  const [phase, setPhase] = useState('loading'); // loading | select | reveal | gameover
  const [p1Card, setP1Card] = useState(null);    // buffered Player 1 card
  const [p1BuildTarget, setP1BuildTarget] = useState(null);       // PVP P1 build target
  const [p1DismantleTarget, setP1DismantleTarget] = useState(null); // PVP P1 dismantle target
  const [roundLog, setRoundLog] = useState([]);    // accumulated round results
  const [error, setError] = useState(null);

  // ---- BUILD card sub-state ----
  const [buildPhase, setBuildPhase] = useState(null);  // null | 'select_target'
  const [pendingBuild, setPendingBuild] = useState(null); // card string buffered for BUILD

  // ---- Pile viewer state ----
  const [pileViewer, setPileViewer] = useState(null); // null | { pid, draw_pile, discard_pile }

  // ---- Load card metadata on mount ----
  useEffect(() => {
    fetchCards().then(data => setCardMeta(data.cards)).catch(() => {});
  }, []);
  useEffect(() => {
    return () => { if (gameId) deleteGame(gameId).catch(() => {}); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Clear error after timeout ----
  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 5000);
    return () => clearTimeout(t);
  }, [error]);

  // =========================================================================
  // Menu & AI Selection
  // =========================================================================

  /** Start a PVP game directly from the menu. */
  function handleStartPVP() {
    setScreen('game');
    setMode('pvp');
    setAiName(null);
    initGame('pvp', null);
  }

  /** Navigate to AI selection screen. */
  async function handleStartPVE() {
    setError(null);
    try {
      const data = await listAis();
      setAis(data.ais);
      setScreen('ai_select');
    } catch (err) {
      setError('Failed to load AI list: ' + err.message);
    }
  }

  /** Create a PVE game with the chosen AI. */
  function handleSelectAI(name) {
    setScreen('game');
    setMode('pve');
    setAiName(name);
    initGame('pve', name);
  }

  /** Go back to main menu. */
  function handleBackToMenu() {
    // Delete the current game if one exists.
    if (gameId) {
      deleteGame(gameId).catch(() => {});
    }
    setScreen('menu');
    setMode(null);
    setAiName(null);
    setGameId(null);
    setState(null);
    setP1Card(null);
    setP1BuildTarget(null);
    setP1DismantleTarget(null);
    setRoundLog([]);
    setError(null);
    setPhase('loading');
    setBuildPhase(null);
    setPendingBuild(null);
  }

  // =========================================================================
  // Game lifecycle
  // =========================================================================

  async function initGame(gameMode, gameAiName) {
    setError(null);
    setPhase('loading');
    try {
      const data = await createGame(gameMode, gameAiName);
      setGameId(data.game_id);
      setState(data.state);
      setRoundLog([]);
      setP1Card(null);
      setPhase('select');
    } catch (err) {
      setError('Failed to create game: ' + err.message);
    }
  }

  // ---- Card selection ----
  async function handleSelectCard(card) {
    if (phase !== 'select') return;
    const hand = state.players['1'].hand;
    if (countInHand(hand, card) === 0) return;

    // BUILD card — show dual-mode choice (design/37)
    if (card === 'build') {
      setPendingBuild(card);
      setBuildPhase('choose_mode');
      return;
    }

    // Normal card — play immediately
    await doPlayCard(card, null, null);
  }

  /** User selected the build mode (build or dismantle) or a build target card. */
  async function handleSelectBuildTarget(card, mode) {
    if (mode === 'build') {
      // Build mode chosen from dual-mode dialog: show target card selection
      setBuildPhase('select_target');
      return;
    }
    // Build target card selected: execute build
    setBuildPhase(null);
    setPendingBuild(null);
    await doPlayCard('build', card, null);
  }

  /** User selected a dismantle target card. */
  async function handleSelectDismantleTarget(card, target) {
    if (!target) {
      // First call: switch to dismantle target selection
      setBuildPhase('select_dismantle');
      return;
    }
    // Second call: target selected, execute dismantle
    setBuildPhase(null);
    setPendingBuild(null);
    await doPlayCard('build', null, target);
  }

  /** Cancel BUILD sub-flow. */
  function handleCancelBuild() {
    setBuildPhase(null);
    setPendingBuild(null);
  }

  /** Common play-round logic. Per design/37: supports build and dismantle targets. */
  async function doPlayCard(card, buildTarget, dismantleTarget = null) {
    if (mode === 'pve') {
      setP1Card(card);
      setPhase('loading');
      try {
        const data = await playRound(gameId, card, null, buildTarget, dismantleTarget);
        // Merge round_result statuses into state so status bar shows current round's effects
        setState({
          ...data.state,
          players: {
            ...data.state.players,
            '1': { ...data.state.players['1'], statuses: data.round_result.after_resolve.statuses_1 },
            '2': { ...data.state.players['2'], statuses: data.round_result.after_resolve.statuses_2 },
          },
        });
        setRoundLog(prev => [...prev, data.round_result]);
        setP1Card(null);
        setBuildPhase(null);
        setPendingBuild(null);
        setPileViewer(null);
        const isGameOver = data.round_result.after_resolve.is_game_over
          || data.round_result.after_end_actions.is_game_over;
        setPhase(isGameOver ? 'gameover' : 'reveal');
      } catch (err) {
        setError(err.message);
        setPhase('select');
      }
    } else {
      // PVP: buffer P1 card + build/dismantle targets, show pass screen
      setP1Card(card);
      setP1BuildTarget(buildTarget);
      setP1DismantleTarget(dismantleTarget);
      setPhase('pvp_pass');
    }
  }

  /** Pass this round (play no card). Per design/28. */
  async function handlePass() {
    if (mode === 'pve') {
      // PVE: send pass directly
      setP1Card(null);
      setPhase('loading');
      try {
        const data = await playRound(gameId, null, null, null);
        setState({
          ...data.state,
          players: {
            ...data.state.players,
            '1': { ...data.state.players['1'], statuses: data.round_result.after_resolve.statuses_1 },
            '2': { ...data.state.players['2'], statuses: data.round_result.after_resolve.statuses_2 },
          },
        });
        setRoundLog(prev => [...prev, data.round_result]);
        setP1Card(null);
        setBuildPhase(null);
        setPendingBuild(null);
        setPileViewer(null);
        const isGameOver = data.round_result.after_resolve.is_game_over
          || data.round_result.after_end_actions.is_game_over;
        setPhase(isGameOver ? 'gameover' : 'reveal');
      } catch (err) {
        setError(err.message);
        setPhase('select');
      }
    } else {
      // PVP: buffer pass, show pass screen for P2
      setP1Card(null);
      setPhase('pvp_pass');
    }
  }

  /** Player 2 selects a card (PVP only). */
  async function handlePlayer2Card(card) {
    if (phase !== 'pvp_pass' || mode !== 'pvp') return;
    const hand = state.players['2'].hand;
    if (countInHand(hand, card) === 0) return;

    setPhase('loading');
    try {
      const data = await playRound(gameId, p1Card, card, p1BuildTarget, p1DismantleTarget);
      setState({
        ...data.state,
        players: {
          ...data.state.players,
          '1': { ...data.state.players['1'], statuses: data.round_result.after_resolve.statuses_1 },
          '2': { ...data.state.players['2'], statuses: data.round_result.after_resolve.statuses_2 },
        },
      });
      setRoundLog(prev => [...prev, data.round_result]);
      setP1Card(null);
      setP1BuildTarget(null);
      setP1DismantleTarget(null);
      const isGameOver = data.round_result.after_resolve.is_game_over
        || data.round_result.after_end_actions.is_game_over;
      setPhase(isGameOver ? 'gameover' : 'reveal');
    } catch (err) {
      setError(err.message);
      setPhase('pvp_pass');
    }
  }

  /** Player 2 passes (PVP only). Per design/28. */
  async function handleP2Pass() {
    if (phase !== 'pvp_pass' || mode !== 'pvp') return;
    setPhase('loading');
    try {
      const data = await playRound(gameId, p1Card, null, p1BuildTarget, p1DismantleTarget);
      setState({
        ...data.state,
        players: {
          ...data.state.players,
          '1': { ...data.state.players['1'], statuses: data.round_result.after_resolve.statuses_1 },
          '2': { ...data.state.players['2'], statuses: data.round_result.after_resolve.statuses_2 },
        },
      });
      setRoundLog(prev => [...prev, data.round_result]);
      setP1Card(null);
      setP1BuildTarget(null);
      setP1DismantleTarget(null);
      const isGameOver = data.round_result.after_resolve.is_game_over
        || data.round_result.after_end_actions.is_game_over;
      setPhase(isGameOver ? 'gameover' : 'reveal');
    } catch (err) {
      setError(err.message);
      setPhase('pvp_pass');
    }
  }

  // ---- Next round ----
  async function handleNextRound() {
    setPhase('loading');
    setBuildPhase(null);
    setPendingBuild(null);
    try {
      const data = await getState(gameId);
      setState(data.state);
      setP1Card(null);
      setPhase('select');
    } catch (err) {
      setError(err.message);
    }
  }

  // ---- Pile viewer ----
  /** View draw/discard piles for any player (design/34: mutual visibility). */
  async function handleViewPiles(pid) {
    if (pileViewer && pileViewer.pid === pid) { setPileViewer(null); return; }
    if (!state) return;
    const p = state.players[pid];
    setPileViewer({
      pid,
      draw_pile: p.draw_pile || [],
      discard_pile: p.discard_pile || [],
    });
  }

  // ---- New game (same mode/AI) ----
  async function handleNewGame() {
    // Delete old game.
    if (gameId) {
      deleteGame(gameId).catch(() => {});
    }
    initGame(mode, aiName);
  }

  // =========================================================================
  // Render helpers
  // =========================================================================

  const renderMenu = () => (
    <div className="menu-screen">
      <h2 className="menu-title">Select Game Mode</h2>
      <div className="menu-buttons">
        <button className="btn btn-primary btn-big menu-btn" onClick={handleStartPVP}>
          👥 Player vs Player
        </button>
        <button className="btn btn-primary btn-big menu-btn" onClick={handleStartPVE}>
          🤖 Player vs AI
        </button>
      </div>
      <div className="menu-hint">Hot-seat — pass the device between players</div>
    </div>
  );

  const renderAISelect = () => (
    <div className="menu-screen">
      <h2 className="menu-title">Choose Your Opponent</h2>
      <div className="menu-buttons">
        {ais.map(ai => (
          <button
            key={ai.name}
            className="btn btn-primary btn-big menu-btn"
            onClick={() => handleSelectAI(ai.name)}
          >
            🤖 {ai.name}
            <span className="menu-ai-desc">{ai.description}</span>
          </button>
        ))}
      </div>
      <button className="btn btn-secondary" onClick={handleBackToMenu}>
        ← Back
      </button>
    </div>
  );

  // ---- Main render ----
  // Menu screen
  if (screen === 'menu') {
    return (
      <>
        <h1 className="app-title">⚔️ Custom Card Game</h1>
        {error && <div className="error-banner">{error}</div>}
        {renderMenu()}
      </>
    );
  }

  // AI selection screen
  if (screen === 'ai_select') {
    return (
      <>
        <h1 className="app-title">⚔️ Custom Card Game</h1>
        {error && <div className="error-banner">{error}</div>}
        {renderAISelect()}
      </>
    );
  }

  // Game screen — loading
  if (phase === 'loading' && !state) {
    return (
      <>
        <h1 className="app-title">⚔️ Custom Card Game</h1>
        <div className="app-subtitle">
          {mode === 'pve' ? `vs AI — ${aiName}` : 'Hot-seat — pass the device between players'}
        </div>
        {error && <div className="error-banner">{error}</div>}
        <div className="loading-text">Loading…</div>
      </>
    );
  }

  return (
    <>
      <h1 className="app-title">⚔️ Custom Card Game</h1>
      <div className="app-subtitle">
        {mode === 'pve' ? `vs AI — ${aiName}` : 'Hot-seat — pass the device between players'}
      </div>

      {error && <div className="error-banner">{error}</div>}
      <GameBoard
        state={state}
        cardMeta={cardMeta}
        phase={phase}
        mode={mode}
        aiName={aiName}
        p1Card={p1Card}
        roundLog={roundLog}
        buildPhase={buildPhase}
        pendingBuild={pendingBuild}
        pileViewer={pileViewer}
        onSelectCard={handleSelectCard}
        onPlayer2Card={handlePlayer2Card}
        onSelectBuildTarget={handleSelectBuildTarget}
        onSelectDismantleTarget={handleSelectDismantleTarget}
        onCancelBuild={handleCancelBuild}
        onViewPiles={handleViewPiles}
        onClosePiles={() => setPileViewer(null)}
        onNextRound={handleNextRound}
        onNewGame={handleNewGame}
        onBackToMenu={handleBackToMenu}
        onPass={handlePass}
        onP2Pass={handleP2Pass}
      />
    </>
  );
}
