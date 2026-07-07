import { useState, useEffect } from 'react';
import { createGame, getState, playRound, deleteGame, listAis } from './api.js';
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
  const [phase, setPhase] = useState('loading'); // loading | select | reveal | gameover
  const [p1Card, setP1Card] = useState(null);    // buffered Player 1 card
  const [roundLog, setRoundLog] = useState([]);    // accumulated round results
  const [error, setError] = useState(null);

  // ---- BUILD card sub-state ----
  const [buildPhase, setBuildPhase] = useState(null);  // null | 'select_target'
  const [pendingBuild, setPendingBuild] = useState(null); // card string buffered for BUILD

  // ---- Cleanup game on unmount ----
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

    // BUILD card requires target selection
    if (card === 'build') {
      setPendingBuild(card);
      setBuildPhase('select_target');
      return;
    }

    // Normal card — play immediately
    await doPlayCard(card, null);
  }

  /** User selected a build target (or cancelled). */
  async function handleSelectBuildTarget(target) {
    if (target === null) {
      // Cancelled
      setBuildPhase(null);
      setPendingBuild(null);
      return;
    }
    setBuildPhase(null);
    setPendingBuild(null);
    await doPlayCard('build', target);
  }

  /** Common play-round logic for both normal cards and BUILD. */
  async function doPlayCard(card, buildTarget) {
    if (mode === 'pve') {
      setP1Card(card);
      setPhase('loading');
      try {
        const data = await playRound(gameId, card, null, buildTarget);
        setState(data.state);
        setRoundLog(prev => [...prev, data.round_result]);
        setP1Card(null);
        setPhase(data.round_result.is_game_over ? 'gameover' : 'reveal');
      } catch (err) {
        setError(err.message);
        setPhase('select');
      }
    } else {
      // PVP: buffer P1 card, show pass screen
      setP1Card(card);
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
      const data = await playRound(gameId, p1Card, card);
      setState(data.state);
      setRoundLog(prev => [...prev, data.round_result]);
      setP1Card(null);
      setPhase(data.round_result.is_game_over ? 'gameover' : 'reveal');
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
        phase={phase}
        mode={mode}
        aiName={aiName}
        p1Card={p1Card}
        roundLog={roundLog}
        buildPhase={buildPhase}
        pendingBuild={pendingBuild}
        onSelectCard={handleSelectCard}
        onPlayer2Card={handlePlayer2Card}
        onSelectBuildTarget={handleSelectBuildTarget}
        onNextRound={handleNextRound}
        onNewGame={handleNewGame}
        onBackToMenu={handleBackToMenu}
      />
    </>
  );
}
