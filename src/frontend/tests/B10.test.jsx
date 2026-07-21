/**
 * B10: PVP Pass Phase highlights Player 2, not Player 1.
 *
 * Bug: During pvp_pass phase, isCurrent is true but only Player 1 gets
 * the .current class highlight. Should highlight Player 2 instead.
 */

import { describe, it, expect } from 'vitest';
import React from 'react';
import { render } from '@testing-library/react';
import GameBoard from '../src/GameBoard.jsx';

const makeState = (overrides = {}) => ({
  round: 1, cycle: 1, round_in_cycle: 1,
  is_over: false, winner: null,
  players: {
    '1': { hp: 30, energy: 2, hand: ['attack'], statuses: [], history: [],
           draw_pile: [], draw_pile_count: 5, discard_pile: [], discard_pile_count: 0 },
    '2': { hp: 30, energy: 2, hand: ['defense'], statuses: [], history: [],
           draw_pile: [], draw_pile_count: 5, discard_pile: [], discard_pile_count: 0 },
  },
  ...overrides,
});

const noop = () => {};

describe('B10: PVP Pass Phase Player Highlight', () => {

  it('highlights Player 2 during pvp_pass phase', () => {
    const state = makeState();
    const { container } = render(
      <GameBoard
        state={state}
        phase="pvp_pass"
        mode="pvp"
        buildPhase={null}
        cardMeta={[]}
        p1Card={null}
        roundLog={[]}
        onSelectCard={noop}
        onPlayer2Card={noop}
        onNextRound={noop}
        onNewGame={noop}
        onBackToMenu={noop}
        onSelectBuildTarget={noop}
        onSelectDismantleTarget={noop}
        onCancelBuild={noop}
        pendingBuild={null}
        pileViewer={null}
        onViewPiles={noop}
        onClosePiles={noop}
        onPass={noop}
        onP2Pass={noop}
      />
    );

    const statusPlayers = container.querySelectorAll('.status-player');
    expect(statusPlayers.length).toBe(2);

    // B10 CHECK: During pvp_pass, Player 2 should have .current class
    expect(statusPlayers[1].classList.contains('current')).toBe(true);

    // B10 CHECK: Player 1 should NOT have .current class during pvp_pass
    expect(statusPlayers[0].classList.contains('current')).toBe(false);
  });

  it('highlights Player 1 during select phase', () => {
    const state = makeState();
    const { container } = render(
      <GameBoard
        state={state}
        phase="select"
        mode="pvp"
        buildPhase={null}
        cardMeta={[]}
        p1Card={null}
        roundLog={[]}
        onSelectCard={noop}
        onPlayer2Card={noop}
        onNextRound={noop}
        onNewGame={noop}
        onBackToMenu={noop}
        onSelectBuildTarget={noop}
        onSelectDismantleTarget={noop}
        onCancelBuild={noop}
        pendingBuild={null}
        pileViewer={null}
        onViewPiles={noop}
        onClosePiles={noop}
        onPass={noop}
        onP2Pass={noop}
      />
    );

    const statusPlayers = container.querySelectorAll('.status-player');

    // Sanity: During select, Player 1 should have .current
    expect(statusPlayers[0].classList.contains('current')).toBe(true);
    // Player 2 should NOT have .current during select
    expect(statusPlayers[1].classList.contains('current')).toBe(false);
  });
});
