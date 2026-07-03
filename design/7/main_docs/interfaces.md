# Interface Design — Sample File Restructuring

Public interfaces only. Private methods omitted per design rule.

---

## 1. `app/selfplay_runner.py` — SelfPlayRunner (Modified)

`run()` is modified to collect game sample dicts in memory and write a single JSON array file at the end. The former `_write_samples()` static method is replaced by `_build_game_sample()`. `_play_one_game()` no longer receives `sample_fh`.

```
class SelfPlayRunner:
    """Execute self-play games and collect results."""

    def __init__(self, sample_action_fn):
        """(unchanged)"""

    def run(self, config: SelfPlayConfig) -> SelfPlayResult:
        """Execute all self-play games and return aggregate results.

        Modified: collects game sample dicts into a list.
        If config.sample_output is set, writes the list as a single
        JSON array file at the end. Otherwise, no sample file is created.

        Returns:
            SelfPlayResult with win/loss/draw counts and avg rounds.
        """

    # ------------------------------------------------------------------
    # Replaces former _write_samples()
    # ------------------------------------------------------------------

    @staticmethod
    def _build_game_sample(
        game_samples: list[dict],
        game_id: int,
        winner: str | None,
        rounds_played: int,
        p1_reward: float,
        p2_reward: float,
        ai1: str,
        ai2: str,
    ) -> dict:
        """Build a structured game sample dict from per-step samples.

        Splits the flat game_samples list into two perspective groups
        (by the 'player' field) and assigns the appropriate reward to
        each step.

        Args:
            game_samples: List of per-step sample dicts with keys
                player, state, action, value, log_prob (as built in
                _play_one_game).
            game_id: 1-based game index.
            winner: "1", "2", or None (draw).
            rounds_played: Number of rounds in this game.
            p1_reward: Reward for all player 1 steps.
            p2_reward: Reward for all player 2 steps.
            ai1: AI name for player 1.
            ai2: AI name for player 2.

        Returns:
            dict with keys game_id, winner, rounds, perspectives.
            Each perspective is {player, ai, steps}.
            Each step is {state, action, reward, value, log_prob}.
        """
```

---

## 2. Sample File Schema

Output file is a single JSON array. Each element is a game object.

```json
[
  {
    "game_id": 1,
    "winner": "1",
    "rounds": 5,
    "perspectives": [
      {
        "player": "1",
        "ai": "rnn_v1.1",
        "steps": [
          {
            "state": [3.0, 2.0, 1.0, 1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "action": "attack",
            "reward": 1.0,
            "value": -0.066,
            "log_prob": -1.341
          }
        ]
      },
      {
        "player": "2",
        "ai": "rnn_v1.1",
        "steps": [
          {
            "state": [3.0, 2.0, 1.0, 1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "action": "defense",
            "reward": -1.0,
            "value": -0.066,
            "log_prob": -1.579
          }
        ]
      }
    ]
  }
]
```

**Field constraints:**

| Field | Type | Constraints |
|-------|------|-------------|
| `game_id` | int | ≥1, sequential |
| `winner` | string or null | `"1"`, `"2"`, or `null` (draw) |
| `rounds` | int | 1–7 |
| `perspectives[].player` | string | `"1"` or `"2"` |
| `perspectives[].ai` | string | AI name from registry |
| `steps[].state` | array[float] | Length exactly 14 |
| `steps[].action` | string | `"attack"`, `"defense"`, `"heavy"`, `"counter"` |
| `steps[].reward` | float | 1.0, -1.0, or 0.0 |
| `steps[].value` | float | In [-1.0, 1.0] |
| `steps[].log_prob` | float or null | float ≤ 0, or null for greedy steps |
