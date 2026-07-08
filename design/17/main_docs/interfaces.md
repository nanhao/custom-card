# Interface Design — RoundResult 字段层次化重组 (intention #17)

## 1. 新增嵌套 dataclass

位于 `game/game_core.py`。与 `RoundResult` 定义在同一文件中。

### 1.1 BeforeResolve

结算前的玩家状态快照。

```python
@dataclass
class BeforeResolve:
    """State snapshot taken BEFORE card resolution.

    Captures HP, energy and statuses as they were at the start of the round,
    before any card effects are applied.
    """
    statuses_1: list[PlayerStatus]
    statuses_2: list[PlayerStatus]
    energy_1: int
    energy_2: int
    hp_1: int
    hp_2: int
```

### 1.2 AfterResolve

卡牌效果结算后的即时结果。

```python
@dataclass
class AfterResolve:
    """State immediately AFTER card resolution (Phase 1).

    Includes resolution details, damage applied, and game-over check
    triggered by HP reaching 0 during card resolution.
    """
    statuses_1: list[PlayerStatus]
    statuses_2: list[PlayerStatus]
    energy_1: int
    energy_2: int
    hp_1: int
    hp_2: int
    details: list[str]
    is_game_over: bool = False
    winner: str | None = None
```

### 1.3 EndActions

回合结束时的动作记录。

```python
@dataclass
class EndActions:
    """Round-end actions applied after card resolution (Phase 2).

    cycle_ended is True only when round_in_cycle reaches ROUNDS_PER_CYCLE.
    bleed_damage values are 0 when cycle_ended is False.
    """
    cycle_ended: bool = False
    bleed_damage_1: int = 0
    bleed_damage_2: int = 0
```

### 1.4 AfterEndActions

回合全部处理完成后的最终状态。

```python
@dataclass
class AfterEndActions:
    """Final state snapshot after all round-end processing (Phase 3).

    Includes round/counter advancement, deck reshuffling (if cycle ended),
    and a second game-over check for bleed damage.
    hand_cards_1/hand_cards_2 reflect the NEW hand after cycle processing.
    """
    is_game_over: bool = False
    winner: str | None = None
    round: int = 0
    cycle: int = 0
    round_in_cycle: int = 0
    hand_cards_1: list[CardType] = field(default_factory=list)
    hand_cards_2: list[CardType] = field(default_factory=list)
    statuses_1: list[PlayerStatus] = field(default_factory=list)
    statuses_2: list[PlayerStatus] = field(default_factory=list)
    energy_1: int = 0
    energy_2: int = 0
    hp_1: int = 0
    hp_2: int = 0
```

## 2. RoundResult — 重构

### 2.1 旧结构（删除）

`RoundResult` 的所有旧字段全部移除，替换为嵌套 dataclass。

删除的字段列表：`round`, `cycle`, `round_in_cycle`, `cards_1`, `cards_2`, `damage_1`, `damage_2`, `hp_1_after`, `hp_2_after`, `energy_1_after`, `energy_2_after`, `statuses_1`, `statuses_2`, `details`, `cycle_ended`, `bleed_damage_1`, `bleed_damage_2`, `after_cycle`, `after_round_in_cycle`, `after_hp_1`, `after_hp_2`, `after_statuses_1`, `after_statuses_2`, `is_game_over`, `winner`。

### 2.2 新结构

```python
@dataclass
class RoundResult:
    """Value object returned by GameCore.play_round().

    Top-level: round identifiers and cards played.
    before: snapshot before card resolution.
    after_resolve: resolution outcome and immediate post-resolve state.
    end_actions: round-end actions (status tick, cycle bleed).
    after_end_actions: final state after all processing (next round start).
    """
    round: int
    cycle: int
    round_in_cycle: int
    cards_1: list[CardType]
    cards_2: list[CardType]
    before: BeforeResolve
    after_resolve: AfterResolve
    end_actions: EndActions
    after_end_actions: AfterEndActions
```

## 3. GameCore.play_round — 变更

位于 `game/game_core.py`。方法签名不变。

### 3.1 快照采集时序

1. **Step 1-2（验证前）**：采集 `before` 快照（HP、能量、状态）
2. **Step 9**：填充 `after_resolve`（与旧 Phase 1 等价）
3. **Step 11b**：填充 `end_actions.cycle_ended`、`bleed_damage`
4. **Step 13**：填充 `after_end_actions`

### 3.2 before 快照采集位置

在 `_validate()` 之后、`resolution = CardResolver.resolve()` 之前采集：

```
before = BeforeResolve(
    statuses_1=[PlayerStatus(ps.status_type, ps.lifetime) for ps in self.player_1.statuses],
    statuses_2=[PlayerStatus(ps.status_type, ps.lifetime) for ps in self.player_2.statuses],
    energy_1=self.player_1.energy,
    energy_2=self.player_2.energy,
    hp_1=self.player_1.hp,
    hp_2=self.player_2.hp,
)
```

### 3.3 游戏结束判定逻辑

旧代码中 `is_game_over` 和 `winner` 是在 Step 10（结算后）和 Step 11b（流血后）分别写入同一个 `result.is_game_over`/`result.winner`，后者会覆盖前者。

新结构中两者独立：
- `result.after_resolve.is_game_over` / `result.after_resolve.winner` — 结算伤害导致的游戏结束
- `result.after_end_actions.is_game_over` / `result.after_end_actions.winner` — 流血导致的游戏结束

若结算时游戏已结束，`after_end_actions` 直接复制 `after_resolve` 的值（不再走流血逻辑）。

## 4. Serializers — 变更

位于 `controller/serializers.py`。`round_result_to_dict()` 适配新结构：

```python
def round_result_to_dict(rr: RoundResult) -> dict:
    return {
        "round": rr.round,
        "cycle": rr.cycle,
        "round_in_cycle": rr.round_in_cycle,
        "cards_1": [card_to_str(c) for c in rr.cards_1],
        "cards_2": [card_to_str(c) for c in rr.cards_2],
        "before": {
            "statuses_1": [ps.to_dict() for ps in rr.before.statuses_1],
            "statuses_2": [ps.to_dict() for ps in rr.before.statuses_2],
            "energy_1": rr.before.energy_1,
            "energy_2": rr.before.energy_2,
            "hp_1": rr.before.hp_1,
            "hp_2": rr.before.hp_2,
        },
        "after_resolve": {
            "statuses_1": [ps.to_dict() for ps in rr.after_resolve.statuses_1],
            "statuses_2": [ps.to_dict() for ps in rr.after_resolve.statuses_2],
            "energy_1": rr.after_resolve.energy_1,
            "energy_2": rr.after_resolve.energy_2,
            "hp_1": rr.after_resolve.hp_1,
            "hp_2": rr.after_resolve.hp_2,
            "details": rr.after_resolve.details,
            "is_game_over": rr.after_resolve.is_game_over,
            "winner": rr.after_resolve.winner,
        },
        "end_actions": {
            "cycle_ended": rr.end_actions.cycle_ended,
            "bleed_damage_1": rr.end_actions.bleed_damage_1,
            "bleed_damage_2": rr.end_actions.bleed_damage_2,
        },
        "after_end_actions": {
            "is_game_over": rr.after_end_actions.is_game_over,
            "winner": rr.after_end_actions.winner,
            "round": rr.after_end_actions.round,
            "cycle": rr.after_end_actions.cycle,
            "round_in_cycle": rr.after_end_actions.round_in_cycle,
            "hand_cards_1": [card_to_str(c) for c in rr.after_end_actions.hand_cards_1],
            "hand_cards_2": [card_to_str(c) for c in rr.after_end_actions.hand_cards_2],
            "statuses_1": [ps.to_dict() for ps in rr.after_end_actions.statuses_1],
            "statuses_2": [ps.to_dict() for ps in rr.after_end_actions.statuses_2],
            "energy_1": rr.after_end_actions.energy_1,
            "energy_2": rr.after_end_actions.energy_2,
            "hp_1": rr.after_end_actions.hp_1,
            "hp_2": rr.after_end_actions.hp_2,
        },
    }
```

## 5. GameCore.get_state — 变更

`get_state()` 中 `round_log` 的序列化从手动逐字段构造变为调用 `round_result_to_dict()`：

```python
"round_log": [round_result_to_dict(rr) for rr in self.round_log],
```

同时需要导入 `round_result_to_dict`。为避免循环依赖，`GameCore.get_state()` 改为接收一个可选的 serializer callable，或直接在方法内调用序列化函数。

**决策**：`round_result_to_dict` 从 `controller/serializers.py` 移至 `game/game_core.py`（与 `RoundResult` 同文件），消除 `game` 层对 `controller` 层的依赖。

## 6. SelfPlayRunner — 变更

位于 `app/selfplay_runner.py`。日志中访问 `RoundResult` 字段路径变更：

```python
# 旧
logger.log("DEBUG", "selfplay.round_result",
    round=last_rr.round,
    dmg1=last_rr.damage_1,
    dmg2=last_rr.damage_2,
    hp1=last_rr.hp_1_after,
    hp2=last_rr.hp_2_after,
)

# 新
logger.log("DEBUG", "selfplay.round_result",
    round=last_rr.round,
    hp1_before=last_rr.before.hp_1,
    hp2_before=last_rr.before.hp_2,
    hp1_after=last_rr.after_resolve.hp_1,
    hp2_after=last_rr.after_resolve.hp_2,
)
```

伤害值由 `before.hp` 和 `after_resolve.hp` 的差值隐式体现。若未来需要精确伤害值（HP 归零时差值可能小于实际伤害），可通过 `after_resolve.details` 文本解析。

## 7. SessionStore — 变更

`_serialize_game()` 中 `round_log` 的序列化，从手动展开改为调用 `round_result_to_dict()`。

## 8. 前端的变更（汇总）

| 组件 | 旧访问路径 | 新访问路径 |
|------|-----------|-----------|
| GameBoard `rr.cards_1[0]` | `rr.cards_1[0]` | 不变（顶层字段） |
| GameBoard `rr.cards_2[0]` | `rr.cards_2[0]` | 不变 |
| GameBoard `rr.details` | `rr.details` | `rr.after_resolve.details` |
| GameBoard `rr.cycle_ended` | `rr.cycle_ended` | `rr.end_actions.cycle_ended` |
| GameBoard `rr.hp_1_after` | `rr.hp_1_after` | `rr.after_resolve.hp_1` |
| GameBoard `rr.hp_2_after` | `rr.hp_2_after` | `rr.after_resolve.hp_2` |
| GameBoard `rr.energy_1_after` | `rr.energy_1_after` | `rr.after_resolve.energy_1` |
| GameBoard `rr.energy_2_after` | `rr.energy_2_after` | `rr.after_resolve.energy_2` |
| App.jsx `data.round_result.statuses_1` | `rr.statuses_1` | `rr.after_resolve.statuses_1` |
| App.jsx `data.round_result.statuses_2` | `rr.statuses_2` | `rr.after_resolve.statuses_2` |
| App.jsx `data.round_result.is_game_over` | `rr.is_game_over` | `rr.after_resolve.is_game_over` |
| cycle-info `round_log[last].cycle_ended` | `...cycle_ended` | `...end_actions.cycle_ended` |

## 9. 不受影响的模块

- `app/converter.py` — 不依赖 RoundResult
- `controller/api.py` — 仅调用 `round_result_to_dict()`，接口不变
- `game/resolver.py` — 不依赖 RoundResult
- `game/card.py` — 不依赖 RoundResult
- `game/player.py` — 不依赖 RoundResult
- `ai/` — 全部模块不依赖 RoundResult
