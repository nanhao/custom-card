# Interface Design — 回合与抽牌参数调整 (intention #26)

## 1. GameCore

### 1.1 类常量变更

```
ROUNDS_PER_CYCLE: int = 3    # 原值 5
CARDS_PER_CYCLE: int = 5     # 新增
```

### 1.2 _start_cycle

行为不变（回收手牌 → 洗入抽牌堆 → 抽牌），仅抽牌数从 7 改为 `CARDS_PER_CYCLE`（5）。

**变更前**：`draw_count = min(7, len(player.draw_pile))`
**变更后**：`draw_count = min(CARDS_PER_CYCLE, len(player.draw_pile))`

### 1.3 play_round

**cycle-end 判定**：`round_in_cycle >= ROUNDS_PER_CYCLE` 语义不变，ROUNDS_PER_CYCLE 改为 3 后，第 3 回合结束时触发。

**RoundResult.end_actions**：
- `cycle_ended = True` 仅当 `round_in_cycle == 3`
- `bleed_damage_1 = bleed_damage_2 = 1` 同前

**RoundResult.after_end_actions**：
- `round_in_cycle` 在 cycle-ended 后重置为 1
- `cycle` 递增

### 1.4 get_state

返回值 `GameState.round_in_cycle` 取值由 1-5 变为 1-3。

### 1.5 不变的部分

- `INITIAL_HP = 20`
- `_tick_status_lifetimes()` — 每回合正常执行
- `_validate()` — 校验逻辑不变
- 其他所有方法签名和行为不变

## 2. GameState

### 2.1 round_in_cycle

docstring 从 "Round position within current cycle (1–5)" 改为 "Round position within current cycle (1–3)"。

字段类型 `int` 不变。

## 3. RoundResult

### 3.1 EndActions.cycle_ended

docstring 从 "cycle_ended is True only when round_in_cycle reaches ROUNDS_PER_CYCLE" 改为仅描述语义（不硬编码数值）。

## 4. 前端

### 4.1 GameBoard.jsx — 回合显示

3 处硬编码 `/5` 改为 `/3`：

- 选牌阶段：`Round {state.round_in_cycle}/5` → `Round {state.round_in_cycle}/3`
- 结算弹窗：`Round {rr.round_in_cycle}/5` → `Round {rr.round_in_cycle}/3`
- 状态栏：`Round {state.round_in_cycle}/5` → `Round {state.round_in_cycle}/3`

## 5. card_def — 不变

`get_initial_draw_pile()` 保持返回 7 张牌，不做修改。
