# Interface Design — 经典 DBG 抽牌机制

## 1. 新增接口

### 1.1 `Player.draw_cards(count: int) -> int`

从抽牌堆抽牌到手中，不足时自动将弃牌堆洗入抽牌堆。

**参数**：
- `count` — 期望抽牌数量，必须 ≥ 0。

**返回值**：
- 实际抽到的牌数（0 ≤ 返回值 ≤ count）。当抽牌堆和弃牌堆均空时，返回值可能小于 count。

**行为规则**：
1. 从 `draw_pile` 头部取最多 `count` 张，加入 `hand` 末尾
2. 若抽完后 `draw_pile` 已空且仍需继续抽牌：将 `discard_pile` 全部随机打乱后移入 `draw_pile`，清空 `discard_pile`，继续从 `draw_pile` 取牌
3. 若 `discard_pile` 也为空，则停止抽牌
4. 不对 `hand` 中原有牌做任何操作

**异常**：不抛出异常。`count=0` 时直接返回 0。

## 2. 变更接口

### 2.1 `GameCore._start_cycle()` 行为变更

**变更前**（design/26）：
```
hand → discard_pile
discard_pile → draw_pile
shuffle draw_pile
draw CARDS_PER_CYCLE from draw_pile
```

**变更后**：
```
hand → discard_pile
player.draw_cards(CARDS_PER_CYCLE)
```

即去掉"弃牌堆 → 抽牌堆"的自动合并步骤，改为调用智能抽牌方法。

### 2.2 `GameCore.play_round()` 行为不变

`play_round()` 内部调用的 `_start_cycle()` 行为随上述变更自动适配，play_round 的外部签名和返回类型不变。

## 3. 不变接口

以下接口签名和行为均不变化：

| 接口 | 说明 |
|------|------|
| `GameCore.start_game()` | 初始牌堆 7 张不变 |
| `GameCore.play_round(cards_1, cards_2, ...)` | 外部行为不变 |
| `GameCore.get_state(player_perspective)` | PlayerState 字段不变 |
| `Player.play_cards(cards)` | 弃牌到 discard_pile 不变 |
| `Player.has_card/cards(...)` | 手牌检查不变 |
| `BuildResolver.resolve(ctx, details)` | 构建的牌进 discard_pile 不变 |
