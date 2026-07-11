# Test Cases — 经典 DBG 抽牌机制

## 1. `Player.draw_cards(count)` 单元测试

### 1.1 正常抽牌 — 抽牌堆充足
- **GIVEN** 玩家抽牌堆有 [ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY]，弃牌堆有 [BUILD, FLEX]，手牌为 []
- **WHEN** 调用 `draw_cards(3)`
- **THEN** 手牌变为 [ATTACK, DEFENSE, HEAVY]，抽牌堆变为 [COUNTER, ENERGY]，弃牌堆保持 [BUILD, FLEX]，返回 3

### 1.2 抽牌堆恰好够
- **GIVEN** 抽牌堆有 [ATTACK, DEFENSE]，弃牌堆有 [HEAVY]，手牌为 []
- **WHEN** 调用 `draw_cards(2)`
- **THEN** 手牌变为 [ATTACK, DEFENSE]，抽牌堆为 []，弃牌堆保持 [HEAVY]，返回 2

### 1.3 抽牌堆不足 — 弃牌堆补足
- **GIVEN** 抽牌堆有 [ATTACK]，弃牌堆有 [DEFENSE, HEAVY, COUNTER, ENERGY]，手牌为 []
- **WHEN** 调用 `draw_cards(5)`
- **THEN** 手牌为 5 张且前 1 张为 ATTACK，抽牌堆为空，弃牌堆为空，返回 5

### 1.4 抽牌堆为空 — 全靠弃牌堆
- **GIVEN** 抽牌堆有 []，弃牌堆有 [ATTACK, ATTACK, DEFENSE]，手牌为 []
- **WHEN** 调用 `draw_cards(3)`
- **THEN** 手牌为 3 张（ATTACK, ATTACK, DEFENSE），抽牌堆为 []，弃牌堆为 []，返回 3

### 1.5 两堆均空 — 提前结束
- **GIVEN** 抽牌堆有 []，弃牌堆有 []，手牌为 []
- **WHEN** 调用 `draw_cards(5)`
- **THEN** 手牌保持 []，抽牌堆保持 []，弃牌堆保持 []，返回 0

### 1.6 两堆合计不足 — 抽到竭尽
- **GIVEN** 抽牌堆有 [ATTACK, DEFENSE]，弃牌堆有 []，手牌为 []
- **WHEN** 调用 `draw_cards(5)`
- **THEN** 手牌变为 [ATTACK, DEFENSE]，抽牌堆为 []，弃牌堆为 []，返回 2

### 1.7 弃牌堆洗入后乱序
- **GIVEN** 抽牌堆有 [ATTACK]，弃牌堆有 [DEFENSE, HEAVY, COUNTER, ENERGY, BUILD]，手牌为 []
- **WHEN** 调用 `draw_cards(5)`，且已知弃牌堆洗牌后顺序不可预测
- **THEN** 手牌第 1 张固定为 ATTACK，后 4 张来自弃牌堆的原 5 张中的 4 张，弃牌堆余 1 张留在抽牌堆中，返回 5；重复多次调用验证非固定顺序（弃牌堆确实被洗牌）

### 1.8 count=0 边界
- **GIVEN** 抽牌堆有 [ATTACK, DEFENSE]，弃牌堆有 [HEAVY]，手牌为 []
- **WHEN** 调用 `draw_cards(0)`
- **THEN** 手牌为 []，抽牌堆为 [ATTACK, DEFENSE]，弃牌堆为 [HEAVY]，返回 0

### 1.9 手牌非空时追加抽牌
- **GIVEN** 手牌为 [COUNTER]，抽牌堆有 [ATTACK, DEFENSE]，弃牌堆为 []
- **WHEN** 调用 `draw_cards(2)`
- **THEN** 手牌变为 [COUNTER, ATTACK, DEFENSE]，抽牌堆为 []，返回 2

## 2. Cycle 开始集成测试

### 2.1 首轮开局抽牌
- **GIVEN** 游戏通过 `start_game()` 启动
- **THEN** 每个玩家手牌为 5 张，抽牌堆余 2 张，弃牌堆为空

### 2.2 Cycle 间过渡 — 抽牌堆有剩余
- **GIVEN** Cycle 1 进行了 3 回合（各打出 1 张牌），cycle 结束时触发 `_start_cycle()`
- **WHEN** 抽牌堆剩余 2 张（初始 7 - 5）、弃牌堆含 3 张（打出）+ 2 张（手牌剩余）
- **THEN** 手牌变为 0 → 手牌（曾是弃牌堆的牌）→ 弃牌堆清空 → `draw_cards(5)` 被调用：先抽抽牌堆 2 张 → 弃牌堆 5 张洗入抽牌堆 → 再抽 3 张 → 最终手牌 5 张、抽牌堆余 2 张、弃牌堆空

### 2.3 Cycle 间过渡 — 抽牌堆恰好为空
- **GIVEN** 多个 cycle 后，当前 cycle 结束，上一 cycle 的抽牌堆已耗尽（最后一张已被抽出）
- **WHEN** 新 cycle 开始
- **THEN** 手牌清空进入弃牌堆，`draw_cards(5)` 先发现抽牌堆空 → 弃牌堆洗入抽牌堆 → 抽 5 张 → 手牌 5 张

### 2.4 Cycle 内回合 — 弃牌堆正常累积
- **GIVEN** Cycle 第 1 回合
- **WHEN** 玩家各打出 1 张牌
- **THEN** 打出的牌进入弃牌堆，手牌减少 1 张，抽牌堆不变

### 2.5 多 Cycle 后总牌数增长 — BUILD 构建
- **GIVEN** 玩家在多个 cycle 中 BUILD 了 3 张新牌
- **WHEN** 新 cycle 开始触发 `_start_cycle()`
- **THEN** 弃牌堆包含 BUILD 构建的牌 + 未打出的手牌，`draw_cards(5)` 正确从混合牌堆中抽牌
