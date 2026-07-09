# Test Cases — 前端维护卡牌图标 (intention #23)

---

## 1. CardDef — icon 字段移除

### TC-CD-01: CardDef 不含 icon 字段
- **Given**: 导入 `from src.backend.game.card_def import CardDef`
- **When**: 检查 `CardDef.__dataclass_fields__`
- **Then**: 仅包含 `card_type`, `build_cost`, `display_name`, `resolvers` 四个字段，不含 `icon`

### TC-CD-02: 访问 icon 抛出 AttributeError
- **Given**: `get_card_def(CardType.ATTACK)`
- **When**: 访问 `cd.icon`
- **Then**: 抛出 AttributeError

### TC-CD-03: get_card_def 不依赖 icon
- **Given**: 遍历全部 7 个 CardType 调用 `get_card_def()`
- **Then**: 返回的 CardDef 实例均不包含 `icon` 属性；card_type, build_cost, display_name, resolvers 字段正常

---

## 2. GET /api/cards — 响应移除 icon

### TC-API-01: /api/cards 响应不含 icon 字段
- **Given**: Flask 测试客户端 `GET /api/cards`
- **When**: 解析响应 JSON
- **Then**: `cards` 数组每个元素仅包含 `type`, `display_name`, `build_cost` 三个字段，不含 `icon`

### TC-API-02: /api/cards 返回 7 张卡牌
- **Given**: Flask 测试客户端 `GET /api/cards`
- **When**: 解析响应 JSON 中 `cards` 数组长度
- **Then**: 长度为 7

### TC-API-03: /api/cards 每张卡牌字段值正确
- **Given**: Flask 测试客户端 `GET /api/cards`
- **When**: 检查 attack 卡牌
- **Then**: `type="attack"`, `display_name="攻击"`, `build_cost=1`

---

## 3. 前端 — CARD_ICONS 常量

### TC-FE-01: CARD_ICONS 包含全部 7 个卡牌类型
- **Given**: 渲染 GameBoard 组件
- **When**: 检查 CARD_ICONS 对象的 keys
- **Then**: 包含 `attack`, `defense`, `heavy`, `counter`, `energy`, `build`, `flex` 共 7 个 key

### TC-FE-02: buildLookups 不依赖 API icon 字段
- **Given**: cardMeta 数组中每项不含 `icon` 字段；仅含 `type`, `display_name`, `build_cost`
- **When**: 调用 `buildLookups(cardMeta)`
- **Then**: 返回对象包含 `names`, `order`, `costs`；不含 `icons` 字段

### TC-FE-03: 卡牌按钮正确显示图标
- **Given**: 渲染 GameBoard 组件；state.players['1'].hand 含 `['attack']`
- **When**: 检查卡牌按钮渲染内容
- **Then**: 按钮显示本地 `CARD_ICONS['attack']` 对应的 emoji（⚔️）

### TC-FE-04: 对手历史记录正确显示图标
- **Given**: state.players['2'].history 含 `[{card: 'attack'}]`
- **When**: 渲染状态栏历史 chips
- **Then**: chip 显示 ⚔️ 图标（使用本地 CARD_ICONS）

### TC-FE-05: Round Log 正确显示图标
- **Given**: roundLog 含 P1 attack vs P2 defense
- **When**: 渲染 Round Log 条目
- **Then**: 显示 `P1 ⚔️ vs AI 🛡️` 格式

### TC-FE-06: 牌堆查看器正确显示图标
- **Given**: pileViewer.draw_pile = `['attack', 'defense']`
- **When**: 渲染牌堆弹窗
- **Then**: 每张卡牌 chip 显示对应 emoji（⚔️, 🛡️）

### TC-FE-07: BUILD 目标选择器正确显示图标
- **Given**: buildPhase = 'select_target'
- **When**: 渲染 BUILD 目标选择器
- **Then**: 每个可选目标按钮显示对应 emoji 和名称

### TC-FE-08: API mock 数据不含 icon
- **Given**: 前端测试 mock 的 `MOCK_CARDS` 数据
- **Then**: 每项仅含 `type`, `display_name`, `build_cost`，不含 `icon`

---

## 4. 回归测试 — 游戏功能不受影响

### TC-RG-01: PVP 游戏创建与回合正常
- **Given**: `POST /api/games` 创建 PVP 游戏
- **When**: 连续 `POST /api/games/{id}/play` 多回合
- **Then**: 每回合返回正确的 RoundResult；HP/energy/statuses 结算正确

### TC-RG-02: PVE 游戏 AI 响应正常
- **Given**: `POST /api/games` 创建 PVE 游戏（mode=pve, ai=rnn_v1.1）
- **When**: `POST /api/games/{id}/play` 提交 P1 卡牌
- **Then**: 自动计算 AI 卡牌；返回正确 RoundResult

### TC-RG-03: BUILD 构建流程正常
- **Given**: P1 hand=[BUILD], energy=2
- **When**: `POST /api/games/{id}/play` 含 build_target_1=COUNTER
- **Then**: P1 energy 减少 2；discard_pile 增加 COUNTER

### TC-RG-04: 测试套件全部通过
- **Given**: 运行 `python -m pytest tests/` 和 `cd src/frontend; npm test`
- **Then**: 所有测试通过（后端 + 前端）
