# Test Cases — 卡牌名字效果变更 + 描述/标签字段 (intention #36)

---

## TC-01: CardDef 新增字段

### TC-01-01 — CardDef 包含 description 字段
- **Given**: `CardDef` dataclass 定义
- **When**: 通过 `get_card_def()` 获取任意卡牌定义
- **Then**: `cd.description` 为非空字符串；`"description" in CardDef.__dataclass_fields__`

### TC-01-02 — CardDef 包含 tags 字段
- **Given**: `CardDef` dataclass 定义
- **When**: 通过 `get_card_def()` 获取任意卡牌定义
- **Then**: `cd.tags` 为 `tuple[str, ...]` 类型；`"tags" in CardDef.__dataclass_fields__`

### TC-01-03 — 连续击 tags = ("攻",)
- **Given**: `CardDef` for `CardType.ATTACK`
- **When**: 读取 `cd.tags`
- **Then**: `cd.tags == ("攻",)`

### TC-01-04 — 格挡 tags = ("防",)
- **Given**: `CardDef` for `CardType.DEFENSE`
- **When**: 读取 `cd.tags`
- **Then**: `cd.tags == ("防",)`

### TC-01-05 — 攻守兼备 tags = ("攻", "防")
- **Given**: `CardDef` for `CardType.FLEX`
- **When**: 读取 `cd.tags`
- **Then**: `cd.tags == ("攻", "防")`

### TC-01-06 — 聚能 tags 为空元组
- **Given**: `CardDef` for `CardType.ENERGY`
- **When**: 读取 `cd.tags`
- **Then**: `cd.tags == ()`

### TC-01-07 — 构建 tags 为空元组
- **Given**: `CardDef` for `CardType.BUILD`
- **When**: 读取 `cd.tags`
- **Then**: `cd.tags == ()`

### TC-01-08 — CardDef 仍为 frozen（不可变）
- **Given**: 任意 CardDef 实例
- **When**: 尝试修改 `cd.description = "new"`
- **Then**: 抛出 `FrozenInstanceError`

---

## TC-02: display_name 变更

### TC-02-01 — 连续击
- **Given**: `get_card_def(CardType.ATTACK)`
- **Then**: `cd.display_name == "连续击"`

### TC-02-02 — 格挡
- **Given**: `get_card_def(CardType.DEFENSE)`
- **Then**: `cd.display_name == "格挡"`

### TC-02-03 — 破防重击
- **Given**: `get_card_def(CardType.HEAVY)`
- **Then**: `cd.display_name == "破防重击"`

### TC-02-04 — COUNTER / ENERGY / BUILD / FLEX 名称不变
- **Given**: `get_card_def()` for COUNTER、ENERGY、BUILD、FLEX
- **Then**: display_name 分别为 "防守反击"、"聚能"、"构建"、"攻守兼备"

---

## TC-03: 连续击（ATTACK）效果

### TC-03-01 — 首次连续击，对方无防御
- **Given**: 我方首次打出连续击，对方无 DEFENSE
- **Then**: 造成 1 点伤害

### TC-03-02 — 连续第 2 回合打出，对方无防御
- **Given**: 我方上轮打出连续击，本轮打出连续击，对方无 DEFENSE
- **Then**: 造成 2 点伤害

### TC-03-03 — 连续第 3+ 回合打出，伤害上限 3
- **Given**: 我方已连续打出 3 回合连续击，对方无 DEFENSE
- **Then**: 造成 3 点伤害（不超过 3）

### TC-03-04 — 对方有防御，伤害被阻挡
- **Given**: 对方有 DEFENSE 状态
- **When**: 我方打出连续击
- **Then**: 不造成伤害

---

## TC-04: 格挡（DEFENSE）效果

### TC-04-01 — 基础 +1 防御
- **Given**: 我方打出格挡，无 FRAGILE
- **Then**: 获得 DEFENSE（lifetime=1）

### TC-04-02 — 脆弱阻止防御
- **Given**: 我方有 FRAGILE
- **When**: 打出格挡
- **Then**: 不获得 DEFENSE

### TC-04-03 — 上轮打出防卡，获得联防
- **Given**: 我方上轮打出 DEFENSE/COUNTER/FLEX
- **When**: 本轮打出格挡
- **Then**: 获得 GUARD（lifetime=1）

### TC-04-04 — 上轮未打防卡，无联防
- **Given**: 我方上轮打出 ATTACK
- **When**: 本轮打出格挡
- **Then**: 不获得 GUARD

---

## TC-05: 破防重击（HEAVY）效果

### TC-05-01 — 对方无 GUARD，造成 1 点伤害
- **Given**: 对方无 GUARD
- **When**: 打出破防重击
- **Then**: 对方受到 1 点伤害

### TC-05-02 — 对方有 GUARD，伤害被阻挡
- **Given**: 对方有 GUARD
- **When**: 打出破防重击
- **Then**: 不造成伤害

### TC-05-03 — 对方有 DEFENSE，移除防御
- **Given**: 对方有 DEFENSE 状态
- **When**: 打出破防重击
- **Then**: 对方 DEFENSE 被移除

### TC-05-04 — 对方无 DEFENSE，不移除（无影响）
- **Given**: 对方无 DEFENSE
- **When**: 打出破防重击
- **Then**: 无异常，不报错

### TC-05-05 — 对方无 GUARD 且已有 1 回合脆弱，脆弱变为 3
- **Given**: 对方有 FRAGILE（lifetime=1），无 GUARD
- **When**: 打出破防重击
- **Then**: 对方 FRAGILE lifetime 变为 3

### TC-05-06 — 对方无 GUARD 且无脆弱，脆弱设为 2
- **Given**: 对方无 FRAGILE，无 GUARD
- **When**: 打出破防重击
- **Then**: 对方获得 FRAGILE（lifetime=2）

### TC-05-07 — 对方有 GUARD 有 DEFENSE，伤害和脆弱被阻挡但防御仍被移除
- **Given**: 对方有 GUARD + DEFENSE
- **When**: 打出破防重击
- **Then**: 不造成伤害、不给予脆弱；但 DEFENSE 被移除

---

## TC-06: 防守反击（COUNTER）效果

### TC-06-01 — 基础 +1 防御
- **Given**: 我方打出防守反击，无 FRAGILE
- **Then**: 获得 DEFENSE（lifetime=1）

### TC-06-02 — 对方出带"攻"标签卡且无防御 → 造成 1 点反击伤害
- **Given**: 对方打出 ATTACK（标签含 "攻"），对方无 DEFENSE
- **When**: 我方打出防守反击
- **Then**: 对方受到 1 点反击伤害

### TC-06-03 — 对方出带"攻"标签卡但有防御 → 无反伤
- **Given**: 对方打出 ATTACK，对方有 DEFENSE
- **When**: 我方打出防守反击
- **Then**: 不造成反击伤害

### TC-06-04 — 对方出不带"攻"标签卡 → 无反伤
- **Given**: 对方打出 DEFENSE（标签不含 "攻"）
- **When**: 我方打出防守反击
- **Then**: 不造成反击伤害

### TC-06-05 — 对方出攻守兼备（标签含"攻"）且无防御 → 触发反伤
- **Given**: 对方打出 FLEX（标签含 "攻"），对方无 DEFENSE
- **When**: 我方打出防守反击
- **Then**: 对方受到 1 点反击伤害

### TC-06-06 — 对方 pass → 无反伤
- **Given**: 对方 pass（`card_played_by_opponent is None`）
- **When**: 我方打出防守反击
- **Then**: 不造成反击伤害

### TC-06-07 — 上轮打防卡，获得联防
- **Given**: 我方上轮打出 DEFENSE
- **When**: 本轮打出防守反击
- **Then**: 获得 GUARD

### TC-06-08 — 上轮未打防卡，无联防
- **Given**: 我方上轮打出 ATTACK
- **When**: 本轮打出防守反击
- **Then**: 不获得 GUARD

---

## TC-07: 聚能（ENERGY）效果

### TC-07-01 — 首次打出，+1 能量
- **Given**: 我方 history 为空或上轮未打 ENERGY
- **When**: 打出聚能
- **Then**: 能量 +1

### TC-07-02 — 连续打出，+2 能量
- **Given**: 我方上轮打出 ENERGY
- **When**: 本轮打出聚能
- **Then**: 能量 +2

### TC-07-03 — 能量不应超过上限（5）
- **Given**: 我方当前能量 = 4，上轮打出 ENERGY
- **When**: 打出聚能
- **Then**: 能量 = 5（+2 但上限截断）

---

## TC-08: 攻守兼备（FLEX）效果

### TC-08-01 — 基础 +1 防御
- **Given**: 我方打出攻守兼备，无 FRAGILE
- **Then**: 获得 DEFENSE（lifetime=1）

### TC-08-02 — 上轮打出"防"标签卡 → 额外防御
- **Given**: 我方上轮打出 DEFENSE（标签含 "防"）
- **When**: 本轮打出攻守兼备
- **Then**: 获得 2 回合 DEFENSE（基础 1 + 额外 1）

### TC-08-03 — 上轮未打"防"标签卡 → 仅基础防御
- **Given**: 我方上轮打出 ATTACK（标签不含 "防"）
- **When**: 本轮打出攻守兼备
- **Then**: 获得 1 回合 DEFENSE（仅基础）

### TC-08-04 — 对方无防御 → 造成 1 点伤害
- **Given**: 对方无 DEFENSE
- **When**: 打出攻守兼备
- **Then**: 对方受到 1 点伤害

### TC-08-05 — 对方有防御 → 伤害被阻挡
- **Given**: 对方有 DEFENSE
- **When**: 打出攻守兼备
- **Then**: 不造成伤害

### TC-08-06 — 不触发联防
- **Given**: 我方上轮打出 DEFENSE
- **When**: 本轮打出攻守兼备
- **Then**: 不获得 GUARD（FLEX resolvers 不含 GrantGuardResolver）

### TC-08-07 — 脆弱阻止所有防御（基础和额外）
- **Given**: 我方有 FRAGILE
- **When**: 打出攻守兼备
- **Then**: 不获得任何 DEFENSE

---

## TC-09: Player.extend_status()

### TC-09-01 — 延长已有状态
- **Given**: Player 有 FRAGILE（lifetime=1）
- **When**: 调用 `extend_status(FRAGILE, 2)`
- **Then**: FRAGILE lifetime = 3

### TC-09-02 — 目标状态不存在，新建状态
- **Given**: Player 无 FRAGILE
- **When**: 调用 `extend_status(FRAGILE, 2)`
- **Then**: 获得 FRAGILE（lifetime=2）

---

## TC-10: Player.remove_status()

### TC-10-01 — 移除现有防御
- **Given**: Player 有 DEFENSE
- **When**: 调用 `remove_status(DEFENSE)`
- **Then**: `has_status(DEFENSE)` 返回 False

### TC-10-02 — 移除不存在的状态无异常
- **Given**: Player 无 DEFENSE
- **When**: 调用 `remove_status(DEFENSE)`
- **Then**: 不抛出异常

---

## TC-11: GET /api/cards 响应

### TC-11-01 — 响应包含 description 字段
- **Given**: `GET /api/cards`
- **Then**: 响应 JSON 中每张卡牌包含 `"description"` 字段，值为非空字符串

### TC-11-02 — 响应包含 tags 字段
- **Given**: `GET /api/cards`
- **Then**: 响应 JSON 中每张卡牌包含 `"tags"` 字段，值为字符串数组

### TC-11-03 — 聚能 tags 为空数组
- **Given**: `GET /api/cards`
- **Then**: ENERGY 卡牌的 `tags` 为 `[]`

### TC-11-04 — display_name 返回新名称
- **Given**: `GET /api/cards`
- **Then**: ATTACK 的 `display_name` 为 "连续击"；DEFENSE 为 "格挡"；HEAVY 为 "破防重击"
