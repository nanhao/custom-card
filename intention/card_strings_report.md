# 卡牌字符串常量分散问题分析报告

> 生成日期：2026-07-08

---

## 1. 问题概述

项目中存在大量硬编码的卡牌字符串常量（如 `"attack"`, `"defense"`, `"build"` 等），分散在后端 6 个模块和前端 2 个组件中，共计 **约 120+ 处**。虽然后端已有 `CardType` 枚举作为名义上的"单点真相源"，但实际使用中大量绕过枚举直接用字符串字面量，导致：

- **修改成本高**：新增/删除卡牌类型需要改动 10+ 个文件中的多处代码
- **缺乏编译时检查**：字符串字面量拼写错误只能在运行时发现
- **前端后端不一致**：前端独立维护一套卡牌元数据，与后端不同步
- **隐式耦合**：多处通过 hardcode 的 `"build"` 字符串做分支判断

---

## 2. 卡牌字符串常量清单

### 2.1 卡牌类型标识（7 种）

| 字符串 | CardType 枚举 | 含义 |
|--------|-------------|------|
| `"attack"` | `CardType.ATTACK` | 攻击 |
| `"defense"` | `CardType.DEFENSE` | 防御 |
| `"heavy"` | `CardType.HEAVY` | 重击 |
| `"counter"` | `CardType.COUNTER` | 反击 |
| `"energy"` | `CardType.ENERGY` | 聚能 |
| `"build"` | `CardType.BUILD` | 构建 |
| `"flex"` | `CardType.FLEX` | 攻守兼备 |

### 2.2 效果类型标识（7 种）

| 字符串 | 含义 |
|--------|------|
| `"damage"` | 造成伤害 |
| `"grant_defense"` | 授予防御状态 |
| `"grant_guard"` | 授予联防状态 |
| `"reflect"` | 反弹伤害 |
| `"gain_energy"` | 获取能量 |
| `"flex_damage"` | 攻守兼备伤害 |
| `"build"` | 构建（无即时效果） |

---

## 3. 各文件硬编码位置明细

### 3.1 `src/backend/game/card.py` — 定义层（已用枚举，相对干净）

```python
class CardType(Enum):
    ATTACK = "attack"     # L20
    DEFENSE = "defense"   # L21
    HEAVY = "heavy"       # L22
    COUNTER = "counter"   # L23
    ENERGY = "energy"     # L24
    BUILD = "build"       # L25
    FLEX = "flex"         # L26
```

**效果字符串**（effect_type 是裸字符串，无枚举保护）：
```python
effects=(("damage", 10),)           # L62
effects=(("grant_defense", 11), ("grant_guard", 11))  # L70
effects=(("damage", 10),)           # L78
effects=(("grant_defense", 11), ("reflect", 10))      # L86
effects=(("gain_energy", 10),)      # L94
effects=(("build", 10),)            # L102
effects=(("grant_defense", 11), ("flex_damage", 10))  # L110
```

**问题**: `CardDef.effects` 的类型注解是 `tuple[tuple[str, int], ...]`，effect_type 是裸 `str`，编译时无法校验。

---

### 3.2 `src/backend/app/converter.py` — 重灾区（6 个独立 dict，全部硬编码）

| 位置 | 变量 | 硬编码内容 | 行号 |
|------|------|-----------|------|
| 类属性 | `ALL_CARD_TYPES` | frozenset of 7 种卡牌字符串 | L31-33 |
| 类属性 | `_PLAY_TO_INDEX` | 6 条 card_str→index 映射 | L40-47 |
| 类属性 | `_BUILD_TO_INDEX` | 7 条 card_str→index 映射 | L50-58 |
| 类属性 | `_INDEX_TO_ACTION` | 13 条 index→(str, str\|None) 映射 | L61-75 |
| 类属性 | `_BUILD_COST` | 7 条 card_str→cost 映射 | L78-86 |
| 类属性 | `_CARD_TYPE_ORDER` | 7 条 card_str→index 映射 | L92-100 |
| 方法内 | `action_to_policy_index()` | `if card_str == "build":` | L207 |

**核心问题**: 6 个 dict 重复声明同一组卡牌字符串 6 次，任何增删卡牌都需要修改 6 处。这些 dict 之间存在严格的数学对应关系，但当前是通过"手动保持同步"来维护的。

---

### 3.3 `src/backend/app/app_manager.py` — mask_fn 硬编码

**位置**: `_make_mask_fn()` 静态方法 (L262-310)

```python
# L286-289: 硬编码 play 映射元组
pure_play_cards = [
    (1, "attack"), (3, "defense"), (5, "heavy"),
    (7, "counter"), (9, "energy"), (11, "flex"),
]

# L297-298: 硬编码 build 映射列表
build_targets = ["build", "attack", "defense", "heavy",
                 "counter", "energy", "flex"]

# L281: 硬编码字符串判断
can_build = "build" in legal_cards and energy > 0
```

**问题**: 这些数据实际上是 `Converter._PLAY_TO_INDEX` 和 `Converter._BUILD_TO_INDEX` 的反向副本，但被独立维护。

---

### 3.4 `src/backend/game/resolver.py` — 效果分支全部硬编码

```python
if effect.effect_type == "grant_defense":    # L200
elif effect.effect_type == "grant_guard":    # L208
elif effect.effect_type == "damage":         # L218
elif effect.effect_type == "reflect":        # L248
elif effect.effect_type == "gain_energy":    # L257
elif effect.effect_type == "flex_damage":    # L265
elif effect.effect_type == "build":          # L279
```

以及 card type 的枚举比较（这里已经用了枚举，OK）：
- `CardType.ATTACK` (L87, L219, L249)
- `CardType.HEAVY` (L234, L249)

**问题**: effect_type 是裸字符串，没有枚举保护。

---

### 3.5 `src/backend/controller/serializers.py` — 相对干净

`_CARD_BY_VALUE` 从 `CardType` 枚举动态构建（L18），只要枚举不变就不需要手动同步。

---

### 3.6 `src/backend/game/session_store.py` — 相对干净

`_CARD_TYPE_BY_VALUE` 从 `CardType` 枚举动态构建（L26）。

---

### 3.7 `src/backend/controller/api.py` — 1 处

```python
# L291: 使用 Converter.ALL_CARD_TYPES 校验，OK
if card_str not in Converter.ALL_CARD_TYPES:
```

---

### 3.8 `src/backend/app/selfplay_runner.py` — 相对干净

使用 `CardType(card1)` 枚举构造（L292-293），OK。

---

### 3.9 前端 `src/frontend/src/GameBoard.jsx` — 3 处独立硬编码

| 位置 | 变量 | 内容 | 行号 |
|------|------|------|------|
| 常量 | `CARD_NAMES` | 7 条 card_str→display 映射 | L17-20 |
| 常量 | `CARD_ICONS` | 7 条 card_str→icon 映射 | L21-24 |
| 常量 | `CARD_ORDER` | 7 种卡牌显示顺序数组 | L25 |
| 方法内 | `renderHandArea()` 内嵌逻辑 | `(card === 'counter' \|\| card === 'flex') ? 2 : 1` | L423 |

**问题**:
- 前端重复维护了与后端 `CardDef` 等效的元数据（名称、图标、费用）
- `CARD_ORDER` 定义了显示顺序，与后端 `Converter._CARD_TYPE_ORDER` 独立但语义相同
- `build_cost` 硬编码在前端逻辑中（`'counter' || 'flex' → 2`）

---

### 3.10 前端 `src/frontend/src/App.jsx` — 2 处

```javascript
// L129
if (card === 'build') {
// L149
await doPlayCard('build', target);
```

---

### 3.11 测试文件 — 大量硬编码字符串

| 文件 | 硬编码内容 |
|------|-----------|
| `tests/unit/test_converter.py` | 硬编码 7 种卡牌字符串在 state dict 和参数化测试中 |
| `tests/unit/test_app_manager.py` | 硬编码 `["attack", "energy"]`, `["attack", "build"]` 等 |
| `tests/unit/test_game_core.py` | 硬编码 `"energy"` 等 |
| `tests/unit/test_player_state.py` | 硬编码 `"defense"`, `"guard"` |
| `tests/integration/test_controller_api.py` | 硬编码卡牌字符串在请求体 |
| `src/frontend/tests/api.test.js` | 硬编码 `'attack'`, `'build'`, `'defense'` |
| `src/frontend/tests/App.test.jsx` | 硬编码 7 种卡牌全套字符串 |

---

## 4. 重构方案

### 4.1 目标

- **单一真相源**: 卡牌类型/效果类型仅在一处定义，其余所有地方引用
- **类型安全**: 编译时（Python type checker / IDE）即可发现拼写错误
- **前端同步**: 前端从后端 API 获取卡牌元数据，不再独立维护
- **易于扩展**: 新增卡牌类型仅需修改 1-2 个文件

---

### 4.2 后端改造

#### 步骤 1: 创建 `EffectType` 枚举（`src/backend/game/card.py`）

将 effect_type 从裸 `str` 提升为枚举类型：

```python
class EffectType(Enum):
    DAMAGE = "damage"
    GRANT_DEFENSE = "grant_defense"
    GRANT_GUARD = "grant_guard"
    REFLECT = "reflect"
    GAIN_ENERGY = "gain_energy"
    FLEX_DAMAGE = "flex_damage"
    BUILD = "build"
```

同步修改 `CardDef.effects` 类型注解和 `_CARD_DEFS` 字典。

#### 步骤 2: 重构 `Converter` — 单一映射表 (`converter.py`)

当前 6 个 dict 的核心数据是**完全同构**的，可以推导：

```python
class Converter:
    # 唯一的主映射表
    _CARD_INDEX_MAP: list[tuple[str, int, int]] = [
        # (卡牌字符串, play索引, build_cost)
        ("attack",  1, 1),
        ("defense", 3, 1),
        ("heavy",   5, 1),
        ("counter", 7, 2),
        ("energy",  9, 1),
        ("build",  None, 1),  # BUILD 不可直接 play，无 play 索引
        ("flex",   11, 2),
    ]

    # 以下全部从 _CARD_INDEX_MAP 计算得出（类初始化时）
    ALL_CARD_TYPES: frozenset[str]        # 自动推导
    _CARD_TYPE_ORDER: dict[str, int]      # 自动推导（按顺序 0-6）
    _PLAY_TO_INDEX: dict[str, int]        # 自动推导
    _BUILD_TO_INDEX: dict[str, int]       # 自动推导（偶数 0,2,4,6,8,10,12）
    _INDEX_TO_ACTION: dict[int, tuple]    # 自动推导（含 build 条目）
    _BUILD_COST: dict[str, int]           # 自动推导
```

**收益**: 新增卡牌只需往 `_CARD_INDEX_MAP` 添加一行，其余全部自动生成。

#### 步骤 3: 清理 `app_manager.py` 的 `_make_mask_fn()`

将 `pure_play_cards` 和 `build_targets` 的硬编码列表改为引用 `Converter` 的映射表：

```python
@staticmethod
def _make_mask_fn(hand, energy):
    legal_cards = set(hand)
    can_build = CardType.BUILD.value in legal_cards and energy > 0  # 改用枚举

    def mask_fn(logits):
        masked = logits.clone()
        # 从 Converter 的映射表推导，而非独立维护
        for idx, card_str in Converter.iter_play_indices():
            if card_str not in legal_cards:
                masked[0, idx] = float("-inf")
        for idx, card_str in Converter.iter_build_indices():
            if not can_build or Converter.build_cost(card_str) > energy:
                masked[0, idx] = float("-inf")
        return masked
    return mask_fn
```

#### 步骤 4: 清理 `resolver.py` 的 effect_type 判断

将字符串比较改为枚举：

```python
# 之前
if effect.effect_type == "grant_defense":
# 之后
if effect.effect_type == EffectType.GRANT_DEFENSE:
```

#### 步骤 5: 后端新增卡牌元数据 API

在 `controller/api.py` 新增 endpoint，便于前端获取：

```python
GET /api/cards → {
    "cards": [
        {"type": "attack", "display_name": "攻击", "icon": "⚔️",
         "build_cost": 1, "priority": 10, "effects": [...]},
        ...
    ]
}
```

---

### 4.3 前端改造

#### 步骤 1: 从 API 获取卡牌元数据

`GameBoard.jsx` 不再硬编码 `CARD_NAMES`、`CARD_ICONS`、`CARD_ORDER`，改为在 `App.jsx` 创建游戏时通过 API 获取：

```javascript
// 启动时加载一次
const cardMeta = await fetchCards();

// GameBoard 接收 cardMeta prop
// CARD_ORDER = cardMeta.map(c => c.type)
// CARD_NAMES = Object.fromEntries(cardMeta.map(c => [c.type, c.display_name]))
// CARD_ICONS  = Object.fromEntries(cardMeta.map(c => [c.type, c.icon]))
```

#### 步骤 2: 消除 App.jsx 中的 `'build'` 硬编码

将 `if (card === 'build')` 改为使用从 API 下载的配置进行判断——但 `'build'` 作为 BUILD 卡的特殊逻辑标识符，其语义是业务规则层面的，不需要完全消除字符串，改为引用 cardMeta 中的 `type` 字段即可。

---

### 4.4 测试文件改造

测试中的卡牌字符串字面量有两种处理方式：
1. **单元测试（converter, resolver）**: 使用 `CardType` 枚举值引用（`CardType.ATTACK.value`），确保编译时检查
2. **集成测试（API）**: 通过 `GET /api/cards` 动态获取有效类型列表，而不是硬编码

---

### 4.5 改造工作量估计

| 步骤 | 文件 | 改动量 | 风险 |
|------|------|--------|------|
| 4.2.1 EffectType 枚举 | `card.py`, `resolver.py` | ~15 行 | 低 |
| 4.2.2 Converter 重构 | `converter.py` | ~40 行 | 中（需跑全量测试） |
| 4.2.3 mask_fn 清理 | `app_manager.py` | ~15 行 | 中 |
| 4.2.4 resolver 清理 | `resolver.py` | ~10 行 | 低 |
| 4.2.5 卡牌元数据 API | `api.py`, `serializers.py` | ~25 行 | 低 |
| 4.3.1 前端去硬编码 | `GameBoard.jsx`, `App.jsx` | ~30 行 | 中 |
| 4.3.2 前端 `'build'` 消除 | `App.jsx` | ~5 行 | 低 |
| 4.4 测试改造 | 7+ 测试文件 | ~50 行 | 低 |
| **总计** | **~14 个文件** | **~190 行** | — |

---

## 5. 建议优先级

1. **高优先级**: EffectType 枚举（4.2.1）+ Converter 单一映射表（4.2.2）— 核心数据去重，立竿见影
2. **中优先级**: mask_fn 清理（4.2.3）+ resolver 清理（4.2.4）— 消除分支硬编码
3. **低优先级**: 卡牌元数据 API（4.2.5）+ 前端去硬编码（4.3）— 前后端解耦，长期收益
4. **随重构进行**: 测试文件改造（4.4）— 同步更新即可

建议按上述顺序分阶段实施，每阶段完成后运行全量测试确保无回归。
