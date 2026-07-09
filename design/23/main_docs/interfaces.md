# Interface Design — 前端维护卡牌图标 (intention #23)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `game/card_def.py` — CardDef 字段变更

### 1.1 CardDef（修改后）

```python
@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata (design/22, design/23).

    Per design/23: icon field removed — display logic moved to frontend.

    Attributes:
        card_type: The CardType enum member this definition belongs to.
        build_cost: Energy cost (1 or 2) to construct this card via BUILD.
        display_name: Human-readable Chinese name for UI display.
        resolvers: Resolver classes for this card, in registration order.
    """
    card_type: CardType
    build_cost: int
    display_name: str
    resolvers: tuple[type, ...]
```

对比 design/22：移除 `icon: str` 字段。

### 1.2 _CARD_DEFS 静态表（修改后）

```python
_CARD_DEFS: dict[CardType, CardDef] = {
    CardType.ATTACK: CardDef(
        card_type=CardType.ATTACK, build_cost=1,
        display_name="攻击",
        resolvers=(AttackDamageResolver,),
    ),
    CardType.DEFENSE: CardDef(
        card_type=CardType.DEFENSE, build_cost=1,
        display_name="防御",
        resolvers=(GrantDefenseResolver, GrantGuardResolver),
    ),
    CardType.HEAVY: CardDef(
        card_type=CardType.HEAVY, build_cost=1,
        display_name="重击",
        resolvers=(HeavyDamageResolver,),
    ),
    CardType.COUNTER: CardDef(
        card_type=CardType.COUNTER, build_cost=2,
        display_name="反击",
        resolvers=(GrantDefenseResolver, ReflectResolver),
    ),
    CardType.ENERGY: CardDef(
        card_type=CardType.ENERGY, build_cost=1,
        display_name="聚能",
        resolvers=(GainEnergyResolver,),
    ),
    CardType.BUILD: CardDef(
        card_type=CardType.BUILD, build_cost=1,
        display_name="构建",
        resolvers=(BuildResolver,),
    ),
    CardType.FLEX: CardDef(
        card_type=CardType.FLEX, build_cost=2,
        display_name="攻守兼备",
        resolvers=(GrantDefenseResolver, FlexDamageResolver),
    ),
}
```

每个条目移除 `icon="..."` 参数。

### 1.3 公共函数（不变）

```python
def get_card_def(card_type: CardType) -> CardDef:
    """Return the CardDef for a given CardType. 接口不变。"""

def get_initial_draw_pile() -> list[CardType]:
    """Return the standard 7-card initial draw pile. 接口不变。"""
```

---

## 2. `controller/api.py` — GET /api/cards 响应变更

### 2.1 响应格式（修改后）

```python
@app.route("/api/cards", methods=["GET"])
def list_cards():
    cards = []
    for ct in CardType:
        cd = get_card_def(ct)
        cards.append({
            "type": ct.value,
            "display_name": cd.display_name,
            "build_cost": cd.build_cost,
        })
    return jsonify({"cards": cards}), 200
```

响应示例：
```json
{
  "cards": [
    {"type": "attack", "display_name": "攻击", "build_cost": 1},
    {"type": "defense", "display_name": "防御", "build_cost": 1}
  ]
}
```

变更点：移除 `"icon": cd.icon` 行。`cards` 数组每个元素从 4 字段减为 3 字段。

---

## 3. `frontend/src/GameBoard.jsx` — 前端图标本地化

### 3.1 新增模块级常量

```js
/** Icon mapping per card type — maintained on frontend (design/23). */
const CARD_ICONS = {
  attack: '\u2694\ufe0f',   // ⚔️
  defense: '\U0001f6e1\ufe0f', // 🛡️
  heavy: '\U0001f4a5',      // 💥
  counter: '\u21a9\ufe0f',  // ↩️
  energy: '\u26a1',         // ⚡
  build: '\U0001f528',      // 🔨
  flex: '\u2696\ufe0f',     // ⚖️
};
```

### 3.2 buildLookups() 修改

```js
function buildLookups(cardMeta) {
  if (!cardMeta) return { names: {}, order: [], costs: {} };
  return {
    names: Object.fromEntries(cardMeta.map(c => [c.type, c.display_name])),
    order: cardMeta.map(c => c.type),
    costs: Object.fromEntries(cardMeta.map(c => [c.type, c.build_cost])),
  };
}
```

变更点：
- 移除 `icons: Object.fromEntries(cardMeta.map(c => [c.type, c.icon]))` 行
- `buildLookups` 返回对象不再包含 `icons` 字段

### 3.3 组件内使用

```js
const { names: CARD_NAMES, order: apiOrder, costs: CARD_COSTS } = buildLookups(cardMeta);
```

变更点：不再从 `buildLookups` 解构 `icons`。所有 `CARD_ICONS[card]` 引用直接使用模块级 `CARD_ICONS` 常量，无需修改。

---

## 4. 前端 `api.js` 接口文档变更

`fetchCards()` 返回值中 `cards` 数组每个元素从 `{type, display_name, icon, build_cost}` 变为 `{type, display_name, build_cost}`。移除 `icon` 字段说明。
