# 设计 #37 — 接口设计

## 1. CardType 枚举

```python
class CardType(Enum):
    ATTACK = "attack"         # 连续击
    DEFENSE = "defense"       # 格挡
    HEAVY = "heavy"           # 破防重击
    COUNTER = "counter"       # 防守反击
    ENERGY = "energy"         # 聚能
    BUILD = "build"           # 构建/拆解
    FLEX = "flex"             # 攻守兼备
    IRON_DEFENSE = "iron_defense"  # 严防死守 ← 新增
```

## 2. CardDef 修改

`_CARD_DEFS` 中 BUILD 条目更新，新增 IRON_DEFENSE 条目：

| CardType | build_cost | display_name | description | tags | resolvers |
|----------|-----------|-------------|-------------|------|-----------|
| BUILD | 1 | 构建/拆解 | (上半部分)构建一张新的卡牌；(下半部分)从弃牌堆中移除一张卡牌 | () | `(BuildResolver, DismantleResolver)` |
| IRON_DEFENSE | 2 | 严防死守 | 调动全身力量进行防守 | ("防",) | `(StrongDefenseResolver, GrantGuardResolver)` |

## 3. Player 接口变更

### 3.1 PlayerHistoryItem — 新增字段

```python
@dataclass
class PlayerHistoryItem:
    card: CardType | None
    build_target: CardType | None = None
    dismantle_target: CardType | None = None  # ← 新增
```

### 3.2 Player — 新增方法

```python
def remove_from_discard(self, card: CardType) -> None:
    """从弃牌堆中移除一张指定卡牌（拆解效果）。

    Raises:
        ValueError: 若弃牌堆中无此卡牌。
    """
```

## 4. ResolveContext — 新增字段

```python
@dataclass
class ResolveContext:
    card_owner: Player
    opponent: Player
    card_played_by_owner: CardType
    card_played_by_opponent: CardType | None
    build_target: CardType | None = None
    build_target_cost: int = 0
    dismantle_target: CardType | None = None  # ← 新增
```

## 5. 新增 Resolver

### 5.1 DismantleResolver (P10)

```python
class DismantleResolver(Resolver):
    """从 card_owner 弃牌堆中移除 dismantle_target，不消耗能量。"""

    def get_priority(self) -> int: ...

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        # 若 ctx.dismantle_target 为 None，直接返回
        # 调用 ctx.card_owner.remove_from_discard(ctx.dismantle_target)
```

### 5.2 StrongDefenseResolver (P11)

```python
class StrongDefenseResolver(Resolver):
    """授予 card_owner DEFENSE 状态，持续 2 回合。受 FRAGILE 阻挡。"""

    def get_priority(self) -> int: ...

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        # 若 card_owner 有 FRAGILE，跳过
        # ctx.card_owner.grant_status(StatusType.DEFENSE, lifetime=2)
```

## 6. BuildResolver — 修改

**唯一变化**：构建的卡牌放入抽牌堆而非弃牌堆。

```python
class BuildResolver(Resolver):
    """执行构建：扣能量，将 build_target 放入 card_owner 的 draw_pile。"""

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.build_target is None:
            return
        cost = ctx.build_target_cost
        ctx.card_owner.energy = max(0, ctx.card_owner.energy - cost)
        ctx.card_owner.draw_pile.append(ctx.build_target)  # ← 改为 draw_pile
        details.add(f"BUILD {ctx.build_target.value} → draw pile (cost {cost})")
```

## 7. GameCore.play_round — 接口变更

```python
def play_round(
    self,
    cards_1: list[CardType],
    cards_2: list[CardType],
    build_target_1: CardType | None = None,
    build_target_2: CardType | None = None,
    dismantle_target_1: CardType | None = None,  # ← 新增
    dismantle_target_2: CardType | None = None,  # ← 新增
) -> RoundResult:
```

**BUILD 验证规则**（替代原逻辑）：
- 若玩家打出 BUILD：`build_target` 和 `dismantle_target` **互斥**，必须有且仅有一个非 None
- 构建模式：`build_target` 非 None → 能量 > 0 且 `get_card_def(build_target).build_cost <= energy`
- 拆解模式：`dismantle_target` 非 None → 弃牌堆中存在该卡

**Resolver 任务构造**：`ResolveContext` 传递 `build_target` 和 `dismantle_target`。

## 8. AI 常量

```python
INPUT_SIZE = 62         # 56 → 62
HIDDEN_SIZE = 32        # 不变
NUM_CARD_TYPES = 8      # 7 → 8
POLICY_OUTPUT_SIZE = 24 # 14 → 24
```

## 9. Converter — 接口变更

### 9.1 _CARD_MAP 重构为四元组

```python
_CARD_MAP: list[tuple[CardType, int | None, int, int]] = [
    # (card_type, play_index, build_index, dismantle_index)
    (CardType.BUILD,         None,  1,  2),
    (CardType.ATTACK,           3,  4,  5),
    (CardType.DEFENSE,          6,  7,  8),
    (CardType.HEAVY,            9, 10, 11),
    (CardType.COUNTER,         12, 13, 14),
    (CardType.ENERGY,          15, 16, 17),
    (CardType.FLEX,            18, 19, 20),
    (CardType.IRON_DEFENSE,    21, 22, 23),
]
```

### 9.2 派生映射表

从 `_CARD_MAP` 自动推导：
- `_PLAY_TO_INDEX`: card_type → play_index（BUILD 无 play）
- `_BUILD_TO_INDEX`: card_type → build_index
- `_DISMANTLE_TO_INDEX`: card_type → dismantle_index（新增）
- `_INDEX_TO_ACTION`: index → (card, build_target, dismantle_target)（新增三元组）

### 9.3 _INDEX_TO_ACTION — 返回三元组

```python
_INDEX_TO_ACTION: dict[int, tuple[CardType|None, CardType|None, CardType|None]] = {
    0: (None, None, None),  # pass
    # build 模式：
    1: (CardType.BUILD, CardType.BUILD, None),           # build→构建/拆解
    4: (CardType.BUILD, CardType.ATTACK, None),           # build→连续击
    ...
    # dismantle 模式：
    2: (CardType.BUILD, None, CardType.BUILD),            # dismantle→构建/拆解
    5: (CardType.BUILD, None, CardType.ATTACK),           # dismantle→连续击
    ...
    # 纯出牌（play）：
    3: (CardType.ATTACK, None, None),                     # 出连续击
    6: (CardType.DEFENSE, None, None),                    # 出格挡
    ...
}
```

### 9.4 action_to_policy_index — 新增 dismantle_target 参数

```python
@staticmethod
def action_to_policy_index(
    card: CardType | None,
    build_target: CardType | None = None,
    dismantle_target: CardType | None = None,  # ← 新增
) -> int:
```

### 9.5 build_input_vector — 62-d

每个卡牌统计段从 7 维扩展到 8 维（按 `_CARD_TYPE_ORDER` 顺序，末尾加入 IRON_DEFENSE）。

### 9.6 NUM_CARD_TYPES

```python
NUM_CARD_TYPES: int = 8
```

## 10. AppManager._make_mask_fn — 24-d

```python
@staticmethod
def _make_mask_fn(hand, energy, discard_pile):  # ← discard_pile 新增参数
    """24-d masking:
    - Index 0 (pass): 始终合法
    - Play indices (3,6,9,12,15,18,21): 手牌中无对应卡 → mask
    - Build indices (1,4,7,10,13,16,19,22): BUILD 不在手 或 能量=0 或 能量不足 → mask
    - Dismantle indices (2,5,8,11,14,17,20,23): BUILD 不在手 或 弃牌堆无对应卡 → mask
    """
```

## 11. 前端接口变更

### 11.1 GameBoard — BUILD 卡选择弹窗

玩家选中构建/拆解卡后，弹出模态框：
- 上半部分按钮："构建"（能量不足时灰显）
- 下半部分按钮："拆解"（弃牌堆为空时灰显）
- 取消按钮

选"构建"后 → 进入目标卡牌选择（与现有逻辑相同）。
选"拆解"后 → 进入弃牌堆中选择要移除的卡牌。

### 11.2 新卡显示

- 新增 严防死守 卡牌（build_cost=2, tags="防"）的图标和名称
- API 响应中 `iron_defense` 卡牌类型的前端展示
