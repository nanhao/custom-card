# 设计 #38 — 接口定义

## 1. 常量模块 `constants.py`

### 变更：`INITIAL_HP`

```
INITIAL_HP: int = 30  # 原值 20
```

### 新增：`INITIAL_ENERGY`

```
INITIAL_ENERGY: int = 2
```

## 2. 卡牌定义模块 `card_def.py`

### 变更：`get_initial_draw_pile()`

```
def get_initial_draw_pile() -> list[CardType]:
    """返回 7 张初始抽牌堆（全新副本）。

    组成：3×ATTACK, 2×DEFENSE, 1×ENERGY, 1×BUILD。
    HEAVY 和 COUNTER 不再出现在初始牌堆中。
    """
```

## 3. 玩家模块 `player.py`

### 变更：`Player` dataclass 默认值

```
@dataclass
class Player:
    hp: int = 30       # 原默认 20
    energy: int = 2    # 原默认 0
    ...
```

## 4. 游戏核心 `game_core.py`

### 变更：`GameCore.start_game()`

`start_game()` 内部逻辑变更：
- 创建 Player 时使用 `INITIAL_HP` 常量替代硬编码的 `hp=20`
- 创建 Player 时使用 `INITIAL_ENERGY` 常量替代硬编码的 `energy=0`
- 其余逻辑不变（抽牌堆初始化、洗牌、_start_cycle）

## 5. 前端 `GameBoard.jsx`

### 变更：状态标签渲染

`renderStatusBar()` 中的 `statusTags` 渲染逻辑变更：
- 当前：`<span>{s.type}</span>` — 仅显示状态类型名
- 目标：`<span>{s.type} ({s.lifetime})</span>` — 同时显示类型和剩余回合数

## 6. 无需变更的模块

| 模块 | 原因 |
|------|------|
| `controller/api.py` | `GET /api/games/{id}` 响应中的 `statuses` 已包含 `lifetime` 字段（来自 `PlayerStatus.to_dict()`） |
| `controller/serializers.py` | 无新增序列化需求 |
| `app/converter.py` | 输入向量维度不变（8 种卡牌 × 7 段位），仅初始分布改变 |
| `app/app_manager.py` | 不涉及游戏创建逻辑 |
| `ai/` | 模型输入层不变（metadata 中的 HP/energy 是运行时值，非编译时常量） |
| `session_store.py` | JSON 序列化格式不变 |
