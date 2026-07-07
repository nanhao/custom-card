# Interface Design — Game v2 (intention #11)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `game/card.py` — CardType + CardDef

```python
class CardType(Enum):
    ATTACK = "attack"
    DEFENSE = "defense"
    HEAVY = "heavy"
    COUNTER = "counter"
    ENERGY = "energy"
    BUILD = "build"
    FLEX = "flex"

@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata."""
    card_type: CardType
    priority: int | None      # 主优先级(None表示多优先级卡), None时使用effects
    effects: tuple[tuple[str, int], ...]  # (effect_type, priority) 如 FLEX→(("grant_defense",11),("damage",10))
    build_cost: int           # 1 or 2 (energy cost to build via 构建卡)
    display_name: str         # e.g. "攻击"
    icon: str                 # e.g. "⚔️"

def get_card_def(card_type: CardType) -> CardDef:
    """Return the CardDef for a given CardType."""

def get_initial_draw_pile() -> list[CardType]:
    """Return [ATK,ATK,DEF,HEAVY,COUNTER,ENERGY,BUILD] (fresh copy)."""
```

---

## 2. `game/player.py` — Player

```python
@dataclass
class Player:
    hp: int = 20
    energy: int = 0
    statuses: list[str] = []            # "defense" | "联防"
    hand: list[CardType] = []            # 初始空，每轮抽7张
    draw_pile: list[CardType] = []       # 抽牌堆
    discard_pile: list[CardType] = []    # 棄牌堆
    history: list[CardType] = []         # 历史出牌(跨轮保留，用于连续攻击计数)

    def play_card(self, card: CardType) -> None:
        """从hand中移除一张card。ValueError if not in hand."""

    def take_damage(self, amount: int) -> None:
        """扣血，clamped to ≥0."""

    def has_card(self, card: CardType) -> bool:
        """判断hand中是否有card."""

    def add_energy(self, amount: int) -> None:
        """增加能量, clamped to [0,5]."""

    def has_status(self, status: str) -> bool:
        """判断是否拥有指定状态."""
```

---

## 3. `game/resolver.py` — CardResolver

```python
@dataclass
class Effect:
    """单个卡牌效果单元, 由CardResolver内部使用."""
    source_player: str    # "1" | "2"
    priority: int         # 10 | 11
    source_card: CardType
    effect_type: str      # "damage" | "grant_defense" | "grant_guard" | "reflect" | "gain_energy" | "build" | "flex_damage"

@dataclass
class ResolutionResult:
    damage_1: int
    damage_2: int
    energy_delta_1: int    # 本回合P1能量变化
    energy_delta_2: int    # 本回合P2能量变化
    statuses_granted_1: list[str]   # 本回合P1获得的状态
    statuses_granted_2: list[str]   # 本回合P2获得的状态
    details: list[str]

class CardResolver:
    """Stateless priority-layered card resolution engine."""

    @staticmethod
    def resolve(
        cards_1: list[CardType],
        cards_2: list[CardType],
        player_1_statuses: list[str],    # 回合开始前的状态
        player_2_statuses: list[str],
        p1_attack_streak: int,           # 本回合前的连续攻击计数
        p2_attack_streak: int,
    ) -> ResolutionResult:
        """Compute all effects for one turn.
        
        内部流程:
        1. 将两张卡拆分Effect列表 (BUILD→"build" effect, FLEX→["grant_defense"(11),"flex_damage"(10)])
        2. 按priority分组(11,10)
        3. 逐组结算: 状态授予即时生效, 伤害累加
        4. 汇总返回 (BUILD 对所有字段贡献为0)
        """

    @staticmethod
    def get_consecutive_attack(history: list[CardType]) -> int:
        """统计history末尾连续attack数量。外部调用(跨轮保留)."""
```

---

## 4. `game/game_core.py` — GameCore

```python
@dataclass
class RoundResult:
    round: int            # 回合号 (全局递增)
    cycle: int            # 轮号
    round_in_cycle: int   # 轮内第几回合 (1-5)
    cards_1: list[CardType]
    cards_2: list[CardType]
    damage_1: int
    damage_2: int
    hp_1_after: int
    hp_2_after: int
    energy_1_after: int
    energy_2_after: int
    statuses_1: list[str]
    statuses_2: list[str]
    details: list[str]
    is_game_over: bool
    winner: str | None

class GameCore:
    ROUNDS_PER_CYCLE = 5

    def __init__(self, mode: str = "pvp") -> None: ...

    def start_game(self) -> None:
        """初始化双方Player(HP=20, energy=0, 抽牌堆初始7张), 调用_start_cycle()."""

    def play_round(self, cards_1: list[CardType], cards_2: list[CardType],
                    build_target_1: CardType | None = None,
                    build_target_2: CardType | None = None) -> RoundResult:
        """执行一回合, 结算后弃牌→检查是否满5回合→cycle末扣血→下一cycle.
        build_target仅在对应方打出BUILD卡时有效."""

    def get_state(self, player_perspective: str | None = None) -> dict:
        """返回完整游戏状态的dict快照, 供API层使用.
        
        返回键: round, cycle, round_in_cycle,
        players{1,2}{hp,energy,statuses,hand,draw_pile,discard_pile,history},
        is_over, winner, mode, round_log.
        
        player_perspective="1"时, P1的draw_pile/discard_pile暴露完整内容(玩家可查看己方牌堆),
        P2的仅暴露count. 反之亦然. None时双方均仅暴露count.
        """
```

---

## 5. `ai/rnn_v1/model.py` — GRUPolicyModel / GRUValueModel

```python
INPUT_SIZE = 25
HIDDEN_SIZE = 32
NUM_CARD_TYPES = 7  # 前7维选卡 + 后7维选构建目标

class GRUPolicyModel(nn.Module):
    """Input(25) → GRU(32,1层) → Linear(32→14) → softmax.

    输出14维: 前7维是出牌动作, 后7维是构建目标(仅出构建卡时有效)。
    """
    INITIAL_HIDDEN_STATE: list[float]  # [0.0]*32

    def forward(
        self,
        x: torch.Tensor,           # (1,1,25)
        hidden: torch.Tensor,      # (1,1,32)
        mask_fn: Callable | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """→ (probs(1,14), log_probs(1,14), new_hidden(1,1,32)).

        mask_fn 负责两段屏蔽:
        - indices 0-6: 屏蔽手牌中没有的卡
        - indices 7-13: 屏蔽能量不足以构建的卡
        训练时 mask_fn=None (不屏蔽后7维)。
        """

class GRUValueModel(nn.Module):
    """Input(25) → GRU(32,1层) → Linear(32→1)."""
    INITIAL_HIDDEN_STATE: list[float]  # [0.0]*32

    def forward(
        self,
        x: torch.Tensor,       # (1,1,25)
        hidden: torch.Tensor,  # (1,1,32)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """→ (value(1,1), new_hidden(1,1,32))."""
```

---

## 6. `app/converter.py` — Converter

```python
class Converter:
    NUM_CARD_TYPES: int = 7
    INPUT_SIZE: int = 25
    ALL_CARD_TYPES: frozenset[str] = frozenset({
        "attack","defense","heavy","counter","energy","build","flex"
    })

    @staticmethod
    def card_to_index(card_str: str) -> int:
        """attack→0 defense→1 heavy→2 counter→3 energy→4 build→5 flex→6.
        -1 if unknown. 此映射供前7维(选卡)和后7维(构建目标)共同使用。"""

    @staticmethod
    def index_to_card(index: int) -> str:
        """0→attack ... 6→flex. IndexError if out of range [0,6]."""

    @staticmethod
    def build_input_vector(state: dict, player_key: str = "2") -> list[float]:
        """Build 25-d input vector.
        
        索引0-6:   手牌中7类卡牌各自数量
        索引7-8:   己方HP, 对方HP (未clamp, 直接传入)
        索引9-15:  己方上回合出牌 one-hot
        索引16-22: 对方上回合出牌 one-hot
        索引23:    当前轮内回合序号(1-5)
        索引24:    当前 cycle 编号
        """
```

---

## 7. `app/app_manager.py` — AppManager (public methods)

```python
class AppManager:
    """Bridge layer — game creation + AI action selection."""

    def create_game(self, mode: str, ai_name: str | None = None) -> tuple[str, dict]:
        """创建游戏, 返回(game_id, state). PVE时初始化 hidden_state [0.0]*32."""

    def ai_select_action(self, state: dict, hidden_state: list[float], ai_name: str) -> tuple[str, str | None, list[float]]:
        """Greedy AI选牌. 返回 (card_str, build_target_str_or_None, new_hidden_state).
        build_target 仅在 card_str="build" 时非None.
        hidden_state 长度32."""

    def ai_sample_action(
        self, state: dict,
        hidden_state_policy: list[float], hidden_state_value: list[float],
        ai_name: str, temperature: float, player_key: str = "2",
    ) -> tuple[str, str | None, list[float], float | None, float, list[float]]:
        """采样AI选牌. 返回 (card_str, build_target_str_or_None, new_h_pol, log_prob, value, new_h_val).
        build_target 仅在 card_str="build" 时非None.
        log_prob 需在温度>0时计算; 选构建目标时需额外记录后7维的log_prob."""

    def run_selfplay(self, config: SelfPlayConfig) -> SelfPlayResult:
        """委托SelfPlayRunner执行."""
```

---

## 8. `ai/registry.py` — 新增条目

```python
# 新增 rnn_v2.1 — 适配 v2 游戏机制的随机权重模型
{
    "name": "rnn_v2.1",
    "description": "v2 game, random weights (gen1)",
    "policy": {
        "weight_path": "src/backend/ai/rnn_v1/gen5/policy.pth",
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUPolicyModel",
    },
    "value": {
        "weight_path": "src/backend/ai/rnn_v1/gen5/value.pth",
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUValueModel",
    },
}
```

---

## 9. `ai/training_runner.py` — TrainingRunner (public)

```python
class TrainingRunner:
    """三阶段PPO训练, 适配v2模型维度."""

    def run(self, config: TrainingConfig) -> None:
        """执行训练. 内部使用 INPUT_SIZE=25, HIDDEN_SIZE=32, POLICY_OUTPUT=14.

        ACTION_TO_INDEX 保持7值(前7维). 训练时mask_fn=None.
        _load_samples 读取 step["input"](25-d) + step["action"] + step["build_target"](可选).
        policy model forward 输出14维 log_probs, 选 action 的 log_prob 取自前7维对应索引.
        若 action="build", 同时计算后7维中 build_target 索引的 log_prob，合并作为总 log_prob.
        """
```

---

## 10. `game/session_store.py` — 序列化格式变更

```python
class SessionStore:
    """save/load/delete 不变, 内部序列化适配新字段.
    
    round_log(历史回合记录列表) 与 GameCore 整体一起持久化,
    无需单独处理。
    """

    # Player 序列化新增键:
    #   "energy": int, "statuses": [str],
    #   "draw_pile": [str], "discard_pile": [str]
    #
    # GameCore 序列化新增键:
    #   "round": int (回合号), "cycle": int (轮号), "round_in_cycle": int
    #
    # ai_state / ai_value_state 长度从 8 → 32
```

---

## 11. `controller/serializers.py` — RoundResult 序列化

```python
def round_result_to_dict(rr: RoundResult) -> dict:
    """序列化新增键: cycle, round_in_cycle, energy_1_after, energy_2_after, statuses_1, statuses_2."""
```

---

## 12. `controller/api.py` — API 层变更

| 端点 | 变更 |
|------|------|
| POST /api/games | create_game 内 ai_state 初始化 [0.0]*32 |
| POST /api/games/{id}/play | PVP/PVE: request body 新增可选字段 `build_target: str`(仅当 `card` 为 "build" 时)。PVE 模式 ai_state 长度32。`cards_*` 字段保持向后兼容 (list[str]) |
| GET /api/games/{id} | state 返回新增 round/cycle/round_in_cycle/energy/statuses |
