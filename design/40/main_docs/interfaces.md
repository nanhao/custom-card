# 设计 #40 — 接口设计

## 1. 模型常量 (`ai/atte_v1/constants.py`)

```python
# Embeddings
ACTION_VOCAB_SIZE = 25      # 24 action IDs + 1 PAD (for last_action=-1)
ACTION_EMBEDDING_DIM = 32   # e1
STATUS_VOCAB_SIZE = 3       # DEFENSE=0, GUARD=1, FRAGILE=2
STATUS_EMBEDDING_DIM = 8    # e2
ORDINAL_VOCAB_SIZE = 20     # 最大序数 19（min(ordinal, 19)），20 足以覆盖
ORDINAL_EMBEDDING_DIM = 2   # e3

# Dimensions
EMB2_DIM = 34
Q_DIM = 34
GRU_INPUT_SIZE = 80
HIDDEN_SIZE = 32
FEATURE_CONCAT_INPUT_SIZE = 169
OUTPUT_FEATURE_DIM = 64
POLICY_HIDDEN_DIM = 32
VALUE_HIDDEN_DIM = 32
POLICY_OUTPUT_SIZE = 24

# Normalization
MAX_HP = 30
MAX_ENERGY = 5

# Padding
PAD_ACTION_ID = 24
```

---

## 2. 数据类

### 2.1 AtteV1PlayerInput — 单玩家输入

```python
@dataclass
class AtteV1PlayerInput:
    """单个玩家在一个时间步的模型输入数据。

    包含己方手牌/弃牌堆/抽牌堆序列，以及对方手牌/弃牌堆/抽牌堆序列。
    """
    hand_ids: list[int]            # 手牌 action ID 序列（不定长）
    discard_ids: list[int]         # 弃牌堆 action ID 序列（不定长）
    draw_ids: list[int]            # 抽牌堆 action ID 序列（不定长）
    last_action: int               # 上回合 action ID，首回合为 -1
    hp: float                      # 归一化血量 [0, 1]
    energy: float                  # 归一化能量 [0, 1]
    status_ids: list[int]          # 状态 ID 序列（不定长）
    status_durations: list[int]    # 状态持续回合数（与 status_ids 等长）
    opp_hand_ids: list[int]        # 对方手牌 action ID 序列（手牌 cross-attention 依赖此数据）
    opp_discard_ids: list[int]     # 对方弃牌堆 action ID 序列
    opp_draw_ids: list[int]        # 对方抽牌堆 action ID 序列
```

### 2.2 AtteV1Input — 完整输入

```python
@dataclass
class AtteV1Input:
    """atte_v1 模型单步完整输入。

    注意：round_in_cycle 取值 -1/0/1，与 GameStateSnapshot.round_in_cycle(1-3) 语义不同。
    Converter 负责映射：1→-1, 2→0, 3→1。
    """
    own: AtteV1PlayerInput
    opp: AtteV1PlayerInput
    round_in_cycle: int            # -1(第1回合)/0(第2回合)/1(第3回合)
```

---

## 3. AtteV1Converter

```python
class AtteV1Converter:
    """atte_v1 专用转换器 — 处理不定长序列输入。"""

    @staticmethod
    def build_input(state: GameStateSnapshot, player_key: str) -> AtteV1Input:
        """从游戏状态构建 AtteV1Input。

        player_key: "1" 或 "2"（必传，无默认值）。

        数据处理规则：
        - HP/30、能量/5 归一化。
        - round_in_cycle 映射：GameStateSnapshot 的 1→-1, 2→0, 3→1。
        - 首回合 last_action = -1，映射为 PAD_ACTION_ID(24)。
        - BUILD 卡：在 hand_ids 中不直接映射；由调用方通过 action_to_policy_index
          选择具体 build/dismantle 目标。
        - 对方手牌/弃牌堆/抽牌堆均填入 opp_hand_ids / opp_discard_ids / opp_draw_ids。
        """

    @staticmethod
    def card_to_action_id(card: CardType) -> int:
        """将 CardType 映射为 play action ID（非 BUILD 卡）。
        BUILD 卡调用此方法触发 ValueError。
        """

    @staticmethod
    def status_to_id(status_type: StatusType) -> int:
        """DEFENSE → 0, GUARD → 1, FRAGILE → 2。"""

    @staticmethod
    def action_to_policy_index(
        card: CardType | None,
        build_target: CardType | None = None,
        dismantle_target: CardType | None = None,
    ) -> int:
        """游戏动作 → policy 输出索引 (24-d)。"""

    @staticmethod
    def policy_index_to_action(
        index: int,
    ) -> tuple[CardType | None, CardType | None, CardType | None]:
        """policy 输出索引 → (card, build_target, dismantle_target)。"""

    @staticmethod
    def build_cost(card_type: CardType) -> int:
        """返回卡牌的构建能量消耗。"""
```

---

## 4. AtteV1PolicyModel

```python
class AtteV1PolicyModel(nn.Module):
    """atte_v1 Policy 网络。

    Policy Head 末层 FC(32→24) 权重矩阵与 e1 嵌入矩阵 (25,32) 的前 24 行转置共享。
    e1 vocab=25（index 0-23 为 action，index 24 为 PAD），末层 FC 共享前 24 个 token 的嵌入。
    """

    INITIAL_HIDDEN_STATE: list[float]  # 32 维全零

    def __init__(self) -> None: ...

    def forward(
        self,
        own_hand_ids: torch.Tensor,          # (batch, seq_len_hand)
        own_discard_ids: torch.Tensor,        # (batch, seq_len_discard)
        own_draw_ids: torch.Tensor,           # (batch, seq_len_draw)
        own_last_action: torch.Tensor,        # (batch, 1) long
        own_status_ids: torch.Tensor,         # (batch, seq_len_status)
        own_status_durations: torch.Tensor,   # (batch, seq_len_status)
        own_opp_hand_ids: torch.Tensor,       # (batch, seq_len_opp_hand)
        own_opp_discard_ids: torch.Tensor,    # (batch, seq_len_opp_discard)
        own_opp_draw_ids: torch.Tensor,       # (batch, seq_len_opp_draw)
        opp_hand_ids: torch.Tensor,
        opp_discard_ids: torch.Tensor,
        opp_draw_ids: torch.Tensor,
        opp_last_action: torch.Tensor,
        opp_status_ids: torch.Tensor,
        opp_status_durations: torch.Tensor,
        opp_opp_hand_ids: torch.Tensor,
        opp_opp_discard_ids: torch.Tensor,
        opp_opp_draw_ids: torch.Tensor,
        own_hp: torch.Tensor,                 # (batch, 1)
        own_energy: torch.Tensor,
        opp_hp: torch.Tensor,
        opp_energy: torch.Tensor,
        round_in_cycle: torch.Tensor,
        hidden: torch.Tensor,                 # (1, batch, 32)
        mask_fn: Callable | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (probs, log_probs, new_hidden).

        mask_fn: Optional callable, signature mask_fn(logits: Tensor(1,24)) -> Tensor(1,24)。
        由 AtteV1Adapter 在 ai_select_action / ai_sample_action 中基于手牌、能量、
        弃牌堆状态构造并传入。mask_fn 将非法动作的 logit 设为 -inf，确保 masked softmax
        后对应概率为 0。
        """
```

---

## 5. AtteV1ValueModel

```python
class AtteV1ValueModel(nn.Module):
    """atte_v1 Value 网络。主体同 Policy，输出头 FC(64→32)→ReLU→FC(32→1)，无 tanh。"""

    INITIAL_HIDDEN_STATE: list[float]

    def __init__(self) -> None: ...

    def forward(
        self,
        # 输入参数与 AtteV1PolicyModel.forward 完全一致（除 mask_fn, temperature）
        ...
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (value, new_hidden)."""
```

---

## 6. AtteV1Adapter（严格实现 AiInterface）

```python
class AtteV1Adapter(AiInterface):
    """atte_v1 AI 适配器，实现 AiInterface。

    Hidden state 由调用方传入/接收，adapter 不持有可变状态（线程安全）。
    模型懒加载：首次 ai_select_action / ai_sample_action 时才加载权重文件，
    首回合有 ~1-3s 冷启动延迟（取决于权重文件大小和硬件）。
    """

    def __init__(
        self,
        policy_module: str,
        policy_class: str,
        policy_e1e2_weight_path: str,   # e1/e2 嵌入权重路径（构造时校验路径在合法模型目录内）
        policy_weight_path: str,        # Policy 主体权重路径
        value_module: str,
        value_class: str,
        value_e1e2_weight_path: str,
        value_weight_path: str,
    ) -> None:
        """构造适配器，不加载模型（懒加载）。

        路径安全：构造时使用 pathlib 规范化路径，验证前缀位于
        {PROJECT_ROOT}/src/backend/ai/atte_v1/ 目录下。
        """

    # -- AiInterface 实现 --

    def ai_select_action(
        self, state: GameStateSnapshot, hidden_state: list[float],
    ) -> tuple[CardType | None, CardType | None, CardType | None, list[float]]:
        """贪心选择动作。首次调用时加载 policy model（冷启动延迟 ~1-2s）。
        mask_fn 由本方法内部构造：从 state 提取 hand/energy/discard_pile，
        调用 self._make_mask_fn() 生成闭包后传入 model.forward()。
        """

    def ai_sample_action(
        self,
        state: GameStateSnapshot,
        hidden_state_policy: list[float],
        hidden_state_value: list[float],
        temperature: float,
        player_key: str = "2",
    ) -> tuple[
        CardType | None, CardType | None, CardType | None,
        list[float], float | None, float, list[float],
    ]:
        """采样动作。首次调用时加载 policy + value model（冷启动延迟 ~1-3s）。
        mask_fn 由本方法内部构造。
        log_prob 在 temperature=0 时为 None。
        """
```

---

## 7. Registry 注册

### 7.1 AI 层 registry (`ai/registry.py`) — 新增条目

```python
{
    "name": "atte_v1",
    "description": "attention-based + GRU",
    "policy": {
        "module": "src.backend.ai.atte_v1.model",
        "class": "AtteV1PolicyModel",
        "weight_path": [
            "src/backend/ai/atte_v1/gen{N}/policy_e1_e2.pth",
            "src/backend/ai/atte_v1/gen{N}/policy.pth",
        ],
    },
    "value": {
        "module": "src.backend.ai.atte_v1.model",
        "class": "AtteV1ValueModel",
        "weight_path": [
            "src/backend/ai/atte_v1/gen{N}/value_e1_e2.pth",
            "src/backend/ai/atte_v1/gen{N}/value.pth",
        ],
    },
}
```

### 7.2 AI Adapter 层 registry (`ai_adapter/ai_registry.py`) — 工厂修改

`AiRegistry.get_ai_interface()` 按 model name 前缀路由：
- `name == "atte_v1"` → 构造 `AtteV1Adapter(8 params)`
- 其余 → 构造 `RnnAiAdapter(6 params)`

---

## 8. AtteV1TrainingRunner

```python
class AtteV1TrainingRunner(BaseTrainingRunner):
    """atte_v1 三阶段 PPO 训练器。

    继承 BaseTrainingRunner.run() 的三阶段循环。
    Phase 1: _process_value_perspective — 按步处理 card 序列 → embedding → attention
             → GRU → value MSE loss，仅更新 value_model。
    Phase 2: _recompute_advantages — 用更新后的 value_model 重新前向 → GAE，
             写回 perspective["new_advantages"] / perspective["new_value_targets"]。
    Phase 3: _process_policy_perspective — 按步处理 → PPO clip loss + entropy，
             仅更新 policy_model。
    """

    def _load_samples(self, sample_path: str) -> list[dict]:
        """覆写 — 从 JSON 重建 AtteV1Input 格式的样本。"""

    def _process_value_perspective(
        self, perspective: dict, value_model: nn.Module,
        config: TrainingConfig, device: torch.device,
    ) -> torch.Tensor | None: ...

    def _recompute_advantages(
        self, perspectives: list[dict], value_model: nn.Module,
        config: TrainingConfig, device: torch.device,
    ) -> None: ...

    def _process_policy_perspective(
        self, perspective: dict, policy_model: nn.Module,
        config: TrainingConfig, device: torch.device,
    ) -> tuple | None: ...
```

---

## 9. TrainingRunnerFactory 修改

```python
class TrainingRunnerFactory:
    @staticmethod
    def create(model_name: str) -> BaseTrainingRunner:
        name, _gen = parse_ai_name(model_name)
        if name == "rnn_v2":
            return V2TrainingRunner()
        if name == "atte_v1":
            return AtteV1TrainingRunner()
        raise ValueError(f"Unknown model: {name!r}")
```

---

## 10. 样本格式

```python
@dataclass
class AtteV1SampleStep:
    input: AtteV1Input       # 该步的模型输入（不定长序列）
    action: int              # policy index (0-23)
    log_prob: float
    reward: float
    value: float

@dataclass
class AtteV1Perspective:
    game_id: str
    player_key: str
    model_name: str
    steps: list[AtteV1SampleStep]
```

**存储开销对比**（单局 ~30 步估算）：
- rnn_v2：62 × 30 × 4 bytes ≈ **7.5 KB**
- atte_v1：每个 step 包含 6 个不定长 list（平均 5-8 个 int）+ 标量，JSON 序列化后 ≈ **25-40 KB**（约 4-5×）

---

## 11. 性能评估

- atte_v1 单次 forward ~25K FLOPs vs rnn_v2 ~5K FLOPs（5×）
- 自对弈吞吐量预计从 ~200 samples/sec 降至 **~40-50 samples/sec**（未优化）
- `torch.compile` 预期可恢复至 ~60-80 samples/sec
- 批量推理（batch=4）可进一步将吞吐量提升 ~2×，但需自对弈引擎支持并行对局

---

## 12. 关键维度速查

| 符号 | 维度 | 说明 |
|------|------|------|
| e1 | vocab=25, dim=32 | 动作嵌入（含 PAD index 24） |
| e2 | vocab=3, dim=8 | 状态嵌入 |
| e3 | vocab=20, dim=2 | 序数嵌入 |
| emb1 | 32 | action → e1 |
| emb2 | 34 | emb1 + ordinal_emb |
| Q1/Q2/Q3 | 34 | 注意力查询向量 |
| Z3 | 34 | 卡组优势特征向量 |
| SS1/SS2 | 8 | 状态特征向量 |
| h_out | 32 | GRU 输出 |
| h_out' | 34 | h_out → Linear(32→34) + LN |
| HAZ | 34 | 手牌优势特征向量 |
| out | 64 | 拼接 → FC+ReLU 输出 |
| Policy Head | 64→32→24 | 末层与 e1 共享权重 |
| Value Head | 64→32→1 | 无 tanh |
| GRU input | 80 | last_act_emb×2 + SS1 + SS2 |
| Feature concat input | 169 | HP×2 + energy×2 + round + last_act_emb×2 + HAZ + Z3 + h_out |
