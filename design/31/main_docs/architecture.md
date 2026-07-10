# Architecture Design — rnn_v2 Skip-Connection 网络 (intention #31)

## 1. Motivation

当前 rnn_v1 的架构为 `Input(42) → GRU(32, 1 layer) → Linear(32→14/1)`。手牌信息仅占 42 维输入的 16.7%，经过 GRU 压缩到 32 维后信号被稀释，两个 head 均为单层 Linear，无法从压缩后的混合表示中有效"挖掘"手牌特征。实测表现为训练到 gen 8 棋力停滞，gen 7 频繁出 pass，权重分析显示两个网络均未充分关注手牌。

本次新增 rnn_v2 版本（独立模块，v1 保持不动）：在 GRU 输出之外增加一条 skip connection，将原始 42 维 input 直接拼入 head，让手牌、HP、能量等即时特征绕过 GRU 压缩瓶颈，同时 GRU 继续负责时序上下文（last action、status 变化）。Head 增加为三层 MLP 以增强从拼接特征中提取模式的能力。

## 2. Network Architecture

```
GRU: Input(42) → GRU(32, 1 layer)  (不变)

Policy head: Concat(GRU_out(32) + Input(42)) = 74
           → Linear(74→64) → ReLU
           → Linear(64→32) → ReLU
           → Linear(32→14) → softmax

Value head:  Concat(GRU_out(32) + Input(42)) = 74
           → Linear(74→64) → ReLU
           → Linear(64→32) → ReLU
           → Linear(32→1)
```

参数估算：
- GRU(42→32, 1 layer): ~7,200
- Policy head (74→64→32→14): ~6,800
- Value head (74→64→32→1): ~6,800
- 总计: ~21,000（v1 约 15,000）

## 3. Key Design Decisions

**D1 — Skip connection 绕过 GRU。** 即时特征（手牌、HP、能量、状态 lifetime、弃牌/抽牌分布）不需要历史记忆，直接拼入 head 避免 GRU 压缩导致的信号稀释。GRU 专注时序上下文（last action 带来的攻防节奏）。

**D2 — GRU 结构不变。** 仍为 Input(42)→GRU(32, 1 layer)。hidden_size 保持 32，因为游戏时序记忆需求有限（主要记录攻防状态切换），不需要更大容量。

**D3 — Head 三层 MLP。** 两层 ReLU 非线性变换，比 v1 单层 Linear 有更强的特征提取能力，能从 74 维拼接输入中学习复杂的特征组合模式。

**D4 — INPUT_SIZE 和 POLICY_OUTPUT_SIZE 不变。** 42 维输入、14 维输出沿用 v1 约定。Converter 无需修改。

**D5 — forward 签名兼容 v1。** `GRUPolicyModelV2.forward(x, hidden, mask_fn, temperature)` 和 `GRUValueModelV2.forward(x, hidden)` 签名与 v1 一致，调用方（AppManager、SelfPlayRunner、TrainingRunner）只需替换类名即可。

**D6 — 新增模块，v1 完全不受影响。** 新建 `src/backend/ai/rnn_v2/` 目录及 `model.py`、`__init__.py`。v1 的 `rnn_v1/model.py` 不做任何修改。registry 新增 rnn_v2 条目，rnn_v1 条目保留不动。两个版本可以共存，各自独立训练、独立推理。

**D7 — TrainingRunner 提取继承体系，子类归入各自模型包。** 原 `TrainingRunner` 重构为 `BaseTrainingRunner`（抽象父类，留在 `ai/training_runner.py`），承载共享逻辑。`V1TrainingRunner` 放在 `ai/rnn_v1/v1_training_runner.py`，`V2TrainingRunner` 放在 `ai/rnn_v2/v2_training_runner.py`，各自只覆写前向策略相关的三个方法。新增 `ai/training_factory.py` 中的 `TrainingRunnerFactory` 工厂类，根据 model name 返回对应子类实例。

## 4. Module Impact

```
src/backend/ai/
├── rnn_v1/
│   ├── model.py               # NO CHANGE
│   └── v1_training_runner.py   # NEW — V1TrainingRunner(BaseTrainingRunner)，GRU bypass 前向
├── rnn_v2/
│   ├── model.py               # NEW — GRUPolicyModelV2, GRUValueModelV2
│   ├── __init__.py            # NEW
│   └── v2_training_runner.py   # NEW — V2TrainingRunner(BaseTrainingRunner)，完整 forward() 前向
├── training_runner.py         # BaseTrainingRunner(abstract) — 共享逻辑：run()、_build_models()、
│                              #   _load_samples()、_compute_gae()、_save_weights()、epoch 批处理循环
├── training_factory.py        # NEW — TrainingRunnerFactory.create(model_name) → 返回 V1/V2 runner
├── registry.py                # ADD rnn_v2 条目（rnn_v1 条目不动）
├── training_config.py         # NO CHANGE
└── training_utils.py          # NO CHANGE

src/backend/app/
├── app_manager.py             # NO CHANGE — _load_model() 已是动态导入（importlib + registry），且 HIDDEN_SIZE/INPUT_SIZE/POLICY_OUTPUT_SIZE v1/v2 相同
├── converter.py               # NO CHANGE — INPUT_SIZE/POLICY_OUTPUT_SIZE 不变
└── selfplay_runner.py         # NO CHANGE

src/backend/scripts/
└── train.py                   # MODIFIED — TrainingRunner() 改为 TrainingRunnerFactory.create(config.model)
```

## 5. Diagrams

- **网络架构图**: [network_architecture.puml](diagrams/network_architecture.puml)
- **类图**: [class_diagram.puml](diagrams/class_diagram.puml)
