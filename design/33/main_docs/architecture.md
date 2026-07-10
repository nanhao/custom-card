# Architecture Design — Value Network Validation Set

## 1. Motivation

当前 PPO 训练的 Phase 1（value 训练）仅使用训练集 loss 监控训练进度，缺少验证集评估，无法判断 value 网络是否过拟合训练样本。本设计引入 game-level validation split 机制，在训练开始前按游戏划分固定 train/val 集合，每个 value epoch 后输出验证集 loss，辅助判断训练收敛状态和早停时机。

## 2. Scope

- **仅影响 Phase 1（value 训练）**，Phase 2（advantage 重算）和 Phase 3（policy 训练）不改动
- 验证集只用于评估 loss，不参与梯度更新

## 3. Key Design Decisions

**D1 — 按游戏划分 train/val。** 将 perspectives 按 `game_id` 分组后随机划分 game 至 train/val。同一局 game 的所有 perspective 归属于同一集合，避免训练集/验证集之间因共享对局过程导致的数据泄露。

**D2 — 固定划分。** 训练开始时用固定 seed 随机划分一次，整个训练过程（所有 value epoch）使用同一 train/val split，保证验证 loss 曲线跨 epoch 可比。

**D3 — validation_split 可配置。** `TrainingConfig` 新增 `validation_split` 字段（默认 0.1），用户可通过 `backend.json` 或 CLI `--validation-split` 覆盖。

**D4 — validation_split=0 时跳过验证。** 允许用户关闭验证集功能，训练行为与当前完全一致。

**D5 — perspective 携带 game_id。** `SampleLoader` 在提取 perspective 时附加 `game_id` 字段，使 runner 可按游戏分组进行划分。

**D6 — 验证时模型处于 eval 模式。** 计算 validation loss 前 `model.eval()`，计算后恢复 `model.train()`，避免影响 GRU 内部状态（当前 GRU 无 dropout/batchnorm，但遵循标准实践）。

## 4. Training Flow

参见 [training_flow.puml](diagrams/training_flow.puml)

## 5. Class Relationships

参见 [class_diagram.puml](diagrams/class_diagram.puml)

## 6. Changed Files

| File | Change |
|------|--------|
| `ai/sample_loader.py` | MODIFIED — 透视字典附加 `game_id` |
| `ai/training_config.py` | MODIFIED — 新增 `validation_split` 字段 |
| `ai/training_runner.py` | MODIFIED — `run()` 增加划分逻辑；`_train_value_epoch()` 增加验证 |
| `ai/training_utils.py` | MODIFIED — CSV/chart 增加 `val_value_loss` 列/曲线 |
| `config/backend.json` | MODIFIED — training 段增加 `validation_split: 0.1` |
| `scripts/train.py` | MODIFIED — 增加 `--validation-split` 参数 |
