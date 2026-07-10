# Test Cases — Value Network Validation Set

所有测试用例应被新增或扩展到现有测试文件中。

---

## TC1: `_split_train_val` — 正常划分

- **给定**: 6 个 game，每个 game 2 个 perspective（共 12 个 perspective），`validation_split=0.5`
- **当**: 调用 `_split_train_val(perspectives, 0.5)`
- **则**:
  - 返回 `(train, val)` 两个列表
  - train + val 总 perspective 数 = 12
  - 同一 game 的所有 perspective 归属于同一集合（train 或 val）
  - val 的 game 数 ≈ 3（6 × 0.5）
  - train 至少 1 个 game

## TC2: `_split_train_val` — validation_split=0

- **给定**: 任意 perspectives, `validation_split=0`
- **当**: 调用 `_split_train_val(perspectives, 0)`
- **则**:
  - val 为空列表
  - train 包含所有 perspectives

## TC3: `_split_train_val` — 仅 1 个 game

- **给定**: 仅 1 个 game, `validation_split=0.2`
- **当**: 调用 `_split_train_val(perspectives, 0.2)`
- **则**:
  - train 包含该 game 的所有 perspective
  - val 为空列表（train 至少保留 1 个 game）

## TC4: `_split_train_val` — 同游戏 perspectives 不分割

- **给定**: 2 个 game（game_a 有 3 个 perspective, game_b 有 2 个 perspective）, `validation_split=0.5`
- **当**: 调用 `_split_train_val(perspectives, 0.5)`
- **则**:
  - 2 个 game 分别完整进入 train 或 val
  - 不存在同一 game 的 perspective 分散在两个集合中

## TC5: `TrainingConfig` — validation_split 校验

- **给定**: `validation_split=-0.1`
- **当**: 构造 `TrainingConfig`
- **则**: 抛出 `ValueError`

- **给定**: `validation_split=1.0`
- **当**: 构造 `TrainingConfig`
- **则**: 抛出 `ValueError`

- **给定**: `validation_split=0.0`
- **当**: 构造 `TrainingConfig`
- **则**: 成功（允许关闭验证）

- **给定**: `validation_split=0.1`
- **当**: 构造 `TrainingConfig`
- **则**: 成功

## TC6: `_compute_val_loss` — 正常计算

- **给定**: 包含已知 value target 的验证 perspectives, `value_model` 已训练
- **当**: 调用 `_compute_val_loss(val_perspectives, value_model, config, device)`
- **则**:
  - 返回一个非 NaN 的 float
  - 返回值等于 perspectives 上 MSE loss 的平均值
  - 计算后模型恢复 `train()` 模式

## TC7: `_compute_val_loss` — 空验证集

- **给定**: 空的 val_perspectives 列表
- **当**: 调用 `_compute_val_loss([], value_model, config, device)`
- **则**: 返回 `nan`

## TC8: Phase 1 记录结构 — 含 val_value_loss

- **给定**: 正常训练，`validation_split=0.1`
- **当**: Phase 1 每个 epoch 完成后
- **则**: 生成的 `phase_records` 中每个 `phase="value"` 的记录包含：
  - `"value_loss"` — 训练集 loss（float）
  - `"val_value_loss"` — 验证集 loss（float，非 NaN）
  - `"phase"` — `"value"`
- Phase 1 的 val_value_loss ≠ `nan`

## TC9: Phase 1 记录结构 — validation_split=0 时

- **给定**: 训练, `validation_split=0`
- **当**: Phase 1 完成后
- **则**: 每个 `phase="value"` 记录的 `"val_value_loss"` 为 `nan`

## TC10: CSV 输出 — 含 val_value_loss 列

- **给定**: 3 个 value epoch + 2 个 policy epoch 的 phase_records
- **当**: 调用 `save_split_training_artifacts(phase_records, tmp_dir)`
- **则**:
  - `loss.csv` 表头为 `phase,epoch,policy_loss,value_loss,val_value_loss,entropy`
  - value 行中 `val_value_loss` 有具体数值（非 "nan"）
  - policy 行中 `val_value_loss` 为 `nan`

## TC11: PNG 图表 — 含 val loss 曲线

- **给定**: 包含有效 `val_value_loss` 的 phase_records
- **当**: 调用 `save_split_training_artifacts(phase_records, tmp_dir)`
- **则**:
  - `loss.png` 生成成功
  - 图表包含 4 条曲线: Value Loss, Val Value Loss, Policy Loss, Entropy
  - Val Value Loss 使用虚线样式

## TC12: `sample_loader` — perspective 含 game_id

- **给定**: 包含多个 game 的样本 JSON 文件
- **当**: 调用 `SampleLoader.load(sample_path)`
- **则**:
  - 每个返回的 perspective 字典包含 `"game_id"` 键
  - `game_id` 值与原 JSON 中 `game["game_id"]` 一致
  - perspective 字典仍包含 `"steps"` 键

## TC13: CLI — `--validation-split` 参数

- **给定**: 命令行 `--validation-split 0.2`
- **当**: 解析参数并构造 `TrainingConfig`
- **则**: `config.validation_split` = 0.2

- **给定**: 命令行不指定 `--validation-split`
- **当**: 解析参数并构造 `TrainingConfig`
- **则**: `config.validation_split` = `backend.json` 中的默认值

## TC14: 端到端 — 完整训练流程含验证

- **给定**: 包含 10 个 game 的样本文件, `validation_split=0.1`, `value_epoch=2`
- **当**: 执行 `BaseTrainingRunner.run(config)`
- **则**:
  - 训练正常完成，无异常
  - 控制台 Phase 1 输出包含 `val_value_loss`
  - `loss/loss.csv` 包含 `val_value_loss` 列
  - `loss/loss.png` 包含验证 loss 曲线
