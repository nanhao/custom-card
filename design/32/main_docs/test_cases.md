# Test Cases — 分离 Policy/Value 训练 Epoch (intention #32)

## 测试范围

- **单元测试**: `tests/unit/test_training_config.py` — `TrainingConfig` 字段验证
- **集成测试**: `tests/integration/test_train_cli.py` — train.py CLI 参数解析

---

## 1. TrainingConfig — 字段校验

### TC-EC-01 — 仅指定 policy_epoch 时 value_epoch 等于 policy_epoch

**Given**: `policy_epoch=10`, `value_epoch` 由 CLI 层默认处理为 10

**When**: `TrainingConfig(policy_epoch=10, value_epoch=10, ...)`

**Then**: 成功创建，`config.policy_epoch == 10`, `config.value_epoch == 10`

---

### TC-EC-02 — 分别指定不同的 policy_epoch 和 value_epoch

**Given**: `policy_epoch=10`, `value_epoch=5`

**When**: `TrainingConfig(policy_epoch=10, value_epoch=5, ...)`

**Then**: 成功创建，`config.policy_epoch == 10`, `config.value_epoch == 5`

---

### TC-EC-03 — policy_epoch = 0 抛出异常

**Given**: `policy_epoch=0`

**When**: `TrainingConfig(policy_epoch=0, ...)`

**Then**: 抛出 `ValueError`，消息包含 "policy_epoch must be >= 1"

---

### TC-EC-04 — value_epoch = 0 抛出异常

**Given**: `value_epoch=0`

**When**: `TrainingConfig(value_epoch=0, ...)`

**Then**: 抛出 `ValueError`，消息包含 "value_epoch must be >= 1"

---

### TC-EC-05 — policy_epoch = -1 抛出异常

**Given**: `policy_epoch=-1`

**When**: `TrainingConfig(policy_epoch=-1, ...)`

**Then**: 抛出 `ValueError`

---

### TC-EC-06 — value_epoch = -1 抛出异常

**Given**: `value_epoch=-1`

**When**: `TrainingConfig(value_epoch=-1, ...)`

**Then**: 抛出 `ValueError`

---

### TC-EC-07 — 原 `epoch` 字段已移除

**Given**: 尝试使用 `epoch` 参数

**When**: `TrainingConfig(epoch=10, ...)`

**Then**: 抛出 `TypeError`（`unexpected keyword argument 'epoch'`）

---

## 2. train.py — CLI 参数解析

### TC-CLI-01 — --policy-epoch 为必选参数

**Given**: 不传 `--policy-epoch`

**When**: `python train.py --sample-path ... --output-dir ... --model rnn_v2.1`

**Then**: argparse 报错 "the following arguments are required: --policy-epoch"，退出码非零

---

### TC-CLI-02 — 仅指定 --policy-epoch 时正常运行

**Given**: `--policy-epoch 10`，不传 `--value-epoch`

**When**: `python train.py --sample-path ... --output-dir ... --model rnn_v2.1 --policy-epoch 10`

**Then**: 正常启动训练，`TrainingConfig.value_epoch` 被设为 10（等于 policy_epoch）

---

### TC-CLI-03 — 同时指定 --policy-epoch 和 --value-epoch 时正常运行

**Given**: `--policy-epoch 10 --value-epoch 5`

**When**: `python train.py --sample-path ... --output-dir ... --model rnn_v2.1 --policy-epoch 10 --value-epoch 5`

**Then**: 正常启动训练，Phase 1 执行 5 轮，Phase 3 执行 10 轮

---

### TC-CLI-04 — 不再接受 --epoch 参数

**Given**: `--epoch 10`

**When**: `python train.py --sample-path ... --output-dir ... --model rnn_v2.1 --epoch 10`

**Then**: argparse 报错 "unrecognized arguments: --epoch"，退出码非零

---

### TC-CLI-05 — --help 显示新参数

**Given**: 无参数

**When**: `python train.py --help`

**Then**: 输出中不包含 `--epoch`，包含 `--policy-epoch`（required）和 `--value-epoch`（optional）

---

## 3. BaseTrainingRunner — 循环轮数验证

### TC-RUN-01 — value_epoch=3 时 Phase 1 打印 3 轮

**Given**: `value_epoch=3`, `policy_epoch=10`

**When**: `BaseTrainingRunner.run(config)`

**Then**: 控制台输出包含 `[Value] Epoch 1/3`, `[Value] Epoch 2/3`, `[Value] Epoch 3/3`，不出现 `[Value] Epoch 4/...`

---

### TC-RUN-02 — policy_epoch=3 时 Phase 3 打印 3 轮

**Given**: `value_epoch=10`, `policy_epoch=3`

**When**: `BaseTrainingRunner.run(config)`

**Then**: 控制台输出包含 `[Policy] Epoch 1/3`, `[Policy] Epoch 2/3`, `[Policy] Epoch 3/3`，不出现 `[Policy] Epoch 4/...`

---

### TC-RUN-03 — phase_records 记录正确长度

**Given**: `value_epoch=3`, `policy_epoch=5`

**When**: `BaseTrainingRunner.run(config)`

**Then**: `phase_records` 总长度为 8（3 value + 5 policy），value phase 记录 epoch 1~3，policy phase 记录 epoch 1~5

---

## 4. 向后兼容性验证

### TC-BC-01 — 仅指定 --policy-epoch 等效于原 --epoch

**Given**: 同一份 sample 文件和相同的超参数

**When**: 旧版 `--epoch 5` 训练一次，新版 `--policy-epoch 5` 训练一次

**Then**: 两次训练的 Phase 1 和 Phase 3 轮数完全相同（各 5 轮），loss 曲线一致
