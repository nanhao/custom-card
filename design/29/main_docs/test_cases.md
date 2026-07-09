# Test Cases — Training Performance Optimization (intention #29)

---

## 1. Sequence Batching Correctness

### TC-SB-01: Value loss unchanged after sequence batching
- **Given**: 相同的样本数据、模型权重（固定随机种子）、训练配置
- **When**: 分别用改造前（逐步）和改造后（全序列）执行 Phase 1 value training 1 epoch
- **Then**: `avg_value_loss` 和 `avg_grad_norm` 数值完全一致（浮点精度内）

### TC-SB-02: Policy loss unchanged after sequence batching
- **Given**: 相同的样本数据、模型权重、Phase 2 重算的 advantages
- **When**: 分别用改造前和改造后执行 Phase 3 policy training 1 epoch
- **Then**: `avg_policy_loss`、`avg_entropy`、`avg_grad_norm` 数值完全一致

### TC-SB-03: Phase 2 recomputed advantages unchanged
- **Given**: 相同的样本数据、训练后的 value model 权重
- **When**: 分别用改造前和改造后执行 `_recompute_advantages`
- **Then**: 所有 perspective 的 `new_advantages` 和 `new_value_targets` 完全一致

### TC-SB-04: Full training produces identical weights
- **Given**: 相同的样本数据、初始模型权重（固定随机种子）、训练配置
- **When**: 分别用改造前和改造后执行完整三阶段训练 1 epoch
- **Then**: 输出的 `policy.pth` 和 `value.pth` 权重数值完全一致（`torch.allclose`）

---

## 2. Load Progress Output

### TC-LP-01: Load progress prints on stdout
- **Given**: 任意有效样本文件
- **When**: 执行训练
- **Then**: stdout 输出包含 "Loading ..."、"Parsed N games"、"Validating... X%"、"Loaded N perspectives" 四类提示

### TC-LP-02: Progress percentage covers full range
- **Given**: 包含 >=10 局游戏的样本文件
- **When**: 执行训练
- **Then**: "Validating..." 从 0% 开始逐步增长至 90%，步长约 10%

### TC-LP-03: Phase 2 start message
- **Given**: 任意有效训练配置
- **When**: 训练进入 Phase 2
- **Then**: stdout 输出 "[Phase 2] Recomputing advantages for N perspectives..."

---

## 3. Edge Cases

### TC-EC-01: Empty perspective (T=0 steps)
- **Given**: 样本包含一个 steps 为空的 perspective
- **When**: 加载样本
- **Then**: `_load_samples` 正常跳过或报出明确错误（取决于现有验证逻辑）

### TC-EC-02: Single-step perspective (T=1)
- **Given**: 样本中所有 perspective 均只有 1 步
- **When**: 执行训练
- **Then**: 全序列 GRU `(1, 1, 42)` 正常运行，loss 与前一致

### TC-EC-03: Very long perspective (T >> 100)
- **Given**: 样本包含 steps 数 >100 的 perspective
- **When**: 执行训练
- **Then**: 全序列 GRU 不触发 OOM，loss 正常

### TC-EC-04: NaN values in input
- **Given**: 样本中某 step input 包含 NaN
- **When**: 执行训练
- **Then**: 保持原有 NaN 检测和跳过行为

---

## 4. Non-Regression

### TC-NR-01: Training output format unchanged
- **Given**: 任意有效训练配置
- **When**: 执行完整训练
- **Then**: 输出文件 `policy.pth` 和 `value.pth` 格式、key 名均不变

### TC-NR-02: Training artifacts unchanged
- **Given**: 任意有效训练配置
- **When**: 执行完整训练
- **Then**: 训练产物（CSV 文件等）内容结构不变

### TC-NR-03: Model forward unchanged
- **Given**: 任意 `GRUPolicyModel` 或 `GRUValueModel` 实例
- **When**: 调用 `model.forward(x, h)`，`x` 为 `(1, 1, 42)`
- **Then**: 行为与改造前完全一致

### TC-NR-04: Self-play compatibility
- **Given**: 改造后的模型权重文件
- **When**: 用于 self-play 推理
- **Then**: 推理行为（action 选择、value 估计）与改造前一致

### TC-NR-05: Registry unchanged
- **Given**: AI registry 中所有模型条目
- **When**: 查询和使用
- **Then**: 行为与变更前完全一致

### TC-NR-06: Existing test suite passes
- **Given**: 变更前全部测试套件
- **When**: 执行全部测试
- **Then**: 所有已有测试通过（本次变更不涉及测试文件更新）
