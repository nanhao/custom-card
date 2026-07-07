# Test Cases — Minor Optimizations (intention #10)

---

## 1. Sample Key Rename: `state` → `input`

### TC-SR-01: SelfPlayRunner generates steps with `input` key
- **Given**: 任意有效的 `SelfPlayConfig`，`sample_output` 已设置
- **When**: 执行至少一局完整对弈
- **Then**: 输出的样本 JSON 中，每个 step dict 包含 `"input"` 键（14-d 向量），不包含 `"state"` 键

### TC-SR-02: TrainingRunner accepts samples with `input` key
- **Given**: 样本文件使用 `"input"` 键（D1 新格式）
- **When**: 执行训练
- **Then**: 正常加载，无报错

### TC-SR-03: TrainingRunner rejects samples with old `state` key
- **Given**: 样本文件使用 `"state"` 键（旧格式）
- **When**: 执行训练
- **Then**: 抛出 `RuntimeError`，错误信息指出缺少必填键 `"input"`

### TC-SR-04: Sample format backward incompatibility is documented
- **Given**: 存在旧格式样本文件（使用 `"state"` 键）
- **Then**: 旧样本文件不可直接用于新训练脚本，需重新生成

---

## 2. SelfPlayConfig Perspective Field

### TC-PC-01: Default perspective is `"both"`
- **Given**: 创建 `SelfPlayConfig` 时未指定 `perspective`
- **Then**: `perspective` 默认值为 `"both"`

### TC-PC-02: Valid perspective values accepted
- **Given**: `perspective` 分别设为 `"ai1"`, `"ai2"`, `"both"`
- **When**: 创建 `SelfPlayConfig`
- **Then**: 构造成功，无异常

### TC-PC-03: Invalid perspective value rejected
- **Given**: `perspective` 设为非法值，如 `"player1"`, `""`, `"none"`
- **When**: 创建 `SelfPlayConfig`
- **Then**: 抛出 `ValueError`

---

## 3. Perspective Filtering in SelfPlayRunner

### TC-PF-01: `perspective="both"` includes both perspectives
- **Given**: `config.perspective = "both"`，`sample_output` 已设置
- **When**: 执行至少一局完整对弈
- **Then**: 样本 JSON 中，每个 game dict 的 `perspectives` 数组包含 2 个元素（`player=1` 和 `player=2`）

### TC-PF-02: `perspective="ai1"` includes only player 1 perspective
- **Given**: `config.perspective = "ai1"`，`sample_output` 已设置
- **When**: 执行至少一局完整对弈
- **Then**: 样本 JSON 中，每个 game dict 的 `perspectives` 数组包含恰好 1 个元素，其 `player` 值为 `"1"`

### TC-PF-03: `perspective="ai2"` includes only player 2 perspective
- **Given**: `config.perspective = "ai2"`，`sample_output` 已设置
- **When**: 执行至少一局完整对弈
- **Then**: 样本 JSON 中，每个 game dict 的 `perspectives` 数组包含恰好 1 个元素，其 `player` 值为 `"2"`

### TC-PF-04: Perspective filtering also applied to step recording buffer
- **Given**: `config.perspective = "ai1"`，`sample_output` 已设置
- **When**: 执行对弈过程中检查内存中的 `game_samples` 缓冲区
- **Then**: 缓冲区仅包含 `player="1"` 的 step，不包含 `player="2"` 的 step

### TC-PF-05: Perspective filtering with `sample_output=None` has no effect
- **Given**: `config.perspective = "ai1"`，`sample_output = None`
- **When**: 执行对弈
- **Then**: 无样本文件生成，无异常，对弈统计正常输出

### TC-PF-06: Empty game with perspective filter produces valid JSON
- **Given**: `config.perspective = "ai1"`，`config.game_count = 0`，`sample_output` 已设置
- **When**: 执行 self-play
- **Then**: 输出文件包含空数组 `[]`

---

## 4. SelfPlay CLI — `--perspective`

### TC-SP-01: CLI accepts `--perspective ai1`
- **Given**: 命令行参数 `--perspective ai1`
- **When**: 解析参数
- **Then**: `args.perspective == "ai1"`

### TC-SP-02: CLI accepts `--perspective ai2`
- **Given**: 命令行参数 `--perspective ai2`
- **When**: 解析参数
- **Then**: `args.perspective == "ai2"`

### TC-SP-03: CLI default is `--perspective both`
- **Given**: 命令行未指定 `--perspective`
- **When**: 解析参数
- **Then**: `args.perspective == "both"`

### TC-SP-04: CLI rejects invalid perspective value
- **Given**: 命令行参数 `--perspective player1`
- **When**: 解析参数
- **Then**: argparse 报错退出

### TC-SP-05: Warning when --perspective set but --sample_output missing
- **Given**: 命令行参数 `--perspective ai1`，未指定 `--sample_output`
- **When**: 执行 self-play
- **Then**: stderr 输出警告信息，提示 perspective 过滤在无 sample output 时不生效

---

## 5. TrainingConfig Unified Parameters

### TC-TC-01: `output_dir` replaces `policy_output_path` + `value_output_path`
- **Given**: 创建 `TrainingConfig` 时指定 `output_dir`
- **Then**: `TrainingConfig` 包含 `output_dir` 字段，不包含 `policy_output_path` 和 `value_output_path` 字段

### TC-TC-02: `epoch` replaces `value_epochs`
- **Given**: 创建 `TrainingConfig` 时指定 `epoch`
- **Then**: `TrainingConfig` 包含 `epoch` 字段，不包含 `value_epochs` 字段

### TC-TC-03: `epoch` validation — must be >=1
- **Given**: `epoch` 设为 0 或负数
- **When**: 创建 `TrainingConfig`
- **Then**: 抛出 `ValueError`

### TC-TC-04: `output_dir` validation — must be non-empty
- **Given**: `output_dir` 设为 `""`
- **When**: 创建 `TrainingConfig`
- **Then**: 抛出 `ValueError`

### TC-TC-05: `sample_path` validation unchanged
- **Given**: `sample_path` 指向不存在的路径
- **When**: 创建 `TrainingConfig`
- **Then**: 抛出 `ValueError`

---

## 6. TrainingRunner Behavior

### TC-TR-01: Phase 1 uses `epoch` iterations
- **Given**: `config.epoch = 5`
- **When**: 执行训练
- **Then**: Phase 1（value training）运行恰好 5 个 epoch

### TC-TR-02: Phase 3 uses `epoch` iterations
- **Given**: `config.epoch = 5`
- **When**: 执行训练
- **Then**: Phase 3（policy training）运行恰好 5 个 epoch

### TC-TR-03: Policy weights saved to `{output_dir}/policy.pth`
- **Given**: `config.output_dir = "saves/gen5"`
- **When**: 训练完成
- **Then**: 文件 `saves/gen5/policy.pth` 被创建

### TC-TR-04: Value weights saved to `{output_dir}/value.pth`
- **Given**: `config.output_dir = "saves/gen5"`
- **When**: 训练完成
- **Then**: 文件 `saves/gen5/value.pth` 被创建

### TC-TR-05: Output directory auto-created if not exists
- **Given**: `config.output_dir = "saves/nonexistent_dir"`
- **When**: 训练完成
- **Then**: 目录 `saves/nonexistent_dir/` 被自动创建，权重文件写入成功

### TC-TR-06: TrainingRunner reads `step["input"]` not `step["state"]`
- **Given**: 包含 `"input"` 键的样本文件
- **When**: 加载样本并训练
- **Then**: 成功读取所有 step 的 14-d 向量，无 KeyError

---

## 7. Train CLI

### TC-CLI-01: CLI requires `--output-dir`
- **Given**: 命令行缺少 `--output-dir`
- **When**: 解析参数
- **Then**: argparse 报错退出

### TC-CLI-02: CLI rejects `--policy-output-path`
- **Given**: 命令行包含 `--policy-output-path`
- **When**: 解析参数
- **Then**: argparse 报 "unrecognized arguments"

### TC-CLI-03: CLI rejects `--value-output-path`
- **Given**: 命令行包含 `--value-output-path`
- **When**: 解析参数
- **Then**: argparse 报 "unrecognized arguments"

### TC-CLI-04: CLI rejects `--value-epochs`
- **Given**: 命令行包含 `--value-epochs`
- **When**: 解析参数
- **Then**: argparse 报 "unrecognized arguments"

### TC-CLI-05: CLI requires `--epoch`
- **Given**: 命令行缺少 `--epoch`
- **When**: 解析参数
- **Then**: argparse 报错退出

### TC-CLI-06: Full correct invocation
- **Given**: 命令行 `--sample-path samples/v3_self_play.json --output-dir saves/gen5 --model rnn_v2.1 --epoch 10`
- **When**: 解析参数并执行训练
- **Then**: 训练正常完成，`saves/gen5/policy.pth` 和 `saves/gen5/value.pth` 被创建

---

## 8. Integration

### TC-INT-01: Full self-play → train pipeline with new format
- **Given**: 使用新 self-play 生成样本（`--perspective both`，`"input"` 键）；使用新 train 加载样本（`--output-dir` + `--epoch`）
- **When**: 完整执行 self-play 后执行训练
- **Then**: 训练正常完成，权重文件写入正确路径

### TC-INT-02: Self-play with `--perspective ai1` → train with filtered samples
- **Given**: self-play 使用 `--perspective ai1`；训练加载生成的样本
- **When**: 完整执行
- **Then**: 训练正常完成（每个 game 只有一个 perspective，仍然构成有效的训练数据）

---

## 9. Non-Regression

### TC-NR-01: Game logic unchanged
- **Given**: 任意测试场景
- **When**: 使用变更后的代码执行对弈
- **Then**: 对弈结果（胜负判定、伤害计算、回合数限制）与变更前一致

### TC-NR-02: PVE mode unchanged
- **Given**: 人机对战模式
- **When**: 使用变更后的代码
- **Then**: AI 行为（greedy 选牌）与变更前一致

### TC-NR-03: AI registry unchanged
- **Given**: 所有已注册的 AI（`rnn_v1.1` ~ `rnn_v1.4`）
- **When**: 查询和使用
- **Then**: 行为与变更前完全一致

### TC-NR-04: Self-play parameters unchanged (except new `--perspective`)
- **Given**: 使用 `--ai1`, `--ai2`, `--game`, `--temperature`, `--temp_step`, `--temp_side`, `--temp_seed`, `--sample_output` 参数
- **When**: 执行 self-play
- **Then**: 行为与变更前一致（当 `--perspective` 为默认 `both` 时）

### TC-NR-05: Existing tests update required
- **Given**: 变更前全部测试套件
- **When**: 执行全部测试
- **Then**: 以下测试文件引用了 `“state”` 键或旧 CLI 参数，必须同步更新：
  - `tests/unit/test_selfplay_runner.py` — `“state”` → `“input”`
  - `tests/integration/test_selfplay_samples.py` — `“state”` → `“input”`
  - `tests/integration/test_selfplay_samples_output.py` — `“state”` → `“input”`
  - 调用 `train.py` 的测试 — `--policy-output-path`/`--value-output-path`/`--value-epochs` → `--output-dir`/`--epoch`
  - 更新后全部测试通过
