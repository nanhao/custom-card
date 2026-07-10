# Test Cases — AI Registry Simplification (intention #30)

---

## 1. `parse_ai_name()` — Name Parsing

### TC-PN-01: Valid `name.gen` format
- **Given**: ai_name = `"rnn_v1.3"`
- **When**: 调用 `parse_ai_name("rnn_v1.3")`
- **Then**: 返回 `("rnn_v1", 3)`

### TC-PN-02: Single-digit gen
- **Given**: ai_name = `"rnn_v1.1"`
- **When**: 调用 `parse_ai_name("rnn_v1.1")`
- **Then**: 返回 `("rnn_v1", 1)`

### TC-PN-03: Multi-digit gen
- **Given**: ai_name = `"rnn_v1.123"`
- **When**: 调用 `parse_ai_name("rnn_v1.123")`
- **Then**: 返回 `("rnn_v1", 123)`

### TC-PN-04: Name contains multiple dots
- **Given**: ai_name = `"rnn_v2.0.1"`  （测试只按最后一个 `.` 分割？还是拒绝？——设计要求：拒绝）
- **When**: 调用 `parse_ai_name("rnn_v2.0.1")`
- **Then**: 抛出 `ValueError`，信息包含 `"invalid ai_name"`

### TC-PN-05: No dot
- **Given**: ai_name = `"rnn_v1"`
- **When**: 调用 `parse_ai_name("rnn_v1")`
- **Then**: 抛出 `ValueError`

### TC-PN-06: Non-integer gen
- **Given**: ai_name = `"rnn_v1.abc"`
- **When**: 调用 `parse_ai_name("rnn_v1.abc")`
- **Then**: 抛出 `ValueError`

### TC-PN-07: gen = 0
- **Given**: ai_name = `"rnn_v1.0"`
- **When**: 调用 `parse_ai_name("rnn_v1.0")`
- **Then**: 抛出 `ValueError`（gen 必须 >= 1）

### TC-PN-08: Negative gen
- **Given**: ai_name = `"rnn_v1.-1"`
- **When**: 调用 `parse_ai_name("rnn_v1.-1")`
- **Then**: 抛出 `ValueError`

---

## 2. `_derive_weight_path()` — Path Derivation

### TC-DP-01: Policy path
- **Given**: name=`"rnn_v1"`, gen=3, suffix=`"policy"`
- **When**: 调用 `_derive_weight_path("rnn_v1", 3, "policy")`
- **Then**: 返回 `"src/backend/ai/rnn_v1/gen3/policy.pth"`

### TC-DP-02: Value path
- **Given**: name=`"rnn_v1"`, gen=3, suffix=`"value"`
- **When**: 调用 `_derive_weight_path("rnn_v1", 3, "value")`
- **Then**: 返回 `"src/backend/ai/rnn_v1/gen3/value.pth"`

### TC-DP-03: Different architecture
- **Given**: name=`"rnn_v2"`, gen=1, suffix=`"policy"`
- **When**: 调用 `_derive_weight_path("rnn_v2", 1, "policy")`
- **Then**: 返回 `"src/backend/ai/rnn_v2/gen1/policy.pth"`

---

## 3. `get_ai_info()` — Registry Lookup

### TC-GI-01: Valid name.gen returns full info with derived weight_path
- **Given**: registry 中有 `rnn_v1` 条目；磁盘上 `src/backend/ai/rnn_v1/gen3/` 存在
- **When**: 调用 `get_ai_info("rnn_v1.3")`
- **Then**: 返回 dict 包含 `policy.weight_path = "src/backend/ai/rnn_v1/gen3/policy.pth"` 和 `value.weight_path = "src/backend/ai/rnn_v1/gen3/value.pth"`；不检查文件是否存在

### TC-GI-02: Unknown architecture name
- **Given**: registry 中无 `"unknown_model"`
- **When**: 调用 `get_ai_info("unknown_model.1")`
- **Then**: 抛出 `ValueError`，信息包含 `"unknown"` 或 `"not found"`

### TC-GI-03: Invalid name.gen format
- **Given**: 任意 registry 状态
- **When**: 调用 `get_ai_info("invalid_format")`
- **Then**: 抛出 `ValueError`

---

## 4. `list_ais()` — Filesystem Scanning

### TC-LA-01: Single architecture, single gen
- **Given**: registry 中有 `rnn_v1`；`src/backend/ai/rnn_v1/` 下仅有 `gen1/`
- **When**: 调用 `list_ais()`
- **Then**: 返回 `[{"name": "rnn_v1.1", "description": "gru hidden-state=32, level=2 (gen1)"}]`

### TC-LA-02: Single architecture, multiple gens
- **Given**: registry 中有 `rnn_v1`；`gen1/`、`gen3/`、`gen5/` 存在（gen2、gen4 缺失）
- **When**: 调用 `list_ais()`
- **Then**: 返回 `rnn_v1.1`、`rnn_v1.3`、`rnn_v1.5` 共 3 条，按 gen 升序

### TC-LA-03: Multiple architectures
- **Given**: registry 中有 `rnn_v1` 和 `rnn_v2`；各有 gen 目录
- **When**: 调用 `list_ais()`
- **Then**: 返回所有 `{name}.{gen}` 组合，按 name 后 gen 排序

### TC-LA-04: Architecture with no gen directories
- **Given**: registry 中有 `rnn_v1`；`src/backend/ai/rnn_v1/` 下无 `gen*` 目录
- **When**: 调用 `list_ais()`
- **Then**: `rnn_v1` 架构不出现在返回列表中

### TC-LA-05: Non-gen directories ignored
- **Given**: `src/backend/ai/rnn_v1/` 下有 `gen1/`、`gen2/`、`__pycache__/`、`some_file.txt`
- **When**: 调用 `list_ais()`
- **Then**: 仅 `gen1` 和 `gen2` 被识别；`__pycache__` 和 `.txt` 被忽略

### TC-LA-06: gen directory with non-integer suffix ignored
- **Given**: `src/backend/ai/rnn_v1/` 下有 `gen1/`、`genX/`、`gen/`
- **When**: 调用 `list_ais()`
- **Then**: 仅 `gen1` 被识别；非数字后缀忽略

### TC-LA-07: Empty registry
- **Given**: `_REGISTRY` 为空列表
- **When**: 调用 `list_ais()`
- **Then**: 返回空列表 `[]`

---

## 5. Integration — End-to-End Consistency

### TC-IT-01: PVE game creation with name.gen format
- **Given**: `rnn_v1` 在 registry 中；`gen1/policy.pth` 和 `gen1/value.pth` 存在
- **When**: `POST /api/games` body `{"mode": "pve", "ai": "rnn_v1.1"}`
- **Then**: 201 Created；后续 `POST /api/games/{id}/play` 正常返回 AI 出牌

### TC-IT-02: Self-play with name.gen format
- **Given**: `rnn_v1` 在 registry 中；`gen1/` 权重存在
- **When**: `python selfplay.py --ai1 rnn_v1.1 --ai2 rnn_v1.1 --game 1`
- **Then**: 正常运行，无报错

### TC-IT-03: Training with name.gen format
- **Given**: `rnn_v1` 在 registry 中；`gen1/` 权重存在
- **When**: `python train.py --model rnn_v1.1 --sample-path samples.json --output-dir gen2 --epoch 1`
- **Then**: 从 `gen1/` 加载初始权重，输出到 `gen2/`

### TC-IT-04: `GET /api/ais` returns filesystem-based list
- **Given**: registry 中有 `rnn_v1`；`gen1/`、`gen2/` 存在
- **When**: `GET /api/ais`
- **Then**: 返回 `{"ais": [{"name": "rnn_v1.1", ...}, {"name": "rnn_v1.2", ...}]}`

---

## 6. Non-Regression

### TC-NR-01: All existing tests pass
- **Given**: 变更前全部测试套件
- **When**: 执行全部测试
- **Then**: 所有已有测试通过（必要时更新测试中的 mock 数据）

### TC-NR-02: PVP mode unaffected
- **Given**: 无 AI 参与的 PVP 对局
- **When**: 创建 PVP 游戏并进行完整对局
- **Then**: 行为与变更前完全一致

### TC-NR-03: Game logic unchanged
- **Given**: 所有 game/ 模块
- **When**: 执行 game 层单元测试
- **Then**: 全部通过，game 层代码无需任何修改

### TC-NR-04: Model architecture unchanged
- **Given**: `GRUPolicyModel` 和 `GRUValueModel`
- **When**: 使用任意 gen 权重文件进行推理
- **Then**: 推理结果与变更前一致
