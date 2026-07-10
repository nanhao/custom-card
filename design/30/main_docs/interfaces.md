# Interface Design — AI Registry Simplification (intention #30)

Public interfaces only. Private methods omitted per design rule.

---

## 1. `registry.py` — Public API

### 1.1 `parse_ai_name(ai_name: str) -> tuple[str, int]`

解析 `name.gen` 格式的 AI 标识字符串，返回 `(name, gen)`。

- `"rnn_v1.3"` → `("rnn_v1", 3)`
- `"rnn_v2.1"` → `("rnn_v2", 1)`

异常：
- `ValueError` — 格式不合法（不含 `.`、gen 非整数、gen < 1）

---

### 1.2 `list_ais() -> list[dict]`

返回所有可用 AI 的列表，每项包含 `name` 和 `description`。

行为：
1. 遍历 `_REGISTRY` 中每个架构条目，提取 `name` 和 `description`
2. 扫描 `src/backend/ai/{name}/` 目录，查找 `gen{i}` 格式的子目录
3. 为每个 `gen{i}` 生成条目：`{"name": "{name}.{i}", "description": "{description} (gen{i})"}`
4. 按 name 后 gen 升序排列

若某架构名下无 gen 目录，则该架构不出现在列表中。

---

### 1.3 `get_ai_info(ai_name: str) -> dict`

返回指定 AI 的完整信息（架构 + 派生权重路径）。

入参 `ai_name` 为 `name.gen` 格式。内部调用 `parse_ai_name()` 解析，再查 `_REGISTRY` 获取架构，按 D2 约定拼接 `weight_path`。

返回格式：
```
{
    "name": "rnn_v1",
    "description": "gru hidden-state=32, level=2",
    "policy": {
        "weight_path": "src/backend/ai/rnn_v1/gen3/policy.pth",
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUPolicyModel",
    },
    "value": {
        "weight_path": "src/backend/ai/rnn_v1/gen3/value.pth",
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUValueModel",
    },
}
```

异常：
- `ValueError` — `name` 不在 registry、`parse_ai_name()` 失败

---

### 1.4 `_derive_weight_path(name: str, gen: int, suffix: str) -> str`

派生权重文件路径。`suffix` 为 `"policy"` 或 `"value"`。

返回：`src/backend/ai/{name}/gen{gen}/{suffix}.pth`（相对于项目根目录）。

---

## 2. `app_manager.py` — Modified Methods

### 2.1 `_load_model(module_name: str, class_name: str, weight_path: str) -> nn.Module`

签名变更为接收显式 `weight_path` 参数（原从 registry 隐式获取）。

行为不变：导入模块 → 实例化类 → 加载权重 → `model.eval()`。

---

## 3. `training_runner.py` — Modified Methods

### 3.1 `_build_models(model_name: str) -> tuple[nn.Module, nn.Module, torch.device]`

改为通过 `parse_ai_name()` + `_derive_weight_path()` 获取权重路径，不再从 registry 的 `weight_path` 字段读取。

其余行为不变。

---

## 4. API Routes — Unchanged

| Route | Status |
|-------|--------|
| `GET /api/ais` | Unchanged — 返回格式不变，内容由 `list_ais()` 新行为自动适配 |
| `POST /api/games` | Unchanged — 请求中 `ai` 字段仍为 `"rnn_v1.3"` 格式 |
| `POST /api/games/{id}/play` | Unchanged |
| `GET /api/games/{id}` | Unchanged |
| `DELETE /api/games/{id}` | Unchanged |

## 5. Script CLIs — Unchanged

| Script | Status |
|--------|--------|
| `selfplay.py` | Unchanged — `--ai1`/`--ai2` 仍为 `"rnn_v1.3"` 格式 |
| `train.py` | Unchanged — `--model` 仍为 `"rnn_v1.3"` 格式 |
