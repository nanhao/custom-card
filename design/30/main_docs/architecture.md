# Architecture Design — AI Registry Simplification (intention #30)

## 1. Motivation

当前 registry 中每个 AI 条目硬编码 `weight_path`，每训练一个 gen 就要新增条目（`rnn_v1.1` ~ `rnn_v1.8` 实为同一架构、不同权重），导致注册繁琐且冗余。

本次重构将权重路径从 registry 剥离：registry 仅声明网络架构（module + class），权重路径由固定的命名约定派生。`name.gen` 格式（如 `rnn_v1.3`）统一标识架构+权重版本。

## 2. Module Overview

```
src/backend/ai/
├── registry.py              # MODIFIED — 移除 weight_path，新增 parse_ai_name()，list_ais() 改为文件系统扫描
├── training_runner.py       # MODIFIED — 通过约定派生 weight_path 替代 registry 读取
├── training_config.py       # Unchanged
└── rnn_v1/model.py          # Unchanged

src/backend/app/
├── app_manager.py           # MODIFIED — _load_model() 接收显式 weight_path 参数
├── selfplay_runner.py       # Unchanged — 通过 ai_name 透传，解析在下游
├── selfplay_config.py       # Unchanged
└── converter.py             # Unchanged

src/backend/controller/
└── api.py                   # Unchanged — list_ais() 返回行为自动适配新格式

src/backend/scripts/
├── train.py                 # Unchanged
└── selfplay.py              # Unchanged
```

## 3. Key Design Decisions

**D1 — `name.gen` 格式**。所有接受 AI 标识的参数（`ai_name`、`--ai1`、`--ai2`、`--model`）统一使用 `{name}.{gen}` 格式。`.` 前为 registry 中的架构名，`.` 后为 gen 编号。

**D2 — 权重路径约定**。policy 和 value 权重路径由固定规则派生：
- `policy_path = src/backend/ai/{name}/gen{gen}/policy.pth`
- `value_path = src/backend/ai/{name}/gen{gen}/value.pth`
路径相对于项目根目录。调用方负责确保文件存在。

**D3 — `list_ais()` 文件系统扫描**。返回的 AI 列表不再来自 registry 的硬编码条目，而是：遍历 registry 中所有架构名 → 扫描 `src/backend/ai/{name}/` 下 `gen{i}` 格式的目录 → 拼接 `{name}.{i}` 作为 AI 名。这样新 gen 无需修改 registry。

**D4 — Registry 格式简化**。条目仅保留架构信息：
```
{
    "name": "rnn_v1",
    "description": "gru hidden-state=32, level=2",
    "policy": {
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUPolicyModel",
    },
    "value": {
        "module": "src.backend.ai.rnn_v1.model",
        "class": "GRUValueModel",
    },
}
```
`rnn_v1.1` ~ `rnn_v1.8` 合并为单条 `rnn_v1`。

**D5 — AppManager 权重路径参数化**。`_load_model()` 不再从 registry 读取 `weight_path`，改为接收显式的 `weight_path` 参数。调用方通过 `parse_ai_name()` 解析 `name.gen` 后按 D2 约定拼接路径。

## 4. What Changes

| File | Change |
|------|--------|
| `ai/registry.py` | REPLACE: 移除 `weight_path`；新增 `parse_ai_name()`；修改 `list_ais()` 为文件系统扫描；修改 `get_ai_info()` 支持 `name.gen` 格式并派生 `weight_path` |
| `app/app_manager.py` | MODIFIED: `_load_model()` 增加 `weight_path` 参数；`_get_policy_model()` / `_get_value_model()` 解析 ai_name 派生路径 |
| `ai/training_runner.py` | MODIFIED: `_build_models()` 通过约定派生 weight_path 替代 registry 读取 |
| `scripts/train.py` | Unchanged |
| `scripts/selfplay.py` | Unchanged |
| `controller/api.py` | Unchanged |
| `game/*` | Unchanged |
| `app/converter.py` | Unchanged |

## 5. Key Flows

- **AI name 解析与权重路径派生**: see [registry_flow.puml](diagrams/registry_flow.puml)
- **类关系图**: see [class_diagram.puml](diagrams/class_diagram.puml)
