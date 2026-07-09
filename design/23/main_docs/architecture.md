# Architecture Design — 前端维护卡牌图标 (intention #23)

## 1. Motivation

当前 `CardDef`（`game/card_def.py`）定义了 `icon: str` 字段，存储卡牌的 emoji 图标（如 ⚔️🛡️💥）。`icon` 是纯显示逻辑，不属于后端游戏层的职责。按项目分层原则，显示元数据应由前端自行维护。

本次重构将 `icon` 从后端 `CardDef` 中移除，前端改为本地维护图标映射。`display_name` 保留在后端（用于 API 响应中的卡牌名称展示）。

## 2. Module Impact

| 文件 | 变更 |
|------|------|
| `game/card_def.py` | **修改** — `CardDef` 移除 `icon` 字段；`_CARD_DEFS` 移除所有 icon 值；更新 docstring |
| `controller/api.py` | **修改** — `GET /api/cards` 响应移除 `icon` 字段 |
| `frontend/src/GameBoard.jsx` | **修改** — 移除从 API 读取 icon 的逻辑；新增本地 `CARD_ICONS` 常量映射 |
| `frontend/tests/App.test.jsx` | **修改** — `MOCK_CARDS` 移除 `icon` 字段 |
| `tests/unit/test_card.py` | **修改** — 移除 `test_icon_set` 测试；`test_card_def_fields` 更新断言 |

不需要变更：`ai/`、`app/`、`log/`、`config/`、`resolver/`、`session_store.py`。

## 3. Key Design Decisions

**D1 — icon 由前端硬编码。** 前端 `GameBoard.jsx` 内定义 `CARD_ICONS` 常量对象，以 card type 字符串为 key、emoji 为 value。图标映射表仅 7 个条目，硬编码比通过 API 获取更简单直接。未来若需动态图标，可由 API 返回图标 URL 或由 CSS 类控制。

**D2 — 不新增前端文件。** 图标常量直接定义在 `GameBoard.jsx` 组件文件内（模块级常量），不单独创建 `constants.js`。7 行映射不构成拆分理由。

**D3 — `/api/cards` 接口收缩。** 响应字段从 `{type, display_name, icon, build_cost}` 缩减为 `{type, display_name, build_cost}`。前端不再依赖该接口获取图标。

**D4 — display_name 保留在后端。** `display_name` 虽也是显示数据，但用于 API 响应中标识卡牌名称，且未来可能用于 AI 训练数据的可读性输出。保留在后端维持单一数据源。

## 4. Icon Mapping Table（前端维护）

| card type | icon |
|-----------|------|
| attack | ⚔️ |
| defense | 🛡️ |
| heavy | 💥 |
| counter | ↩️ |
| energy | ⚡ |
| build | 🔨 |
| flex | ⚖️ |

与当前后端 `_CARD_DEFS` 中的值完全相同，不改变任何图标。

## 5. Key Flows

- **卡牌显示数据流**: 见 [data_flow.puml](diagrams/data_flow.puml)
- **类图及依赖关系**: 见 [class_diagram.puml](diagrams/class_diagram.puml)
