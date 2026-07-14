# 设计 #40 — 测试用例集

## T1. AtteV1Converter — card_to_action_id

| ID | 场景 | 预期 |
|----|------|------|
| T1.1 | `card_to_action_id(CardType.ATTACK)` | 返回 3 |
| T1.2 | `card_to_action_id(CardType.DEFENSE)` | 返回 6 |
| T1.3 | `card_to_action_id(CardType.HEAVY)` | 返回 9 |
| T1.4 | `card_to_action_id(CardType.COUNTER)` | 返回 12 |
| T1.5 | `card_to_action_id(CardType.ENERGY)` | 返回 15 |
| T1.6 | `card_to_action_id(CardType.FLEX)` | 返回 18 |
| T1.7 | `card_to_action_id(CardType.IRON_DEFENSE)` | 返回 21 |
| T1.8 | `card_to_action_id(CardType.BUILD)` | 抛出 ValueError |

## T2. AtteV1Converter — status_to_id

| ID | 场景 | 预期 |
|----|------|------|
| T2.1 | `status_to_id(StatusType.DEFENSE)` | 返回 0 |
| T2.2 | `status_to_id(StatusType.GUARD)` | 返回 1 |
| T2.3 | `status_to_id(StatusType.FRAGILE)` | 返回 2 |

## T3. AtteV1Converter — build_input

| ID | 场景 | 预期 |
|----|------|------|
| T3.1 | 手牌 [ATTACK, DEFENSE] | own.hand_ids = [3, 6] |
| T3.2 | 弃牌堆 [ATTACK, ATTACK] | own.discard_ids = [3, 3] |
| T3.3 | 抽牌堆 [ENERGY] | own.draw_ids = [15] |
| T3.4 | 空手牌 | own.hand_ids = [] |
| T3.5 | 首回合（无历史） | own.last_action = -1 |
| T3.6 | 上回合打出 ATTACK | own.last_action = 3 |
| T3.7 | HP=30, energy=5 | own.hp = 1.0, own.energy = 1.0 |
| T3.8 | HP=15, energy=2 | own.hp = 0.5, own.energy = 0.4 |
| T3.9 | DEFENSE(lifetime=2) | own.status_ids = [0], own.status_durations = [2] |
| T3.10 | 无状态 | own.status_ids = [], own.status_durations = [] |
| T3.11 | player_key="2" 视角 | own=玩家2, opp=玩家1 |
| T3.12 | player_key 非法值 | 抛出 ValueError |
| T3.13 | 对方弃牌堆/抽牌堆/手牌可见 | own.opp_discard_ids, own.opp_draw_ids, own.opp_hand_ids 正确填充 |
| T3.14 | GameStateSnapshot.round_in_cycle=1/2/3 | AtteV1Input.round_in_cycle = -1/0/1（Converter 做 -2 偏移） |
| T3.15 | last_action=-1 → PAD | Converter 将 -1 映射为 24（PAD_ACTION_ID） |
| T3.16 | 序数编码计算 | 手牌[ATTACK,ENERGY,ATTACK]、弃牌堆[ENERGY,ATTACK]、抽牌堆[ATTACK]：ATTACK 序数 [1,3,4]，ENERGY 序数 [1,2] |
| T3.17 | 手牌包含 BUILD | BUILD 在 hand_ids 中不出现（被过滤）；由调用方通过 action_to_policy_index 处理 |

## T4. AtteV1Converter — action ↔ policy index

| ID | 场景 | 预期 |
|----|------|------|
| T4.1 | `action_to_policy_index(None)` | 返回 0 |
| T4.2 | `action_to_policy_index(ATTACK)` | 返回 3 |
| T4.3 | `action_to_policy_index(BUILD, build_target=ATTACK)` | 返回 4 |
| T4.4 | `action_to_policy_index(BUILD, dismantle_target=ATTACK)` | 返回 5 |
| T4.5 | BUILD 同时指定 build_target 和 dismantle_target | 抛出 ValueError |
| T4.6 | `policy_index_to_action(0)` | 返回 (None, None, None) |
| T4.7 | `policy_index_to_action(21)` | 返回 (IRON_DEFENSE, None, None) |
| T4.8 | `policy_index_to_action(24)` | 抛出 IndexError |
| T4.9 | `build_cost(ATTACK)` | 返回 1 |

## T5. AtteV1PolicyModel — 初始化和前向传播

| ID | 场景 | 预期 |
|----|------|------|
| T5.1 | 实例化 | 不抛异常 |
| T5.2 | INITIAL_HIDDEN_STATE | 长度 32，全零 |
| T5.3 | forward batch=1，单卡手牌 | probs (1,24), log_probs (1,24), new_hidden (1,1,32) |
| T5.4 | probs 之和 | ≈ 1.0 |
| T5.5 | 空手牌（hand_ids=[]） | 不抛异常 |
| T5.6 | 空状态（status_ids=[]） | 不抛异常 |
| T5.7 | last_action=PAD=24 | 不抛异常 |
| T5.8 | mask_fn（mask 所有非 pass） | probs[0] ≈ 1.0 |
| T5.9 | temperature=0.0 | 确定性输出 |
| T5.10 | temperature=100.0 | 近似均匀分布 |
| T5.11 | e1 vocab=25 | embedding.weight shape (25, 32) |
| T5.12 | **全空输入**（所有序列为空，仅标量特征） | 不抛异常，probs 和 value 有合法输出值 |
| T5.13 | **batch=2，两样本序列长度不同**（样本A手牌2张，样本B手牌5张） | 正确处理 padding/对齐，probs (2,24), new_hidden (1,2,32) |
| T5.14 | **极端序列**（手牌 10 张，弃牌堆 20 张，抽牌堆 20 张） | 不抛异常，显存使用在合理范围 |

## T6. AtteV1ValueModel

| ID | 场景 | 预期 |
|----|------|------|
| T6.1 | 实例化 | 不抛异常 |
| T6.2 | INITIAL_HIDDEN_STATE | 长度 32，全零 |
| T6.3 | forward batch=1 | value (1,1), new_hidden (1,1,32) |
| T6.4 | 不同 hidden → 不同 value | value 随 hidden 变化 |

## T7. 权重共享

| ID | 场景 | 预期 |
|----|------|------|
| T7.1 | e1.weight shape | (25, 32) |
| T7.2 | Policy Head fc shape | (24, 32) |
| T7.3 | 修改 e1 权重后 forward | Policy Head 输出变化 |
| T7.4 | 一次训练 step（backward） | e1 权重梯度非零，两路梯度正确累加 |

## T8. AtteV1Adapter — ai_select_action

| ID | 场景 | 预期 |
|----|------|------|
| T8.1 | 首次调用 | 加载 policy model，返回合法动作 + new_hidden |
| T8.2 | 手牌 [ATTACK], energy=1 | 返回 mask_fn 过滤后的合法动作 |
| T8.3 | 空手牌 | 仅返回 pass |
| T8.4 | 相同 state+h 多次调用 | 返回相同动作 |
| T8.5 | 连续两回合 | hidden 正确传递 |
| T8.6 | 签名匹配 AiInterface | `ai_select_action(state, hidden_state)` |
| T8.7 | **权重文件缺失**（policy_weight_path 指向不存在的文件） | 抛出 RuntimeError，错误信息包含文件路径 |

## T9. AtteV1Adapter — ai_sample_action

| ID | 场景 | 预期 |
|----|------|------|
| T9.1 | 首次调用 temp=1.0 | 加载双模型，返回完整 tuple |
| T9.2 | temp > 0 | log_prob 为 float |
| T9.3 | temp=0.0 | log_prob=None，贪心 |
| T9.4 | value | 为 float，无 tanh |
| T9.5 | new_h_pol, new_h_val | 独立更新 |
| T9.6 | 签名匹配 AiInterface | `ai_sample_action(state, h_pol, h_val, temp, player_key)` |
| T9.7 | energy=0, hand=[BUILD], discard=[ATTACK] | build mask, 仅 dismantle 或 pass |

## T10. mask function

| ID | 场景 | 预期 |
|----|------|------|
| T10.1 | hand=[ATTACK], energy=2, discard=[ATTACK] | index 3/4/5 不 mask |
| T10.2 | BUILD 不在手 | 所有 build/dismantle mask |
| T10.3 | energy=0, discard=[ATTACK] | build 全 mask, dismantle ATTACK 合法 |
| T10.4 | discard=[] | 所有 dismantle mask |
| T10.5 | index 0 | 始终不 mask |

## T11. Registry

| ID | 场景 | 预期 |
|----|------|------|
| T11.1 | `list_ais()` 含 atte_v1 | 包含 |
| T11.2 | `get_ai_info("atte_v1.1")` | weight_path 为 list[str]（长度 2） |
| T11.3 | `parse_ai_name("atte_v1.3")` | ("atte_v1", 3) |
| T11.4 | `AiRegistry.get_ai_interface("atte_v1.1")` | AtteV1Adapter（8 参数） |
| T11.5 | `AiRegistry.get_ai_interface("rnn_v2.1")` | RnnAiAdapter（6 参数） |

## T12. TrainingRunnerFactory

| ID | 场景 | 预期 |
|----|------|------|
| T12.1 | `create("rnn_v2.1")` | V2TrainingRunner |
| T12.2 | `create("atte_v1.1")` | AtteV1TrainingRunner |
| T12.3 | `create("unknown.1")` | ValueError |

## T13. AtteV1TrainingRunner

| ID | 场景 | 预期 |
|----|------|------|
| T13.1 | run() 完整流程 | 三阶段完成，不抛异常 |
| T13.2 | 输出四个权重文件 | 均存在 |
| T13.3 | value_loss / policy_loss | 非 NaN |
| T13.4 | 空样本 | 跳过 |
| T13.5 | _load_samples JSON→AtteV1Input | steps[i]["input"] 为 AtteV1Input |

## T14. 样本格式

| ID | 场景 | 预期 |
|----|------|------|
| T14.1 | input.own.hand_ids | list[int] |
| T14.2 | 相邻 step last_action | step[i+1].last_action = step[i].action |
| T14.3 | 首 step last_action | -1 |

## T15. 集成测试

| ID | 场景 | 预期 |
|----|------|------|
| T15.1 | atte_v1 自对弈一局 | 两个 AtteV1Perspective |
| T15.2 | 奖励 | winner=1, loser=-1 |
| T15.3 | JSON 往返 | AtteV1Perspective ↔ JSON 一致 |
| T15.4 | run() 端到端 | 样本→训练→权重 |
| T15.5 | atte_v1 + rnn_v2 共存 | AiRegistry 双缓存 |
| T15.6 | SelfPlayRunner 路由 | atte_v1→AtteV1SampleCollector, rnn_v2→现有 Collector |
| T15.7 | 混合自对弈 | 不抛异常，各自格式生成 |
