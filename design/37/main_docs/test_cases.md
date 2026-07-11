# 设计 #37 — 测试用例集

## T1. CardType 枚举

| ID | 场景 | 预期 |
|----|------|------|
| T1.1 | `CardType.IRON_DEFENSE.value` | 返回 `"iron_defense"` |
| T1.2 | `CardType("iron_defense")` | 返回 `CardType.IRON_DEFENSE` |
| T1.3 | `len(CardType)` | 返回 8 |
| T1.4 | `get_card_def(CardType.IRON_DEFENSE)` | 返回 build_cost=2, display_name="严防死守", tags=("防",), resolvers=(StrongDefenseResolver, GrantGuardResolver) |

## T2. CardDef — BUILD 卡更新

| ID | 场景 | 预期 |
|----|------|------|
| T2.1 | `get_card_def(CardType.BUILD).display_name` | 返回 `"构建/拆解"` |
| T2.2 | `get_card_def(CardType.BUILD).description` | 包含"上半部分"和"下半部分"描述 |
| T2.3 | `get_card_def(CardType.BUILD).resolvers` | 返回 `(BuildResolver, DismantleResolver)` |
| T2.4 | `get_initial_draw_pile()` | 仍返回 7 张（无 IRON_DEFENSE，无 FLEX） |

## T3. Player — 新增方法

| ID | 场景 | 预期 |
|----|------|------|
| T3.1 | 弃牌堆 `[ATTACK, DEFENSE]`，调用 `remove_from_discard(ATTACK)` | 弃牌堆变为 `[DEFENSE]` |
| T3.2 | 弃牌堆 `[ATTACK, ATTACK]`，调用 `remove_from_discard(ATTACK)` | 弃牌堆变为 `[ATTACK]`（仅移除一个） |
| T3.3 | 弃牌堆 `[DEFENSE]`，调用 `remove_from_discard(ATTACK)` | 抛出 `ValueError` |
| T3.4 | 空弃牌堆，调用 `remove_from_discard(ATTACK)` | 抛出 `ValueError` |

## T4. BuildResolver — 构建放入抽牌堆

| ID | 场景 | 预期 |
|----|------|------|
| T4.1 | 玩家打出 BUILD + build_target=ATTACK, energy=2 | energy 减 1，抽牌堆新增 ATTACK，弃牌堆不变 |
| T4.2 | 玩家打出 BUILD + build_target=COUNTER, energy=2 | energy 减 2，抽牌堆新增 COUNTER |
| T4.3 | ctx.build_target 为 None | 不修改任何状态，不报错 |

## T5. DismantleResolver — 拆解

| ID | 场景 | 预期 |
|----|------|------|
| T5.1 | 弃牌堆 `[ATTACK, DEFENSE]`, dismantle_target=ATTACK | ATTACK 从弃牌堆移除，能量不变 |
| T5.2 | ctx.dismantle_target 为 None | 不修改任何状态，不报错 |
| T5.3 | 弃牌堆 `[ATTACK]`, dismantle_target=DEFENSE | 抛出 ValueError（由 Player.remove_from_discard 抛出） |

## T6. StrongDefenseResolver — 严防死守防御

| ID | 场景 | 预期 |
|----|------|------|
| T6.1 | 玩家无 FRAGILE | 获得 DEFENSE 状态，lifetime=2 |
| T6.2 | 玩家有 FRAGILE | 不获得 DEFENSE，details 记录被 FRAGILE 阻挡 |
| T6.3 | 玩家已有 DEFENSE(lifetime=1)，无 FRAGILE | 新增一个 DEFENSE(lifetime=2)，状态列表有 2 个 DEFENSE |

## T7. GameCore.play_round — BUILD 双模式

| ID | 场景 | 预期 |
|----|------|------|
| T7.1 | P1 打出 `[BUILD]`，`build_target_1=ATTACK`（无 dismantle_target） | 构建模式：能量-1，抽牌堆+ATTACK |
| T7.2 | P1 打出 `[BUILD]`，`dismantle_target_1=ATTACK`（无 build_target），弃牌堆有 ATTACK | 拆解模式：弃牌堆移除 ATTACK，能量不变 |
| T7.3 | P1 打出 `[BUILD]`，`build_target_1=ATTACK` 且 `dismantle_target_1=DEFENSE` | 抛出 ValueError（互斥） |
| T7.4 | P1 打出 `[BUILD]`，`build_target_1` 和 `dismantle_target_1` 都为 None | 抛出 ValueError（必须有且仅有一个） |
| T7.5 | P1 打出 `[BUILD]`，`build_target_1=COUNTER`，energy=1 | 抛出 ValueError（能量不足） |
| T7.6 | P1 打出 `[BUILD]`，`dismantle_target_1=ATTACK`，弃牌堆为空 | 抛出 ValueError（弃牌堆无此卡） |
| T7.7 | P1 打出 `[BUILD]`，energy=0，`build_target_1=ATTACK` | 抛出 ValueError（能量为 0 无法构建） |
| T7.8 | P1 打出 `[BUILD]`，energy=0，`dismantle_target_1=ATTACK`（弃牌堆有 ATTACK） | **合法**（拆解不消耗能量） |
| T7.9 | P1 打出 `[ATTACK]`，`build_target_1` 或 `dismantle_target_1` 非 None | 抛出 ValueError |

## T8. 严防死守卡牌效果

| ID | 场景 | 预期 |
|----|------|------|
| T8.1 | P1 打出 IRON_DEFENSE，无 FRAGILE | 获得 DEFENSE(lifetime=2) |
| T8.2 | P1 上回合打出 DEFENSE（带"防"标签），本回合打出 IRON_DEFENSE | 获得 DEFENSE(lifetime=2) + GUARD(lifetime=1) |
| T8.3 | P1 首回合打出 IRON_DEFENSE | 仅获得 DEFENSE(lifetime=2)，无 GUARD（上回合无历史） |
| T8.4 | P1 有 FRAGILE | DEFENSE 和 GUARD 均被阻挡 |

## T9. Converter — 24-d 输出映射

| ID | 场景 | 预期 |
|----|------|------|
| T9.1 | `action_to_policy_index(None)` | 返回 0 |
| T9.2 | `action_to_policy_index(ATTACK, build_target=None, dismantle_target=None)` | 返回 3 |
| T9.3 | `action_to_policy_index(BUILD, build_target=ATTACK)` | 返回 4 |
| T9.4 | `action_to_policy_index(BUILD, dismantle_target=ATTACK)` | 返回 5 |
| T9.5 | `action_to_policy_index(BUILD, build_target=ATTACK, dismantle_target=DEFENSE)` | 抛出 ValueError（互斥） |
| T9.6 | `policy_index_to_action(0)` | 返回 `(None, None, None)` |
| T9.7 | `policy_index_to_action(4)` | 返回 `(BUILD, ATTACK, None)` |
| T9.8 | `policy_index_to_action(5)` | 返回 `(BUILD, None, ATTACK)` |
| T9.9 | `policy_index_to_action(21)` | 返回 `(IRON_DEFENSE, None, None)` |
| T9.10 | `policy_index_to_action(22)` | 返回 `(BUILD, IRON_DEFENSE, None)` |
| T9.11 | `policy_index_to_action(23)` | 返回 `(BUILD, None, IRON_DEFENSE)` |
| T9.12 | `POLICY_OUTPUT_SIZE` | 返回 24 |

## T10. Converter — 62-d 输入向量

| ID | 场景 | 预期 |
|----|------|------|
| T10.1 | `build_input_vector(state)` | 返回长度 62 的 list[float] |
| T10.2 | 手牌包含 1 张 IRON_DEFENSE | 输入向量 index 13（己方手牌 IRON_DEFENSE 计数）= 1.0 |
| T10.3 | INPUT_SIZE | 返回 62 |
| T10.4 | NUM_CARD_TYPES | 返回 8 |
| T10.5 | 手牌 8 种卡牌的 count 段 | 每段长度为 8（非 7） |

## T11. AppManager — mask_fn 24-d

| ID | 场景 | 预期 |
|----|------|------|
| T11.1 | hand=[ATTACK], energy=2, discard=[ATTACK] | mask_fn: index 3 (play ATTACK) 合法；index 4 (build ATTACK) 合法；index 5 (dismantle ATTACK) 合法 |
| T11.2 | hand=[ATTACK], BUILD 不在手, energy=2 | 所有 build/dismantle 索引被 mask（无 BUILD 卡） |
| T11.3 | hand=[BUILD], energy=0, discard=[ATTACK] | index 5 (dismantle ATTACK) 合法；所有 build 索引被 mask；play 索引被 mask（ATTACK 不在手） |
| T11.4 | hand=[BUILD], energy=1, discard=[] | 所有 dismantle 索引被 mask（弃牌堆空） |
| T11.5 | hand=[BUILD, IRON_DEFENSE], energy=2, discard=[ATTACK] | play IRON_DEFENSE(21) 合法；build IRON_DEFENSE(22) 合法（cost 2 ≤ energy 2） |

## T12. PlayerHistoryItem — dismantle_target

| ID | 场景 | 预期 |
|----|------|------|
| T12.1 | `PlayerHistoryItem(ATTACK)` | card=ATTACK, build_target=None, dismantle_target=None |
| T12.2 | `PlayerHistoryItem(BUILD, build_target=ATTACK)` | card=BUILD, build_target=ATTACK, dismantle_target=None |
| T12.3 | `PlayerHistoryItem(BUILD, dismantle_target=DEFENSE)` | card=BUILD, build_target=None, dismantle_target=DEFENSE |

## T13. 前端 — BUILD 选择弹窗

| ID | 场景 | 预期 |
|----|------|------|
| T13.1 | 选中构建/拆解卡，能量=2，弃牌堆=[ATTACK] | "构建"和"拆解"按钮均可点击 |
| T13.2 | 选中构建/拆解卡，能量=0 | "构建"按钮灰显，"拆解"按钮可点击（若弃牌堆非空） |
| T13.3 | 选中构建/拆解卡，弃牌堆为空 | "拆解"按钮灰显，"构建"按钮可点击（若能量>0） |
| T13.4 | 选中构建/拆解卡，能量=0 且弃牌堆为空 | 两个按钮均灰显 |
| T13.5 | 点击"取消" | 弹窗关闭，不发出任何请求 |

## T14. 前端 — 新卡显示

| ID | 场景 | 预期 |
|----|------|------|
| T14.1 | 手牌中有 IRON_DEFENSE | 显示"严防死守"名称和对应图标 |
| T14.2 | API 返回 cards 包含 `"iron_defense"` | 前端不报错，正确渲染 |
