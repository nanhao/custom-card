# Design #28: 测试用例

所有测试用例使用文字和符号描述，不包含代码。

---

## 一、模型层（model.py）

### T1.1 POLICY_OUTPUT_SIZE 常量
- **给定**：导入模块
- **断言**：`POLICY_OUTPUT_SIZE == 14`

### T1.2 GRUPolicyModel 输出维度
- **给定**：创建 GRUPolicyModel 实例
- **断言**：`fc_policy` 层输出维度为 14

### T1.3 forward 输出 logits 维度
- **给定**：输入 x(1,1,42)、hidden(1,1,32)
- **断言**：probs 形状为 (1,14)，log_probs 形状为 (1,14)

---

## 二、转换器层（converter.py）

### T2.1 _CARD_MAP 索引验证
- **断言**：
  - `_CARD_MAP` 中无索引 0（pass 不在映射表中）
  - BUILD.play_index = None, build_index = 1
  - ATTACK.play_index = 2, build_index = 3
  - ... FLEX.play_index = 12, build_index = 13

### T2.2 _INDEX_TO_ACTION 包含 pass
- **断言**：
  - `_INDEX_TO_ACTION[0]` → (None, None)
  - `_INDEX_TO_ACTION[1]` → (BUILD, BUILD)
  - 共 14 个条目（0~13）

### T2.3 action_to_policy_index — pass
- **给定**：card=None, build_target=None
- **断言**：返回 0

### T2.4 action_to_policy_index — 纯出牌
- **给定**：card=ATTACK, build_target=None
- **断言**：返回 2

### T2.5 action_to_policy_index — 构建
- **给定**：card=BUILD, build_target=COUNTER
- **断言**：返回 9

### T2.6 policy_index_to_action — pass
- **给定**：index=0
- **断言**：返回 (None, None)

### T2.7 policy_index_to_action — 纯出牌
- **给定**：index=6
- **断言**：返回 (HEAVY, None)

### T2.8 policy_index_to_action — 构建
- **给定**：index=13
- **断言**：返回 (BUILD, FLEX)

### T2.9 policy_index_to_action — 越界校验
- **给定**：index=-1 或 index=14
- **断言**：抛出 IndexError

### T2.10 build_cost 不变
- **给定**：card_type=COUNTER
- **断言**：build_cost 返回 2（与 CardDef 一致）

### T2.11 _action_item_to_index — pass 历史项
- **给定**：PlayerHistoryItem(card=None, build_target=None)
- **断言**：返回 0

---

## 三、游戏核心层（game_core.py）

### T3.1 一方 pass，另一方出攻击
- **给定**：P1 空列表（pass），P2 [ATTACK]，P1 无防御状态
- **断言**：
  - P1 手牌不变（无消耗）
  - P1 弃牌堆不变
  - P2 手牌减少 1 张 ATTACK
  - P2 弃牌堆新增 1 张 ATTACK
  - P1 掉血（受攻击伤害）
  - RoundResult.cards_1 = []
  - 游戏继续

### T3.2 一方 pass，另一方出防御
- **给定**：P1 空列表（pass），P2 [DEFENSE]
- **断言**：
  - 双方均不掉血
  - P2 获得防御状态
  - RoundResult.cards_1 = []

### T3.3 双方都 pass
- **给定**：双方空列表
- **断言**：
  - 双方手牌不变、弃牌堆不变
  - 双方 HP 不变
  - RoundResult.cards_1 = [], cards_2 = []
  - 回合正常推进（round_number +1, round_in_cycle +1）

### T3.4 pass 重置连续攻击计数
- **给定**：P1 连续出了 2 回合 ATTACK，然后本回合 pass
- **断言**：
  - P1 的 get_consecutive_attack_count() 在下回合为 0

### T3.5 P1 pass，P2 出 BUILD
- **给定**：P1 空列表（pass），P2 [BUILD] + build_target=ATTACK，P2 能量 2
- **断言**：
  - P1 不受影响
  - P2 能量减少 1（BUILD 消耗）
  - P2 手牌 BUILD 移除，弃牌堆新增 BUILD
  - 成功构建的 ATTACK 进入 P2 弃牌堆

### T3.6 pass 不影响对手状态结算
- **给定**：P1 空列表（pass），P2 空列表（pass），轮末（round_in_cycle=3）
- **断言**：
  - 双方各掉 1 滴血（轮末流血）
  - cycle_number +1
  - 新轮开始，双方抽 5 张牌

### T3.7 pass 后游戏正常结束
- **给定**：P1 HP=1，P2 空列表（pass），P1 空列表（pass），轮末（round_in_cycle=3）
- **断言**：
  - 双方各掉 1 滴血（轮末流血）
  - P1 HP 降为 0
  - 游戏结束，winner="2" 或 "draw"（取决于 P2 HP）

### T3.8 P1 pass，P2 出重击，P1 有防御状态
- **给定**：P1 空列表（pass），P1 有防御状态，P2 [HEAVY]
- **断言**：
  - P1 掉血（重击无法被防御状态阻挡）
  - P1 获得脆弱状态

### T3.9 P1 pass，P2 出重击，P1 有联防状态
- **给定**：P1 空列表（pass），P1 有联防状态，P2 [HEAVY]
- **断言**：
  - P1 不掉血

---

## 四、App 层（app_manager.py）

### T4.1 mask_fn — pass 永不被屏蔽
- **给定**：mask_fn = _make_mask_fn(hand=[], energy=0)，logits 中索引 0 初始值为任意有限值
- **断言**：屏蔽后 masked[0, 0] ≠ -inf

### T4.2 mask_fn — 14 维偶数位（纯出牌）屏蔽逻辑
- **给定**：hand=[ATTACK, DEFENSE]，energy=3
- **断言**：
  - 索引 2（attack）✓ 不被屏蔽
  - 索引 4（defense）✓ 不被屏蔽
  - 索引 6（heavy）被屏蔽为 -inf
  - 索引 8（counter）被屏蔽为 -inf

### T4.3 mask_fn — 14 维奇数位（构建）屏蔽逻辑
- **给定**：hand=[BUILD, ATTACK]，energy=1
- **断言**：
  - 索引 3（build→attack, cost=1）✓ 不被屏蔽
  - 索引 9（build→counter, cost=2）被屏蔽为 -inf
  - 索引 13（build→flex, cost=2）被屏蔽为 -inf

### T4.4 mask_fn — 无 BUILD 卡时所有构建位被屏蔽
- **给定**：hand=[ATTACK]，energy=5
- **断言**：
  - 所有奇数索引（1,3,5,7,9,11,13）被屏蔽为 -inf

### T4.5 ai_select_action 返回 pass
- **给定**：模型权重使得输出概率最大在索引 0
- **断言**：
  - 返回 (None, None, hidden_state)
  - 不抛出异常

### T4.6 ai_sample_action 采样到 pass
- **给定**：temperature=1.0，模型输出分布
- **断言**：
  - 可能采样到 card_type=None, build_target=None
  - log_prob 正确记录

---

## 五、API 层（controller/api.py）

### T5.1 POST /play — cards_1 为空数组（pass）
- **给定**：活跃游戏，POST body `{"cards_1": []}`
- **断言**：
  - 返回 200（非 400 MISSING_CARDS_1）
  - round_result.cards_1 为 []

### T5.2 POST /play — cards_2 为空数组（PVP pass）
- **给定**：活跃 PVP 游戏，POST body `{"cards_1": ["attack"], "cards_2": []}`
- **断言**：
  - 返回 200
  - round_result.cards_2 为 []

### T5.3 PVE — AI 返回 pass
- **给定**：AI 选择 pass（card_type=None）
- **断言**：
  - cards_2 为空数组
  - 游戏正常进行，返回 200

---

## 六、玩家历史（player.py）

### T6.1 PlayerHistoryItem 支持 card=None
- **给定**：PlayerHistoryItem(card=None, build_target=None)
- **断言**：
  - card 属性为 None
  - build_target 属性为 None

### T6.2 get_consecutive_attack_count — pass 打断连续攻击
- **给定**：history = [ATTACK, ATTACK, pass_item(card=None), ATTACK]
- **断言**：get_consecutive_attack_count() 返回 1

---

## 七、RoundResult 序列化

### T7.1 to_dict — 空 cards 序列化
- **给定**：RoundResult(cards_1=[], cards_2=[ATTACK], ...)
- **断言**：序列化后 `"cards_1": []`, `"cards_2": ["attack"]`

### T7.2 to_dict — 双方都空
- **给定**：RoundResult(cards_1=[], cards_2=[], ...)
- **断言**：序列化后 `"cards_1": [], "cards_2": []`

---

## 八、前端

### T8.1 Pass 按钮可见性
- **断言**：select 阶段和 pvp_pass 阶段均显示 Pass 按钮

### T8.2 Pass 按钮功能
- **给定**：select 阶段，用户点击 Pass 按钮
- **断言**：调用 onPass 回调

### T8.3 PVE pass 流程
- **给定**：用户点击 Pass
- **断言**：`POST /play` 发送 `cards_1: []`，返回正常结果

### T8.4 PVP 双方 pass
- **给定**：P1 点 Pass → P2 点 Pass
- **断言**：双方手牌不变，回合推进，结果面板显示双方均为"不出牌"

### T8.5 reveal 阶段 pass 显示
- **给定**：round_result.cards_1 为空数组
- **断言**：结果面板 P1 位置显示"不出牌"文字而非卡牌图标

---

## 九、回归测试

### T9.1 正常出牌流程不受影响
- **断言**：所有已有测试用例通过（card 映射索引 +1 已由设计保证语义一致）

### T9.2 旧版权重文件兼容性
- **给定**：13 维输出的旧版权重文件
- **断言**：加载报错（维度不匹配），用户需重新训练
