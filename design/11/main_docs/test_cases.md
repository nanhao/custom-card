# Test Cases — Game v2 (intention #11)

---

## 1. CardType + CardDef

### TC-CD-01: CardType 枚举包含7个值
- **Given**: 导入 CardType 枚举
- **Then**: 包含 ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, BUILD, FLEX 共7个成员

### TC-CD-02: get_card_def 返回正确的优先级
- **Given**: 分别传入 CardType.ATTACK / DEFENSE / FLEX
- **Then**: CardDef.priority 依次为 10 / 11 / 11; FLEX 有两个效果(优先级11和10)

### TC-CD-03: get_card_def 返回正确的构建消耗
- **Given**: 分别传入 CardType.COUNTER / FLEX / ATTACK
- **Then**: CardDef.build_cost 依次为 2 / 2 / 1

### TC-CD-04: get_initial_draw_pile 返回标准初始牌堆
- **Given**: 调用 get_initial_draw_pile()
- **Then**: 返回7张卡: 2×ATTACK, 1×DEFENSE, 1×HEAVY, 1×COUNTER, 1×ENERGY, 1×BUILD; 不包含 FLEX

### TC-CD-05: CardDef 不可变
- **Given**: 获取一个 CardDef 实例
- **When**: 尝试修改其 priority 字段
- **Then**: 抛出 FrozenInstanceError

---

## 2. Player

### TC-PL-01: 默认值
- **Given**: 创建 Player()
- **Then**: hp=20, energy=0, statuses=[], hand=[], draw_pile=[], discard_pile=[], history=[]

### TC-PL-02: take_damage 正常扣血
- **Given**: Player(hp=20)
- **When**: take_damage(5)
- **Then**: hp=15

### TC-PL-03: take_damage 不低于0
- **Given**: Player(hp=3)
- **When**: take_damage(10)
- **Then**: hp=0

### TC-PL-04: add_energy 正常增加
- **Given**: Player(energy=0)
- **When**: add_energy(2)
- **Then**: energy=2

### TC-PL-05: add_energy 上限为5
- **Given**: Player(energy=4)
- **When**: add_energy(3)
- **Then**: energy=5

### TC-PL-06: add_energy 已达上限后无效
- **Given**: Player(energy=5)
- **When**: add_energy(1)
- **Then**: energy=5

### TC-PL-07: has_status 正确判断
- **Given**: Player(statuses=["defense"])
- **When**: has_status("defense") / has_status("联防")
- **Then**: 前者 True, 后者 False

### TC-PL-08: play_card 从hand中移除
- **Given**: Player(hand=[ATTACK, DEFENSE])
- **When**: play_card(ATTACK)
- **Then**: hand=[DEFENSE]

### TC-PL-09: play_card 不存在的card抛错
- **Given**: Player(hand=[ATTACK])
- **When**: play_card(DEFENSE)
- **Then**: 抛出 ValueError

---

## 3. CardResolver — 优先级结算

### TC-RS-01: 优先级11先于优先级10结算
- **Given**: P1出 DEFENSE(优先级11), P2出 ATTACK(优先级10); P1/P2均无状态
- **When**: resolve()
- **Then**: P1先获得防御状态, P2的攻击被防御状态阻挡, damage_1=0, damage_2=0

### TC-RS-02: 同优先级同时结算
- **Given**: P1出 ATTACK(10), P2出 ATTACK(10); 双方无状态; P1连续攻击计数=0, P2连续攻击计数=1
- **When**: resolve()
- **Then**: damage_1 = max(P2攻击×2, 0) = 2; damage_2 = max(P1攻击×1, 0) = 1; 双方HP分别扣2和1(非串联)

### TC-RS-03: 攻守兼备双优先级独立结算
- **Given**: P1出 FLEX, P2出 ATTACK(10); P1无防御状态, P2无防御状态
- **When**: resolve()
- **Then**: 优先级11阶段: P1获得防御状态; 优先级10阶段: P1攻守兼备攻击部分检测P2无防御→P2扣1; P2攻击被P1防御阻挡→P1扣0; damage_1=0, damage_2=1

### TC-RS-04: 防御获得联防状态
- **Given**: P1 statuses=["defense"]; P1出 DEFENSE, P2出 HEAVY
- **When**: resolve()
- **Then**: P1获得联防状态, P2重击被联防阻挡, damage_1=0

### TC-RS-05a: 反击 vs 普通攻击 — 弹伤触发
- **Given**: P1出 COUNTER, P2出 ATTACK; P1/P2无状态
- **When**: resolve()
- **Then**: P1获得防御状态; P2受到反击弹伤1点; damage_1=0, damage_2=1

### TC-RS-05b: 反击 vs 重击 — 弹伤触发
- **Given**: P1出 COUNTER, P2出 HEAVY; P1/P2无状态
- **When**: resolve()
- **Then**: P1获得防御状态; P2受到反击弹伤1点; P1受到重击1点(防御挡不住重击); damage_1=1, damage_2=1

### TC-RS-05c: 反击 vs 防御 — 无弹伤
- **Given**: P1出 COUNTER, P2出 DEFENSE; 双方无状态
- **When**: resolve()
- **Then**: P1获得防御状态; P2获得防御状态; 双方均无伤害; damage_1=0, damage_2=0

### TC-RS-05d: 反击 vs 聚能 — 无弹伤
- **Given**: P1出 COUNTER, P2出 ENERGY; 双方无状态
- **When**: resolve()
- **Then**: P1获得防御状态; P2 energy_delta=1; 无弹伤; damage_1=0, damage_2=0

### TC-RS-06: 聚能增加能量
- **Given**: P1 energy=2, P1出 ENERGY
- **When**: resolve()
- **Then**: energy_delta_1=1; 最终 energy=3

### TC-RS-07: 聚能达上限无效
- **Given**: P1 energy=5, P1出 ENERGY
- **When**: resolve()
- **Then**: energy_delta_1=0; 最终 energy=5

### TC-RS-08: 构建不产生伤害和能量
- **Given**: P1出 BUILD, P2出 ATTACK(无状态)
- **When**: resolve()
- **Then**: damage_1=1(P2攻击), damage_2=0; energy_delta 均为0; statuses_granted_2为空。构建逻辑由GameCore在结算后处理

### TC-RS-09a: 攻击连续计数常规
- **Given**: P1 history=[ATTACK, ATTACK]; P1出 ATTACK, P2无防御状态
- **When**: resolve() (传入 p1_attack_streak=2)
- **Then**: P2受到 damage=min(2+1,3)=3

### TC-RS-09b: 攻击中断 — 末尾非ATTACK时不计数
- **Given**: P1 history=[ATTACK, DEFENSE, ATTACK]; P1出 ATTACK, P2无防御
- **When**: resolve() (传入 p1_attack_streak, 仅末尾连续)
- **Then**: 连续攻击计数=1 (仅最后一个ATTACK), P2受到 damage=min(1+1,3)=2

### TC-RS-09c: 空 history — 连续计数为0
- **Given**: P1 history=[]; P1出 ATTACK, P2无防御
- **When**: resolve() (传入 p1_attack_streak=0)
- **Then**: P2受到 damage=min(0+1,3)=1

### TC-RS-09d: COUNTER 不延续攻击计数
- **Given**: P1 history=[ATTACK, COUNTER]; P1出 ATTACK, P2无防御
- **When**: resolve() (传入 p1_attack_streak=0, COUNTER隔断)
- **Then**: 连续攻击计数=0, P2受到 damage=min(0+1,3)=1

### TC-RS-10: 重击被联防阻挡
- **Given**: P1 statuses=["联防"], P2出 HEAVY
- **When**: resolve()
- **Then**: P1不受伤害, damage_1=0

### TC-RS-11: 重击不被普通防御阻挡
- **Given**: P1 statuses=["defense"], P2出 HEAVY
- **When**: resolve()
- **Then**: P1受1点伤害, damage_1=1 (普通防御挡不住重击)

---

## 4. GameCore — cycle 机制

### TC-GC-01: start_game 初始化正确
- **Given**: 创建 GameCore, mode="pvp"
- **When**: start_game()
- **Then**: P1/P2 hp=20, energy=0, hand各有7张(从抽牌堆抽的), draw_pile空, discard_pile空; round_number=1, cycle_number=1, round_in_cycle=1

### TC-GC-02: play_round 后弃牌进discard_pile
- **Given**: 游戏已开始, P1 hand包含ATTACK, P2 hand包含DEFENSE
- **When**: play_round([ATTACK], [DEFENSE])
- **Then**: P1 discard_pile=[ATTACK], P2 discard_pile=[DEFENSE]; P1/P2 hand各减少1张

### TC-GC-03: 5回合后触发 cycle 末扣血
- **Given**: 游戏已开始, 当前为第5回合(round_in_cycle=5)
- **When**: play_round() 完成第5回合后
- **Then**: P1和P2 HP各减1; 触发_start_cycle()进入下一cycle, round_in_cycle=1

### TC-GC-04: cycle 末扣血可能导致游戏结束
- **Given**: P1 hp=1, P2 hp=1, 第5回合结算后无人死亡
- **When**: 触发 cycle 末扣血
- **Then**: 双方hp=0, is_over=True, winner="draw"

### TC-GC-05: 新 cycle 开始时回收手牌+洗牌+重抽
- **Given**: P1 hand剩下3张; P1 discard_pile有2张; 第5回合结束
- **When**: 进入新 cycle _start_cycle()
- **Then**: P1 hand中3张进入discard_pile→discard_pile全部进入draw_pile→洗牌→抽7张; P1 hand=7, draw_pile=0

### TC-GC-06: 新 cycle 保留连续攻击计数
- **Given**: P1 history=[ATTACK, ATTACK]; 第5回合结束进入新 cycle
- **When**: 新 cycle _start_cycle()
- **Then**: P1 history 仍为 [ATTACK, ATTACK]; 连续攻击计数保留

### TC-GC-07: 游戏仅在HP归零时结束
- **Given**: P1 hp=10, P2 hp=15, 已进行50 cycle
- **When**: 检查 _check_game_end()
- **Then**: is_over=False (不再有回合数上限)

### TC-GC-08a: HP归零立即结束
- **Given**: P1 hp=5, P2出ATTACK造成6伤害(含连续攻击)
- **When**: play_round()
- **Then**: P1 hp=0, is_over=True, winner="2"

### TC-GC-08b: 同回合双方同时攻击致死→平局
- **Given**: P1 hp=1, P2 hp=1; P1出 ATTACK, P2出 ATTACK; 双方均无防御状态, 攻击计数均为0
- **When**: play_round([ATTACK], [ATTACK])
- **Then**: 双方各受1伤→hp均归零→is_over=True, winner="draw"

### TC-GC-09: game over后play_round抛错
- **Given**: 游戏已结束(is_over=True)
- **When**: play_round([ATTACK], [DEFENSE])
- **Then**: 抛出 ValueError("game is over")

### TC-GC-10: get_state 返回完整状态
- **Given**: 游戏进行中, cycle=2, round_in_cycle=3, round=8
- **When**: get_state()
- **Then**: 返回dict包含 cycle=2, round_in_cycle=3, round=8, players.{1,2}.{hp/energy/statuses/hand/draw_pile_count/discard_pile_count}, 不暴露draw_pile/discard_pile具体内容

### TC-GC-11: 能量为0时无法打出构建卡
- **Given**: P1 energy=0, P1 hand包含 BUILD
- **When**: play_round([BUILD], [ATTACK])
- **Then**: 抛出 ValueError (能量不足无法打构建卡)

---

## 5. AI Model

### TC-MD-01: GRUPolicyModel 维度正确
- **Given**: 创建 GRUPolicyModel()
- **When**: forward(x(1,1,25), hidden(1,1,32))
- **Then**: probs shape=(1,14), log_probs shape=(1,14), new_hidden shape=(1,1,32)

### TC-MD-02: GRUValueModel 维度正确
- **Given**: 创建 GRUValueModel()
- **When**: forward(x(1,1,25), hidden(1,1,32))
- **Then**: value shape=(1,1), new_hidden shape=(1,1,32)

### TC-MD-03: INITIAL_HIDDEN_STATE 长度为32
- **Given**: 访问 GRUPolicyModel.INITIAL_HIDDEN_STATE
- **Then**: len=32, 全部为0.0

### TC-MD-04: mask_fn 正确屏蔽7卡牌中的非法动作
- **Given**: 模型forward, mask_fn将energy/build/flex以外的logits置为-inf
- **When**: forward() temperature=1.0
- **Then**: probs 中 attack/defense/heavy/counter(0-3)概率为0, energy/build/flex(4-6)概率>0

### TC-MD-05: mask_fn 后7维屏蔽能量不足构建目标
- **Given**: 模型forward, player energy=2, mask_fn 将后7维中 build_cost=2 的卡(counter=3, flex=6)对应索引 logits 置为 -inf
- **When**: forward() temperature=1.0
- **Then**: probs[10](counter-build)和probs[13](flex-build)概率为0

### TC-MD-06: 训练时mask策略 — 前段屏蔽手牌, 后段不屏蔽
- **Given**: 模型forward, hand=[ATTACK, DEFENSE](仅2种), mask_fn = 训练模式(前段屏蔽不在手牌的卡, 后段不屏蔽)
- **When**: forward() temperature=1.0
- **Then**: probs[0]和probs[1]概率>0 (手牌中有), probs[2-6]概率=0 (手牌中无); 后7维 probs[7-13] 均>0 (不屏蔽)

---

## 6. Converter

### TC-CV-01: NUM_CARD_TYPES=7, INPUT_SIZE=25
- **Given**: 访问 Converter 类属性
- **Then**: NUM_CARD_TYPES=7, INPUT_SIZE=25

### TC-CV-02: card_to_index 映射完整
- **Given**: "attack"/"defense"/"heavy"/"counter"/"energy"/"build"/"flex"
- **Then**: 依次返回 0/1/2/3/4/5/6

### TC-CV-03: index_to_card 映射完整
- **Given**: 0/1/2/3/4/5/6
- **Then**: 依次返回对应卡牌字符串; 索引-1或7抛 IndexError

### TC-CV-04: build_input_vector 长度为25
- **Given**: 有效game state dict
- **When**: build_input_vector(state)
- **Then**: 返回list长度为25

### TC-CV-05: build_input_vector 索引23-24正确
- **Given**: state 中 cycle=3, round_in_cycle=2
- **Then**: input[23]=2.0 (round_in_cycle), input[24]=3.0 (cycle)

### TC-CV-06: build_input_vector 手牌计数正确
- **Given**: P2 hand=[ATTACK, ATTACK, DEFENSE, HEAVY, COUNTER, ENERGY, BUILD]
- **When**: build_input_vector(state, player_key="2")
- **Then**: input[0]=2.0, input[1]=1.0, input[2]=1.0, input[3]=1.0, input[4]=1.0, input[5]=1.0, input[6]=0.0

---

## 7. SessionStore — 序列化

### TC-SS-01: save/load 轮转 Player 新字段
- **Given**: Player(energy=3, statuses=["defense"], draw_pile=[FLEX], discard_pile=[ATTACK])
- **When**: save→load
- **Then**: 恢复后 energy=3, statuses=["defense"], draw_pile=[FLEX], discard_pile=[ATTACK]

### TC-SS-02: ai_state 长度32正确序列化
- **Given**: ai_state=[0.0]*32
- **When**: save→load
- **Then**: 恢复后长度=32

### TC-SS-03: 旧存档加载失败
- **Given**: 旧格式存档文件(无energy/statuses/piles字段, ai_state长度8)
- **When**: load()
- **Then**: 抛出 ValueError (缺少必填键)

---

## 8. Self-Play + Training

### TC-SP-01: selfplay 生成 25-d input 样本
- **Given**: 使用 rnn_v2.1 运行 selfplay, sample_output 已设置
- **When**: 完成至少一局
- **Then**: 样本 JSON 中 step["input"] 长度为25

### TC-SP-02: selfplay 样本 action 为7种卡牌之一
- **Given**: selfplay 运行完成
- **Then**: 所有 step["action"] 值在 {attack,defense,heavy,counter,energy,build,flex} 中

### TC-SP-03: selfplay 样本含 build_target 字段
- **Given**: selfplay 运行中AI出了构建卡
- **When**: 样本写入
- **Then**: 对应 step 包含 "build_target" 字段, 值为7种卡牌之一

### TC-SP-04: train 接受 25-d input + build_target 样本
- **Given**: 25-d input 样本文件(含build_target字段)
- **When**: 执行 train --model rnn_v2.1
- **Then**: 训练正常完成, 无维度不匹配错误

### TC-SP-05: train 拒绝 14-d input 旧样本
- **Given**: 14-d input 旧格式样本
- **When**: 执行 train
- **Then**: 抛出 RuntimeError (维度不匹配)

### TC-SP-06: 多 BUILD 连续构建能量递减
- **Given**: 经多轮构建后, P1 hand=[BUILD, BUILD], P1 energy=3
- **When**: 连续打出两张 BUILD 分别构建 cost=1 的 ATTACK 和 DEFENSE
- **Then**: 能量 3→2→1; 弃牌堆增加 ATTACK 和 DEFENSE; hand 剩余 card_count-2

---

## 9. AppManager

### TC-AM-01: create_game PVE 初始化 hidden_state=[0.0]*32
- **Given**: mode="pve", ai_name="rnn_v2.1"
- **When**: create_game()
- **Then**: 存档中 ai_state 长度为32, 全部为0.0

### TC-AM-02: ai_select_action 适配7卡牌
- **Given**: state 中 P2 hand 包含 FLEX
- **When**: ai_select_action() 被调用
- **Then**: AI 可能在7种合法卡牌中选到 FLEX; 不会选到hand中没有的卡

### TC-AM-03: mask_fn 双段屏蔽(手牌+构建目标)
- **Given**: P2 hand=[ENERGY, BUILD], P2 energy=1
- **When**: _make_mask_fn 生成 mask_fn
- **Then**: 前7维: ATTACK/DEFENSE/HEAVY/COUNTER/FLEX 对应索引置为 -inf\n后7维: 仅 ENERGY(4, build_cost=1)不屏蔽, 其余均屏蔽(build_cost=1或2但 energy=1只够构建cost=1的卡)

### TC-AM-03b: mask_fn energy=0时屏蔽BUILD
- **Given**: P2 hand=[BUILD, ATTACK], P2 energy=0
- **When**: _make_mask_fn 生成 mask_fn
- **Then**: 前7维: BUILD(索引5)的logit置为 -inf, ATTACK(0)不屏蔽

### TC-AM-04: ai_sample_action 返回构建目标
- **Given**: AI energy=3, AI hand含 BUILD
- **When**: ai_sample_action() 选到 BUILD(前7维索引5)
- **Then**: 返回的 build_target_str 非None, 为能量足以构建的卡之一

---

## 10. API

### TC-API-01: GET /api/games/{id} 返回新字段
- **Given**: 游戏进行中
- **When**: GET state
- **Then**: state 包含 round/cycle/round_in_cycle/players.{1,2}.{energy,statuses,draw_pile_count,discard_pile_count}

### TC-API-02: POST /api/games/{id}/play PVE 模式 32-d hidden state
- **Given**: PVE 游戏, P1 选 ATTACK
- **When**: POST play
- **Then**: AI 返回有效卡牌, 存档中 ai_state 长度=32

### TC-API-03: RoundResult 序列化含新字段
- **Given**: 回合(round)结算完成
- **When**: round_result_to_dict()
- **Then**: 返回dict包含 round/cycle/round_in_cycle/energy_1_after/energy_2_after/statuses_1/statuses_2

---

## 11. Non-Regression（显式声明不回归）

### TC-NR-01: 旧测试文件全部不兼容
- **Given**: 变更前全部测试套件
- **Then**: 以下文件引用了旧 CardType(4值)、旧 Player 结构、旧 GameCore(7回合限制)、旧模型维度, 必须重写或删除:
  - `tests/unit/test_card.py`
  - `tests/unit/test_player.py`
  - `tests/unit/test_game_core.py`
  - `tests/unit/test_resolver.py`
  - `tests/unit/test_ai_model.py`
  - `tests/unit/test_ai_model_split.py`
  - `tests/unit/test_converter.py`
  - `tests/unit/test_app_manager.py`
  - `tests/unit/test_app_manager_selfplay.py`
  - `tests/unit/test_serializers.py`
  - `tests/unit/test_session_store.py`
  - `tests/unit/test_selfplay_runner.py`
  - `tests/unit/test_selfplay_perspective.py`
  - `tests/integration/test_ai_api.py`
  - `tests/integration/test_controller_api.py`
  - `tests/integration/test_full_game.py`
  - `tests/integration/test_selfplay.py`
  - `tests/integration/test_selfplay_samples.py`
  - `tests/integration/test_selfplay_samples_output.py`

























