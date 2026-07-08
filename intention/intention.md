# intention_id=1
这是一个自定义卡牌游戏项目。
游戏流程：
2人对抗。每人初始手牌7张（3攻击2防御1重击1反击）、血量3点。每回合双方各出一张牌，出牌时面朝下扣牌，双方都口号牌后同时翻牌，进行结算，结算完成后弃掉当前回合出的牌并开始下一回合，直到游戏结束。
游戏结束条件：
一方或双方血量降到0，或者7个回合结束
胜者判断：
若有且仅有一方血量为0，则血量不为0的一方获胜。若7个回合结束后双方血量都不为0，则血量高者获胜。若7个回合结束后双方血量都不为0且相等，则平局。
卡牌定义：
1，攻击。打出攻击时，如果对方出的牌不是防御或反击，则对方掉x滴血。x等于我方连续出攻击的回合数（包括本回合）
2，防御。打出防御时，如果对方出的牌是攻击、防御、反击，则不掉血。如果对方出的是重击，则看上一回合我方是否出的也是防御，如果是，我方不掉血，如果不是，我方掉血
3，重击。打出重击时，除非对方上一回合及当前回合出的都是防御，否则对方掉一滴血
4，反击。打出反击时，如果对方出的牌是攻击，对方掉一滴血，我方不掉血；如果对方出的是重击，双方各掉一滴血；否则双方都不掉血
举例：
例子1：a、b都出攻击（连续一次），双方都掉1血
例子2：a、b都出防御，双方都不掉血
例子3：a出攻击，b出防御，双方都不掉血
例子4：a出攻击，b出反击，a掉血
例子5：a出重击，b出反击，双方都掉血
例子6：a出攻击（连续2次），b出攻击（连续1次），a掉1滴血，b掉2滴血
例子7：a出防御（连续2次），b出重击，双方都不掉血

你的任务是为这个游戏设计原型和 prd，prd 是对原型的介绍，最简洁但精确即可，不要啰嗦

# intention_id=2
增加一个人机对战模式。
在 src/backend/ai/rnn_v1 目录下创建一个 gru 模型，网络结构如下：
input(14) -> gru(8) -> 
                              -> （policy head）fc-linear(8->4) -> 屏蔽非法动作 -> softmax 
                              -> （value head）fc-linear(8->1) -> tanh
input 的维度安排：
1，我方剩余攻击牌数
2，我方剩余防御牌数
3，我方剩余重击牌数
4，我方剩余反击牌数
5，我方当前血量
6，对方当前血量
7-10，我方上一回合出的牌，one-hot
11-14, 对方上一回合出的牌，one-hot
policy head 的输出是一个概率分布，其最接近的 one-hot 向量表示己方这一步要出的牌。
value head 的输出是一个标量，表示对当前局面（出牌前）的胜率估计。-1 表示负，- 表示赢，0表示平局

模型的pytorch代码放在 src\backend\ai\rnn_v1。

第一版的权重文件放在 src\backend\ai\rnn_v1/gen1/rnn_v1.pth，由随机初始化生成，不经过训练。

新增一个 ai 注册文件，在里面注册当前支持的 ai。每个 ai 有一个名字和一个权重文件路径。，第一版的ai叫做 rnn_v1.1，使用上文的随机初始化权重文件。

Ui上增加一个主菜单，可选择 hot-seat pvp，也可以选择人机对战。选择 hot-seat pvp 进入现有界面，选择人机对战进入新界面。需要设计新界面的原型。

# intention_id=3
这是一个重构任务。
新增一个包 backend/app，作用是用来桥接 game 包和ai包，从而把 ai 包变成一个纯 ai 包，即不依赖 game。
本次重构不能改变已有功能。
其中几个需要注意的要点：
- app 包里要实现一个 AppManager 类，专门管理 ai_select_action、create_game 等操作
- 非法动作屏蔽 + softmax 都是模型自身的一部分，应该定义在 forward 里
- 非法动作屏蔽的方法，在 app 层的 AppManager 里面定义，然后以闭包的形式通过 forward 函数传入到 model 里面，在 forward 当中使用。forward 把 vector 传进去负责屏蔽的方法，该方法返回屏蔽后的 vector 给  forward
- 把 select_action 的逻辑挪到 app/AppManager.ai_select_action 中，model 只包含模型自身的行为，不再涉及游戏规则和流程方面的知识
- one-hot 向量与卡牌之间的双向转换逻辑、input 的构造 都放到 app 包的 Converter 类里
- controller 可以直接使用 ai.registery
- 重构之后，model.py 只包含模型本身的类定义，不再有其他东西。所有访问方都应该改为调用 app 包中的代码
- 重构之后，register.py 不再负责加载 model 模块，由 app/AppManager 负责实例化模型类
- init_hidden_state() 移除，改为作为一个常量定义在模型类中

做设计的时候，只需画以下图：
- 类图
- 所有 endpoint 的交互图，放在 diagrams/endpoints

# intention_id=4
refactor: reanme GameManager to GameCore. don't forget to also rename the file

# intention_id=5
新增一个 ai 自对弈脚本，支持以下参数
- ai1 player1 使用的 ai，对应 register.py 里面的 name
- ai2 player2 使用的 ai，对应 register.py 里面的 name
- game 对局次数，默认100
- sample_output ppo 样本的路径。如果指定，则在指定路径生成 ppo 样本文件；如果不指定，则不生成样本文件
- temperature 采样温度，默认1.0，传入 0 则关闭采样
- temp_step 每一局的前几步进行采样，当 step_count > temp_step 时，切换到不采样恶贪婪模式
- temp-seed 采用种子，便于复现
- temp-side 采样方，可选值 player1, player2, both 三种。默认值 both

# intention_id=6
add logs into the whole lifesycle of scripts/selfplay.py.
add a switch to controll whether logs are enabled. The switch should be placed in a config file which in under src/backend/config/.
the log destination defaults to logs and can be overrided by config file.
the file name of the log is {yyyy}-{mm}-{dd}.log.
the writing logic of logging should be encapsulated into a class.
The logs should meet the following critiria:
- each line includes a unique traceId which representing a request of a cli running instance
- each line includes a threadId and processId
- each line includes a timestamp which is of miliseconds
- the content of logs should be able to be used to imply the correctness of the code
- all logs should following a consistant format so that scripts can easily read them and analyse them. It also should be easy to append so don't use json

# intention_id=7
现在的 sample 文件有问题，首先，每一局游戏应该组成一个单元，然后每局游戏里应该有两个玩家视角，每个玩家视角的动作序列应该又组成一个子单元才对。因为训练的时候，是按照子单元为单位喂给模型的。一个子单元喂完，h要清空重来的。

# intention_id=8
修复了 sample 里面 reward 全部为终局奖励的问题

# intention_id=9
在训练的时候发现 PPO 的经典现象：两个头互相干扰。于是决定拆成两个网络，分开训练。
另外，把优化器改成 adamW

# intention_id=10
这次做几个小优化：
1，sample 文件里面的 state 进行重命名，叫 input，不要叫 state。不然老以为是记录某种状态，跟隐藏状态混淆
2，selfplay 要增加一个参数，可以指定样本文件中要包含哪方的 perspective。默认是 both，也可以指定 ai1 或者 ai2
3, train.py 不要 --policy-output-path 、--value-output-path 两个参数，太麻烦了。合并成一个 --output-dir 就好。policy权重就固定叫做policy.pth，value权重就固定叫做 value.pth。 epoc 也是，不要两个参数，统一用同一个 --epoch 就好

# intention_id=11
游戏机制改变。
1，增加能量条机制
- 每位玩家都有一个能量条用于存储能量。初始能量为0，最高为5。能量达到5之后，所有增加能量的效果都视为没有效果
- 可以通过打出构建卡并消耗能量以构建新卡牌。每张卡牌都有一个固定的能量消耗量。各个卡牌的能量消耗量详见第7点。
- 特别澄清：打出卡牌不需要消耗能量。仅构建卡牌需要消耗能量
2，增加牌堆机制
- 每个玩家都有一个抽牌堆，初始存放7张卡：2攻击，1防御，1重击，1反击，1聚能，1构建。双方初始没有手牌。
- 每个玩家都有一个弃牌堆，每回合结束时，打出的牌都进入各自玩家的弃牌堆
- 牌堆的卡牌面朝下放置，对方不能查看，但玩家能查看自己的牌堆
3，增加“轮”机制
- 游戏开始后，立刻进入第一轮
- 每轮开始时，双方将手上所有卡牌（若有）放入弃牌堆，然后将弃牌堆都加入到抽牌堆并洗牌，然后各抽7张卡
- 每轮开始时，连续攻击计数、状态、hp都保留
- 进行5个回合
- 5个回合结束后，如果游戏没有结束，则进入下一轮。如此循环直到游戏结束
4，双方初始血量改为20点
5，每个玩家都有一个“状态”栏。可以同时存在多种不同状态。某些卡牌会给玩家赋予状态
6，每张卡牌都赋予一个优先级属性。当结算卡牌效果时，优先级高的卡牌率先结算。优先级相同的同时结算
7，新的卡牌定义：
- 攻击。如果对方没有防御状态，则对方掉x滴血。x等于我方连续出攻击的回合数（包括本回合，但最高不超过3）。优先级：10，消耗能量：1
- 防御。两个效果：
    a，如果己方没有防御状态，则赋予己方防御状态，该状态持续到本回合结束。否则防御状态持续回合数+1
    b，如果己方已有防御状态，则赋予己方联防状态，持续到本回合结束。
  优先级：11，消耗能量：1
- 重击。除非对方有联防状态，否则对方掉一滴血。优先级：10，消耗能量：1
- 反击。三个效果：
    a，如果己方没有防御状态，则赋予己方防御状态，该状态持续到本回合结束，否则防御状态持续回合数+1。优先级：11
    b, 把对方的连续攻击次数重置为0。优先级：11
    c，如果对方出的牌是攻击或重击，对方掉一滴血。优先级：10
  消耗能量：2
- 聚能。己方能量+1.优先级：10，消耗能量：1
- 构建。选择任意一张可以构建的卡牌进行构建，可以构建的判断方式为： can_build = 卡牌消耗能量数 <= 己方能量条中的能量数。选中后，消耗选中卡牌的能量消耗量，然后把一张选中卡牌直接放入弃牌堆。如果己方能量为0，无法打出构建牌。优先级：10，消耗能量：1
- 攻守兼备。两个效果：
    a，如果己方没有防御状态，则赋予己方防御状态，该状态持续到本回合结束，否则防御状态持续回合数+1。优先级：11
    b，如果对方没有防御状态，则对方掉1滴血。优先级：10
  消耗能量：2
8，每轮结束时，双方各掉1滴血。这是为了防止游戏陷入无限循环。

由于增加了卡牌，模型结构也要做出调整。
1，input 的长度要增加 11，从而可以涵盖所有7张卡，并且加入了当前回合数和当前轮数的信息。维度安排如下：
    1，我方手牌剩余攻击牌数
    2，我方手牌剩余防御牌数
    3，我方手牌剩余重击牌数
    4，我方手牌剩余反击牌数
    5，我方手牌剩余聚能牌数
    6，我方手牌剩余构建牌数
    7，我方手牌剩余攻守兼备牌数
    8，我方当前血量
    9，对方当前血量
    10-16，我方上一回合出的牌，one-hot
    17-23, 对方上一回合出的牌，one-hot
    24，当前回合数（1-5）
    25，当前轮数
2，policy 网络的输出长度也要改成 7 + 7 = 14。其中前7维是要出的卡，后7维只在出了构建卡的时候用，用来表示要构建的卡。能量不够的卡要通过 mask_fn 屏蔽掉。训练的时候不用屏蔽。
3，由于游戏变得复杂了很多，policy 网络和 value 网络的 hidden state 的长度也要增加到 32

# intention_id=12
修复 intention11 遗留的几个 bug

# intention_id=13
把模型的 input 重新调整一下，方便后面扩展新的卡牌。
    1，我方当前血量
    2，对方当前血量
    3，当前回合数（1-5）
    4，当前轮数
    5，我方手牌剩余攻击牌数
    6，我方手牌剩余防御牌数
    7，我方手牌剩余重击牌数
    8，我方手牌剩余反击牌数
    9，我方手牌剩余聚能牌数
    10，我方手牌剩余构建牌数
    11，我方手牌剩余攻守兼备牌数
    12-18，我方上一回合出的牌，one-hot，
    19-25, 对方上一回合出的牌，one-hot

# intention_id=14
1，把模型的 input 重新调整一下，方便后面扩展新的卡牌。调整后维度降为15维，定义如下：
    0，我方当前血量
    1，对方当前血量
    2，我方当前能量
    3，对方当前能量
    4，当前回合数（1-5）
    5，当前轮数
    6，我方手牌剩余攻击牌数
    7，我方手牌剩余防御牌数
    8，我方手牌剩余重击牌数
    9，我方手牌剩余反击牌数
    10，我方手牌剩余聚能牌数
    11，我方手牌剩余构建牌数
    12，我方手牌剩余攻守兼备牌数
    13，我方上回合出的牌，是一个整数，对应下面第二点的维度下标。首回合为-1
    14，对方上一回合出的牌，是一个整数，对应下面第二点的维度下标。首回合为-1

2，policy 网络的输出 vector 的维度应该调整为13维，且每一维都对应如下理解：
    0, 出构建，并且构建构建
    1，出攻击
    2, 出构建，并且构建攻击
    3，出防御
    4, 出构建，并且构建防御
    5，出重击
    6, 出构建，并且构建重击
    7，出反击
    8, 出构建，并且构建反击
    9，出聚能
    10, 出构建，并且构建聚能
    11，出攻守平衡
    12，出构建，并且构建攻守平衡


# intention_id=15
这是个代码优化任务。目前，玩家的状态缺少一个专门的类来管理。这个任务就是抽象出一个玩家状态类 PlayerState。
首先需要一个 enum，用来定义所有可能的状态，然后，PlayerState 引用这个 enum。
PlayerState 还要维护一个生存期字段。
然后 Player 的 statuses 改为引用 PlayerState。
序列化、RoundResult 等等及其他有引用到 status 的地方，都改为引用类，避免代码里到处分散着各种 status 字符串字面量。

# intention_id=16
这是个优化任务。目前，RoundResult 的字段非常散：
    # Phase 1: After card resolution
    round: int
    cycle: int
    round_in_cycle: int
    cards_1: list[CardType]
    cards_2: list[CardType]
    damage_1: int
    damage_2: int
    hp_1_after: int
    hp_2_after: int
    energy_1_after: int
    energy_2_after: int
    statuses_1: list[PlayerState]
    statuses_2: list[PlayerState]
    details: list[str]

    # Phase 2: Cycle-end actions
    cycle_ended: bool = False
    bleed_damage_1: int = 0
    bleed_damage_2: int = 0

    # Phase 3: Final state after all processing
    after_cycle: int = 0
    after_round_in_cycle: int = 0
    after_hp_1: int = 0
    after_hp_2: int = 0
    after_statuses_1: list[PlayerState]
    after_statuses_2: list[PlayerState]

    is_game_over: bool = False
    winner: str | None = None

要重新组织一下。分一下层次，我用 json 来表达新结构如下：
```json
{
    "round": 23,
    "cycle": 2,
    "round_in_cycle": 3,
    "cards_1": [Card],
    "cards_2": [Card],
    "statuses_1": [...],
    "statuses_2": [...],
    "energy_1": 2,
    "energy_2": 0,
    "resolve": {
        "is_game_over": false,
        "winner": None,
        "damage_1": 1,
        "damage_2": 2,
        "hp_1_after": 12,
        "hp_2_after": 11,
        "energy_1_after": 1,
        "energy_2_after": 0,
        "details": [...]
    },
    "end_actions": {
        "cycle_ended": false,
        "bleed_damage_1": 0,
        "bleed_damage_2": 0
    },
    "after": {
        "round": 23,
        "cycle": 2,
        "round_in_cycle": 3,
        "cards": [Card],
        "statuses_1": [...],
        "statuses_2": [...],
        "hp_1": 12,
        "hp_2": 11,
        "energy_1": 2,
        "energy_2": 0,
    }
}
```
所有消费了RoundResult的地方都要相应改动，包括前端。