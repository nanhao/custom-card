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