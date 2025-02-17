# GridMind AI

A sophisticated deep reinforcement learning system that masters block elimination gameplay through advanced neural network architectures and optimized training strategies.

一个基于深度强化学习的方块消除游戏AI系统。

## Project Description | 项目描述

GridMind AI represents a cutting-edge implementation of deep reinforcement learning in game AI. The project focuses on training an artificial intelligence system to excel at a block elimination puzzle game through:

- Advanced Neural Architecture: Utilizing a custom-designed CNN with attention mechanisms
- Optimized Learning Strategy: Implementing prioritized experience replay and n-step learning
- Efficient CPU Utilization: Leveraging multi-threading and asynchronous processing
- Adaptive Training: Featuring dynamic learning rate adjustment and exploration strategies

The game itself offers unique challenges compared to traditional Tetris-like games:

- Strategic Placement: Blocks must be placed in their original shape without rotation
- Dual Elimination: Both rows and columns can be cleared simultaneously
- Unlimited Time: Focus on strategic decision-making rather than speed
- Space Management: Game continues until no valid moves remain

Key Features:
- Real-time visualization of AI decision-making process
- Comprehensive performance monitoring and analysis
- Automated checkpointing and model persistence
- Scalable architecture for future enhancements

Block Blast AI 是一个使用深度强化学习（Deep Reinforcement Learning）来训练AI玩方块消除游戏的项目。游戏规则类似俄罗斯方块，但有以下特点：

- 方块不能旋转，必须以原始形状放置
- 可以同时消除行和列
- 没有时间限制
- 当无法放置新方块时游戏结束

### Technical Features | 技术特点

- Improved version of Deep Q-Network (DQN)
- CNN for game state processing
- Prioritized Experience Replay
- Multi-threaded training optimization
- CPU multi-core support
- Dynamic learning rate adjustment
- Adaptive exploration strategy

- 使用深度Q网络（DQN）的改进版本
- 采用CNN处理游戏状态
- 实现了优先经验回放
- 使用多线程优化训练速度
- 支持CPU多核心训练
- 动态学习率调整
- 自适应探索策略

## System Requirements | 系统要求

- Python 3.8+
- CPU: Multi-core processor recommended (optimized for AMD Ryzen 9 5950X)
- RAM: 8GB+
- OS: Windows 10/11, Linux, macOS

- Python 3.8+
- CPU: 推荐多核心处理器（代码针对AMD Ryzen 9 5950X优化）
- RAM: 8GB+
- 操作系统: Windows 10/11, Linux, macOS

## Installation | 安装说明

1. Clone the repository | 克隆仓库：
```bash
git clone https://github.com/kicofy/Block-Blast-GridMind-AI.git
cd Block-Blast-GridMind-AI
```

2. Create virtual environment (optional but recommended) | 创建虚拟环境（可选但推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Install dependencies | 安装依赖：
```bash
pip install -r requirements.txt
```

## Usage | 使用方法

### Start Training | 开始训练

Basic training | 基础训练：
```bash
python train_ai.py
```

Continue training from existing model | 从已有模型继续训练：
```bash
python train_ai.py -model path/to/model.pth
```

### Training Controls | 训练控制

- Spacebar: Switch display mode (Fast/Normal/Slow)
- Close window: Save model and exit

- 空格键：切换显示模式（快速/正常/慢速）
- 关闭窗口：保存模型并退出

### Training Parameters | 训练参数

Main adjustable parameters in `train_ai.py` | 主要可调参数（在`train_ai.py`中）：

```python
self.batch_size = 1024        # Batch size | 批处理大小
self.gamma = 0.99             # Discount factor | 折扣因子
self.epsilon = 1.0            # Initial exploration rate | 初始探索率
self.epsilon_min = 0.05       # Minimum exploration rate | 最小探索率
self.epsilon_decay = 0.9998   # Exploration rate decay | 探索率衰减
self.n_steps = 5              # N-step learning | n步学习
```

## Project Structure | 项目结构

```
block-blast/
├── train_ai.py          # Training main program | 训练主程序
├── game.py              # Game core logic | 游戏核心逻辑
├── game_hard.py         # Hard mode game logic | 困难模式游戏逻辑
├── requirements.txt     # Project dependencies | 项目依赖
└── train/               # Training data and model save directory | 训练数据和模型保存目录
    ├── train1/         
    ├── train2/
    └── ...
```

## Training Features | 训练特性

1. Automatic Saving | 自动保存：
   - Checkpoint every 2000 episodes | 每2000轮自动保存检查点
   - Auto-save on unexpected exit | 意外退出时自动保存
   - Save on target score (50000) | 达到目标分数（50000分）时保存

2. Dynamic Optimization | 动态优化：
   - Adaptive learning rate adjustment | 自适应学习率调整
   - Dynamic exploration rate adjustment | 动态探索率调整
   - Random weight reset (prevent local optima) | 网络权重随机重置（防止局部最优）

3. Training Monitoring | 训练监控：
   - Real-time training status | 实时显示训练状态
   - Multiple display modes | 支持多种显示模式
   - Detailed training logs | 详细的训练日志

## Model Architecture | 模型架构

1. Grid Feature Extractor | 网格特征提取器：
   - 4-layer CNN for pattern recognition | 4层CNN用于识别行列填充模式
   - BatchNorm for training stability | BatchNorm用于训练稳定性
   - Attention mechanism for important areas | 注意力机制关注重要区域

2. Preview Block Analyzer | 预览块分析器：
   - Dedicated CNN for preview blocks | 专门的CNN处理预览方块
   - Shape feature extraction | 形状特征提取

3. State Encoder | 状态编码器：
   - Game state processing | 处理游戏状态信息
   - Score and combo feature encoding | 分数、连击等特征编码

4. Decision System | 决策系统：
   - Dual evaluation (row/column + position) | 双重评估（行列评估 + 位置评估）
   - Space utilization and dead corner consideration | 考虑空间利用率和死角

## Reward System | 奖励系统

- Basic placement reward: +1 | 基础放置奖励：+1
- Elimination rewards | 消除奖励：
  - Single row/column: 100 points/line | 单行/列：100分/行
  - Multiple rows/columns: Exponential bonus | 多行/列：指数增长奖励
  - Combo bonus: 100 points/combo | 连击奖励：100分/连击数
- Penalties | 惩罚：
  - Dead corner: -100/each | 死角惩罚：-100/个
  - Height penalty: Dynamically calculated | 高度惩罚：动态计算
  - Game over: -2000 | 游戏结束：-2000

## Performance Optimization | 性能优化

1. CPU Optimization | CPU优化：
   - Using 75% of logical cores | 使用75%的逻辑核心
   - Thread pool optimization | 线程池优化
   - Asynchronous data processing | 异步数据处理

2. Memory Optimization | 内存优化：
   - Experience replay buffer size optimization | 经验回放缓冲区大小优化
   - Batch tensor caching | 批处理张量缓存
   - Efficient state encoding | 高效的状态编码

3. Training Optimization | 训练优化：
   - Prioritized experience replay | 优先经验回放
   - N-step learning | N步学习
   - Soft target network updates | 软目标网络更新

## Common Issues | 常见问题

1. Slow Training | 训练速度慢：
   - Check CPU usage | 检查CPU使用率
   - Use fast training mode | 使用快速训练模式
   - Reduce batch size | 减小批处理大小

2. Poor Learning Performance | 学习效果不佳：
   - Adjust exploration parameters | 调整探索率参数
   - Increase training episodes | 增加训练轮数
   - Check reward settings | 检查奖励设置

3. High Memory Usage | 内存使用过高：
   - Reduce replay buffer size | 减小经验回放缓冲区大小
   - Lower batch size | 降低批处理大小
   - Increase garbage collection frequency | 增加垃圾回收频率

## License | 许可证

MIT License

Copyright (c) 2024 Block Blast AI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

MIT 许可证

版权所有 (c) 2024 Block Blast AI

特此免费授予任何获得本软件和相关文档文件（"软件"）副本的人不受限制地处理本软件的权利，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或出售本软件的副本的权利，并允许向其提供本软件的人这样做，但须符合以下条件：

上述版权声明和本许可声明应包含在本软件的所有副本或重要部分中。

本软件按"原样"提供，不提供任何形式的明示或暗示的保证，包括但不限于对适销性、特定用途的适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责，无论是在合同诉讼、侵权行为或其他方面，由软件或软件的使用或其他交易引起、由软件引起或与之相关。

## Contributing | 贡献

We welcome contributions to GridMind AI! Feel free to:
- Report issues
- Submit pull requests
- Suggest new features
- Improve documentation

Please visit our [GitHub repository](https://github.com/kicofy/Block-Blast-GridMind-AI) for more information.

欢迎为GridMind AI做出贡献！您可以：
- 报告问题
- 提交拉取请求
- 建议新功能
- 改进文档

请访问我们的[GitHub仓库](https://github.com/kicofy/Block-Blast-GridMind-AI)了解更多信息。

## Contact | 联系方式

- GitHub: [@kicofy](https://github.com/kicofy)
- Email: ha22y.xing@gmail.com
- Project Repository: [Block-Blast-GridMind-AI](https://github.com/kicofy/Block-Blast-GridMind-AI)