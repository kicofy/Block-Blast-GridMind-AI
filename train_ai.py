import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random  
from collections import deque
from game import BlockBlast, GRID_WIDTH, GRID_HEIGHT, SHAPES, PREVIEW_COUNT
import pygame
import shutil
import argparse
import time
from contextlib import nullcontext
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import psutil

# 设置进程数量为CPU核心数的75%
NUM_WORKERS = int(psutil.cpu_count(logical=True) * 0.75)
# 设置线程池大小
THREAD_POOL_SIZE = NUM_WORKERS * 2

class DQN(nn.Module):
    def __init__(self, input_size, output_size, game_state_size):
        super(DQN, self).__init__()
        
        # 网格特征提取器 - 使用更深的CNN来识别行列填充模式
        self.grid_conv = nn.Sequential(
            # 第一层：检测基本图案
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 第二层：检测行列填充状态
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 第三层：检测潜在的消除机会
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 第四层：检测死角和空间利用
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 预览块分析器 - 专注于形状特征
        self.preview_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 状态编码器 - 增加对游戏状态的理解
        self.state_encoder = nn.Sequential(
            nn.Linear(game_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 注意力机制 - 关注重要区域
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合层 - 整合所有信息
        self.fusion = nn.Sequential(
            nn.Linear(64 * GRID_HEIGHT * GRID_WIDTH + 64 * 25 + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 行列评估器 - 专门评估行列消除的可能性
        self.line_evaluator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, GRID_WIDTH + GRID_HEIGHT)  # 评估每行每列的价值
        )
        
        # 位置评估器 - 评估具体放置位置的价值
        self.position_evaluator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, grid, preview, state):
        # 数据类型转换
        if not isinstance(grid, torch.Tensor):
            grid = torch.FloatTensor(grid)
        if not isinstance(preview, torch.Tensor):
            preview = torch.FloatTensor(preview)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # 提取网格特征
        grid_features = self.grid_conv(grid)
        
        # 应用注意力机制
        attention_weights = self.attention(grid_features)
        grid_features = grid_features * attention_weights
        grid_features = grid_features.view(grid_features.size(0), -1)
        
        # 处理预览块
        preview_features = self.preview_conv(preview)
        preview_features = preview_features.view(preview_features.size(0), -1)
        
        # 处理游戏状态
        state_features = self.state_encoder(state)
        
        # 特征融合
        combined = torch.cat([grid_features, preview_features, state_features], dim=1)
        features = self.fusion(combined)
        
        # 评估行列消除机会
        line_values = self.line_evaluator(features)
        
        # 评估具体位置
        position_values = self.position_evaluator(features)
        
        # 结合两种评估
        final_values = position_values + line_values.mean(dim=1, keepdim=True)
        
        return final_values

    def to(self, device):
        super().to(device)
        return self

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        self._size = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self._size = len(self.memory)
        self.memory[self.position] = experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self._size == 0:
            return None, None, None

        # 确保优先级都是正数且非零
        priorities = np.clip(self.priorities[:self._size], 1e-6, None)
        
        # 计算采样概率
        probs = priorities ** self.alpha
        probs_sum = np.sum(probs)
        
        if probs_sum == 0 or np.isnan(probs_sum):
            # 如果概率和为0或NaN，使用均匀分布
            probs = np.ones(self._size) / self._size
        else:
            probs = probs / probs_sum

        # 采样
        try:
            indices = np.random.choice(self._size, batch_size, p=probs, replace=False)
        except ValueError as e:
            print(f"采样错误: {str(e)}")
            print(f"优先级统计: min={np.min(priorities)}, max={np.max(priorities)}, mean={np.mean(priorities)}")
            print(f"概率统计: min={np.min(probs)}, max={np.max(probs)}, sum={np.sum(probs)}")
            # 使用均匀采样作为后备方案
            indices = np.random.choice(self._size, batch_size, replace=False)
        
        # 计算重要性权重
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # 归一化权重
        self.beta = min(1.0, self.beta + self.beta_increment)

        experiences = [self.memory[idx] for idx in indices]
        return experiences, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        # 确保优先级是正数且非零
        priorities = np.clip(priorities, 1e-6, 1e6)
        
        # 更新优先级
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
        
        # 安全地更新最大优先级
        valid_priorities = self.priorities[self.priorities > 0]
        if len(valid_priorities) > 0:
            self.max_priority = float(np.max(valid_priorities))
        else:
            self.max_priority = 1.0  # 如果没有有效的优先级，重置为默认值

class AITrainer:
    def __init__(self, model_path=None):
        self.game = BlockBlast()
        
        # 强制使用CPU
        self.device = torch.device("cpu")
        print("使用CPU进行训练")
        
        # 添加时间记录
        self.start_time = time.time()
        self.last_episode_time = self.start_time
        
        self.batch_tensors = {}  # 缓存批处理张量

        # 创建训练文件夹
        self.train_dir = self.create_train_dir()
        print(f"训练文件夹: {self.train_dir}")

        # 记录最近100轮的分数
        self.recent_scores = deque(maxlen=100)
        
        # 记录最高分和无进展回合数
        self.best_avg_score = 0
        self.no_progress_episodes = 0

        self.grid_size = GRID_WIDTH * GRID_HEIGHT
        self.preview_size = 3 * 5 * 5
        self.state_size = 3
        self.input_size = self.grid_size + self.preview_size + self.state_size
        self.output_size = GRID_WIDTH * GRID_HEIGHT * PREVIEW_COUNT

        self.policy_net = DQN(self.input_size, self.output_size, self.state_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size, self.state_size).to(self.device)
        
        # 如果提供了模型路径，加载已有模型
        if model_path:
            self.load_model(model_path)
            print(f"加载模型: {model_path}")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 设置线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
        
        # 设置torch多线程
        torch.set_num_threads(NUM_WORKERS)
        
        # 增加批处理大小以提高训练效率
        self.batch_size = 1024  # 增大批处理大小
        
        # 优化内存管理
        self.memory = PrioritizedReplayBuffer(100000, alpha=0.6, beta=0.4, beta_increment=0.001)
        
        # 使用AdamW优化器并调整学习率
        self.optimizer = optim.AdamW(self.policy_net.parameters(), 
                                   lr=0.001,
                                   weight_decay=1e-4,
                                   amsgrad=True)
        
        # 使用余弦退火学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-5
        )
        
        # 优化训练参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9998
        
        # 增加经验回放缓冲区大小
        self.n_steps = 5
        self.n_step_buffer = deque(maxlen=self.n_steps)
        
        # 使用torch的DataLoader进行批处理
        self.train_loader = None

        # 显示控制
        self.show_training = True
        self.display_speed = 1
        self.fps_limits = [0, 30, 10]
        self.frame_count = 0
        self.display_frame_skip = 2
        self.clock = pygame.time.Clock()

        # 保存代码副本
        self.save_code_copy()

        # 自适应学习率参数
        self.initial_lr = 0.0005
        self.min_lr = 0.00001
        self.lr_decay = 0.999
        self.lr_increase = 1.002
        self.performance_window = deque(maxlen=100)  # 减小性能窗口大小
        self.best_performance = float('-inf')
        self.patience = 50  # 减小耐心值
        self.no_improvement_count = 0

    def create_train_dir(self):
        """创建新的训练文件夹"""
        # 确保主训练文件夹存在
        main_train_dir = "train"
        if not os.path.exists(main_train_dir):
            os.makedirs(main_train_dir)
        
        # 查找最新的训练子文件夹编号
        i = 1
        while True:
            train_dir = os.path.join(main_train_dir, f"train{i}")
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
                return train_dir
            i += 1

    def save_code_copy(self):
        """保存代码副本到训练文件夹"""
        code_path = os.path.abspath(__file__)
        backup_path = os.path.join(self.train_dir, "train_ai_backup.py")
        shutil.copy2(code_path, backup_path)

    def save_checkpoint(self, episode):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        checkpoint_path = os.path.join(self.train_dir, f"checkpoint_{episode}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"保存检查点: {checkpoint_path}")

    def encode_state(self):
        """将游戏状态编码为神经网络的输入"""
        # 创建网格张量
        grid = torch.FloatTensor(self.game.grid).unsqueeze(0).unsqueeze(0)
        
        # 创建预览方块张量
        preview = torch.zeros((1, 3, 5, 5), dtype=torch.float32)
        if self.game.preview_blocks:
            for i, (shape_name, color) in enumerate(self.game.preview_blocks[:3]):
                shape_grid = torch.zeros((5, 5), dtype=torch.float32)
                shape = SHAPES[shape_name]
                
                height = len(shape)
                width = len(shape[0])
                
                if height <= 5 and width <= 5:
                    start_y = (5 - height) // 2
                    start_x = (5 - width) // 2
                    
                    for y in range(height):
                        for x in range(width):
                            if shape[y][x]:
                                shape_grid[start_y + y, start_x + x] = 1.0
                preview[0, i] = shape_grid
        
        # 创建状态张量
        state = torch.FloatTensor([
            self.game.score / 50000.0,  # 归一化分数
            self.game.combo / 10.0,     # 归一化连击数
            len(self.game.preview_blocks) / PREVIEW_COUNT  # 归一化预览块数量
        ]).unsqueeze(0)

        # 移动到设备
        if self.device.type == 'cuda':
            grid = grid.cuda(non_blocking=True)
            preview = preview.cuda(non_blocking=True)
            state = state.cuda(non_blocking=True)
            torch.cuda.synchronize()  # 确保数据传输完成
        
        return grid, preview, state

    def select_action(self, grid, preview, state):
        if random.random() < self.epsilon:
            # 获取所有有效动作
            valid_actions = self.get_valid_actions()
            if not valid_actions:
                return None  # 没有有效动作
            return random.choice(valid_actions)

        with torch.no_grad():
            # 确保输入数据在CPU上
            if isinstance(grid, torch.Tensor) and grid.device.type == 'cuda':
                grid = grid.cpu()
            if isinstance(preview, torch.Tensor) and preview.device.type == 'cuda':
                preview = preview.cpu()
            if isinstance(state, torch.Tensor) and state.device.type == 'cuda':
                state = state.cpu()

            # 转换为张量（如果不是）
            if not isinstance(grid, torch.Tensor):
                grid = torch.FloatTensor(grid)
            if not isinstance(preview, torch.Tensor):
                preview = torch.FloatTensor(preview)
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)

            # 获取所有动作的Q值
            q_values = self.policy_net(grid, preview, state)
            
            # 创建动作掩码
            action_mask = self.create_action_mask()
            
            # 将无效动作的Q值设为负无穷
            q_values = q_values.squeeze()
            q_values[~action_mask] = float('-inf')
            
            # 选择最佳有效动作
            best_action = q_values.argmax().item()
            return best_action

    def get_valid_actions(self):
        """获取所有有效的动作"""
        valid_actions = []
        
        if not self.game.preview_blocks:
            return valid_actions
            
        for preview_idx, (shape, _) in enumerate(self.game.preview_blocks):
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.game.can_place_block(shape, x, y):
                        action = y * GRID_WIDTH + x + preview_idx * (GRID_WIDTH * GRID_HEIGHT)
                        valid_actions.append(action)
        
        return valid_actions

    def create_action_mask(self):
        """创建动作掩码，True表示有效动作"""
        mask = torch.zeros(self.output_size, dtype=torch.bool)
        
        if not self.game.preview_blocks:
            return mask
            
        for preview_idx, (shape, _) in enumerate(self.game.preview_blocks):
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.game.can_place_block(shape, x, y):
                        action = y * GRID_WIDTH + x + preview_idx * (GRID_WIDTH * GRID_HEIGHT)
                        mask[action] = True
        
        return mask

    def decode_action(self, action):
        """将动作值解码为位置和预览方块索引"""
        grid_pos = action % (GRID_WIDTH * GRID_HEIGHT)
        preview_idx = action // (GRID_WIDTH * GRID_HEIGHT)
        x = grid_pos % GRID_WIDTH
        y = grid_pos // GRID_WIDTH
        return x, y, preview_idx

    def train_step(self):
        if len(self.memory.memory) < self.batch_size:
            return

        # 使用多线程采样
        future = self.thread_pool.submit(self.memory.sample, self.batch_size)
        batch, indices, weights = future.result()
        
        if batch is None:
            return

        # 使用torch.jit优化forward pass
        with torch.jit.optimized_execution(True):
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # 使用异步数据传输
            grid_batch = torch.zeros((self.batch_size, 1, GRID_HEIGHT, GRID_WIDTH), 
                                   dtype=torch.float32)
            preview_batch = torch.zeros((self.batch_size, 3, 5, 5), 
                                      dtype=torch.float32)
            state_batch = torch.zeros((self.batch_size, self.state_size), 
                                    dtype=torch.float32)
            
            # 并行处理数据填充
            def fill_batch_data(i, state):
                grid_batch[i].copy_(state[0].squeeze(0))
                preview_batch[i].copy_(state[1].squeeze(0))
                state_batch[i].copy_(state[2].squeeze(0))
            
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                executor.map(fill_batch_data, range(self.batch_size), states)
            
            # 获取当前状态的Q值
            current_q_values = self.policy_net(grid_batch, preview_batch, state_batch)
            
            # 获取下一状态的Q值
            next_q_values = self.target_net(grid_batch, preview_batch, state_batch)
            
            # 创建下一状态的动作掩码
            next_state_masks = torch.stack([self.create_action_mask() for _ in range(self.batch_size)])
            
            # 将无效动作的Q值设为负无穷
            next_q_values[~next_state_masks] = float('-inf')
            
            # 使用Double DQN
            next_actions = self.policy_net(grid_batch, preview_batch, state_batch).argmax(1)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # 使用向量化操作
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            # 计算TD误差（使用GAE）
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 使用带权重的Huber损失
            td_errors = torch.abs(current_q_values - expected_q_values)
            loss = (weights * torch.where(td_errors < 1, 
                                        0.5 * td_errors.pow(2),
                                        td_errors - 0.5)).mean()
        
        # 梯度裁剪和优化
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 软更新目标网络
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 
                                  0.005 * policy_param.data)
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors.detach().numpy() + 1e-8)

    def toggle_display_speed(self):
        """切换显示速度"""
        self.display_speed = (self.display_speed + 1) % 3
        if self.display_speed == 0:
            print("切换到快速训练模式")
            self.show_training = False
        else:
            self.show_training = True
            speed_text = "正常" if self.display_speed == 1 else "慢速"
            print(f"切换到{speed_text}显示模式")

    def handle_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n检测到窗口关闭,正在保存最终模型...")
                model_path = os.path.join(self.train_dir, "final_model.pth")
                torch.save(self.policy_net.state_dict(), model_path)
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.toggle_display_speed()
        return True

    def update_display(self):
        """更新显示"""
        if not self.show_training:
            return

        self.frame_count += 1
        if self.frame_count % self.display_frame_skip == 0:
            self.game.draw()
            
            # 显示训练信息
            font = pygame.font.Font(None, 24)
            avg_score = sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0
            info_texts = [
                f"训练信息:",
                f"Episode Score: {self.game.score}",
                f"Epsilon: {self.epsilon:.3f}",
                f"Avg Score (100): {avg_score:.1f}"
            ]
            
            # 计算训练信息的显示位置
            window_width = self.game.screen.get_width()
            info_x = window_width - 220  # 在窗口右侧显示，留出20像素边距
            
            # 绘制信息背景
            info_bg_rect = pygame.Rect(info_x - 10, 10, 200, len(info_texts) * 25 + 10)
            pygame.draw.rect(self.game.screen, (0, 0, 0), info_bg_rect)
            pygame.draw.rect(self.game.screen, (100, 100, 100), info_bg_rect, 1)
            
            # 显示训练信息
            for i, text in enumerate(info_texts):
                text_surface = font.render(text, True, (255, 255, 255))
                self.game.screen.blit(text_surface, (info_x, 20 + i * 25))
            
            pygame.display.flip()
            if self.display_speed > 0:
                self.clock.tick(self.fps_limits[self.display_speed])

    def calculate_reward(self, lines_cleared, score_gained, combo):
        """计算奖励值"""
        reward = 0
        
        # 基础放置奖励
        reward += 1
        
        # 消除奖励
        if lines_cleared > 0:
            # 基础消除奖励
            base_clear_reward = 100 * lines_cleared
            
            # 多行/列同时消除的额外奖励（指数增长）
            if lines_cleared >= 2:
                bonus = 2 ** (lines_cleared - 1) * 200
                reward += base_clear_reward + bonus
            else:
                reward += base_clear_reward
            
            # 连击奖励
            if combo > 1:
                reward += combo * 100
        
        # 空间管理奖励
        holes = self.count_holes()
        height_var = self.calculate_height_variance()
        max_height = self.get_max_height()
        
        # 计算死角惩罚
        dead_corners = self.count_dead_corners()
        reward -= dead_corners * 100
        
        # 空间利用率奖励
        space_efficiency = self.calculate_space_efficiency()
        reward += space_efficiency * 50
        
        # 动态高度惩罚
        if max_height > GRID_HEIGHT * 0.7:  # 当高度超过70%时开始惩罚
            height_penalty = ((max_height - GRID_HEIGHT * 0.7) ** 2) * 100
            reward -= height_penalty
        
        # 空洞惩罚（基于位置的动态权重）
        for hole_depth in self.get_hole_depths():
            reward -= (hole_depth ** 1.5) * 30
        
        # 游戏结束惩罚
        if self.game.game_over:
            reward -= 2000
        
        return reward

    def count_holes(self):
        """计算网格中的空洞数"""
        holes = 0
        for x in range(GRID_WIDTH):
            found_block = False
            for y in range(GRID_HEIGHT):
                if self.game.grid[y][x]:
                    found_block = True
                elif found_block and not self.game.grid[y][x]:
                    holes += 1
        return holes

    def calculate_height_variance(self):
        """计算网格高度差异"""
        heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if self.game.grid[y][x]:
                    heights.append(GRID_HEIGHT - y)
                    break
            else:
                heights.append(0)
        
        if not heights:
            return 0
            
        avg_height = sum(heights) / len(heights)
        variance = sum((h - avg_height) ** 2 for h in heights) / len(heights)
        return variance

    def get_max_height(self):
        """获取当前最大高度"""
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game.grid[y][x]:
                    return GRID_HEIGHT - y
        return 0

    def count_dead_corners(self):
        """计算死角数量"""
        dead_corners = 0
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if not self.game.grid[y][x]:  # 如果是空位
                    # 检查周围是否被封死
                    surrounded = True
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        new_x, new_y = x + dx, y + dy
                        if (0 <= new_x < GRID_WIDTH and 
                            0 <= new_y < GRID_HEIGHT and 
                            not self.game.grid[new_y][new_x]):
                            surrounded = False
                            break
                    if surrounded:
                        dead_corners += 1
        return dead_corners

    def calculate_space_efficiency(self):
        """计算空间利用率"""
        total_cells = GRID_WIDTH * GRID_HEIGHT
        filled_cells = sum(sum(row) for row in self.game.grid)
        return filled_cells / total_cells

    def get_hole_depths(self):
        """获取每个空洞的深度"""
        depths = []
        for x in range(GRID_WIDTH):
            depth = 0
            found_block = False
            for y in range(GRID_HEIGHT):
                if self.game.grid[y][x]:
                    found_block = True
                    depth = 0
                elif found_block:
                    depth += 1
                    if depth > 0:
                        depths.append(depth)
        return depths

    def adjust_exploration(self, avg_score):
        """动态调整探索率"""
        # 更新最佳平均分
        if avg_score > self.best_avg_score:
            self.best_avg_score = avg_score
            self.no_progress_episodes = 0
        else:
            self.no_progress_episodes += 1

        # 如果长时间没有进展，大幅增加探索
        if self.no_progress_episodes >= 500:  # 降低无进展判定门槛
            self.epsilon = min(0.8, self.epsilon + 0.2)  # 更激进的探索重置
            self.no_progress_episodes = 0
            print(f"\n[探索重置] 连续500轮无进展，重置探索率至: {self.epsilon:.3f}")
            
            # 随机重置部分网络权重
            if random.random() < 0.3:  # 30%概率重置部分网络
                print("[网络重置] 随机重置部分网络权重")
                for param in self.policy_net.parameters():
                    if random.random() < 0.3:  # 随机重置30%的参数
                        if len(param.data.shape) >= 2:
                            # 对于2维及以上的张量使用xavier初始化
                            nn.init.xavier_normal_(param.data)
                        else:
                            # 对于1维张量（偏置项）使用均匀分布初始化
                            nn.init.uniform_(param.data, -0.1, 0.1)

        # 正常衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def calculate_n_step_returns(self):
        """计算n步回报"""
        if len(self.n_step_buffer) < self.n_steps:
            return None
        
        reward = 0
        for i in range(self.n_steps):
            reward += self.gamma ** i * self.n_step_buffer[i][2]  # index 2 is the reward
        
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]
        
        return (state, action, reward, next_state, done)

    def adjust_learning_rate(self):
        """动态调整学习率"""
        if len(self.performance_window) < self.performance_window.maxlen:
            return

        current_performance = sum(self.performance_window) / len(self.performance_window)
        
        # 更新最佳性能
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # 根据性能调整学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if self.no_improvement_count >= self.patience:
            # 性能停滞，降低学习率
            new_lr = max(current_lr * self.lr_decay, self.min_lr)
            if new_lr != current_lr:
                print(f"\n[学习率调整] 降低学习率: {current_lr:.6f} -> {new_lr:.6f}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                self.no_improvement_count = 0
        elif current_performance > self.best_performance:
            # 性能提升，轻微增加学习率
            new_lr = min(current_lr * self.lr_increase, self.initial_lr)
            if new_lr != current_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

    def train(self):
        episode = 0
        print("\n开始训练...")
        print("控制说明：")
        print("- 空格键：切换显示模式（快速/正常/慢速）")
        print("- 关闭窗口：保存模型并退出")
        
        try:
            while True:
                episode += 1
                current_time = time.time()
                total_elapsed = current_time - self.start_time
                episode_elapsed = current_time - self.last_episode_time
                self.last_episode_time = current_time
                
                self.game.reset_game()
                grid, preview, state = self.encode_state()
                total_reward = 0
                lines_cleared_total = 0
                self.n_step_buffer.clear()
                steps_since_train = 0

                if episode % 25 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"更新目标网络 - Episode {episode}")

                while not self.game.game_over:
                    if not self.handle_events():
                        return

                    # 选择动作
                    action = self.select_action(grid, preview, state)
                    if action is None:  # 没有有效动作
                        self.game.game_over = True
                        break
                        
                    x, y, preview_idx = self.decode_action(action)
                    
                    if not self.game.preview_blocks:
                        self.game.generate_new_block_set()
                        if not self.get_valid_actions():  # 使用get_valid_actions检查
                            self.game.game_over = True
                            break
                    
                    # 执行动作
                    shape, color = self.game.preview_blocks[preview_idx]
                    self.game.selected_block = self.game.preview_blocks[preview_idx]
                    
                    # 放置方块（现在我们确定这是有效的放置）
                    old_score = self.game.score
                    old_combo = self.game.combo
                    self.game.place_block(shape, x, y)
                    self.game.preview_blocks.pop(preview_idx)
                    self.game.check_lines()
                    
                    # 计算奖励
                    score_gained = self.game.score - old_score
                    lines_cleared = 0
                    if score_gained > 0:
                        if score_gained >= 2000:
                            lines_cleared = 4
                        elif score_gained >= 1000:
                            lines_cleared = 3
                        elif score_gained >= 400:
                            lines_cleared = 2
                        else:
                            lines_cleared = 1
                    
                    lines_cleared_total += lines_cleared
                    reward = self.calculate_reward(lines_cleared, score_gained, self.game.combo)
                    
                    # 生成新的预览方块
                    if not self.game.preview_blocks:
                        self.game.generate_new_block_set()
                        if not self.get_valid_actions():  # 使用get_valid_actions检查
                            self.game.game_over = True
                            reward = -100
                    
                    # 获取下一个状态
                    next_grid, next_preview, next_state = self.encode_state()
                    
                    # 存储经验
                    self.n_step_buffer.append((
                        (grid, preview, state),
                        action,
                        reward,
                        (next_grid, next_preview, next_state),
                        self.game.game_over
                    ))
                    
                    if len(self.n_step_buffer) >= self.n_steps:
                        n_step_experience = self.calculate_n_step_returns()
                        if n_step_experience:
                            self.memory.push(n_step_experience)
                    
                    grid, preview, state = next_grid, next_preview, next_state
                    total_reward += reward

                    steps_since_train += 1
                    if steps_since_train >= 4:
                        self.train_step()
                        steps_since_train = 0

                    self.update_display()

                # 处理剩余的N步经验
                while self.n_step_buffer:
                    n_step_experience = self.calculate_n_step_returns()
                    if n_step_experience:
                        self.memory.push(n_step_experience)
                    self.n_step_buffer.popleft()

                # 记录性能并调整学习率
                self.performance_window.append(self.game.score)
                self.adjust_learning_rate()
                
                # 记录本轮分数并调整探索率
                self.recent_scores.append(self.game.score)
                avg_score = sum(self.recent_scores) / len(self.recent_scores)
                self.adjust_exploration(avg_score)
                
                print(f"Episode {episode} - Score: {self.game.score}, Lines: {lines_cleared_total}, "
                      f"Avg Score (100): {avg_score:.1f}, Epsilon: {self.epsilon:.3f}, "
                      f"总训练时间: {total_elapsed:.1f}秒, 本轮用时: {episode_elapsed:.1f}秒")
                
                if episode % 2000 == 0:
                    self.save_checkpoint(episode)
                
                if self.game.score >= 50000:
                    print(f"训练完成！最终得分: {self.game.score}")
                    self.save_checkpoint(episode)
                    break

        except KeyboardInterrupt:
            print("\n检测到训练中断，正在保存模型...")
            self.save_checkpoint(episode)
        
        finally:
            pygame.quit()

    def load_model(self, model_path):
        """加载已有模型"""
        try:
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict):
                # 如果是检查点文件
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.epsilon = checkpoint.get('epsilon', 0.3)  # 如果没有epsilon信息，使用0.3
                print(f"从检查点加载模型成功，当前epsilon: {self.epsilon}")
            else:
                # 如果只是模型状态字典
                self.policy_net.load_state_dict(checkpoint)
                print("加载模型状态成功")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            print("将使用新模型开始训练")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='俄罗斯方块AI训练程序')
    parser.add_argument('-model', type=str, help='要加载的模型路径')
    args = parser.parse_args()

    # 创建训练器实例
    trainer = AITrainer(model_path=args.model)
    trainer.train()
