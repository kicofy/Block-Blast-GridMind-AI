import pygame
import random
import numpy as np

# 初始化Pygame
pygame.init()

# 游戏常量
BLOCK_SIZE = 60
GRID_WIDTH = 8
GRID_HEIGHT = 8
PREVIEW_COUNT = 3
GRID_OFFSET_X = BLOCK_SIZE  # 网格的X偏移
GRID_OFFSET_Y = BLOCK_SIZE  # 网格的Y偏移
SCREEN_WIDTH = BLOCK_SIZE * (GRID_WIDTH + 4)
SCREEN_HEIGHT = BLOCK_SIZE * (GRID_HEIGHT + 4)

# 颜色定义
BACKGROUND_COLOR = (59, 89, 152)
GRID_COLOR = (40, 59, 112)
TEXT_COLOR = (255, 255, 255)
PREVIEW_BG_COLOR = (49, 69, 132)
GAME_OVER_COLOR = (0, 0, 0, 180)
HINT_BUTTON_COLOR = (70, 130, 180)  # 提示按钮颜色
HINT_BUTTON_HOVER_COLOR = (100, 160, 210)  # 提示按钮悬停颜色

# 方块颜色列表 - 更鲜艳的颜色
BLOCK_COLORS = [
    (255, 50, 50),    # 亮红色
    (50, 255, 50),    # 亮绿色
    (50, 50, 255),    # 亮蓝色
    (255, 255, 50),   # 亮黄色
    (255, 50, 255),   # 亮紫色
    (50, 255, 255),   # 亮青色
    (255, 150, 50),   # 亮橙色
    (150, 50, 255),   # 亮紫蓝色
]

# 方块形状定义
SHAPES = {
    'I2': [(0,0), (0,1)],                    # 2格直线
    'I3': [(0,0), (0,1), (0,2)],             # 3格直线
    'I4': [(0,0), (0,1), (0,2), (0,3)],      # 4格直线
    'I5': [(0,0), (0,1), (0,2), (0,3), (0,4)], # 5格直线
    'L2': [(0,0), (0,1), (1,1)],             # 2x2 L形
    'L3': [(0,0), (0,1), (0,2), (1,2)],      # 2x3 L形
    'T': [(0,1), (1,0), (1,1), (1,2)],       # T形
    'S': [(1,0), (1,1), (0,1), (0,2)],       # S形
    'Z': [(0,0), (0,1), (1,1), (1,2)],       # Z形
    'SQ2': [(0,0), (0,1), (1,0), (1,1)],     # 2x2方形
    'SQ3': [(0,0), (0,1), (0,2),             # 3x3方形
           (1,0), (1,1), (1,2),
           (2,0), (2,1), (2,2)],
    'R23': [(0,0), (0,1), (0,2),             # 2x3矩形
           (1,0), (1,1), (1,2)],
    'D': [(0,0), (1,1)]                      # 对角形
}

class BlockBlast:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Block Blast")
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        """重置游戏状态"""
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)  # 用于标记格子是否被占用
        self.grid_colors = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=int)  # 存储颜色信息
        self.score = 0
        self.combo = 0
        self.font_large = pygame.font.Font(None, 72)
        self.font = pygame.font.Font(None, 36)
        self.preview_blocks = []
        self.selected_block = None
        self.selected_index = None
        self.dragging = False
        self.drag_pos = None
        self.game_over = False
        self.generate_new_block_set()

    def get_block_offset(self, shape):
        """获取方块的偏移量和尺寸"""
        shape_blocks = SHAPES[shape]
        min_x = min(bx for bx, _ in shape_blocks)
        min_y = min(by for _, by in shape_blocks)
        max_x = max(bx for bx, _ in shape_blocks)
        max_y = max(by for _, by in shape_blocks)
        width = (max_x - min_x + 1) * BLOCK_SIZE
        height = (max_y - min_y + 1) * BLOCK_SIZE
        return min_x, min_y, width, height

    def screen_to_grid(self, screen_x, screen_y):
        """将屏幕坐标转换为网格坐标"""
        grid_x = (screen_x - GRID_OFFSET_X) // BLOCK_SIZE
        grid_y = (screen_y - GRID_OFFSET_Y) // BLOCK_SIZE
        return grid_x, grid_y

    def grid_to_screen(self, grid_x, grid_y):
        """将网格坐标转换为屏幕坐标"""
        screen_x = grid_x * BLOCK_SIZE + GRID_OFFSET_X
        screen_y = grid_y * BLOCK_SIZE + GRID_OFFSET_Y
        return screen_x, screen_y

    def check_game_over(self):
        """检查游戏是否结束"""
        # 检查当前预览区域中的方块是否有任何一个可以放置
        for shape, _ in self.preview_blocks:
            if self.can_block_be_placed_anywhere(shape):
                return False
        return True

    def rotate_90(self, shape_blocks):
        """90度旋转方块"""
        # 找到形状的边界
        max_x = max(x for x, _ in shape_blocks)
        
        # 进行90度旋转：(x,y) -> (-y,x)，然后平移使坐标非负
        rotated = []
        for x, y in shape_blocks:
            new_x = -y
            new_y = x
            rotated.append((new_x, new_y))
            
        # 规范化坐标（确保从0,0开始）
        min_x = min(x for x, _ in rotated)
        min_y = min(y for _, y in rotated)
        return [(x - min_x, y - min_y) for x, y in rotated]

    def mirror(self, shape_blocks):
        """水平镜像方块"""
        # 找到形状的边界
        max_x = max(x for x, _ in shape_blocks)
        
        # 水平镜像：x -> max_x - x
        mirrored = [(max_x - x, y) for x, y in shape_blocks]
        
        # 规范化坐标
        min_x = min(x for x, _ in mirrored)
        min_y = min(y for _, y in mirrored)
        return [(x - min_x, y - min_y) for x, y in mirrored]

    def generate_rotated_shape(self, base_shape):
        """生成基础形状的随机旋转或镜像版本"""
        shape_blocks = SHAPES[base_shape]
        variations = [shape_blocks]  # 原始形状
        
        # 添加90度、180度、270度旋转的版本
        current = shape_blocks
        for _ in range(3):
            current = self.rotate_90(current)
            if not any(self.shapes_are_same(current, v) for v in variations):
                variations.append(current)
        
        # 添加镜像版本
        mirrored = self.mirror(shape_blocks)
        if not any(self.shapes_are_same(mirrored, v) for v in variations):
            variations.append(mirrored)
            # 对镜像后的形状也进行旋转
            current = mirrored
            for _ in range(3):
                current = self.rotate_90(current)
                if not any(self.shapes_are_same(current, v) for v in variations):
                    variations.append(current)
        
        # 随机选择一个变体
        return random.choice(variations)

    def shapes_are_same(self, shape1, shape2):
        """检查两个形状是否相同"""
        # 将坐标转换为集合进行比较
        set1 = {(x, y) for x, y in shape1}
        set2 = {(x, y) for x, y in shape2}
        return set1 == set2

    def simulate_block_placement(self, shape, x, y, color=None):
        """模拟在指定位置放置方块（不依赖selected_block）"""
        blocks = SHAPES[shape]
        for block_x, block_y in blocks:
            grid_x = x + block_x
            grid_y = y + block_y
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                self.grid[grid_y][grid_x] = 1  # 标记为已占用
                if color:
                    self.grid_colors[grid_y][grid_x] = color  # 存储颜色信息

    def analyze_grid_state(self):
        """分析当前网格状态"""
        # 计算空洞数量
        holes = sum(1 for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT-1)
                   if self.grid[y][x] == 0 and self.grid[y+1][x] == 1)
        
        # 计算高度差异
        column_heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if self.grid[y][x] == 1:
                    column_heights.append(GRID_HEIGHT - y)
                    break
            else:
                column_heights.append(0)
        height_diff = sum(abs(column_heights[i] - column_heights[i+1]) 
                         for i in range(len(column_heights)-1))
        
        # 计算平均高度
        avg_height = sum(column_heights) / GRID_WIDTH if column_heights else 0
        
        # 计算可消除的行和列
        clearable_rows = sum(1 for y in range(GRID_HEIGHT) 
                           if sum(self.grid[y]) >= GRID_WIDTH - 2)
        clearable_cols = sum(1 for x in range(GRID_WIDTH) 
                           if sum(self.grid[:, x]) >= GRID_HEIGHT - 2)
        
        return {
            'holes': holes,
            'height_diff': height_diff,
            'avg_height': avg_height,
            'clearable_rows': clearable_rows,
            'clearable_cols': clearable_cols,
            'column_heights': column_heights
        }

    def evaluate_block_placement(self, shape, x, y, grid_state):
        """评估方块放置的效果"""
        temp_grid = self.grid.copy()
        score = 0
        
        # 模拟放置方块
        blocks = SHAPES[shape]
        for block_x, block_y in blocks:
            grid_x = x + block_x
            grid_y = y + block_y
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                temp_grid[grid_y][grid_x] = 1
        
        # 计算放置后的状态
        lines_before = sum(all(temp_grid[y]) for y in range(GRID_HEIGHT)) + \
                      sum(all(temp_grid[:, x]) for x in range(GRID_WIDTH))
        
        # 消除完整的行和列
        for y in range(GRID_HEIGHT):
            if all(temp_grid[y]):
                temp_grid[y] = np.zeros(GRID_WIDTH)
        for x in range(GRID_WIDTH):
            if all(temp_grid[:, x]):
                temp_grid[:, x] = np.zeros(GRID_HEIGHT)
        
        lines_after = sum(all(temp_grid[y]) for y in range(GRID_HEIGHT)) + \
                     sum(all(temp_grid[:, x]) for x in range(GRID_WIDTH))
        lines_cleared = lines_before - lines_after
        
        # 基础得分：消除行列
        score += lines_cleared * 1000
        
        # 评估放置位置
        score += (GRID_HEIGHT - y) * 10  # 底部优先
        score += 50 if x == 0 or x == GRID_WIDTH-1 else 0  # 边缘优先
        
        # 评估对网格状态的改善
        new_holes = sum(1 for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT-1)
                       if temp_grid[y][x] == 0 and temp_grid[y+1][x] == 1)
        score -= (new_holes - grid_state['holes']) * 200  # 空洞变化惩罚
        
        # 评估高度差异变化
        new_heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if temp_grid[y][x] == 1:
                    new_heights.append(GRID_HEIGHT - y)
                    break
            else:
                new_heights.append(0)
        new_height_diff = sum(abs(new_heights[i] - new_heights[i+1]) 
                            for i in range(len(new_heights)-1))
        score -= (new_height_diff - grid_state['height_diff']) * 100
        
        return score, temp_grid

    def get_difficulty_level(self):
        """根据分数计算当前难度等级"""
        base_difficulty = min(self.score // 2000, 5)  # 每2000分增加一个难度等级，最高5级
        return base_difficulty

    def get_mode_probabilities(self, grid_state, remaining_blocks):
        """根据网格状态、剩余方块数和难度等级动态调整模式概率"""
        difficulty = self.get_difficulty_level()
        
        # 基础概率
        base_probs = {
            'single': 0.5 - (difficulty * 0.02),  # 随难度降低单一模式概率（每级降低2%）
            'combo': 0.4,
            'challenge': 0.1 + (difficulty * 0.02)  # 随难度提高挑战模式概率（每级提高2%）
        }
        
        # 根据局面状态调整概率
        if grid_state['avg_height'] > GRID_HEIGHT * 0.7:
            # 高度较高时，增加单独清理概率
            return {
                'single': min(0.8, base_probs['single'] + 0.2),
                'combo': base_probs['combo'] - 0.15,
                'challenge': max(0.05, base_probs['challenge'] - 0.05)
            }
        elif grid_state['holes'] > 5:
            # 空洞较多时，增加组合清理概率
            return {
                'single': base_probs['single'] - 0.1,
                'combo': min(0.8, base_probs['combo'] + 0.15),
                'challenge': max(0.05, base_probs['challenge'] - 0.05)
            }
        elif remaining_blocks == PREVIEW_COUNT:
            # 第一个方块，根据难度调整
            first_block_challenge = 0.1 + (difficulty * 0.01)  # 随难度增加挑战概率（每级增加1%）
            return {
                'single': max(0.4, 0.6 - (difficulty * 0.02)),
                'combo': 0.3,
                'challenge': first_block_challenge
            }
        else:
            return base_probs

    def select_shapes_for_mode(self, mode, grid_state):
        """根据模式和网格状态选择合适的形状集"""
        if mode == 'single':
            if grid_state['avg_height'] > GRID_HEIGHT * 0.6:
                # 高度较高时优先选择小方块
                shapes = ['I2', 'L2', 'SQ2', 'T', 'S']
            else:
                # 正常高度可以选择所有基础方块
                shapes = ['I2', 'I3', 'L2', 'SQ2', 'T', 'S', 'Z']
                
        elif mode == 'combo':
            if grid_state['holes'] > 3:
                # 有较多空洞时选择适合填补的方块
                shapes = ['L3', 'T', 'SQ2', 'S', 'Z', 'R23']
            else:
                # 正常情况选择所有可用方块
                shapes = ['I4', 'T', 'L3', 'S', 'Z', 'I5', 'R23']
                
        else:  # challenge
            if grid_state['avg_height'] < GRID_HEIGHT * 0.4:
                # 高度较低时可以选择大方块
                shapes = ['I4', 'I5', 'SQ3', 'D', 'R23']
            else:
                # 高度较高时选择灵活的方块
                shapes = ['T', 'L3', 'S', 'Z', 'D', 'R23']
        
        # 随机打乱形状顺序
        random.shuffle(shapes)
        return shapes

    def try_generate_clearing_blocks(self):
        """尝试生成能直接清屏的方块组合"""
        # 分析当前网格
        grid_state = self.analyze_grid_state()
        difficulty = self.get_difficulty_level()
        
        # 根据难度调整生成概率（5%到15%之间）
        base_chance = 0.05 + (difficulty * 0.02)  # 每级增加2%概率
        if random.random() > base_chance:
            return None
            
        # 决定使用多少个方块来清屏（1-3个）
        blocks_count = random.choices([1, 2, 3], weights=[1, 2, 3])[0]
        
        # 创建临时网格用于模拟
        temp_grid = self.grid.copy()
        clearing_blocks = []
        
        # 尝试找到能清除当前网格的方块组合
        all_shapes = list(SHAPES.keys())
        max_attempts = 1000
        attempts = 0
        
        while len(clearing_blocks) < blocks_count and attempts < max_attempts:
            attempts += 1
            
            # 随机选择一个形状
            shape = random.choice(all_shapes)
            rotated_blocks = self.generate_rotated_shape(shape)
            
            # 尝试在所有可能的位置放置
            best_clear_count = 0
            best_position = None
            best_rotation = None
            
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.can_place_block(shape, x, y, temp_grid):
                        # 模拟放置并计算清除的行列数
                        test_grid = temp_grid.copy()
                        for bx, by in rotated_blocks:
                            grid_x = x + bx
                            grid_y = y + by
                            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                                test_grid[grid_y][grid_x] = 1
                        
                        # 计算可以清除的行列数
                        clear_count = sum(all(test_grid[y]) for y in range(GRID_HEIGHT)) + \
                                    sum(all(test_grid[:, x]) for x in range(GRID_WIDTH))
                        
                        if clear_count > best_clear_count:
                            best_clear_count = clear_count
                            best_position = (x, y)
                            best_rotation = rotated_blocks
            
            if best_position and best_rotation:
                # 将找到的方块添加到结果中
                temp_shape = f'TEMP_CLEAR_{len(clearing_blocks)}'
                SHAPES[temp_shape] = best_rotation
                clearing_blocks.append((temp_shape, random.choice(BLOCK_COLORS)))
                
                # 更新临时网格
                x, y = best_position
                for bx, by in best_rotation:
                    grid_x = x + bx
                    grid_y = y + by
                    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                        temp_grid[grid_y][grid_x] = 1
                        
                # 模拟消除完整的行和列
                for y in range(GRID_HEIGHT):
                    if all(temp_grid[y]):
                        temp_grid[y] = np.zeros(GRID_WIDTH)
                for x in range(GRID_WIDTH):
                    if all(temp_grid[:, x]):
                        temp_grid[:, x] = np.zeros(GRID_HEIGHT)
        
        # 如果找到了足够的方块，返回结果
        if len(clearing_blocks) == blocks_count:
            return clearing_blocks
        return None

    def generate_new_block_set(self):
        """生成新的一组方块，使用多阶段生成策略"""
        # 尝试生成清屏方块组合
        clearing_blocks = self.try_generate_clearing_blocks()
        if clearing_blocks:
            self.preview_blocks = clearing_blocks
            while len(self.preview_blocks) < PREVIEW_COUNT:
                # 用普通方块填充剩余位置
                shape = random.choice(['I2', 'I3', 'L2', 'SQ2'])
                temp_shape = f'TEMP_{len(self.preview_blocks)}'
                SHAPES[temp_shape] = SHAPES[shape]
                self.preview_blocks.append((temp_shape, random.choice(BLOCK_COLORS)))
            return

        # 如果没有生成清屏方块组合，使用原来的生成逻辑
        grid_state = self.analyze_grid_state()
        self.preview_blocks = []
        used_shapes = set()
        
        def generate_single_block(remaining_blocks):
            """生成单个方块"""
            # 获取当前状态的模式概率
            mode_probs = self.get_mode_probabilities(grid_state, remaining_blocks)
            
            # 选择生成模式
            mode = random.choices(
                ['single', 'combo', 'challenge'],
                weights=[mode_probs['single'], mode_probs['combo'], mode_probs['challenge']]
            )[0]
            
            # 获取当前模式下的可选形状
            all_shapes = self.select_shapes_for_mode(mode, grid_state)
            
            # 根据已使用的形状调整概率
            shapes = []
            for shape in all_shapes:
                if shape in used_shapes:
                    # 如果形状已经使用过，只有20%的概率被再次选择（原来是30%）
                    if random.random() < 0.2:
                        shapes.append(shape)
                else:
                    shapes.append(shape)
            
            # 如果没有可用形状（极少情况），使用所有形状
            if not shapes:
                shapes = all_shapes
            
            # 为每个形状评估所有可能的位置
            best_score = float('-inf')
            best_shape = None
            best_rotation = None
            
            # 随难度增加随机性
            difficulty = self.get_difficulty_level()
            randomness = min(0.2, difficulty * 0.02)  # 每级增加2%随机性，最高20%（原来是30%）
            
            for shape in shapes:
                # 生成多个旋转和镜像变体
                for _ in range(4):  # 尝试最多4种不同的旋转/镜像
                    rotated_blocks = self.generate_rotated_shape(shape)
                    temp_shape = f'TEMP_{len(self.preview_blocks)}_{_}'
                    SHAPES[temp_shape] = rotated_blocks
                    
                    # 评估这个形状变体的最佳得分
                    max_shape_score = float('-inf')
                    for y in range(GRID_HEIGHT):
                        for x in range(GRID_WIDTH):
                            if self.can_place_block(temp_shape, x, y):
                                score, _ = self.evaluate_block_placement(temp_shape, x, y, grid_state)
                                # 添加随机波动
                                score *= (1 + random.uniform(-randomness, randomness))
                                max_shape_score = max(max_shape_score, score)
                    
                    if max_shape_score > best_score:
                        best_score = max_shape_score
                        best_shape = shape
                        best_rotation = rotated_blocks
            
            if best_shape and best_rotation:
                used_shapes.add(best_shape)  # 记录使用的形状
                temp_shape = f'TEMP_{len(self.preview_blocks)}'
                SHAPES[temp_shape] = best_rotation
                return temp_shape
            
            # 如果没有找到合适的方块，返回基础方块
            temp_shape = f'TEMP_{len(self.preview_blocks)}'
            SHAPES[temp_shape] = SHAPES['I2']
            used_shapes.add('I2')
            return temp_shape

        # 生成每个预览方块
        for i in range(PREVIEW_COUNT):
            shape = generate_single_block(PREVIEW_COUNT - i)
            if shape:
                color = random.choice(BLOCK_COLORS)
                self.preview_blocks.append((shape, color))
        
        # 检查是否有任何方块可以放置
        if not any(self.can_block_be_placed_anywhere(shape) 
                  for shape, _ in self.preview_blocks):
            self.game_over = True

    def can_block_be_placed_anywhere(self, shape):
        """检查方块是否可以放置在任何位置"""
        return any(self.can_place_block(shape, x, y) 
                  for y in range(GRID_HEIGHT) 
                  for x in range(GRID_WIDTH))

    def draw_preview(self):
        """绘制预览区域"""
        preview_y = GRID_OFFSET_Y + GRID_HEIGHT * BLOCK_SIZE + BLOCK_SIZE
        preview_width = SCREEN_WIDTH - 2 * GRID_OFFSET_X
        preview_height = 2 * BLOCK_SIZE
        
        # 绘制预览区背景
        pygame.draw.rect(self.screen, PREVIEW_BG_COLOR,
                        (GRID_OFFSET_X, preview_y, preview_width, preview_height))
        
        # 绘制预览方块
        block_spacing = preview_width // (PREVIEW_COUNT + 1)
        for i, (shape, color) in enumerate(self.preview_blocks):
            if self.dragging and i == self.selected_index:
                continue
            x = GRID_OFFSET_X + block_spacing * (i + 1)
            y = preview_y + preview_height // 2
            self.draw_block(shape, color, (x, y), True)

    def draw_block(self, shape, color, pos, is_preview=False, is_ghost=False):
        """绘制方块"""
        shape_blocks = SHAPES[shape]
        screen_x, screen_y = pos
        min_x, min_y, width, height = self.get_block_offset(shape)

        # 预览区域的方块缩小显示
        scale = 0.6 if is_preview else 1.0
        block_size = int(BLOCK_SIZE * scale)

        if is_preview:
            # 预览区域的方块居中显示
            screen_x -= (width * scale) // 2
            screen_y -= (height * scale) // 2
        elif is_ghost:
            # 预测位置直接使用网格坐标，无需额外调整
            pass
        else:
            # 拖动时的方块完全跟随鼠标
            screen_x -= width // 2
            screen_y -= height // 2

        alpha = 128 if is_ghost else 255
        
        for bx, by in shape_blocks:
            if is_preview:
                x = screen_x + (bx - min_x) * block_size
                y = screen_y + (by - min_y) * block_size
                rect = pygame.Rect(x, y, block_size - 1, block_size - 1)
            else:
                x = screen_x + (bx - min_x) * BLOCK_SIZE
                y = screen_y + (by - min_y) * BLOCK_SIZE
                rect = pygame.Rect(x, y, BLOCK_SIZE - 1, BLOCK_SIZE - 1)

            block_color = (*color[:3], alpha)
            pygame.draw.rect(self.screen, block_color, rect)
            
            if not is_ghost:
                # 光泽效果也要根据缩放调整
                highlight_size = block_size if is_preview else BLOCK_SIZE
                highlight = pygame.Surface((highlight_size - 1, highlight_size//4))
                highlight.fill((255, 255, 255))
                highlight.set_alpha(50)
                self.screen.blit(highlight, (rect.x, rect.y))

    def get_nearest_grid_pos(self, screen_x, screen_y):
        """获取最接近的网格位置"""
        # 考虑方块的中心点偏移
        shape = self.selected_block[0]
        min_x, min_y, width, height = self.get_block_offset(shape)
        
        # 调整鼠标位置，考虑方块中心点
        adjusted_x = screen_x - width // 2
        adjusted_y = screen_y - height // 2
        
        # 计算相对于网格原点的位置
        rel_x = adjusted_x - GRID_OFFSET_X
        rel_y = adjusted_y - GRID_OFFSET_Y
        
        # 计算最近的网格坐标
        grid_x = round(rel_x / BLOCK_SIZE)
        grid_y = round(rel_y / BLOCK_SIZE)
        
        return grid_x, grid_y

    def draw_game_over(self):
        """绘制游戏结束画面"""
        if not self.game_over:
            return
            
        # 创建半透明遮罩
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        # 绘制游戏结束文本
        game_over_text = self.font_large.render("GAME OVER", True, TEXT_COLOR)
        score_text = self.font.render(f"FINAL SCORE: {self.score}", True, TEXT_COLOR)
        restart_text = self.font.render("PRESS SPACE TO RESTART", True, TEXT_COLOR)
        
        self.screen.blit(game_over_text, 
                        game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 40)))
        self.screen.blit(score_text, 
                        score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 20)))
        self.screen.blit(restart_text, 
                        restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 60)))

    def draw(self):
        """绘制游戏界面"""
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_preview()
        self.draw_score()
        
        # 绘制拖动中的方块
        if self.dragging and self.selected_block and self.drag_pos:
            shape, color = self.selected_block
            self.draw_block(shape, color, self.drag_pos)
        
        if self.game_over:
            self.draw_game_over()
        
        pygame.display.flip()

    def handle_click(self, pos):
        """处理鼠标点击"""
        if self.game_over or not self.preview_blocks:
            return

        mouse_x, mouse_y = pos
        preview_y = GRID_OFFSET_Y + GRID_HEIGHT * BLOCK_SIZE + BLOCK_SIZE
        
        if preview_y <= mouse_y <= preview_y + 2 * BLOCK_SIZE:
            preview_width = SCREEN_WIDTH - 2 * GRID_OFFSET_X
            block_spacing = preview_width // (PREVIEW_COUNT + 1)
            
            for i in range(PREVIEW_COUNT):
                block_x = GRID_OFFSET_X + block_spacing * (i + 1)
                if abs(mouse_x - block_x) < BLOCK_SIZE and i < len(self.preview_blocks):
                    self.selected_block = self.preview_blocks[i]
                    self.selected_index = i
                    self.dragging = True
                    self.drag_pos = pos
                    break

    def handle_release(self, pos):
        """处理鼠标释放"""
        if not self.dragging or not self.selected_block:
            return
            
        # 使用最接近的网格位置
        grid_x, grid_y = self.get_nearest_grid_pos(*pos)
        shape, color = self.selected_block
        
        if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
            if self.can_place_block(shape, grid_x, grid_y):
                self.place_block(shape, grid_x, grid_y)
                self.preview_blocks.pop(self.selected_index)
                self.check_lines()
                
                # 如果没有预览方块了，生成新的一组
                if not self.preview_blocks:
                    self.generate_new_block_set()
                # 如果当前没有任何方块可以放置，游戏结束
                elif self.check_game_over():
                    self.game_over = True
        
        self.dragging = False
        self.selected_block = None
        self.selected_index = None
        self.drag_pos = None

    def handle_keydown(self, key):
        """处理键盘按键"""
        if key == pygame.K_SPACE and self.game_over:
            self.reset_game()

    def run(self):
        """运行游戏��循环"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.handle_release(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.handle_motion(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event.key)

            self.draw()
            self.clock.tick(60)

        pygame.quit()

    def handle_motion(self, pos):
        """处理鼠标移动"""
        if self.dragging:
            self.drag_pos = pos

    def place_block(self, shape, x, y):
        """在指定位置放置方块"""
        blocks = SHAPES[shape]
        _, color = self.selected_block
        
        # 计算方块大小（方块数量）
        block_size = len(blocks)
        
        # 放置方块
        for block_x, block_y in blocks:
            grid_x = x + block_x
            grid_y = y + block_y
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                self.grid[grid_y][grid_x] = 1  # 标记为已占用
                self.grid_colors[grid_y][grid_x] = color  # 存储颜色信息
        
        # 增加基于方块大小的分数
        placement_score = block_size * 1  # 每个方块格子1分
        self.score += placement_score

    def can_place_block(self, shape, grid_x, grid_y, test_grid=None):
        """检查是否可以放置方块"""
        if test_grid is None:
            test_grid = self.grid
            
        shape_blocks = SHAPES[shape]
        # 计算方块的偏移量
        min_x = min(bx for bx, _ in shape_blocks)
        min_y = min(by for _, by in shape_blocks)
        
        for bx, by in shape_blocks:
            # 应用偏移量来正确检查位置
            x = grid_x + (bx - min_x)
            y = grid_y + (by - min_y)
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                return False
            if test_grid[y][x] != 0:
                return False
        return True

    def check_lines(self):
        """检查并消除完整的行和列"""
        lines_cleared = 0
        # 检查行
        for y in range(GRID_HEIGHT):
            if all(self.grid[y]):
                self.grid[y] = np.zeros(GRID_WIDTH)
                self.grid_colors[y] = np.zeros((GRID_WIDTH, 3))  # 清除颜色信息
                lines_cleared += 1

        # 检查列
        for x in range(GRID_WIDTH):
            if all(self.grid[:, x]):
                self.grid[:, x] = np.zeros(GRID_HEIGHT)
                self.grid_colors[:, x] = np.zeros((GRID_HEIGHT, 3))  # 清除颜色信息
                lines_cleared += 1

        if lines_cleared > 0:
            self.combo += 1
            base_score = lines_cleared * 100  # 每行/列100分
            combo_bonus = (self.combo - 1) * 100  # 连击奖励每次增加100分
            self.score += base_score + combo_bonus
        else:
            self.combo = 0

    def draw_grid(self):
        """绘制游戏网格"""
        grid_surface = pygame.Surface((GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE))
        grid_surface.fill(GRID_COLOR)
        
        # 绘制已放置的方块
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == 1:
                    rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE,
                                     BLOCK_SIZE - 1, BLOCK_SIZE - 1)
                    color = tuple(self.grid_colors[y][x])
                    pygame.draw.rect(grid_surface, color, rect)

        self.screen.blit(grid_surface, (GRID_OFFSET_X, GRID_OFFSET_Y))

    def draw_score(self):
        """绘制分数和难度等级"""
        score_text = f"Score: {self.score}"
        level_text = f"Level: {self.get_difficulty_level()}"
        score_surface = self.font.render(score_text, True, TEXT_COLOR)
        level_surface = self.font.render(level_text, True, TEXT_COLOR)
        self.screen.blit(score_surface, (BLOCK_SIZE, BLOCK_SIZE // 2))
        self.screen.blit(level_surface, (SCREEN_WIDTH - 3 * BLOCK_SIZE, BLOCK_SIZE // 2))

    def get_state(self):
        """获取游戏状态（用于AI训练）"""
        return {
            'grid': self.grid.copy(),
            'preview_blocks': self.preview_blocks.copy(),
            'score': self.score
        }

if __name__ == "__main__":
    game = BlockBlast()
    game.run() 