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
    'D': [(0,0), (1,1)]                      # 对角形
}

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

class BlockBlast:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Block Blast")
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        """重置游戏状态"""
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
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

    def generate_new_block_set(self):
        """生成新的一组方块，确保有解"""
        self.preview_blocks = []
        base_shapes = list(SHAPES.keys())
        max_attempts = 100  # 防止无限循环
        
        while len(self.preview_blocks) < PREVIEW_COUNT and max_attempts > 0:
            if not self.preview_blocks:
                # 尝试生成一个可放置的方块
                for _ in range(len(base_shapes)):
                    base_shape = random.choice(base_shapes)
                    rotated_blocks = self.generate_rotated_shape(base_shape)
                    temp_shape = f'TEMP_{len(self.preview_blocks)}'
                    SHAPES[temp_shape] = rotated_blocks
                    if self.can_block_be_placed_anywhere(temp_shape):
                        # 为每个方块随机选择不同的颜色
                        block_color = random.choice(BLOCK_COLORS)
                        self.preview_blocks.append((temp_shape, block_color))
                        break
                    del SHAPES[temp_shape]
            else:
                base_shape = random.choice(base_shapes)
                rotated_blocks = self.generate_rotated_shape(base_shape)
                temp_shape = f'TEMP_{len(self.preview_blocks)}'
                SHAPES[temp_shape] = rotated_blocks
                # 为每个方块随机选择不同的颜色
                block_color = random.choice(BLOCK_COLORS)
                self.preview_blocks.append((temp_shape, block_color))
            
            max_attempts -= 1
        
        if len(self.preview_blocks) < PREVIEW_COUNT or self.check_game_over():
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
        
        # 绘制拖动中的方块和预览
        if self.dragging and self.selected_block and self.drag_pos:
            shape, color = self.selected_block
            
            # 获取最接近的网格位置
            grid_x, grid_y = self.get_nearest_grid_pos(*self.drag_pos)
            
            # 如果在有效位置上，先绘制预览
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                if self.can_place_block(shape, grid_x, grid_y):
                    preview_pos = self.grid_to_screen(grid_x, grid_y)
                    self.draw_block(shape, color, preview_pos, is_ghost=True)
            
            # 绘制拖动的方块（完全跟随鼠标）
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
                # 如果当前没��任何方块可以放置，游戏结束
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
        """运行游戏主循环"""
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

    def place_block(self, shape, grid_x, grid_y):
        """放置方块"""
        shape_blocks = SHAPES[shape]
        min_x = min(bx for bx, _ in shape_blocks)
        min_y = min(by for _, by in shape_blocks)
        
        # 获取当前方块的颜色
        _, color = self.selected_block
        
        for bx, by in shape_blocks:
            x = grid_x + (bx - min_x)
            y = grid_y + (by - min_y)
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                self.grid[y][x] = 1
                self.grid_colors[y][x] = color  # 存储颜色信息

    def can_place_block(self, shape, grid_x, grid_y):
        """检查是否可以放置方块"""
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
            if self.grid[y][x] != 0:
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
            base_score = lines_cleared * 100
            combo_bonus = (self.combo - 1) * 50
            self.score += base_score + combo_bonus
        else:
            self.combo = 0

    def draw_grid(self):
        """绘制游戏网格"""
        grid_surface = pygame.Surface((GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE))
        grid_surface.fill(GRID_COLOR)
        
        # 绘制网格线
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(grid_surface, BACKGROUND_COLOR,
                           (x * BLOCK_SIZE, 0),
                           (x * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE))
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(grid_surface, BACKGROUND_COLOR,
                           (0, y * BLOCK_SIZE),
                           (GRID_WIDTH * BLOCK_SIZE, y * BLOCK_SIZE))

        # 绘制已放置的方块
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == 1:
                    rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE,
                                     BLOCK_SIZE - 1, BLOCK_SIZE - 1)
                    color = tuple(self.grid_colors[y][x])  # 使用存储的颜色
                    pygame.draw.rect(grid_surface, color, rect)
                    # 添加光泽效果
                    highlight = pygame.Surface((BLOCK_SIZE - 1, BLOCK_SIZE//4))
                    highlight.fill((255, 255, 255))
                    highlight.set_alpha(50)
                    grid_surface.blit(highlight, (rect.x, rect.y))

        self.screen.blit(grid_surface, (BLOCK_SIZE, BLOCK_SIZE))

    def draw_score(self):
        """绘制分数和标题"""
        # 绘制标题
        title_text = str(self.score)
        title_surface = self.font_large.render(title_text, True, TEXT_COLOR)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH//2, BLOCK_SIZE//2))
        self.screen.blit(title_surface, title_rect)

        # 绘制皇冠图标
        crown_text = "SCORE"  # 改用文字替代表情符号
        crown_surface = self.font.render(crown_text, True, (255, 215, 0))
        self.screen.blit(crown_surface, (BLOCK_SIZE, BLOCK_SIZE//2))

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