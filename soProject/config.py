"""
配置中心
"""
import os

class Config:
    # --- 输入配置 ---
    # 定义被测系统的参数（每个参数的取值个数）
    # 例如: [3, 3, 4, 5, 2] 表示5个参数，分别有3,3,4,5,2个取值
    FACTORS = [3, 3, 4, 5, 2]
    T_WAY = 2  # 组合强度

    # --- 算法参数 ---
    POP_SIZE = 50   # 种群大小 (蛇的数量)
    MAX_ITER = 100  # 最大迭代次数

    # --- 路径配置 ---
    OUTPUT_DIR = "output"
    PLOT_PATH = os.path.join(OUTPUT_DIR, "convergence.png")

    def __init__(self):
        # 确保输出目录存在
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

# 全局配置实例
cfg = Config()