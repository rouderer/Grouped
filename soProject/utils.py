"""
utils.py - 工具函数库 (包含日志、绘图和验证)
"""
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 日志系统 (整合在 utils 中，避免循环导入) ---
# 创建一个全局的日志实例
log = logging.getLogger("CTSO_Tool")
log.setLevel(logging.INFO)

# 防止重复添加 Handler (多次运行 main 时)
if not log.handlers:
    # 控制台输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

# --- 2. 配置与工具 ---
def setup_matplotlib():
    """
    解决 Matplotlib 中文显示问题
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    log.info("Matplotlib 字体配置完成")

def save_plot(history, path):
    """
    保存收敛曲线图
    """
    setup_matplotlib()
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-o', label='最优适应度')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.title('蛇优化算法收敛曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    log.info(f"收敛曲线已保存至: {path}")

def calculate_coverage(test_suite, factors, t_way=2):
    """
    独立计算测试套件的覆盖率 (用于验证结果)
    """
    if t_way != 2:
        raise ValueError("仅支持 t=2")

    covered = set()
    total_possible = 0

    # 计算理论最大组合数
    k = len(factors)
    for i in range(k):
        for j in range(i+1, k):
            total_possible += factors[i] * factors[j]

    # 统计实际覆盖
    for tc in test_suite:
        for i in range(k):
            for j in range(i+1, k):
                pair = (i, tc[i], j, tc[j])
                covered.add(pair)

    coverage_rate = len(covered) / total_possible if total_possible > 0 else 0
    log.info(f"覆盖率验证: {len(covered)}/{total_possible} ({coverage_rate:.2%})")
    return coverage_rate, len(covered), total_possible