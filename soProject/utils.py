import logging
import os
import matplotlib.pyplot as plt
import numpy as np

# 日志系统
log = logging.getLogger("CTSO_Tool")
log.setLevel(logging.INFO)

# 防止重复添加 Handler
if not log.handlers:

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

def setup_matplotlib():

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    log.info("Matplotlib 字体配置完成")

def save_plot(history, path):

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


    if t_way != 2:
        raise ValueError("仅支持 t=2")

    covered = set()
    total_possible = 0


    k = len(factors)
    for i in range(k):
        for j in range(i+1, k):
            total_possible += factors[i] * factors[j]


    for tc in test_suite:
        for i in range(k):
            for j in range(i+1, k):
                pair = (i, tc[i], j, tc[j])
                covered.add(pair)

    coverage_rate = len(covered) / total_possible if total_possible > 0 else 0
    log.info(f"覆盖率验证: {len(covered)}/{total_possible} ({coverage_rate:.2%})")
    return coverage_rate, len(covered), total_possible