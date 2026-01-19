"""
main.py - 系统启动文件
"""
from core import CombinatorialTestingSO
from config import cfg
from utils import save_plot, calculate_coverage, log


def main():
    log.info("=== 基于蛇优化算法的组合测试用例生成系统 启动 ===")

    # 1. 初始化优化器
    ctso = CombinatorialTestingSO(
        factors=cfg.FACTORS,
        t_way=cfg.T_WAY,
        pop_size=cfg.POP_SIZE,
        max_iter=cfg.MAX_ITER
    )

    # 2. 执行优化
    test_suite, final_score = ctso.optimize()

    # 3. 结果验证与输出
    print("\n" + "=" * 60)
    print("优化完成！最终结果报告")
    print("=" * 60)

    coverage_rate, covered, total = calculate_coverage(test_suite, cfg.FACTORS, cfg.T_WAY)

    print(f"参数配置     : {cfg.FACTORS}")
    print(f"理论组合数   : {total}")
    print(f"实际覆盖数   : {covered}")
    print(f"覆盖率       : {coverage_rate:.2%}")
    print(f"生成用例数量 : {len(test_suite)}")
    print(f"适应度评分   : {final_score:.4f}")

    print("\n生成的测试用例集:")
    for i, case in enumerate(test_suite):
        print(f"TC{i + 1:2d}: {case}")

    # 4. 保存图表
    save_plot(ctso.best_fitness_history, cfg.PLOT_PATH)


if __name__ == "__main__":
    main()