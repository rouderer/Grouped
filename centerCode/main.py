import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple


class CombinatorialTestingSO:
    """
    基于蛇优化算法(SO)的组合测试用例生成器
    """

    def __init__(self, factors: List[int], t_way: int = 2, pop_size: int = 20, max_iter: int = 50):
        """
        初始化
        :param factors: 因素列表，例如 [3, 3, 4] 表示3个参数，分别有3,3,4个个值
        :param t_way: 组合强度，通常为2（成对组合）
        :param pop_size: 种群大小（蛇的数量）
        :param max_iter: 最大迭代次数
        """
        self.factors = factors
        self.k = len(factors)  # 参数个数
        self.t_way = t_way
        self.pop_size = pop_size
        self.max_iter = max_iter

        # 计算所有需要覆盖的t-way组合总数 (用于计算覆盖率)
        self.total_interactions = self._calculate_total_interactions()

        # 用于记录最优解的历史
        self.best_fitness_history = []

    def _calculate_total_interactions(self) -> int:
        """计算t-way组合的总数 (简化版，假设t=2)"""
        if self.t_way != 2:
            raise NotImplementedError("仅实现 t=2 的情况")
        total = 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                total += self.factors[i] * self.factors[j]
        return total

    def _generate_random_test_case(self) -> List[int]:
        """生成一个随机的测试用例"""
        return [random.randint(0, self.factors[i] - 1) for i in range(self.k)]

    def _create_initial_population(self) -> np.ndarray:
        """
        初始化种群。
        每个个体（蛇）代表一个测试用例集。
        这里我们简化模型：每个个体是一个固定长度的测试用例序列。
        """
        # 简化假设：我们尝试构建一个大小为 N 的测试套件
        # 为了简单，我们固定每个个体的测试用例数量为一个估计值（例如 2 * max(factor)）
        self.N = max(self.factors) * 2

        population = []
        for _ in range(self.pop_size):
            suite = []
            for _ in range(self.N):
                suite.extend(self._generate_random_test_case())
            population.append(suite)
        return np.array(population)

    def _fitness_function(self, individual: List[int]) -> float:
        """
        适应度函数。
        目标：覆盖所有成对组合，且测试用例越少越好。
        这里我们先展平个体计算覆盖，但惩罚冗余用例。
        """
        # 将个体解码为测试用例列表
        test_cases = []
        for i in range(0, len(individual), self.k):
            if i + self.k <= len(individual):
                tc = individual[i:i + self.k]
                test_cases.append(tc)

        # 统计覆盖的成对组合 (使用集合去重)
        covered = set()
        for tc in test_cases:
            for idx1 in range(self.k):
                for idx2 in range(idx1 + 1, self.k):
                    # 创建一个标识符来表示参数idx1的值tc[idx1]和参数idx2的值tc[idx2]的组合
                    pair_key = (idx1, tc[idx1], idx2, tc[idx2])
                    covered.add(pair_key)

        # 覆盖率
        coverage = len(covered) / self.total_interactions

        # 惩罚：如果覆盖率未满100%，适应度主要看覆盖率；
        # 如果覆盖率满了，适应度看用例数量（越少越好）
        if coverage < 1.0:
            # 优先提升覆盖率
            fitness = coverage
        else:
            # 覆盖率满时，惩罚用例数量（用例越少，适应度越高/惩罚越低）
            # 这里我们用 (N - k) 作为惩罚项，k是参数个数（理论最小值下限）
            penalty = len(test_cases)
            # 归一化惩罚，使其与 coverage 同量纲
            fitness = 1.0 + (1.0 / penalty)  # 用例越少，1/penalty越大

        return fitness

    def _discrete_update(self, x_i: np.ndarray, x_target: np.ndarray, temp: float, is_exploration: bool) -> np.ndarray:
        """
        离散化更新操作。
        在连续空间中是加减法，在离散空间中我们将其解释为“基于概率的变异”或“混合”。

        :param x_i: 当前个体（蛇）
        :param x_target: 目标个体（食物/最优解/异性）
        :param temp: 温度（控制开发/探索程度）
        :param is_exploration: 是否处于探索阶段
        :return: 更新后的新个体
        """
        x_new = x_i.copy()
        dim = len(x_i)

        # 基于温度和阶段调整变异概率
        if is_exploration:
            # 探索阶段：高变异，随机搜索
            mutation_rate = 0.3 * (1 + temp)  # 温度高时变异大
        else:
            # 开发阶段：低变异，精细调整
            mutation_rate = 0.05 * (1 - temp)  # 温度低时微调

        for d in range(dim):
            if random.random() < mutation_rate:
                # 变异操作：随机改变该位置的基因（取值需符合该参数的水平数）
                # 计算这是第几个参数 (根据索引 d 计算)
                param_idx = d % self.k
                max_level = self.factors[param_idx]
                x_new[d] = random.randint(0, max_level - 1)

        # 额外的“交叉”操作：如果在开发阶段，部分基因直接继承自目标（最优解）
        if not is_exploration and random.random() < 0.5:
            # 从目标解中复制一段基因
            start = random.randint(0, dim - 5)
            end = start + 5
            x_new[start:end] = x_target[start:end]

        return x_new

    def optimize(self) -> Tuple[List[List[int]], float]:
        """
        蛇优化算法主循环
        """
        # 1. 初始化种群
        population = self._create_initial_population()
        fitness = np.array([self._fitness_function(ind) for ind in population])

        # 2. 找到初始最优解
        best_idx = np.argmax(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        print(f"初始最优适应度: {best_fitness:.4f}")

        # 3. 迭代优化
        for t in range(self.max_iter):
            # 计算环境参数
            Temp = np.exp(-t / self.max_iter)  # 温度，随迭代降低
            Q = 0.5 * np.exp((t - self.max_iter) / self.max_iter)  # 食物量，初期高，后期低

            new_population = []

            for i in range(self.pop_size):
                x_i = population[i]
                current_fitness = fitness[i]

                # --- 行为决策 ---
                if Q < 0.25:
                    # 探索阶段 (Exploration): 随机搜索
                    # 模拟公式 (6, 8)
                    j = random.randint(0, self.pop_size - 1)
                    x_rand = population[j]
                    x_new = self._discrete_update(x_i, x_rand, Temp, is_exploration=True)

                else:
                    # 开发阶段 (Exploitation)
                    if Temp > 0.6:
                        # 高温：向食物移动 (开发)
                        # 模拟公式 (10)
                        x_new = self._discrete_update(x_i, best_individual, Temp, is_exploration=False)
                    else:
                        # 低温：战斗与交配模式
                        # 模拟公式 (11-18)
                        # 简化：有一定概率进行更激烈的变异或与最优解混合
                        if random.random() < 0.5:
                            # 战斗/竞争：与邻近的较好解混合
                            candidates = [population[idx] for idx in np.argsort(fitness)[-5:]]  # 选前5个优胜者
                            x_rival = random.choice(candidates)
                            x_new = self._discrete_update(x_i, x_rival, Temp, is_exploration=False)
                        else:
                            # 交配/孵化：直接引入部分最优解的基因或随机重生部分
                            # 这里简化为对当前解进行小幅度扰动
                            x_new = self._discrete_update(x_i, best_individual, Temp, is_exploration=False)

                # 边界处理 (确保值在合法范围内)
                for d in range(len(x_new)):
                    param_idx = d % self.k
                    x_new[d] = int(x_new[d]) % self.factors[param_idx]  # 取模确保在范围内

                new_population.append(x_new)

            # 更新种群
            population = np.array(new_population)
            fitness = np.array([self._fitness_function(ind) for ind in population])

            # 更新全局最优
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()

            # 记录历史
            self.best_fitness_history.append(best_fitness)

            if t % 10 == 0:
                print(f"迭代 {t}: 最优适应度 = {best_fitness:.4f}")

        # 4. 解码最优解为测试用例集
        final_test_suite = []
        for i in range(0, len(best_individual), self.k):
            if i + self.k <= len(best_individual):
                tc = best_individual[i:i + self.k]
                final_test_suite.append([int(x) for x in tc])

        return final_test_suite, best_fitness

    def plot_convergence(self):
        """绘制收敛曲线"""
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='最优适应度')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值')
        plt.title('蛇优化算法收敛曲线')
        plt.legend()
        plt.grid(True)
        plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    # 定义测试问题：例如 5 个参数，分别有 3, 3, 4, 5, 2 个取值
    # 这是一个经典的组合测试基准问题
    factors = [3, 3, 4, 5, 2]
    t_way = 2

    print("基于蛇优化算法的组合测试用例生成")
    print(f"测试配置: {factors}")

    # 创建优化器
    # 注意：为了演示快速运行，这里将种群和迭代设得较小，实际工程需调大
    ctso = CombinatorialTestingSO(factors, t_way=t_way, pop_size=50, max_iter=100)

    # 执行优化
    test_suite, score = ctso.optimize()

    print("\n" + "=" * 50)
    print(f"优化完成！生成的测试用例集 (共 {len(test_suite)} 条):")
    for i, case in enumerate(test_suite):
        print(f"TC{i + 1}: {case}")

    # 绘制收敛图
    ctso.plot_convergence()