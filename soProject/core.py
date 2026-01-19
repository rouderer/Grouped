"""
core.py - 基于蛇优化算法(SO)的组合测试用例生成核心逻辑 (GUI兼容版)
"""
import numpy as np
import random
from typing import List, Tuple

class CombinatorialTestingSO:
    """
    基于蛇优化算法(SO)的组合测试用例生成器
    """

    def __init__(self, factors: List[int], t_way: int = 2, pop_size: int = 20, max_iter: int = 50):
        self.factors = factors
        self.k = len(factors) # 参数个数
        self.t_way = t_way
        self.pop_size = pop_size
        self.max_iter = max_iter

        # 计算所有需要覆盖的t-way组合总数
        self.total_interactions = self._calculate_total_interactions()

        # 用于记录最优解的历史
        self.best_fitness_history = []

    def _calculate_total_interactions(self) -> int:
        """计算t-way组合的总数 (仅支持t=2)"""
        if self.t_way != 2:
            raise NotImplementedError("仅实现 t=2 的情况")
        total = 0
        for i in range(self.k):
            for j in range(i+1, self.k):
                total += self.factors[i] * self.factors[j]
        return total

    def _generate_random_test_case(self) -> List[int]:
        """生成一个随机的测试用例"""
        return [random.randint(0, self.factors[i]-1) for i in range(self.k)]

    def _create_initial_population(self) -> np.ndarray:
        """
        初始化种群。
        """
        # 估计测试套件大小
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
        适应度函数：最大化覆盖率，最小化用例数。
        """
        test_cases = self._decode_individual(individual)

        # 统计覆盖的成对组合
        covered = set()
        for tc in test_cases:
            for idx1 in range(self.k):
                for idx2 in range(idx1+1, self.k):
                    pair_key = (idx1, tc[idx1], idx2, tc[idx2])
                    covered.add(pair_key)

        coverage = len(covered) / self.total_interactions

        # 两阶段评估：先保覆盖，后减规模
        if coverage < 1.0:
            return coverage # 优先追求全覆盖
        else:
            # 全覆盖时，用例越少分数越高
            return 1.0 + (1.0 / len(test_cases))

    def _decode_individual(self, individual: List[int]) -> List[List[int]]:
        """将个体向量解码为测试用例列表"""
        test_cases = []
        for i in range(0, len(individual), self.k):
            if i + self.k <= len(individual):
                tc = individual[i:i+self.k]
                test_cases.append([int(x) for x in tc])
        return test_cases

    def _discrete_update(self, x_i: np.ndarray, x_target: np.ndarray, temp: float, is_exploration: bool) -> np.ndarray:
        """
        离散化更新：模拟蛇的移动行为。
        """
        x_new = x_i.copy()
        dim = len(x_i)

        # 动态调整变异率
        mutation_rate = 0.3 * (1 + temp) if is_exploration else 0.05 * (1 - temp)

        # 策略1: 随机变异 (探索/微调)
        for d in range(dim):
            if random.random() < mutation_rate:
                param_idx = d % self.k
                x_new[d] = random.randint(0, self.factors[param_idx] - 1)

        # 策略2: 基因片段继承 (开发/学习)
        if not is_exploration and random.random() < 0.5:
            start = random.randint(0, max(1, dim - 5))
            end = min(start + 5, dim)
            x_new[start:end] = x_target[start:end]

        return x_new

    def optimize(self) -> Tuple[List[List[int]], float, List[float]]:
        """
        蛇优化算法主循环
        :returns: (测试用例集, 最终适应度, 收敛历史)
        """
        # 1. 初始化
        population = self._create_initial_population()
        fitness = np.array([self._fitness_function(ind) for ind in population])

        best_idx = np.argmax(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # 2. 迭代优化
        for t in range(self.max_iter):
            Temp = np.exp(-t / self.max_iter) # 温度递减
            Q = 0.5 * np.exp((t - self.max_iter) / self.max_iter) # 食物量

            new_population = []

            for i in range(self.pop_size):
                x_i = population[i]
                # --- 行为决策 ---
                if Q < 0.25:
                    # 探索：随机搜索
                    j = random.randint(0, self.pop_size - 1)
                    x_new = self._discrete_update(x_i, population[j], Temp, True)
                else:
                    # 开发：向最优解学习
                    if Temp > 0.6:
                        x_new = self._discrete_update(x_i, best_individual, Temp, False)
                    else:
                        # 低温模式：竞争或交配
                        if random.random() < 0.5:
                            # 竞争：向优秀个体学习
                            sorted_idx = np.argsort(fitness)[::-1] # 降序排列
                            candidates = [population[idx] for idx in sorted_idx[:5]]
                            x_rival = random.choice(candidates)
                            x_new = self._discrete_update(x_i, x_rival, Temp, False)
                        else:
                            x_new = self._discrete_update(x_i, best_individual, Temp, False)

                # 边界修复
                for d in range(len(x_new)):
                    param_idx = d % self.k
                    if self.factors[param_idx] > 0: # 防止除以0
                        x_new[d] = int(x_new[d]) % self.factors[param_idx]

                new_population.append(x_new)

            # 更新种群与最优解
            population = np.array(new_population)
            fitness = np.array([self._fitness_function(ind) for ind in population])

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()

            self.best_fitness_history.append(best_fitness)

        # 3. 输出结果
        best_test_suite = self._decode_individual(best_individual)
        return best_test_suite, best_fitness, self.best_fitness_history