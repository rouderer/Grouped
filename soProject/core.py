"""
core.py - 基于蛇优化算法(SO)的组合测试用例生成核心逻辑 (变长编码版)
"""
import numpy as np
import random
from typing import List, Tuple, Set

class CombinatorialTestingSO:
    """
    基于蛇优化算法(SO)的组合测试用例生成器
    采用变长编码策略，解决覆盖率难以达到1的问题
    """

    def __init__(self, factors: List[int], t_way: int = 2, pop_size: int = 20, max_iter: int = 50, max_test_size: int = 50):
        self.factors = factors
        self.k = len(factors) # 参数个数
        self.t_way = t_way
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.max_test_size = max_test_size # 允许生成的最大测试用例数量

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

    def _create_initial_population(self) -> List[List[int]]:
        """
        初始化种群。
        使用变长编码：每个个体的长度（测试用例数）在 [1, max_test_size] 之间随机
        """
        population = []
        for _ in range(self.pop_size):
            # 随机决定该个体包含多少个测试用例
            num_cases = random.randint(1, self.max_test_size)
            individual = []
            for _ in range(num_cases):
                individual.extend(self._generate_random_test_case())
            population.append(individual)
        return population

    def _decode_individual(self, individual: List[int]) -> List[List[int]]:
        """
        将个体向量解码为测试用例列表
        自动处理长度不足或超出的情况
        """
        test_cases = []
        # 计算理论上最多能切分出多少个完整的测试用例
        max_possible = len(individual) // self.k
        # 限制数量不超过最大允许值
        num_cases = min(max_possible, self.max_test_size)

        for i in range(num_cases):
            start_idx = i * self.k
            end_idx = start_idx + self.k
            tc = individual[start_idx:end_idx]
            test_cases.append([int(x) for x in tc])
        return test_cases

    def _fitness_function(self, individual: List[int]) -> float:
        """
        适应度函数：两阶段评估
        阶段一（未全覆盖）：最大化覆盖率
        阶段二（全覆盖）：最小化测试用例数量
        """
        test_cases = self._decode_individual(individual)

        if not test_cases:
            return 0.0

        # 统计覆盖的成对组合
        covered = set()
        for tc in test_cases:
            for idx1 in range(self.k):
                for idx2 in range(idx1+1, self.k):
                    pair_key = (idx1, tc[idx1], idx2, tc[idx2])
                    covered.add(pair_key)

        coverage = len(covered) / self.total_interactions

        # --- 核心逻辑修改：两阶段评估 ---
        if coverage < 1.0:
            # 阶段一：未全覆盖，优先提升覆盖率
            # 这里可以加入微小的长度惩罚，鼓励用更少的用例达到当前覆盖率
            length_penalty = 0.001 * len(test_cases) # 微弱的惩罚，主要看覆盖率
            return coverage - length_penalty
        else:
            # 阶段二：已全覆盖，适应度取决于用例数量越少越好
            # 归一化：(max_size - actual_size) / max_size，值域 (0, 1]
            # 这样保证全覆盖的解永远优于未全覆盖的解
            normalized_efficiency = (self.max_test_size - len(test_cases)) / self.max_test_size
            return 1.0 + normalized_efficiency # 基础分1.0 + 效率分

    def _discrete_update(self, x_i: List[int], x_target: List[int], temp: float, is_exploration: bool) -> List[int]:
        """
        离散化更新：模拟蛇的移动行为（适配变长编码）
        """
        # 转换为 numpy 方便索引，最后再转回 list
        x_new = np.array(x_i).copy()
        dim = len(x_new)

        if dim == 0:
            # 如果为空，直接返回一个随机测试用例
            return self._generate_random_test_case()

        # 动态调整变异率
        mutation_rate = 0.3 * (1 + temp) if is_exploration else 0.05 * (1 - temp)

        # --- 策略1: 随机变异 (探索/微调) ---
        for d in range(dim):
            if random.random() < mutation_rate:
                param_idx = d % self.k
                x_new[d] = random.randint(0, self.factors[param_idx] - 1)

        # --- 策略2: 结构化调整 (增加/删除/替换) ---
        # 随机决定是否进行结构化调整 (20% 概率)
        if random.random() < 0.2:
            action = random.choice(['add', 'remove', 'swap'])
            if action == 'add' and len(x_new) < self.max_test_size * self.k:
                # 添加一个随机基因 (增加测试用例长度)
                new_gene = random.randint(0, max(self.factors)-1)
                insert_pos = random.randint(0, len(x_new))
                x_new = np.insert(x_new, insert_pos, new_gene)

            elif action == 'remove' and len(x_new) > self.k:
                # 移除一个基因 (减少长度，但不能少于1个完整用例)
                remove_pos = random.randint(0, len(x_new)-1)
                x_new = np.delete(x_new, remove_pos)

            elif action == 'swap' and len(x_target) > 0:
                # 随机交换一小段基因
                min_len = min(len(x_new), len(x_target))
                if min_len >= 2:
                    start = random.randint(0, min_len - 2)
                    end = start + 2
                    x_new[start:end] = x_target[start:end]

        # --- 边界修复 ---
        # 确保数值在合法范围内
        for i in range(len(x_new)):
            param_idx = i % self.k
            if self.factors[param_idx] > 0:
                x_new[i] = int(x_new[i]) % self.factors[param_idx]

        return x_new.tolist()

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

        self.best_fitness_history.append(best_fitness)

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
                        # 低温模式：竞争
                        sorted_idx = np.argsort(fitness)[::-1] # 降序排列
                        candidates = [population[idx] for idx in sorted_idx[:5]]
                        x_rival = random.choice(candidates)
                        x_new = self._discrete_update(x_i, x_rival, Temp, False)

                new_population.append(x_new)

            # 更新种群与最优解
            population = new_population
            fitness = np.array([self._fitness_function(ind) for ind in population])

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()

            self.best_fitness_history.append(best_fitness)

        # 3. 输出结果
        best_test_suite = self._decode_individual(best_individual)
        return best_test_suite, best_fitness, self.best_fitness_history