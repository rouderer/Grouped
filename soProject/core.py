import numpy as np
import random
from typing import List, Tuple, Set
import pandas as pd  # 导入 pandas 用于处理表格
import os


class CombinatorialTestingSO:
    """
    基于蛇优化算法(SO)的组合测试用例生成器
    采用变长编码策略
    """
    def __init__(self, factors: List[int], t_way: int = 2, pop_size: int = 20, max_iter: int = 50,
                 max_test_size: int = 50):
        self.factors = factors
        self.k = len(factors)
        self.t_way = t_way
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.max_test_size = max_test_size

        self.total_interactions = self._calculate_total_interactions()
        self.best_fitness_history = []

    def _calculate_total_interactions(self) -> int:
        if self.t_way != 2:
            raise NotImplementedError("仅实现 t=2 的情况")
        total = 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                total += self.factors[i] * self.factors[j]
        return total

    def _generate_random_test_case(self) -> List[int]:
        return [random.randint(0, self.factors[i] - 1) for i in range(self.k)]

    def _create_initial_population(self) -> List[List[int]]:
        population = []
        for _ in range(self.pop_size):
            num_cases = random.randint(1, self.max_test_size)
            individual = []
            for _ in range(num_cases):
                individual.extend(self._generate_random_test_case())
            population.append(individual)
        return population

    def _decode_individual(self, individual: List[int]) -> List[List[int]]:
        test_cases = []
        max_possible = len(individual) // self.k
        num_cases = min(max_possible, self.max_test_size)
        for i in range(num_cases):
            start_idx = i * self.k
            end_idx = start_idx + self.k
            tc = individual[start_idx:end_idx]
            test_cases.append([int(x) for x in tc])
        return test_cases

    def _fitness_function(self, individual: List[int]) -> float:
        test_cases = self._decode_individual(individual)
        if not test_cases:
            return 0.0

        covered = set()
        for tc in test_cases:
            for idx1 in range(self.k):
                for idx2 in range(idx1 + 1, self.k):
                    pair_key = (idx1, tc[idx1], idx2, tc[idx2])
                    covered.add(pair_key)

        coverage = len(covered) / self.total_interactions

        if coverage < 1.0:
            length_penalty = 0.001 * len(test_cases)
            return coverage - length_penalty
        else:
            normalized_efficiency = (self.max_test_size - len(test_cases)) / self.max_test_size
            return 1.0 + normalized_efficiency

    def _discrete_update(self, x_i: List[int], x_target: List[int], temp: float, is_exploration: bool) -> List[int]:
        x_new = np.array(x_i).copy()
        dim = len(x_new)

        if dim == 0:
            return self._generate_random_test_case()

        mutation_rate = 0.3 * (1 + temp) if is_exploration else 0.05 * (1 - temp)

        for d in range(dim):
            if random.random() < mutation_rate:
                param_idx = d % self.k
                x_new[d] = random.randint(0, self.factors[param_idx] - 1)

        if random.random() < 0.2:
            action = random.choice(['add', 'remove', 'swap'])
            if action == 'add' and len(x_new) < self.max_test_size * self.k:
                new_gene = random.randint(0, max(self.factors) - 1)
                insert_pos = random.randint(0, len(x_new))
                x_new = np.insert(x_new, insert_pos, new_gene)
            elif action == 'remove' and len(x_new) > self.k:
                remove_pos = random.randint(0, len(x_new) - 1)
                x_new = np.delete(x_new, remove_pos)
            elif action == 'swap' and len(x_target) > 0:
                min_len = min(len(x_new), len(x_target))
                if min_len >= 2:
                    start = random.randint(0, min_len - 2)
                    end = start + 2
                    x_new[start:end] = x_target[start:end]

        for i in range(len(x_new)):
            param_idx = i % self.k
            if self.factors[param_idx] > 0:
                x_new[i] = int(x_new[i]) % self.factors[param_idx]

        return x_new.tolist()

    def optimize(self) -> Tuple[List[List[int]], float, List[float]]:
        # 初始化
        population = self._create_initial_population()
        fitness = np.array([self._fitness_function(ind) for ind in population])

        best_idx = np.argmax(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.best_fitness_history.append(best_fitness)

        # 迭代优化
        for t in range(self.max_iter):
            Temp = np.exp(-t / self.max_iter)
            Q = 0.5 * np.exp((t - self.max_iter) / self.max_iter)

            new_population = []
            for i in range(self.pop_size):
                x_i = population[i]

                if Q < 0.25:
                    j = random.randint(0, self.pop_size - 1)
                    x_new = self._discrete_update(x_i, population[j], Temp, True)
                else:
                    if Temp > 0.6:
                        x_new = self._discrete_update(x_i, best_individual, Temp, False)
                    else:
                        sorted_idx = np.argsort(fitness)[::-1]
                        candidates = [population[idx] for idx in sorted_idx[:5]]
                        x_rival = random.choice(candidates)
                        x_new = self._discrete_update(x_i, x_rival, Temp, False)

                new_population.append(x_new)

            population = new_population
            fitness = np.array([self._fitness_function(ind) for ind in population])

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()

            self.best_fitness_history.append(best_fitness)

        best_test_suite = self._decode_individual(best_individual)
        return best_test_suite, best_fitness, self.best_fitness_history

    # 导出到 Excel
    def export_to_excel(self, test_cases: List[List[int]], filename: str = "测试用例结果.xlsx") -> bool:
        try:
            # 创建 DataFrame
            columns = [f"参数{i}" for i in range(len(test_cases[0])) if test_cases]
            df = pd.DataFrame(test_cases, columns=columns)

            # 写入 Excel
            df.to_excel(filename, index=False, engine='openpyxl')

            print(f"\n[导出成功] 测试用例已保存至当前目录: {os.path.abspath(filename)}")
            return True
        except Exception as e:
            print(f"[导出失败] 发生错误: {e}")
            return False

    def run_and_export(self, filename: str = "测试用例结果.xlsx") -> Tuple[List[List[int]], float]:
        print("[开始] 算法执行...")
        best_suite, best_fitness, history = self.optimize()

        print(f"[优化完成] 最优适应度: {best_fitness}")
        print(f"           生成用例数: {len(best_suite)}")
        self.export_to_excel(best_suite, filename)

        return best_suite, best_fitness


if __name__ == "__main__":
    factors = [3,3,4,5,2]
    optimizer = CombinatorialTestingSO(
        factors=factors,
        t_way=2,
        pop_size=50,
        max_iter=100,
        max_test_size=50
    )
    best_cases, fitness, history = optimizer.optimize()
    optimizer.export_to_excel(best_cases, "my_results.xlsx")