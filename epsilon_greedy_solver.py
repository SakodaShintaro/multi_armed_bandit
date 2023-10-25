"""一定確率でランダムに、それ以外は今までの平均報酬が最も良いものを選択するSolver."""
import random

import numpy as np

from base_solver import BaseSolver


class EpsilonGreedySolver(BaseSolver):
    """一定確率でランダムに、それ以外は今までの平均報酬が最も良いものを選択するSolver."""

    def __init__(self, machine_num: int) -> None:
        """初期化.

        Args:
            machine_num (int): マシンの数
        """
        super().__init__(machine_num)
        self.epsilon = 0.1
        self.reward_sums = [0] * machine_num
        self.selection_counts = [0] * machine_num

    def select(self) -> int:
        """実行."""
        if random.random() < self.epsilon:
            # ランダムに選択
            return random.randrange(self.machine_num)
        else:
            # 最良の選択肢を選ぶ
            # まだ選ばれていない選択肢がある場合はそれを選ぶ
            if 0 in self.selection_counts:
                return self.selection_counts.index(0)
            else:
                average_rewards = [
                    reward_sum / count
                    for reward_sum, count in zip(
                        self.reward_sums, self.selection_counts
                    )
                ]
                return int(np.argmax(average_rewards))

    def update(self, selected: int, reward: int) -> None:
        """更新."""
        self.reward_sums[selected] += reward
        self.selection_counts[selected] += 1
