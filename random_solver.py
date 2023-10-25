"""ランダムに選択するSolver."""

import random

from base_solver import BaseSolver


class RandomSolver(BaseSolver):
    """ランダムに選択するSolver."""

    def select(self) -> int:
        """実行."""
        # ランダムに選択
        return random.randrange(0, self.machine_num)

    def update(self, selected: int, reward: int) -> None:
        """更新."""
        pass
