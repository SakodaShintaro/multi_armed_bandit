"""Solverの基底クラス."""


class BaseSolver:
    """Solverの基底クラス."""

    def __init__(self, machine_num: int) -> None:
        """初期化.

        Args:
            machine_num (int): マシンの数
        """
        self.machine_num = machine_num

    def select(self) -> int:
        """実行."""
        raise NotImplementedError()

    def update(self, selected: int, reward: int) -> None:
        """更新.

        Args:
            selected (int): 選択したマシンのインデックス
            reward (int): 報酬
        """
        raise NotImplementedError()
