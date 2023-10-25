"""多腕バンディット問題のマシンの実装."""

import numpy as np


class NormalMachine:
    """正規分布に従うマシン."""

    MAX_SCORE = 100
    MIN_SCORE = 0

    def __init__(self, mean: float, stddev: float) -> None:
        """正規分布マシンの初期化.

        Args:
            mean (float): 平均値
            stddev (float): 標準偏差
        """
        self.mean = mean
        self.stddev = stddev

    def sample(self) -> int:
        """サンプル値を取得して返す.

        Returns:
            int: サンプル値
        """
        r = np.random.normal(self.mean, self.stddev)
        r = np.clip(r, self.MIN_SCORE, self.MAX_SCORE)
        r = int(r)
        return r
