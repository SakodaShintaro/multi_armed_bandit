"""ChatGPT3で選択するSolver."""

import logging
import os
from datetime import datetime

import openai

from base_solver import BaseSolver

openai.api_key = os.environ["OPENAI_API_KEY"]


class ChatGPT3Solver(BaseSolver):
    """ChatGPT3で選択するSolver."""

    def __init__(self, machine_num: int, trial_num: int) -> None:
        """初期化.

        Args:
            machine_num (int): マシンの数
            trial_num (int): 試行回数
        """
        super().__init__(machine_num)

        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"{log_dir}/{current_time}_chat_gpt3_solver.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.conversation_history = []
        system_prompt = f"""あなたは多腕バンディット問題のSolverです。
        これから{trial_num}回、マシンを1つ選択して報酬を得ることを繰り返します。
        マシンの数は{self.machine_num}個です。
        得られる報酬の和が最大になるように選択してください。
        この問題を解くプログラムを書くのではなく、あなた自身が選択してください
        """
        self.conversation_history.append({"role": "system", "content": system_prompt})
        self.count = 0

    def select(self) -> int:
        """実行."""
        self.count += 1
        input_str = f"""{self.count}回目の選択です。どのマシンを選びますか？
        選択するための思考を出力しても構いません。
        最終的な選択は0から{self.machine_num - 1}の数字を1つだけ最終行に[]で囲って出力してください。
        最終行にはそれ以外の文字は何も出力しないでください。
        出力すべき最終行の例) [0]
        """
        self.conversation_history.append({"role": "user", "content": input_str})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.conversation_history
        )

        response_text = response["choices"][0]["message"]["content"].strip()
        self.conversation_history.append(
            {"role": "assistant", "content": response_text}
        )
        self.logger.info(f"response_text: {response_text}")

        selected_index = int(response_text[-2])
        return selected_index

    def update(self, selected: int, reward: int) -> None:
        """更新."""
        input_str = f"{self.count}回目の選択として{selected}番目のマシンを選んだところ、今回の報酬は{reward}でした。"
        self.conversation_history.append({"role": "user", "content": input_str})
        self.logger.info(f"update: {input_str}")
