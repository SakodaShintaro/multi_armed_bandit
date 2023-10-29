"""StableLMで選択するSolver."""

import logging
import os
import time
from datetime import datetime
from base_solver import BaseSolver
from transformers import AutoModelForCausalLM, AutoTokenizer


class StableLMSolver(BaseSolver):
    """StableLMで選択するSolver."""

    def __init__(self, machine_num: int, trial_num: int) -> None:
        """初期化.

        Args:
            machine_num (int): マシンの数
            trial_num (int): 試行回数
        """
        super().__init__(machine_num)
        self.trial_num = trial_num

        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"{log_dir}/{current_time}_stable_lm_solver.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/japanese-stablelm-instruct-gamma-7b"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/japanese-stablelm-instruct-gamma-7b",
            torch_dtype="auto",
        )
        self.model.eval()
        self.model.to("cuda")

        self.conversation_history = []
        self.count = 0

    def build_prompt(self) -> str:
        """Promptを作成する."""
        p = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ## 指示
        あなたは多腕バンディット問題のSolverです。
        これから{self.trial_num}回、マシンを1つ選択して報酬を得ることを繰り返します。
        マシンの数は{self.machine_num}個です。
        得られる報酬の和が最大になるように選択してください。

        ## 応答
        """

        for item in self.conversation_history:
            p += f"{item['role']}: {item['content']}\n"

        p += "\nあなたの回答:"

        return p

    def select(self) -> int:
        """実行."""
        self.count += 1
        input_str = f"""{self.count}回目の選択です。どのマシンを選びますか？
        最終的な選択は0から{self.machine_num - 1}の数字を1つだけ最終行に[]で囲って出力してください。
        最終行にはそれ以外の文字は何も出力しないでください。
        出力すべき最終行の例) [0]
        """
        self.conversation_history.append({"role": "マシン", "content": input_str})

        while True:
            try:
                self.logger.info(f"推論開始: {len(self.conversation_history)}")

                prompt = self.build_prompt()
                self.logger.info(prompt)
                input_ids = self.tokenizer.encode(
                    prompt, add_special_tokens=False, return_tensors="pt")

                tokens = self.model.generate(
                    input_ids.to(device=self.model.device),
                    max_new_tokens=512,
                    temperature=1,
                    top_p=0.95,
                    do_sample=True,
                )

                response_text = self.tokenizer.decode(
                    tokens[0][input_ids.shape[1] :], skip_special_tokens=True
                ).strip()

                self.conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )
                self.logger.info(f"response_text: {response_text}")

                selected_index = int(response_text[-2])
                return selected_index
            except ValueError as e:
                self.logger.error(f"ValueError: {e}")
                error_str = (
                    f"最終的な選択は0から{self.machine_num - 1}の数字を1つだけ最終行に[]で囲って出力してください。"
                )
                self.conversation_history.append({"role": "マシン", "content": error_str})
                time.sleep(1)

    def update(self, selected: int, reward: int) -> None:
        """更新."""
        input_str = f"{self.count}回目の選択として{selected}番目のマシンを選んだところ、今回の報酬は{reward}でした。"
        self.conversation_history.append({"role": "マシン", "content": input_str})
        self.logger.info(f"update: {input_str}")
