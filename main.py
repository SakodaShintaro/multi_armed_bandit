"""メインモジュール."""

import argparse

from base_solver import BaseSolver
from chat_gpt3_solver import ChatGPT3Solver
from environment import NormalMachine
from epsilon_greedy_solver import EpsilonGreedySolver
from random_solver import RandomSolver


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する.

    Returns:
        argparse.Namespace: 解析された引数オブジェクト
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "solver_type",
        type=str,
        default="epsilon_greedy",
        choices=["random", "epsilon_greedy", "chat_gpt3"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solver_type = args.solver_type
    machine_list = [
        NormalMachine(mean=10, stddev=5),
        NormalMachine(mean=20, stddev=5),
        NormalMachine(mean=30, stddev=5),
    ]
    TRIAL_NUM = 100
    solver: BaseSolver
    if solver_type == "random":
        solver = RandomSolver(len(machine_list))
    elif solver_type == "epsilon_greedy":
        solver = EpsilonGreedySolver(len(machine_list))
    elif solver_type == "chat_gpt3":
        solver = ChatGPT3Solver(len(machine_list), TRIAL_NUM)
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")
    reward_sum = 0
    for _ in range(TRIAL_NUM):
        selected = solver.select()
        reward = machine_list[selected].sample()
        solver.update(selected, reward)
        print(f"selected: {selected}, reward: {reward}")
        reward_sum += reward

    reward_average = reward_sum / TRIAL_NUM
    print(f"reward_average: {reward_average:.1f}")
