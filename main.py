"""メインモジュール."""

from environment import NormalMachine

from random_solver import RandomSolver

if __name__ == "__main__":
    machine_list = [
        NormalMachine(mean=10, stddev=5),
        NormalMachine(mean=20, stddev=5),
        NormalMachine(mean=30, stddev=5),
    ]
    solver = RandomSolver(len(machine_list))
    reward_sum = 0
    TRIAL_NUM = 100
    for _ in range(TRIAL_NUM):
        selected = solver.select()
        reward = machine_list[selected].sample()
        solver.update(selected, reward)
        print(f"selected: {selected}, reward: {reward}")
        reward_sum += reward

    reward_average = reward_sum / TRIAL_NUM
    print(f"reward_average: {reward_average:.1f}")
