"""メインモジュール."""

from environment import NormalMachine

if __name__ == "__main__":
    machine_dict = {
        "A": NormalMachine(60, 10),
        "B": NormalMachine(40, 20),
        "C": NormalMachine(20, 30),
    }

    while True:
        print("wait input: ", end="", flush=True)
        a = input()
        if a == "q":
            break
        v = None
        if a == "A":
            v = machine_dict["A"].sample()
        elif a == "B":
            v = machine_dict["B"].sample()
        elif a == "C":
            v = machine_dict["C"].sample()
        else:
            print(f"invalid input: {a}")
            continue
        print(f"あなたは{a}を選び、スコアは{v}でした")
