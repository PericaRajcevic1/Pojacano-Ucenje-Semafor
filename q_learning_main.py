# q_learning_main.py
import random
from statistics import mean
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt

from env import TrafficEnv, TrafficEnvConfig
from value_iteration import TrafficMDPModel, value_iteration
from q_learning import q_learning_train, greedy_policy_from_Q


State = Tuple[int, int, int]


def run_random(env: TrafficEnv, num_episodes: int = 30) -> List[float]:
    returns = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        G = 0.0
        while not done:
            action = random.randint(0, 1)
            next_state, reward, done, info = env.step(action)
            G += reward
            state = next_state
        returns.append(G)
    return returns


def run_policy(env: TrafficEnv, policy: Dict[State, int], num_episodes: int = 30) -> List[float]:
    returns = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        G = 0.0
        while not done:
            action = policy.get(state, 0)  # default STAY ako nema u politici
            next_state, reward, done, info = env.step(action)
            G += reward
            state = next_state
        returns.append(G)
    return returns


def moving_average(data, window=50):
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start : i + 1]) / (i - start + 1))
    return result


def main():
    cfg = TrafficEnvConfig(
        max_queue=5,
        p_arrival_ns=0.4,
        p_arrival_ew=0.4,
        cars_through_green=1,
        switch_penalty=1.0,
        max_steps=200,
    )

    # 1) Treniranjem dobijemo Q tablicu
    print("=== Treniranje Q-learning agenta ===")
    Q, train_returns = q_learning_train(
        cfg,
        num_episodes=3000,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    )

    # 2) Greedy politika iz Q tablice
    q_policy = greedy_policy_from_Q(Q, cfg)

    print("\nPrimjer nekoliko stanja i akcija iz Q-learning politike:")
    example_states = list(q_policy.keys())[:10]
    for s in example_states:
        print(f"  {s} -> {q_policy[s]} (0=STAY, 1=SWITCH)")

    # 3) Value iteration politika (za usporedbu)
    mdp_model = TrafficMDPModel(cfg, gamma=0.95)
    _, vi_policy = value_iteration(mdp_model)

    # 4) Evaluacija sve tri politike u stohastičkom env-u
    eval_env_random = TrafficEnv(cfg)
    eval_env_vi = TrafficEnv(cfg)
    eval_env_q = TrafficEnv(cfg)

    num_eval_episodes = 50

    print("\n=== Evaluacija RANDOM politike ===")
    random_returns = run_random(eval_env_random, num_episodes=num_eval_episodes)
    print(f"Prosječna nagrada (random): {mean(random_returns):.2f}")

    print("\n=== Evaluacija VI politike ===")
    vi_returns = run_policy(eval_env_vi, vi_policy, num_episodes=num_eval_episodes)
    print(f"Prosječna nagrada (Value Iteration): {mean(vi_returns):.2f}")

    print("\n=== Evaluacija Q-learning politike ===")
    q_returns = run_policy(eval_env_q, q_policy, num_episodes=num_eval_episodes)
    print(f"Prosječna nagrada (Q-learning): {mean(q_returns):.2f}")

    # 5) Graf učenja Q-learninga
    plt.figure()
    plt.plot(train_returns, alpha=0.4, label="Epizodni return")
    plt.plot(moving_average(train_returns, window=50), label="Moving average (50)")
    plt.xlabel("Epizoda")
    plt.ylabel("Ukupna nagrada (return)")
    plt.title("Q-learning: učenje kroz epizode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 6) Usporedba prosječnih nagrada
    labels = ["Random", "VI", "Q-learning"]
    avgs = [mean(random_returns), mean(vi_returns), mean(q_returns)]

    plt.figure()
    plt.bar(labels, avgs)
    plt.ylabel("Prosječna ukupna nagrada")
    plt.title("Usporedba politika: Random vs VI vs Q-learning")
    plt.grid(axis="y")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
