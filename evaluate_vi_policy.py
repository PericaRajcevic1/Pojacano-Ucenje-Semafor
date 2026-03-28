# evaluate_vi_policy.py
import random
import matplotlib.pyplot as plt

from env import TrafficEnv, TrafficEnvConfig
from value_iteration import TrafficMDPModel, value_iteration


def run_policy(env: TrafficEnv, policy: dict, num_episodes: int = 20):
    """
    Pokreće zadanu politiku u stohastičkom env-u.
    policy: dict[state] -> action (0=STAY, 1=SWITCH)
    """
    all_episode_returns = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_rewards = []

        while not done:
            # politika: ako nekim čudom nema stanja u dict-u, default STAY
            action = policy.get(state, 0)
            next_state, reward, done, info = env.step(action)
            ep_rewards.append(reward)
            state = next_state

        G = sum(ep_rewards)
        all_episode_returns.append(G)
        print(f"[POLICY] Epizoda {ep+1}: ukupna nagrada = {G:.2f}")

    return all_episode_returns


def run_random(env: TrafficEnv, num_episodes: int = 20):
    """
    Random agent za usporedbu.
    """
    all_episode_returns = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_rewards = []

        while not done:
            action = random.randint(0, 1)
            next_state, reward, done, info = env.step(action)
            ep_rewards.append(reward)
            state = next_state

        G = sum(ep_rewards)
        all_episode_returns.append(G)
        print(f"[RANDOM] Epizoda {ep+1}: ukupna nagrada = {G:.2f}")

    return all_episode_returns


def plot_comparison(random_returns, policy_returns):
    plt.figure()
    plt.plot(random_returns, label="Random politika", marker="o")
    plt.plot(policy_returns, label="VI politika", marker="s")
    plt.xlabel("Epizoda")
    plt.ylabel("Ukupna nagrada (return)")
    plt.title("Usporedba: random vs. Value Iteration politika")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # isti config za sve
    cfg = TrafficEnvConfig(
        max_queue=5,
        p_arrival_ns=0.4,
        p_arrival_ew=0.4,
        cars_through_green=1,
        switch_penalty=1.0,
        max_steps=200,
    )

    # 1) value iteration na aproksimiranom MDP modelu
    mdp_model = TrafficMDPModel(cfg, gamma=0.95)
    V, policy = value_iteration(mdp_model)

    print(f"\nBroj stanja u MDP modelu: {mdp_model.n_states}")
    print("Primjer nekoliko stanja i optimalnih akcija:")
    example_states = list(policy.keys())[:10]
    for s in example_states:
        print(f"  {s} -> {policy[s]} (0=STAY, 1=SWITCH)")

    # 2) evaluacija u stohastičkom env-u
    env_for_random = TrafficEnv(cfg)
    env_for_policy = TrafficEnv(cfg)

    print("\n--- Pokrećem RANDOM agenta ---")
    random_returns = run_random(env_for_random, num_episodes=20)

    print("\n--- Pokrećem VI politiku ---")
    policy_returns = run_policy(env_for_policy, policy, num_episodes=20)

    # 3) kratka tekstualna usporedba
    avg_random = sum(random_returns) / len(random_returns)
    avg_policy = sum(policy_returns) / len(policy_returns)

    print("\n=== Sažetak ===")
    print(f"Prosječna ukupna nagrada (random): {avg_random:.2f}")
    print(f"Prosječna ukupna nagrada (VI politika): {avg_policy:.2f}")

    # 4) grafička usporedba
    plot_comparison(random_returns, policy_returns)


if __name__ == "__main__":
    main()
