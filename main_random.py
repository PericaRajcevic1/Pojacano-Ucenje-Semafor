# main_random.py
import random
import matplotlib.pyplot as plt

from env import TrafficEnv, TrafficEnvConfig


def run_random_policy(num_episodes: int = 10):
    cfg = TrafficEnvConfig(
        max_queue=5,
        p_arrival_ns=0.4,
        p_arrival_ew=0.4,
        cars_through_green=1,
        switch_penalty=1.0,
        max_steps=200,
    )
    env = TrafficEnv(cfg)

    all_episode_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_rewards = []

        while not done:
            # nasumična akcija: 0 ili 1
            action = random.randint(0, 1)
            next_state, reward, done, info = env.step(action)
            ep_rewards.append(reward)
            state = next_state

        total_return = sum(ep_rewards)
        all_episode_rewards.append(total_return)
        print(f"Epizoda {ep + 1}: ukupna nagrada = {total_return:.2f}")

    return all_episode_rewards


def plot_rewards(rewards):
    plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Epizoda")
    plt.ylabel("Ukupna nagrada (return)")
    plt.title("Random politika - ukupna nagrada po epizodi")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rewards = run_random_policy(num_episodes=20)
    plot_rewards(rewards)
