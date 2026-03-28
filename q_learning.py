# q_learning.py
import random
from typing import Dict, Tuple, List

import numpy as np

from env import TrafficEnv, TrafficEnvConfig


def num_states(cfg: TrafficEnvConfig) -> int:
    n = cfg.max_queue + 1
    return n * n * 2  # q_ns * q_ew * phase(2)


def state_to_index(state: Tuple[int, int, int], cfg: TrafficEnvConfig) -> int:
    """
    Mapira stanje (q_ns, q_ew, phase) u indeks [0, N_states-1].
    """
    q_ns, q_ew, phase = state
    n = cfg.max_queue + 1
    return phase * (n * n) + q_ns * n + q_ew


def all_states(cfg: TrafficEnvConfig):
    """
    Generator svih diskretnih stanja.
    """
    n = cfg.max_queue + 1
    for phase in [0, 1]:
        for q_ns in range(n):
            for q_ew in range(n):
                yield (q_ns, q_ew, phase)


def q_learning_train(
    cfg: TrafficEnvConfig,
    num_episodes: int = 3000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> Tuple[np.ndarray, List[float]]:
    """
    Treniranje tabličnog Q-learning agenta u TrafficEnv okruženju.

    :return: (Q tablica, lista ukupnih nagrada po epizodi)
    """
    env = TrafficEnv(cfg)

    n_states = num_states(cfg)
    n_actions = 2  # STAY, SWITCH

    Q = np.zeros((n_states, n_actions), dtype=float)
    episode_returns: List[float] = []

    epsilon = epsilon_start

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        G = 0.0

        while not done:
            s_idx = state_to_index(state, cfg)

            # epsilon-greedy odabir akcije
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                action = int(np.argmax(Q[s_idx]))

            next_state, reward, done, info = env.step(action)
            G += reward

            ns_idx = state_to_index(next_state, cfg)

            # Q-learning update
            best_next = np.max(Q[ns_idx])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[s_idx, action]
            Q[s_idx, action] += alpha * td_error

            state = next_state

        episode_returns.append(G)

        # eksponencijalno smanjivanje epsilona
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (ep + 1) % 100 == 0:
            avg_last = sum(episode_returns[-100:]) / 100
            print(
                f"Epizoda {ep+1}/{num_episodes}, "
                f"epsilon={epsilon:.3f}, "
                f"prosječna nagrada zadnjih 100 epizoda = {avg_last:.2f}"
            )

    return Q, episode_returns


def greedy_policy_from_Q(Q: np.ndarray, cfg: TrafficEnvConfig) -> Dict[Tuple[int, int, int], int]:
    """
    Iz Q tablice izvlači determinističku greedy politiku: π(s) = argmax_a Q(s,a)
    Vraća dict: state -> action
    """
    policy: Dict[Tuple[int, int, int], int] = {}
    for state in all_states(cfg):
        s_idx = state_to_index(state, cfg)
        best_action = int(np.argmax(Q[s_idx]))
        policy[state] = best_action
    return policy
