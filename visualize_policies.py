# visualize_policies.py
import numpy as np
import matplotlib.pyplot as plt

from env import TrafficEnvConfig
from value_iteration import TrafficMDPModel, value_iteration
from q_learning import q_learning_train, greedy_policy_from_Q


def policy_to_grid(policy: dict, phase: int, cfg: TrafficEnvConfig) -> np.ndarray:
    """
    Pretvara politiku (state -> action) u 2D grid za zadani phase.
    Grid dimenzije: (max_queue+1, max_queue+1)
      y-os: q_ns
      x-os: q_ew
      vrijednost: akcija (0=STAY, 1=SWITCH)
    """
    n = cfg.max_queue + 1
    grid = np.zeros((n, n), dtype=float)

    for q_ns in range(n):
        for q_ew in range(n):
            state = (q_ns, q_ew, phase)
            action = policy.get(state, 0)
            grid[q_ns, q_ew] = action

    return grid


def plot_policy_heatmaps(cfg: TrafficEnvConfig):
    # 1) Value Iteration politika
    mdp_model = TrafficMDPModel(cfg, gamma=0.95)
    _, vi_policy = value_iteration(mdp_model)

    # 2) Q-learning politika (ponovno treniranje – možeš i smanjiti broj epizoda za bržu vizualizaciju)
    Q, _ = q_learning_train(
        cfg,
        num_episodes=3000,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    )
    q_policy = greedy_policy_from_Q(Q, cfg)

    # 3) Gridovi
    vi_phase0 = policy_to_grid(vi_policy, phase=0, cfg=cfg)
    vi_phase1 = policy_to_grid(vi_policy, phase=1, cfg=cfg)
    q_phase0 = policy_to_grid(q_policy, phase=0, cfg=cfg)
    q_phase1 = policy_to_grid(q_policy, phase=1, cfg=cfg)

    n = cfg.max_queue + 1
    x_ticks = list(range(n))
    y_ticks = list(range(n))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Heatmap politika semafora (0=STAY, 1=SWITCH)")

    # Helper funkcija za crtanje jedne slike
    def show_heat(ax, grid, title):
        im = ax.imshow(grid, origin="lower", vmin=0, vmax=1)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlabel("q_EW (vozila na EW)")
        ax.set_ylabel("q_NS (vozila na NS)")
        ax.set_title(title)
        # bočna ljestvica samo jednom je dovoljna, ali radi jednostavnosti svugdje isto
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["STAY", "SWITCH"])

    show_heat(axes[0, 0], vi_phase0, "Value Iteration politika (phase=NS zeleno)")
    show_heat(axes[0, 1], vi_phase1, "Value Iteration politika (phase=EW zeleno)")
    show_heat(axes[1, 0], q_phase0, "Q-learning politika (phase=NS zeleno)")
    show_heat(axes[1, 1], q_phase1, "Q-learning politika (phase=EW zeleno)")

    plt.tight_layout()
    plt.show()


def main():
    cfg = TrafficEnvConfig(
        max_queue=5,
        p_arrival_ns=0.4,
        p_arrival_ew=0.4,
        cars_through_green=1,
        switch_penalty=1.0,
        max_steps=200,
    )
    plot_policy_heatmaps(cfg)


if __name__ == "__main__":
    main()
