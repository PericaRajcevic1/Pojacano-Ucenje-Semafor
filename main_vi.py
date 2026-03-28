# main_vi.py
from env import TrafficEnvConfig
from value_iteration import TrafficMDPModel, value_iteration


def main():
    cfg = TrafficEnvConfig(
        max_queue=5,
        p_arrival_ns=0.4,
        p_arrival_ew=0.4,
        cars_through_green=1,
        switch_penalty=1.0,
    )

    model = TrafficMDPModel(cfg, gamma=0.95)
    V, policy = value_iteration(model)

    print("\nPrimjer nekoliko stanja i optimalne akcije:")
    for state in list(policy.keys())[:10]:
        print(f"Stanje {state} -> akcija {policy[state]} (0=STAY, 1=SWITCH)")


if __name__ == "__main__":
    main()
