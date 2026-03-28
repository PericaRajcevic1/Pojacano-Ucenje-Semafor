# value_iteration.py
import itertools
import numpy as np
from env import TrafficEnvConfig


class TrafficMDPModel:
    """
    Deterministička aproksimacija očekivanog prijelaza
    za potrebe Value Iteration.
    """

    def __init__(self, config: TrafficEnvConfig, gamma=0.95):
        self.cfg = config
        self.gamma = gamma

        self.states = self._generate_states()
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)
        self.n_actions = 2  # STAY, SWITCH

    def _generate_states(self):
        states = []
        for q_ns, q_ew, phase in itertools.product(
            range(self.cfg.max_queue + 1),
            range(self.cfg.max_queue + 1),
            [0, 1]
        ):
            states.append((q_ns, q_ew, phase))
        return states

    def expected_transition(self, state, action):
        """
        Aproksimacija očekivanog prijelaza (bez stohastičkog sample-a).
        """
        q_ns, q_ew, phase = state
        cfg = self.cfg

        # 1) primjena akcije
        if action == 1:  # SWITCH
            phase = 1 - phase

        # 2) očekivani dolazak vozila
        q_ns = min(q_ns + cfg.p_arrival_ns, cfg.max_queue)
        q_ew = min(q_ew + cfg.p_arrival_ew, cfg.max_queue)

        # 3) prolazak vozila (zaokružimo na cijele)
        if phase == 0:
            q_ns = max(q_ns - cfg.cars_through_green, 0)
        else:
            q_ew = max(q_ew - cfg.cars_through_green, 0)

        # diskretizacija
        q_ns = int(round(q_ns))
        q_ew = int(round(q_ew))

        next_state = (q_ns, q_ew, phase)

        # nagrada
        reward = -(q_ns + q_ew)
        if action == 1:
            reward -= cfg.switch_penalty

        return next_state, reward


def value_iteration(model: TrafficMDPModel, theta=1e-4, max_iterations=1000):
    V = np.zeros(model.n_states)

    for iteration in range(max_iterations):
        delta = 0
        for i, state in enumerate(model.states):
            v = V[i]

            action_values = []
            for action in range(model.n_actions):
                next_state, reward = model.expected_transition(state, action)
                j = model.state_to_idx[next_state]
                action_value = reward + model.gamma * V[j]
                action_values.append(action_value)

            V[i] = max(action_values)
            delta = max(delta, abs(v - V[i]))

        if delta < theta:
            print(f"Value iteration konvergirala nakon {iteration+1} iteracija.")
            break

    # ekstrakcija politike
    policy = {}
    for i, state in enumerate(model.states):
        action_values = []
        for action in range(model.n_actions):
            next_state, reward = model.expected_transition(state, action)
            j = model.state_to_idx[next_state]
            action_value = reward + model.gamma * V[j]
            action_values.append(action_value)

        best_action = int(np.argmax(action_values))
        policy[state] = best_action

    return V, policy
