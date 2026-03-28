# env.py
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class TrafficEnvConfig:
    max_queue: int = 5              # maksimalan broj vozila po smjeru (5 = "5 ili više")
    p_arrival_ns: float = 0.3       # vjerojatnost dolaska vozila NS u jednom koraku
    p_arrival_ew: float = 0.3       # vjerojatnost dolaska vozila EW u jednom koraku
    cars_through_green: int = 1     # koliko vozila može proći na zeleno u jednom koraku
    switch_penalty: float = 1.0     # kazna za promjenu faze
    max_steps: int = 200            # duljina jedne epizode (broj koraka)


class TrafficEnv:
    """
    Jednostavno simulirano okruženje za upravljanje semaforom na raskrižju
    s dva smjera: NS (north-south) i EW (east-west).

    Stanje: (q_ns, q_ew, phase)
      q_ns, q_ew ∈ {0, ..., max_queue}
      phase ∈ {0, 1}  # 0 = zeleno NS, 1 = zeleno EW

    Akcije:
      0 = STAY   (zadrži trenutnu fazu)
      1 = SWITCH (promijeni fazu: 0 -> 1, 1 -> 0)
    """

    def __init__(self, config: TrafficEnvConfig | None = None):
        self.config = config or TrafficEnvConfig()

        # unutarnje varijable stanja
        self.q_ns: int = 0
        self.q_ew: int = 0
        self.phase: int = 0  # 0 = NS zeleno, 1 = EW zeleno

        self.step_count: int = 0
        self.done: bool = False

    def reset(self) -> Tuple[int, int, int]:
        """Postavlja početno stanje i vraća ga."""
        self.q_ns = 0
        self.q_ew = 0
        self.phase = 0  # npr. kreni sa NS zelenim
        self.step_count = 0
        self.done = False
        return self._get_state()

    def _get_state(self) -> Tuple[int, int, int]:
        return self.q_ns, self.q_ew, self.phase

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        """
        Izvršava jedan korak simulacije.

        :param action: 0 = STAY, 1 = SWITCH
        :return: next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("Epizoda je već završila. Pozovi reset().")

        cfg = self.config
        self.step_count += 1

        # 1) Primjena akcije
        switch_happened = False
        if action == 1:  # SWITCH
            self.phase = 1 - self.phase
            switch_happened = True
        elif action != 0:
            raise ValueError("Nepoznata akcija: expected 0 or 1, got {}".format(action))

        # 2) Dolazak novih vozila
        if random.random() < cfg.p_arrival_ns:
            self.q_ns = min(self.q_ns + 1, cfg.max_queue)
        if random.random() < cfg.p_arrival_ew:
            self.q_ew = min(self.q_ew + 1, cfg.max_queue)

        # 3) Prolazak vozila na zeleno
        if self.phase == 0:
            # NS ima zeleno
            cars_can_go = min(self.q_ns, cfg.cars_through_green)
            self.q_ns -= cars_can_go
        else:
            # EW ima zeleno
            cars_can_go = min(self.q_ew, cfg.cars_through_green)
            self.q_ew -= cars_can_go

        # 4) Nagrada
        total_queue = self.q_ns + self.q_ew
        reward = -float(total_queue)
        if switch_happened:
            reward -= cfg.switch_penalty

        # 5) Provjera završetka epizode
        if self.step_count >= cfg.max_steps:
            self.done = True

        next_state = self._get_state()
        info = {
            "total_queue": total_queue,
            "switch_happened": switch_happened,
        }
        return next_state, reward, self.done, info

    def render(self) -> None:
        """Jednostavan ispis stanja u konzolu (za debug)."""
        phase_str = "NS" if self.phase == 0 else "EW"
        print(f"Step={self.step_count} | Phase={phase_str} | q_NS={self.q_ns} | q_EW={self.q_ew}")
