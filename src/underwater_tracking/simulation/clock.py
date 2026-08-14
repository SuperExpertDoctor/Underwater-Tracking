from dataclasses import dataclass


@dataclass(slots=True)
class SimulationClock:
    step_s: int = 10
    sim_time_s: int = 0

    def tick(self) -> int:
        self.sim_time_s += self.step_s
        return self.sim_time_s
