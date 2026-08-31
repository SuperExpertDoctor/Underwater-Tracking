"""USV entity backed by the shared continuous-motion integrator."""

from __future__ import annotations

from dataclasses import dataclass

from underwater_tracking.domain.platforms import MotionLimits
from underwater_tracking.simulation.kinematics import MotionCommand, MotionState, advance_motion


@dataclass(slots=True)
class USVEntity:
    usv_id: str
    platform_index: int
    motion: MotionState
    energy_fraction: float
    limits: MotionLimits
    transit_energy_per_m: float
    hotel_energy_per_s: float
    command: MotionCommand | None = None

    def set_motion_command(self, command: MotionCommand) -> None:
        self.command = command

    def step(self, dt_s: float) -> None:
        command = self.command or MotionCommand(
            desired_heading_rad=self.motion.heading_rad,
            desired_speed_mps=0.0,
        )
        before = self.motion.position_xy
        self.motion = advance_motion(self.motion, command, self.limits, dt_s)
        dx = self.motion.position_xy[0] - before[0]
        dy = self.motion.position_xy[1] - before[1]
        distance = (dx * dx + dy * dy) ** 0.5
        used = distance * self.transit_energy_per_m + dt_s * self.hotel_energy_per_s
        self.energy_fraction = max(0.0, self.energy_fraction - used)
