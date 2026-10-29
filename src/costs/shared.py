"""Shared physical-state cost; independent optimizer and evaluation instances."""

from dataclasses import dataclass

import numpy as np

from dynamics.bicycle import wrap_angle
from scenarios.environment import clearance


@dataclass
class SharedCost:
    goal: np.ndarray
    obstacles: list
    scale: np.ndarray
    vehicle_radius: float
    weights: dict
    dt: float

    def running(self, states, controls, previous_controls):
        w = self.weights
        d2 = np.sum((states[..., :2] - self.goal[:2]) ** 2, axis=-1)
        heading = wrap_angle(states[..., 2] - self.goal[2])
        # Slow down near the target; this reference uses only the shared goal.
        target_speed = np.minimum(w["cruise_speed"], np.sqrt(d2))
        gap = clearance(states, self.obstacles, self.vehicle_radius)
        return self.dt * (
            w["goal"] * d2
            + w["heading"] * heading**2
            + w["speed"] * (states[..., 3] - target_speed) ** 2
            + w["obstacle"] * np.maximum(w["margin"] - gap, 0) ** 2
            + w["collision"] * (gap <= 0)
            + w["control"] * np.sum((controls / self.scale) ** 2, axis=-1)
            + w["variation"] * np.sum(((controls - previous_controls) / self.scale) ** 2, axis=-1)
        )

    def terminal(self, states):
        return (
            self.weights["terminal"] * np.sum((states[..., :2] - self.goal[:2]) ** 2, axis=-1)
            + self.weights["terminal_heading"] * wrap_angle(states[..., 2] - self.goal[2]) ** 2
            + self.weights["terminal_speed"] * (states[..., 3] - self.goal[3]) ** 2
        )
