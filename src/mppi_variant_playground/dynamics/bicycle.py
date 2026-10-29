"""Vectorized Euler bicycle dynamics. Inputs and states use physical units."""

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Bicycle:
    dt: float = 0.15
    wheelbase: float = 1.0
    velocity_min: float = 0.0
    velocity_max: float = 3.0
    acceleration_limit: float = 2.0
    steering_limit: float = 0.5
    steering_gain: float = 1.0
    acceleration_gain: float = 1.0

    def __post_init__(self):
        values = np.array(list(self.__dict__.values()))
        if not np.isfinite(values).all():
            raise ValueError("Dynamics parameters must be finite")
        if min(self.dt, self.wheelbase, self.acceleration_limit, self.steering_limit) <= 0:
            raise ValueError("Time, length and actuator limits must be positive")
        if self.velocity_min >= self.velocity_max:
            raise ValueError("Invalid velocity interval")
        if abs(self.steering_limit * self.steering_gain) >= np.pi / 2:
            raise ValueError("Steering must stay away from the tangent singularity")

    @property
    def control_scale(self):
        return np.array([self.acceleration_limit, self.steering_limit])

    def clip_control(self, control):
        return np.clip(control, -self.control_scale, self.control_scale)

    def step(self, state, control):
        x = np.asarray(state, dtype=float)
        u = self.clip_control(np.asarray(control, dtype=float))
        y = np.empty_like(x)
        v, heading = x[..., 3], x[..., 2]
        y[..., 0] = x[..., 0] + v * np.cos(heading) * self.dt
        y[..., 1] = x[..., 1] + v * np.sin(heading) * self.dt
        y[..., 2] = wrap_angle(
            heading + v / self.wheelbase * np.tan(u[..., 1] * self.steering_gain) * self.dt
        )
        y[..., 3] = np.clip(
            v + u[..., 0] * self.acceleration_gain * self.dt,
            self.velocity_min,
            self.velocity_max,
        )
        return y
