"""Explicit Euler five-state bicycle; original acceleration/angle model is unchanged."""

from dataclasses import dataclass

import numpy as np

from dynamics.bicycle import Bicycle


@dataclass(frozen=True)
class SteeringRateBicycle:
    bicycle: Bicycle
    steering_rate_limit: float = 0.8
    state_dimension: int = 5

    def __post_init__(self):
        if not np.isfinite(self.steering_rate_limit) or self.steering_rate_limit <= 0:
            raise ValueError("Positive steering-rate limit required")

    @property
    def dt(self):
        return self.bicycle.dt

    @property
    def control_scale(self):
        return np.array([self.bicycle.acceleration_limit, self.steering_rate_limit])

    def clip_control(self, control):
        return np.clip(control, -self.control_scale, self.control_scale)

    def step(self, state, control):
        state, control = np.asarray(state, dtype=float), self.clip_control(control)
        actual = np.stack([control[..., 0], state[..., 4]], axis=-1)
        next_state = np.empty_like(state)
        next_state[..., :4] = self.bicycle.step(state[..., :4], actual)
        next_state[..., 4] = np.clip(
            state[..., 4] + control[..., 1] * self.dt,
            -self.bicycle.steering_limit,
            self.bicycle.steering_limit,
        )
        return next_state


def make_model(config):
    values = dict(config)
    limit = values.pop("steering_rate_limit")
    return SteeringRateBicycle(Bicycle(**values), limit)
