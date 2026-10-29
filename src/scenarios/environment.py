"""Controller-independent environment and disk/swept-segment collision geometry."""

import numpy as np

from dynamics.bicycle import Bicycle, wrap_angle


def clearance(state, obstacles, vehicle_radius):
    shape = np.asarray(state).shape[:-1]
    if not obstacles:
        return np.full(shape, np.inf)
    obs = np.asarray(obstacles)
    d = np.linalg.norm(np.asarray(state)[..., None, :2] - obs[:, :2], axis=-1)
    return np.min(d - obs[:, 2] - vehicle_radius, axis=-1)


def swept_clearance(start, end, obstacles, vehicle_radius):
    if not obstacles:
        return np.full(np.asarray(start).shape[:-1], np.inf)
    obs = np.asarray(obstacles)
    a, b = np.asarray(start)[..., :2], np.asarray(end)[..., :2]
    segment = b - a
    den = np.sum(segment**2, axis=-1)
    t = np.sum((obs[:, :2] - a[..., None, :]) * segment[..., None, :], axis=-1)
    t = np.clip(t / np.maximum(den[..., None], 1e-20), 0, 1)
    closest = a[..., None, :] + t[..., None] * segment[..., None, :]
    return np.min(
        np.linalg.norm(closest - obs[:, :2], axis=-1) - obs[:, 2] - vehicle_radius, axis=-1
    )


class Environment:
    def __init__(self, scenario, dynamics, seed=42):
        self.scenario = scenario
        self.model = Bicycle(**{**dynamics, **scenario.get("execution_overrides", {})})
        self.rng = np.random.default_rng(seed)
        self.state = np.array(scenario["initial_state"], dtype=float)
        self.index = 0

    def step(self, control):
        self.state = self.model.step(self.state, control)
        disturbance = self.scenario.get("disturbance", {})
        if self.index == disturbance.get("step", -1):
            self.state += np.array(disturbance["delta"])
        std = np.asarray(self.scenario.get("noise_std", [0, 0, 0, 0]))
        # RNG advances on every execution step: identical exogenous noise across controllers.
        self.state += self.rng.normal(size=4) * std
        self.state[2] = wrap_angle(self.state[2])
        self.state[3] = np.clip(self.state[3], self.model.velocity_min, self.model.velocity_max)
        self.index += 1
        return self.state.copy()
