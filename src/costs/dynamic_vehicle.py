"""Common seven-state task cost, independent of proposal and evaluated on physical inputs."""

import numpy as np

from dynamics.bicycle import wrap_angle


class VehicleCost:
    substep_sweep = True

    def __init__(self, geometry, model, weights, speed):
        self.geometry, self.model, self.weights, self.speed = geometry, model, weights, speed
        self.dt = model.dt
        self.obstacles = getattr(geometry, "obstacles", [])
        self.vehicle_radius = geometry.vehicle_radius

    def transition_clearance(self, start, end):
        return self.geometry.transition_clearance(start, end)

    def path_clearance(self, path):
        return np.min(self.transition_clearance(path[..., :-1, :], path[..., 1:, :]), axis=-1)

    def running(self, states, controls, previous):
        w = self.weights
        _, progress, index = self.geometry.project(states)
        remaining = np.maximum(self.geometry.center.arc[-1] - progress, 0)
        tangent = self.geometry.center.b[index] - self.geometry.center.a[index]
        heading = np.arctan2(tangent[..., 1], tangent[..., 0])
        gap = self.geometry.state_clearance(states)
        return self.dt * (
            w["remaining_progress"] * remaining**2
            + w["speed"] * (states[..., 4] - self.speed) ** 2
            + w["heading"] * wrap_angle(states[..., 2] - heading) ** 2
            + w["clearance"] * np.maximum(w["safety_margin"] - gap, 0) ** 2
            + w["collision"] * (gap <= 0)
            + w["lateral_velocity"] * states[..., 5] ** 2
            + w["yaw_rate"] * states[..., 6] ** 2
            + w["roll"] * states[..., 3] ** 2
            + w["control"] * np.sum((controls / self.model.control_scale) ** 2, axis=-1)
            + w["variation"]
            * np.sum(((controls - previous) / self.model.control_scale) ** 2, axis=-1)
        )

    def terminal(self, states):
        w = self.weights
        goal = self.geometry.goal
        _, progress, _ = self.geometry.project(states)
        remaining = np.maximum(self.geometry.center.arc[-1] - progress, 0)
        return (
            w["terminal_progress"] * remaining**2
            + w["terminal_goal"]
            * (np.minimum(states[..., 0] - goal[0], 0) ** 2 + (states[..., 1] - goal[1]) ** 2)
            + w["terminal_heading"] * wrap_angle(states[..., 2] - goal[2]) ** 2
            + w["terminal_speed"] * (states[..., 4] - goal[4]) ** 2
        )
