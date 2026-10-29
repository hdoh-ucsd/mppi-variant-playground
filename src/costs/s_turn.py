"""Common progress/clearance objective; the centerline is not a tracking constraint."""

import numpy as np

from dynamics.bicycle import wrap_angle


class STurnCost:
    def __init__(self, corridor, model, weights, speed):
        self.corridor, self.model, self.weights, self.speed = corridor, model, weights, speed
        self.dt = model.dt
        self.obstacles = []
        self.vehicle_radius = corridor.vehicle_radius

    def transition_clearance(self, start, end):
        return self.corridor.transition_clearance(start, end)

    def running(self, states, controls, previous):
        w = self.weights
        _, progress, index = self.corridor.project(states)
        remaining = np.maximum(self.corridor.center.arc[-1] - progress, 0)
        tangent = self.corridor.center.b[index] - self.corridor.center.a[index]
        heading = np.arctan2(tangent[..., 1], tangent[..., 0])
        desired_speed = self.speed
        gap = self.corridor.state_clearance(states)
        scale = self.model.control_scale
        return self.dt * (
            w["remaining_progress"] * remaining**2
            + w["speed"] * (states[..., 3] - desired_speed) ** 2
            + w["heading"] * wrap_angle(states[..., 2] - heading) ** 2
            + w["clearance"] * np.maximum(w["safety_margin"] - gap, 0) ** 2
            + w["collision"] * (gap <= 0)
            + w["control"] * np.sum((controls / scale) ** 2, axis=-1)
            + w["variation"] * np.sum(((controls - previous) / scale) ** 2, axis=-1)
        )

    def terminal(self, states):
        w = self.weights
        _, progress, _ = self.corridor.project(states)
        remaining = np.maximum(self.corridor.center.arc[-1] - progress, 0)
        return (
            w["terminal_progress"] * remaining**2
            + w["terminal_goal"]
            * (
                np.minimum(states[..., 0] - self.corridor.goal[0], 0) ** 2
                + (states[..., 1] - self.corridor.goal[1]) ** 2
            )
            + w["terminal_heading"] * wrap_angle(states[..., 2] - self.corridor.goal[2]) ** 2
            + w["terminal_speed"] * (states[..., 3] - self.corridor.goal[3]) ** 2
        )
