"""Seven-state planar single-track physics with overdamped dynamic roll.

State: px, py, heading, roll, vx, vy, yaw_rate. Controls: steering [rad],
normalized signed throttle [-1,1]. Positive steering and lateral force turn left;
positive roll raises the left side (outward roll during a left turn).
"""

from dataclasses import dataclass

import numpy as np

from dynamics.bicycle import wrap_angle


@dataclass(frozen=True)
class DynamicVehicle:
    dt: float = 0.15
    substeps: int = 3
    mass: float = 35.0
    yaw_inertia: float = 8.0
    front_distance: float = 0.4
    rear_distance: float = 0.4
    cornering_front: float = 450.0
    cornering_rear: float = 450.0
    friction: float = 0.9
    gravity: float = 9.81
    engine_force: float = 70.0
    brake_force: float = 105.0
    drag_linear: float = 0.4
    drag_quadratic: float = 0.5
    steering_limit: float = 0.5
    throttle_limit: float = 1.0
    velocity_max: float = 4.0
    low_speed: float = 0.5
    low_speed_damping_time: float = 0.15
    cg_height: float = 0.2
    roll_stiffness: float = 120.0
    roll_damping: float = 48.0
    state_dimension: int = 7

    def __post_init__(self):
        values = np.array(list(self.__dict__.values()), dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Finite positive dynamic-vehicle parameters required")
        if not isinstance(self.substeps, int) or self.steering_limit >= np.pi / 2:
            raise ValueError("Integer substeps and steering limit below pi/2 required")
        if self.dt / self.substeps > 0.05 + 1e-12:
            raise ValueError("Integration substep must be at most .05 s; validate finer steps")

    @property
    def wheelbase(self):
        return self.front_distance + self.rear_distance

    @property
    def control_scale(self):
        return np.array([self.steering_limit, self.throttle_limit])

    def clip_control(self, control):
        return np.clip(control, -self.control_scale, self.control_scale)

    def forces(self, state, control):
        vx, vy, r = state[..., 4], state[..., 5], state[..., 6]
        delta, throttle = control[..., 0], control[..., 1] / self.throttle_limit
        den = np.sqrt(vx**2 + self.low_speed**2)
        blend = vx**2 / (vx**2 + self.low_speed**2)
        alpha_front = delta - np.arctan2(vy + self.front_distance * r, den)
        alpha_rear = -np.arctan2(vy - self.rear_distance * r, den)
        front_limit = self.friction * self.mass * self.gravity * self.rear_distance / self.wheelbase
        rear_limit = self.friction * self.mass * self.gravity * self.front_distance / self.wheelbase
        front = blend * front_limit * np.tanh(self.cornering_front * alpha_front / front_limit)
        rear = blend * rear_limit * np.tanh(self.cornering_rear * alpha_rear / rear_limit)
        longitudinal = np.where(
            throttle >= 0, self.engine_force * throttle, self.brake_force * throttle
        )
        longitudinal -= self.drag_linear * vx + self.drag_quadratic * vx * np.abs(vx)
        lateral = (
            front * np.cos(delta)
            + rear
            - self.mass * (1 - blend) * vy / self.low_speed_damping_time
        )
        moment = self.front_distance * front * np.cos(delta) - self.rear_distance * rear
        moment -= self.yaw_inertia * (1 - blend) * r / self.low_speed_damping_time
        return longitudinal - front * np.sin(delta), lateral, moment

    def derivative(self, state, control):
        state = np.asarray(state, dtype=float)
        control = self.clip_control(np.asarray(control, dtype=float))
        fx, fy, moment = self.forces(state, control)
        heading, roll, vx, vy, r = [state[..., i] for i in (2, 3, 4, 5, 6)]
        result = np.empty_like(state)
        result[..., 0] = vx * np.cos(heading) - vy * np.sin(heading)
        result[..., 1] = vx * np.sin(heading) + vy * np.cos(heading)
        result[..., 2] = r
        result[..., 3] = (self.cg_height * fy - self.roll_stiffness * roll) / self.roll_damping
        result[..., 4] = fx / self.mass + vy * r
        result[..., 5] = fy / self.mass - vx * r
        result[..., 6] = moment / self.yaw_inertia
        return result

    def step(self, state, control, *, return_path=False):
        state = np.asarray(state, dtype=float)
        if state.shape[-1] != 7 or not np.isfinite(state).all():
            raise ValueError("Expected finite seven-state input")
        control = self.clip_control(np.asarray(control, dtype=float))
        result = state.copy()
        path = [result.copy()] if return_path else None
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            first = self.derivative(result, control)
            midpoint = result + 0.5 * h * first
            midpoint[..., 4] = np.clip(midpoint[..., 4], 0, self.velocity_max)
            result = result + h * self.derivative(midpoint, control)
            result[..., 4] = np.clip(result[..., 4], 0, self.velocity_max)
            if return_path:
                path.append(result.copy())
        result[..., 2] = wrap_angle(result[..., 2])
        if not np.isfinite(result).all():
            raise FloatingPointError("Dynamic vehicle integration failed; no model fallback")
        if return_path:
            path[-1] = result.copy()
            return result, np.stack(path, axis=-2)
        return result
