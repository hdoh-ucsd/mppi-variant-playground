"""Seven-state scene adapters reusing existing circle and polygon collision kernels."""

from copy import deepcopy

import numpy as np
from matplotlib.path import Path as PolygonPath

from dynamics.dynamic_vehicle import DynamicVehicle
from scenarios.environment import clearance, swept_clearance
from scenarios.s_turn import Corridor, Polyline


class Circles:
    def __init__(self, obstacles, radius, start, goal):
        self.obstacles = obstacles
        self.vehicle_radius = radius
        self.goal = np.array(goal, dtype=float)
        self.center = Polyline(np.array([start[:2], goal[:2]]))

    def state_clearance(self, state):
        return clearance(state, self.obstacles, self.vehicle_radius)

    def transition_clearance(self, a, b):
        return swept_clearance(a, b, self.obstacles, self.vehicle_radius)

    def project(self, state):
        return self.center.nearest(np.asarray(state)[..., :2])


class DynamicCorridor(Corridor):
    def __init__(self, reference, clearance_value, radius, extension):
        # Offset normals follow the course tangent, not body heading when vy != 0.
        ref = reference.copy()
        xy = ref[:, :2]
        delta = np.diff(xy, axis=0)
        keep = np.r_[np.linalg.norm(delta, axis=1) > 1e-10, True]
        ref = ref[keep]
        xy = ref[:, :2]
        tangent = np.diff(xy, axis=0)
        course = np.arctan2(tangent[:, 1], tangent[:, 0])
        ref[:, 2] = np.r_[course, course[-1]]
        super().__init__(ref, clearance_value, radius, extension)
        self.goal = reference[-1].copy()
        self.polygon = PolygonPath(self.boundary.points)

    def inside(self, xy):
        xy = np.asarray(xy)
        shape = xy.shape[:-1]
        return self.polygon.contains_points(xy.reshape(-1, 2)).reshape(shape)


def reference_maneuver(config, speed=None):
    """Physically executed steering schedule with speed feedback, separate from MPPI."""
    model = DynamicVehicle(**config["vehicle"])
    scene = config["s_turn"]
    speed = config["speed_target"] if speed is None else speed
    state = np.zeros(7)
    states = [state.copy()]
    controls = []

    def advance(steer):
        # Rear propulsion plus explicit drag compensation; no trajectory feedback.
        drag = model.drag_linear * state[4] + model.drag_quadratic * state[4] ** 2
        force = drag + model.mass * 2 * (speed - state[4])
        throttle = force / (model.engine_force if force >= 0 else model.brake_force)
        control = model.clip_control([steer, throttle])
        nxt = model.step(state, control)
        controls.append(control)
        states.append(nxt)
        state[:] = nxt

    for _ in range(int(np.ceil(3 / model.dt))):
        advance(0)
    for _ in range(int(np.ceil(scene["entry_length"] / speed / model.dt))):
        advance(0)
    n = max(8, int(np.ceil(scene["turn_separation"] / speed / model.dt)))
    angle = np.arctan(model.wheelbase / scene["turn_radius"])
    for sign, count in ((1, n), (-1, 2 * n), (1, n)):
        for j in range(count):
            advance(sign * angle * np.sin(np.pi * (j + 1) / count) ** 2)
    for _ in range(int(np.ceil(scene["exit_length"] / speed / model.dt))):
        advance(0)
    return np.array(states), np.array(controls)


def make_scene(config, name):
    start = np.zeros(7)
    goal = np.array([10.0, 0.0, 0.0, 0.0, config["speed_target"], 0.0, 0.0])
    geometry = []
    if name == "narrow_passage":
        geometry = [[5.0, 1.0, 0.55], [5.0, -1.0, 0.55]]
    elif name == "two_route_obstacle":
        geometry = [[4.0, 0.0, 0.75]]
        start[1] = -0.2
    elif name not in {"open_space", "disturbance", "model_mismatch"}:
        raise ValueError(f"Unknown seven-state scene {name}")
    scenario = dict(
        name=name,
        initial_state=start.tolist(),
        goal_state=goal.tolist(),
        obstacles=geometry,
        vehicle_radius=config["vehicle_radius"],
    )
    if name == "disturbance":
        scenario["disturbance"] = dict(step=20, delta=[0, 0.7, 0.10, 0.01, -0.2, 0.3, 0.12])
    if name == "model_mismatch":
        scenario["execution_overrides"] = dict(
            cornering_front=0.7 * config["vehicle"]["cornering_front"],
            cornering_rear=0.7 * config["vehicle"]["cornering_rear"],
        )
    return scenario, Circles(geometry, config["vehicle_radius"], start, goal)


def make_s_turn(config, clearance_value, speed=None):
    resolved = deepcopy(config)
    if speed is not None:
        resolved["speed_target"] = speed
    states, controls = reference_maneuver(resolved)
    corridor = DynamicCorridor(
        states, clearance_value, config["vehicle_radius"], config["s_turn"]["end_extension"]
    )
    model = DynamicVehicle(**config["vehicle"])
    minima = []
    for state, control in zip(states[:-1], controls, strict=True):
        _, path = model.step(state, control, return_path=True)
        minima.append(float(corridor.transition_clearance(path[:-1], path[1:]).min()))
    signs = np.sign(controls[:, 0][np.abs(controls[:, 0]) > 0.01])
    rate = np.max(np.abs(np.diff(np.r_[0.0, controls[:, 0]]))) / model.dt
    check = dict(
        feasible=bool(min(minima) > 0 and len(controls) <= config["execution"]["max_steps"]),
        min_swept_clearance=float(min(minima)),
        reference_steps=len(controls),
        max_steering=float(abs(controls[:, 0]).max()),
        max_throttle=float(abs(controls[:, 1]).max()),
        steering_reversals=int(np.count_nonzero(np.diff(signs))),
        max_profile_rate=float(rate),
        max_abs_vy=float(abs(states[:, 5]).max()),
        max_abs_yaw_rate=float(abs(states[:, 6]).max()),
        max_abs_roll=float(abs(states[:, 3]).max()),
        goal_state=states[-1].tolist(),
        duration_s=len(controls) * model.dt,
        clearance=clearance_value,
    )
    if check["steering_reversals"] < 2 or rate > config["s_turn"]["steering_profile_rate_limit"]:
        check["feasible"] = False
    scenario = dict(
        name="s_turn",
        initial_state=states[0].tolist(),
        goal_state=states[-1].tolist(),
        obstacles=[],
        vehicle_radius=config["vehicle_radius"],
        clearance=clearance_value,
    )
    return scenario, corridor, states, controls, check


def geometry_for_config(config):
    scene = config["scenario"]
    if scene["name"] == "s_turn":
        return make_s_turn(config, scene["clearance"])[1]
    return Circles(
        scene["obstacles"], scene["vehicle_radius"], scene["initial_state"], scene["goal_state"]
    )
