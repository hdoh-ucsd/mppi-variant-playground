"""Deterministic physics/integration validation before any seven-state MPC evaluation."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import root

from analysis.artifacts import begin
from dynamics.dynamic_vehicle import DynamicVehicle
from evaluation.runner import write_json


def simulate(model, state, controls):
    states = [np.asarray(state, dtype=float)]
    for control in controls:
        states.append(model.step(states[-1], control))
    return np.array(states)


def steady_turn(model, speed, steering):
    """Independent force-balance oracle: nonlinear saturated tire equations, not v*tan(delta)/L."""
    m = model

    def balances(values):
        vy, r, throttle = values
        den = np.sqrt(speed**2 + m.low_speed**2)
        w = speed**2 / (speed**2 + m.low_speed**2)
        nf = m.friction * m.mass * m.gravity * m.rear_distance / m.wheelbase
        nr = m.friction * m.mass * m.gravity * m.front_distance / m.wheelbase
        ff = (
            w
            * nf
            * np.tanh(
                m.cornering_front * (steering - np.arctan((vy + m.front_distance * r) / den)) / nf
            )
        )
        fr = w * nr * np.tanh(-m.cornering_rear * np.arctan((vy - m.rear_distance * r) / den) / nr)
        fy = ff * np.cos(steering) + fr - m.mass * (1 - w) * vy / m.low_speed_damping_time
        moment = (
            m.front_distance * ff * np.cos(steering)
            - m.rear_distance * fr
            - m.yaw_inertia * (1 - w) * r / m.low_speed_damping_time
        )
        fx = (
            m.engine_force * throttle
            - m.drag_linear * speed
            - m.drag_quadratic * speed**2
            - ff * np.sin(steering)
        )
        return [fx / m.mass + vy * r, fy / m.mass - speed * r, moment / m.yaw_inertia]

    solution = root(balances, [0.01, speed * steering / m.wheelbase, 0.05])
    if not solution.success or np.linalg.norm(balances(solution.x)) > 1e-8:
        raise RuntimeError("Independent steady-turn oracle failed")
    vy, r, throttle = solution.x
    return dict(
        vy=float(vy),
        yaw_rate=float(r),
        throttle=float(throttle),
        roll=float(m.cg_height * m.mass * speed * r / m.roll_stiffness),
        curvature=float(r / np.hypot(speed, vy)),
        force_balance_residual=float(np.linalg.norm(balances(solution.x))),
    )


def validate_model(model):
    traces = {}
    checks = {}
    n = int(round(8 / model.dt))
    zero = np.zeros(7)

    def add(name, state, controls):
        controls = np.asarray(controls, dtype=float)
        states = simulate(model, state, controls)
        traces[name] = dict(states=states.tolist(), controls=controls.tolist(), dt=model.dt)
        return states

    rest = add("zero", zero, np.zeros((n, 2)))
    checks["zero_equilibrium"] = bool(np.max(np.abs(rest)) < 1e-14)
    straight = add("straight", zero, np.tile([0, 0.3], (n, 1)))
    checks["straight_motion"] = bool(
        straight[-1, 0] > 1 and np.max(abs(straight[:, [1, 2, 3, 5, 6]])) < 1e-12
    )
    speed, angle = 2.0, 0.04
    oracle = steady_turn(model, speed, angle)
    initial = zero.copy()
    initial[4] = speed
    left = add("left_turn", initial, np.tile([angle, oracle["throttle"]], (n, 1)))
    right = add("right_turn", initial, np.tile([-angle, oracle["throttle"]], (n, 1)))
    mirror = np.array([1, -1, -1, -1, 1, -1, -1])
    checks["left_right_symmetry"] = bool(np.max(abs(left * mirror - right)) < 1e-10)
    final = left[-1]
    actual_oracle = steady_turn(model, float(final[4]), angle)
    checks["dynamic_turn_oracle"] = bool(
        abs(final[6] - actual_oracle["yaw_rate"]) < 0.02 * abs(actual_oracle["yaw_rate"])
    )
    checks["lateral_and_roll_signs"] = bool(final[5] > 0 and final[6] > 0 and final[3] > 0)
    xy = left[-min(30, len(left)) :, :2]
    center = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], np.sum(xy**2, axis=1), rcond=None)[0]
    radius = np.sqrt(center[2] + center[0] ** 2 + center[1] ** 2)
    checks["constant_radius_geometry"] = bool(
        abs(1 / radius - actual_oracle["curvature"]) < 0.03 * abs(actual_oracle["curvature"])
    )
    controls = np.tile([0.12, 0.06], (2 * n, 1))
    controls[n:, 0] = -0.12
    reversal = add("steering_reversal", initial, controls)
    checks["steering_reversal"] = bool(
        reversal[n, 6] > 0 and reversal[-1, 6] < 0 and reversal[n, 3] > 0 and reversal[-1, 3] < 0
    )
    braking = add("braking", initial, np.tile([0, -1], (n, 1)))
    checks["braking_no_reverse"] = bool(braking[-1, 4] == 0 and braking[:, 4].min() >= 0)
    perturb = zero.copy()
    perturb[[3, 5, 6]] = [0.01, 0.01, 0.01]
    low = add("near_zero", perturb, np.zeros((n, 2)))
    checks["near_zero_stability"] = bool(
        np.isfinite(low).all() and np.max(abs(low[-1, [3, 5, 6]])) < 1e-4
    )
    extremes = add("physical_bounds", zero, np.tile([10, 10], (n, 1)))
    clipped = simulate(model, zero, np.tile(model.control_scale, (n, 1)))
    checks["physical_bounds"] = bool(
        np.max(abs(extremes - clipped)) < 1e-12
        and extremes[:, 4].min() >= 0
        and extremes[:, 4].max() <= model.velocity_max
    )
    values = []
    for multiplier in (1, 2, 4, 8):
        refined = replace(model, substeps=model.substeps * multiplier)
        values.append(simulate(refined, initial, np.tile([0.12, 0.06], (n, 1)))[-1])
    errors = [float(np.linalg.norm(x - values[-1])) for x in values[:-1]]
    checks["integration_convergence"] = bool(
        errors[1] < errors[0] and errors[2] < errors[1] and errors[0] < 0.03
    )
    transient = add("lateral_yaw_roll_transient", initial, np.tile([0.12, 0.06], (n, 1)))
    checks["dynamic_roll_not_algebraic"] = bool(
        transient[1, 3] != 0 and abs(transient[1, 3]) < abs(transient[10, 3])
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        independent_oracle=actual_oracle,
        measured=dict(
            vy=float(final[5]),
            yaw_rate=float(final[6]),
            roll=float(final[3]),
            fitted_xy_curvature=float(1 / radius),
        ),
        integration_errors=errors,
        parameters=asdict(model),
    ), traces


def run_validation(config, output):
    model = DynamicVehicle(**config["vehicle"])
    output, manifest = begin(output, config, "seven_state_dynamics_validation")
    evidence, traces = validate_model(model)
    write_json(output / "validation.json", evidence)
    write_json(output / "trajectories.json", traces)
    manifest.update(
        complete=evidence["passed"],
        dynamics_model="dynamic_single_track_7",
        state_fields=["px", "py", "heading", "roll", "vx", "vy", "yaw_rate"],
        control_fields=["steering", "throttle"],
        numerical_integration="explicit midpoint, fixed substeps",
        integration_substep_s=model.dt / model.substeps,
    )
    write_json(output / "manifest.json", manifest)
    if not evidence["passed"]:
        raise RuntimeError(f"Dynamics validation failed: {evidence['checks']}")
    return evidence


def figures(output):
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualization.figures import save

    output = Path(output)
    data = json.loads((output / "trajectories.json").read_text())
    evidence = json.loads((output / "validation.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for name in ("left_turn", "right_turn", "steering_reversal"):
        x = np.array(data[name]["states"])
        t = np.arange(len(x)) * data[name]["dt"]
        axes[0, 0].plot(x[:, 0], x[:, 1], label=name)
        for ax, index, label in zip(
            axes.flat[1:5],
            [4, 5, 6, 3],
            ["vx [m/s]", "vy [m/s]", "yaw rate [rad/s]", "roll [rad]"],
            strict=True,
        ):
            ax.plot(t, x[:, index], label=name)
            ax.set(xlabel="Time [s]", ylabel=label)
    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].legend(fontsize=7)
    axes[1, 2].loglog([1, 2, 4], evidence["integration_errors"], "o-")
    axes[1, 2].set(xlabel="Substep refinement factor", ylabel="Endpoint error vs finest")
    fig.suptitle(
        "Seven-state model validation · actual simulations, independent force-balance oracle"
    )
    save(fig, output / "figures", "dynamic_vehicle_validation")
