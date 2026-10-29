"""Metrics computed from executed physical states, separate from sampling costs."""

import numpy as np
from scipy.stats import t

from mppi_variant_playground.dynamics.bicycle import wrap_angle
from mppi_variant_playground.scenarios.environment import swept_clearance


def reached(state, scenario):
    goal = np.array(scenario["goal_state"])
    return bool(
        np.linalg.norm(state[:2] - goal[:2]) <= scenario["position_tolerance"]
        and abs(wrap_angle(state[2] - goal[2])) <= scenario["heading_tolerance"]
        and abs(state[3] - goal[3]) <= scenario["speed_tolerance"]
    )


def cross_track(states, scenario):
    start = np.array(scenario["initial_state"][:2])
    vector = np.array(scenario["goal_state"][:2]) - start
    return np.abs(
        (states[..., 0] - start[0]) * vector[1] - (states[..., 1] - start[1]) * vector[0]
    ) / max(np.linalg.norm(vector), 1e-12)


def episode_metrics(
    states, controls, step_data, scenario, model, cumulative_cost, prediction_errors
):
    goal = np.array(scenario["goal_state"])
    delta = np.diff(np.vstack([np.zeros(2), controls]), axis=0)
    gaps = swept_clearance(
        states[:-1], states[1:], scenario["obstacles"], scenario["vehicle_radius"]
    )
    min_gap = float(gaps.min()) if np.isfinite(gaps).any() else None
    success = reached(states[-1], scenario) and not np.any(gaps <= 0)
    recovery = None
    disturbance_step = scenario.get("disturbance", {}).get("step")
    if disturbance_step is not None and len(controls) > disturbance_step:
        errors = cross_track(states, scenario)
        for i in range(disturbance_step + 1, len(states) - 4):
            if np.all(errors[i : i + 5] <= 0.25):
                recovery = (i - (disturbance_step + 1)) * model.dt
                break
    return {
        "success": bool(success),
        "goal_position_error": float(np.linalg.norm(states[-1, :2] - goal[:2])),
        "goal_orientation_error": float(abs(wrap_angle(states[-1, 2] - goal[2]))),
        "time_to_completion_s": len(controls) * model.dt if success else None,
        "executed_duration_s": len(controls) * model.dt,
        "cumulative_evaluation_cost": float(cumulative_cost),
        "collision": bool(np.any(gaps <= 0)),
        "minimum_obstacle_clearance": min_gap,
        "constraint_violations": int(
            np.sum(np.abs(controls) > model.control_scale + 1e-12)
            + np.sum(states[:, 3] < model.velocity_min - 1e-12)
            + np.sum(states[:, 3] > model.velocity_max + 1e-12)
        ),
        "mean_ess": float(step_data.ess.mean()),
        "mean_weight_entropy": float(step_data.weight_entropy.mean()),
        "degenerate_weight_fraction": float(step_data.degenerate.mean()),
        "mean_trajectory_spread": float(step_data.trajectory_spread.mean()),
        "disturbance_recovery_time_s": recovery,
        "prediction_tracking_rmse": float(np.sqrt(np.mean(np.square(prediction_errors)))),
        "reference_cross_track_rmse": float(np.sqrt(np.mean(cross_track(states, scenario) ** 2))),
        "dynamics_evaluations_per_step": int(step_data.dynamics_evaluations.iloc[0]),
        "total_dynamics_evaluations": int(step_data.dynamics_evaluations.sum()),
        "total_rollout_evaluations": int(step_data.rollout_evaluations.sum()),
        "mean_planning_latency_s": float(step_data.planning_latency_s.mean()),
        "p95_planning_latency_s": float(step_data.planning_latency_s.quantile(0.95)),
        "total_normalized_control_effort": float(
            model.dt * np.sum((controls / model.control_scale) ** 2)
        ),
        "total_normalized_control_variation": float(np.sum((delta / model.control_scale) ** 2)),
        "maximum_steering_change_rad": float(np.max(np.abs(delta[:, 1]))),
        "steps": len(controls),
    }


def mean_interval(values):
    """Descriptive Student-t interval for independent seed means, not a significance test."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    radius = (
        float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
        if len(values) > 1
        else None
    )
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "ci95_low": mean - radius if radius is not None else None,
        "ci95_high": mean + radius if radius is not None else None,
    }
