"""Reproducible paired execution, provenance, exact YAML and compact rollout replay."""

import json
import os
import platform
import resource
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from controllers.mppi import MPPI
from costs.shared import SharedCost
from dynamics.bicycle import Bicycle, wrap_angle
from evaluation.metrics import episode_metrics, reached
from scenarios.environment import Environment, swept_clearance
from utils.config import validate_experiment


def git_info():
    def run(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(
            run("status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)runs")
        ),
        "status_scope": "repository excluding generated results/ and runs/",
    }


def make_controller(config):
    if config.get("model_kind") == "dynamic7":
        from costs.dynamic_vehicle import VehicleCost
        from dynamics.dynamic_vehicle import DynamicVehicle
        from scenarios.dynamic_scenes import geometry_for_config

        model = DynamicVehicle(**config["vehicle"])
        geometry = geometry_for_config(config)
        return MPPI(
            model,
            VehicleCost(geometry, model, config["cost"], config["speed_target"]),
            config["controller"],
        )
    model = Bicycle(**config["dynamics"])
    s = config["scenario"]
    cost = SharedCost(
        np.array(s["goal_state"]),
        s["obstacles"],
        model.control_scale,
        s["vehicle_radius"],
        config["costs"]["optimization"],
        model.dt,
    )
    return MPPI(model, cost, config["controller"])


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run_episode(config, seed, output, mode="native", *, observer=None, controller_factory=None):
    validate_experiment(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    started_at = datetime.now(UTC).isoformat()
    provenance = git_info()
    controller = (controller_factory or make_controller)(config)
    scenario = config["scenario"]
    environment_seed = np.random.SeedSequence([seed, 1001])
    env = Environment(scenario, config["dynamics"], environment_seed)
    controller.reset(env.state, seed)
    model = controller.model
    score = SharedCost(
        np.array(scenario["goal_state"]),
        scenario["obstacles"],
        model.control_scale,
        scenario["vehicle_radius"],
        config["costs"]["evaluation"],
        model.dt,
    )
    states, controls, logs, rollouts, plans = [env.state.copy()], [], [], [], []
    errors, previous, total_cost = [], np.zeros(2), 0.0
    started = perf_counter()
    for index in range(scenario["max_steps"]):
        observed = env.state.copy()
        control = controller.command(observed)
        diag = controller.get_diagnostics()
        # Preserve the compact baseline archive when selected steps capture all samples.
        rollouts.append(
            diag.pop("saved_rollouts")[: config["controller"].get("saved_rollouts", 32)]
        )
        plans.append(diag.pop("planned_trajectory"))
        diag.pop("weights")
        mu, cov = diag.pop("perturbation_mean"), diag.pop("perturbation_covariance")
        diag.update(
            {
                "perturbation_mean_acceleration": mu[0],
                "perturbation_mean_steering": mu[1],
                "perturbation_variance_acceleration": cov[0, 0],
                "perturbation_variance_steering": cov[1, 1],
                "perturbation_cross_covariance": cov[0, 1],
            }
        )
        diag.update({"step": index, "time_s": index * model.dt})
        if mode == "wall_clock":
            diag["deadline_missed"] = diag["planning_latency_s"] > config["budget"]["target_s"]
        logs.append(diag)
        expected = model.step(observed, control)
        actual = env.step(control)
        error = actual - expected
        error[2] = wrap_angle(error[2])
        errors.append(error)
        total_cost += score.running(observed, control, previous)
        gap = swept_clearance(observed, actual, scenario["obstacles"], scenario["vehicle_radius"])
        total_cost += model.dt * score.weights["collision"] * (gap <= 0)
        if observer is not None:
            snapshot_getter = getattr(controller, "get_rollout_snapshot", None)
            observer(
                {
                    "step": index,
                    "state": observed.copy(),
                    "next_state": actual.copy(),
                    "control": control.copy(),
                    "diagnostics": dict(diag),
                    "cumulative_running_cost": float(total_cost),
                    "collision": bool(gap <= 0),
                    "snapshot": snapshot_getter() if snapshot_getter is not None else None,
                }
            )
        previous = control
        states.append(actual)
        controls.append(control)
        if reached(actual, scenario) or gap <= 0:
            break
    elapsed = perf_counter() - started
    states, controls = np.array(states), np.array(controls)
    total_cost += score.terminal(states[-1])
    step_data = pd.DataFrame(logs)
    metrics = episode_metrics(states, controls, step_data, scenario, model, total_cost, errors)
    metrics["total_simulation_wall_time_s"] = elapsed
    metrics["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if mode == "wall_clock":
        metrics["deadline_miss_fraction"] = float(step_data.deadline_missed.mean())
    step_data.to_csv(output / "diagnostics.csv", index=False)
    trajectory = pd.DataFrame(states, columns=["px", "py", "theta", "velocity"])
    trajectory.insert(0, "time_s", np.arange(len(states)) * model.dt)
    trajectory.to_csv(output / "trajectory.csv", index=False)
    pd.DataFrame(controls, columns=["acceleration", "steering_angle"]).to_csv(
        output / "controls.csv", index=False
    )
    np.savez_compressed(
        output / "rollouts.npz",
        rollouts=np.array(rollouts, dtype=np.float32),
        plans=np.array(plans, dtype=np.float32),
    )
    metadata = {
        "schema_version": 1,
        "controller": config["controller"]["kind"],
        "scenario": scenario["name"],
        "seed": seed,
        "environment_seed_entropy": [seed, 1001],
        "mode": mode,
        "git": provenance,
        "started_at_utc": started_at,
        "backend": "numpy-cpu-float64",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ["numpy", "scipy", "pandas", "matplotlib", "pyyaml", "imageio"]
        },
        "threads": {k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]},
        "metrics": metrics,
    }
    write_json(output / "result.json", metadata)
    return metadata
