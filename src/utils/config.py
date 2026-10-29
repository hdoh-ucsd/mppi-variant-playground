from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from dynamics.bicycle import Bicycle

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "configs"


def read_yaml(path):
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def load_experiment(controller="standard", scenario="open_space", root=DEFAULT_ROOT):
    root = Path(root)
    config = {
        "dynamics": read_yaml(root / "dynamics/bicycle.yaml"),
        "controller": read_yaml(root / f"controllers/mppi_{controller}.yaml"),
        "scenario": read_yaml(root / f"scenarios/{scenario}.yaml"),
        "costs": read_yaml(root / "costs.yaml"),
    }
    validate_experiment(config)
    return deepcopy(config)


def validate_experiment(config):
    model = Bicycle(**config["dynamics"])
    scenario = config["scenario"]
    for field in ("initial_state", "goal_state"):
        if np.shape(scenario[field]) != (4,) or not np.isfinite(scenario[field]).all():
            raise ValueError(f"Invalid {field}")
    if scenario["max_steps"] < 1 or scenario["vehicle_radius"] <= 0:
        raise ValueError("Positive episode length and vehicle radius required")
    for obstacle in scenario["obstacles"]:
        if len(obstacle) != 3 or not np.isfinite(obstacle).all() or obstacle[2] <= 0:
            raise ValueError("Obstacles must be finite [x,y,positive radius]")
    execution = Bicycle(**{**config["dynamics"], **scenario.get("execution_overrides", {})})
    if execution.dt != model.dt or not np.array_equal(execution.control_scale, model.control_scale):
        raise ValueError("Execution and prediction must share frequency and actuator limits")
    disturbance = scenario.get("disturbance", {})
    if disturbance and (
        np.shape(disturbance["delta"]) != (4,) or not np.isfinite(disturbance["delta"]).all()
    ):
        raise ValueError("Disturbance delta must have four finite entries")
    noise = scenario.get("noise_std", [0, 0, 0, 0])
    if np.shape(noise) != (4,) or not np.isfinite(noise).all() or np.any(np.array(noise) < 0):
        raise ValueError("Invalid disturbance standard deviations")
