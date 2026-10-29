import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from mppi_variant_playground.dynamics.bicycle import Bicycle
from mppi_variant_playground.evaluation.benchmark import summarize
from mppi_variant_playground.evaluation.metrics import episode_metrics, mean_interval, reached
from mppi_variant_playground.evaluation.runner import run_episode
from mppi_variant_playground.utils.config import load_experiment, validate_experiment


@pytest.mark.parametrize(
    "scenario",
    ["open_space", "narrow_passage", "multiple_obstacles", "disturbance", "model_mismatch"],
)
def test_all_scenarios_closed_loop_and_artifacts(scenario, tmp_path):
    config = load_experiment(scenario=scenario)
    config["controller"].update(samples=16, horizon=6)
    config["scenario"]["max_steps"] = 3
    out = tmp_path / scenario
    result = run_episode(config, 42, out)
    assert result["metrics"]["steps"] == 3
    assert result["metrics"]["constraint_violations"] == 0
    assert np.isfinite(result["metrics"]["cumulative_evaluation_cost"])
    for name in [
        "config.yaml",
        "result.json",
        "diagnostics.csv",
        "controls.csv",
        "trajectory.csv",
        "rollouts.npz",
    ]:
        assert (out / name).exists()
    assert json.loads((out / "result.json").read_text())["seed"] == 42


def test_covariance_fairness_and_config_validation():
    configs = [load_experiment(kind) for kind in ["standard", "isotropic", "diagonal", "full"]]
    for config in configs:
        assert np.trace(config["controller"]["covariance"]) == pytest.approx(0.44)
        np.testing.assert_array_equal(
            config["controller"]["base_covariance"], configs[0]["controller"]["base_covariance"]
        )
    bad = deepcopy(configs[0])
    bad["scenario"]["execution_overrides"] = {"dt": 0.2}
    with pytest.raises(ValueError):
        validate_experiment(bad)


def test_metrics_on_hand_computed_trajectory():
    scenario = load_experiment()["scenario"]
    scenario.update(initial_state=[0, 0, 0, 0], goal_state=[1, 0, 0, 0], obstacles=[[3, 0, 0.5]])
    model = Bicycle(dt=0.1)
    states = np.array([[0.0, 0, 0, 0], [1.0, 0, 0, 0]])
    controls = np.array([[2.0, 0.5]])
    logs = pd.DataFrame(
        {
            "ess": [4.0],
            "weight_entropy": [1.0],
            "degenerate": [False],
            "trajectory_spread": [2.0],
            "dynamics_evaluations": [20],
            "rollout_evaluations": [4],
            "planning_latency_s": [0.01],
        }
    )
    metrics = episode_metrics(states, controls, logs, scenario, model, 7.0, [np.zeros(4)])
    assert metrics["success"]
    assert metrics["goal_position_error"] == 0
    assert metrics["total_normalized_control_effort"] == pytest.approx(0.2)
    assert metrics["total_normalized_control_variation"] == 2
    assert metrics["minimum_obstacle_clearance"] == pytest.approx(1.28)
    assert reached(states[-1], scenario)
    interval = mean_interval([1, 2, 3])
    assert interval["mean"] == 2 and interval["std"] == 1
    assert interval["ci95_low"] < 2 < interval["ci95_high"]


def test_summary_preserves_paired_seed_differences(tmp_path):
    rows = []
    for seed in [42, 43]:
        for name, offset in [("standard", 0), ("full", 1)]:
            rows.append(
                {
                    "controller": name,
                    "scenario": "open_space",
                    "seed": seed,
                    "mode": "dynamics",
                    "metrics": {
                        key: seed + offset
                        for key in [
                            "cumulative_evaluation_cost",
                            "goal_position_error",
                            "mean_ess",
                            "total_normalized_control_variation",
                            "mean_planning_latency_s",
                        ]
                    },
                }
            )
    summarize(rows, tmp_path)
    pairs = json.loads((tmp_path / "paired_differences.json").read_text())
    assert all(row["mean"] == 1 and row["n"] == 2 for row in pairs)


def test_seed42_open_space_converges(tmp_path):
    result = run_episode(load_experiment(), 42, tmp_path / "regression")
    assert result["metrics"]["success"]
    assert not result["metrics"]["collision"]


@pytest.mark.parametrize("scenario", ["narrow_passage", "disturbance"])
def test_seed42_scenario_regression(scenario, tmp_path):
    result = run_episode(load_experiment(scenario=scenario), 42, tmp_path / scenario)
    metrics = result["metrics"]
    assert not metrics["collision"]
    assert metrics["constraint_violations"] == 0
    assert metrics["goal_position_error"] < 1.0
    if scenario == "narrow_passage":
        assert metrics["success"]
    else:
        # Preserve a real observed failure rather than claiming robustness from replanning.
        assert not metrics["success"]
        assert metrics["steps"] == 120


def test_sustained_disturbance_recovery_metric():
    config = load_experiment(scenario="disturbance")
    scenario = config["scenario"]
    scenario.update(
        initial_state=[0, 0, 0, 0],
        goal_state=[8, 0, 0, 0],
        disturbance={"step": 0, "delta": [0, 1, 0, 0]},
    )
    states = np.zeros((9, 4))
    states[:, 0] = np.arange(9)
    states[:, 1] = [0, 1, 0.8, 0.4, 0.2, 0.1, 0, 0, 0]
    controls = np.zeros((8, 2))
    logs = pd.DataFrame(
        {
            "ess": np.ones(8),
            "weight_entropy": np.zeros(8),
            "degenerate": np.zeros(8),
            "trajectory_spread": np.ones(8),
            "dynamics_evaluations": np.ones(8),
            "rollout_evaluations": np.ones(8),
            "planning_latency_s": np.zeros(8),
        }
    )
    metrics = episode_metrics(
        states, controls, logs, scenario, Bicycle(dt=0.1), 0, np.zeros((8, 4))
    )
    assert metrics["disturbance_recovery_time_s"] == pytest.approx(0.3)
