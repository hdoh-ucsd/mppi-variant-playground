import numpy as np
import pandas as pd

from mppi_variant_playground.evaluation.runner import make_controller, run_episode
from mppi_variant_playground.scenarios.environment import Environment
from mppi_variant_playground.utils.config import load_experiment


def test_reset_and_different_seeds():
    config = load_experiment()
    config["controller"].update(samples=32, horizon=8)
    controller = make_controller(config)
    state = np.array(config["scenario"]["initial_state"])
    controller.reset(state, 42)
    a = controller.command(state)
    controller.reset(state, 42)
    np.testing.assert_array_equal(controller.command(state), a)
    controller.reset(state, 43)
    assert not np.array_equal(controller.command(state), a)


def test_paired_exogenous_noise_is_independent_of_controls():
    config = load_experiment()
    scenario = config["scenario"]
    scenario["noise_std"] = [0.1, 0.1, 0.01, 0.01]
    scenario["initial_state"][3] = 1.5
    a, b = [Environment(scenario, config["dynamics"], 42) for _ in range(2)]
    for _ in range(5):
        pa, pb = a.model.step(a.state, [1, 0.1]), b.model.step(b.state, [-1, -0.1])
        np.testing.assert_allclose(a.step([1, 0.1]) - pa, b.step([-1, -0.1]) - pb, atol=1e-14)


def test_saved_episode_reproducibility(tmp_path):
    config = load_experiment(scenario="disturbance")
    config["controller"].update(samples=32, horizon=8)
    config["scenario"]["max_steps"] = 24
    a = run_episode(config, 42, tmp_path / "a")
    b = run_episode(config, 42, tmp_path / "b")
    for name in ["trajectory.csv", "controls.csv"]:
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "a" / name), pd.read_csv(tmp_path / "b" / name)
        )
    for key in ["cumulative_evaluation_cost", "mean_ess", "goal_position_error"]:
        assert a["metrics"][key] == b["metrics"][key]
