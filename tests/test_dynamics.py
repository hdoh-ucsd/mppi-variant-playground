import numpy as np
import pytest

from mppi_variant_playground.dynamics.bicycle import Bicycle, wrap_angle
from mppi_variant_playground.scenarios.environment import Environment, clearance, swept_clearance
from mppi_variant_playground.utils.config import load_experiment


def test_euler_and_vectorization():
    model = Bicycle(dt=0.1, wheelbase=2)
    state = np.array([1.0, 2.0, 0.0, 3.0])
    expected = [1.3, 2.0, 3 / 2 * np.tan(0.2) * 0.1, 2.9]
    np.testing.assert_allclose(model.step(state, [-1.0, 0.2]), expected)
    np.testing.assert_allclose(
        model.step(np.tile(state, (7, 1)), [-1.0, 0.2]), np.tile(expected, (7, 1))
    )


def test_saturation_velocity_and_angle():
    model = Bicycle()
    np.testing.assert_equal(model.clip_control([100, -100]), [2, -0.5])
    assert model.step([0.0, 0.0, 0.0, 3.0], [100, 0])[3] == 3
    assert model.step([0.0, 0.0, 0.0, 0.0], [-100, 0])[3] == 0
    assert -np.pi <= wrap_angle(27) < np.pi


def test_collision_geometry_including_tunneling():
    obs = [[0, 0, 1.0]]
    assert clearance([1.5, 0, 0, 0], obs, 0.5) == 0
    assert clearance([2.0, 0, 0, 0], obs, 0.5) == 0.5
    assert swept_clearance(np.array([-2, 0, 0, 0]), np.array([2, 0, 0, 0]), obs, 0.5) == -1.5
    assert swept_clearance(np.zeros(4), np.zeros(4), obs, 0.5) == -1.5
    assert np.isinf(clearance(np.zeros(4), [], 0.2))


def test_execution_only_disturbance_and_mismatch():
    config = load_experiment(scenario="disturbance")
    scenario = config["scenario"]
    env = Environment(scenario, config["dynamics"], 42)
    prediction = Bicycle(**config["dynamics"])
    state = env.state.copy()
    for _ in range(21):
        state = prediction.step(state, [0, 0])
        actual = env.step([0, 0])
    np.testing.assert_allclose(actual[:3] - state[:3], [0, 0.9, 0.18])
    mismatch = Environment(
        load_experiment(scenario="model_mismatch")["scenario"], config["dynamics"]
    )
    assert mismatch.model.wheelbase != prediction.wheelbase
    assert prediction.wheelbase == 1


@pytest.mark.parametrize("params", [{"wheelbase": 0}, {"dt": -1}, {"steering_limit": 2}])
def test_invalid_dynamics(params):
    with pytest.raises(ValueError):
        Bicycle(**params)
