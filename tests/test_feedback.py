"""Independent tracking optimum, local plant derivative, and tube-selection invariants."""

import numpy as np
import pytest

from analysis.variant_campaign import case_config, equal_work_samples
from controllers.feedback import linearize, lqr_gains, state_error
from controllers.tube import select_actual
from evaluation.runner import make_controller
from utils.config import read_yaml


def test_riccati_matches_independent_batch_least_squares():
    a = np.array([[[1.0, 0.1], [0, 1]]] * 5)
    b = np.array([[[0.005], [0.1]]] * 5)
    q = np.diag([3.0, 1.0])
    r = np.array([[0.2]])
    qf = 4 * q
    gains = lqr_gains(a, b, q, r, qf)
    x = np.array([1.0, -0.2])
    controls = []
    for at, bt, k in zip(a, b, gains, strict=True):
        u = k @ x
        controls.append(u)
        x = at @ x + bt @ u

    # Independently construct all stacked state responses to each control basis.
    def residual(us):
        x = np.array([1.0, -0.2])
        values = []
        for at, bt, u in zip(a, b, us.reshape(5, 1), strict=True):
            values.extend(np.sqrt(q) @ x)
            values.extend(np.sqrt(r) @ u)
            x = at @ x + bt @ u
        values.extend(np.sqrt(qf) @ x)
        return np.array(values)

    zero = residual(np.zeros(5))
    matrix = np.stack([residual(e) - zero for e in np.eye(5)], axis=1)
    expected = np.linalg.lstsq(matrix, -zero, rcond=None)[0]
    np.testing.assert_allclose(np.array(controls).ravel(), expected, atol=1e-10)


def make_tube(samples=32):
    options = read_yaml("configs/variants/benchmark.yaml")
    c, _ = case_config(options, "disturbance", "tube_mppi", samples)
    return make_controller(c), np.asarray(c["scenario"]["initial_state"])


def test_linearization_predicts_small_vehicle_perturbations():
    c, x = make_tube()
    u = np.array([0.03, 0.2])
    x[4] = 1.2
    states = np.array([x, c.model.step(x, u)])
    before = c.model.transitions
    a, b = linearize(c.model, states, u[None])
    assert c.model.transitions - before == 18
    dx = np.array([1, -1, 0.2, 0.1, 0.3, 0.2, -0.1]) * 1e-6
    du = np.array([0.2, -0.3]) * 1e-6
    actual = state_error(c.model.step(x + dx, u + du * c.model.control_scale), states[1])
    np.testing.assert_allclose(actual, a[0] @ dx + b[0] @ du, atol=1e-9)


def test_selection_threshold_and_yaw_wrap():
    assert select_actual(12, 10, 2)
    assert not select_actual(12.01, 10, 2)
    np.testing.assert_allclose(state_error([0, 0, -np.pi + 0.1], [0, 0, np.pi - 0.1]), [0, 0, 0.2])


def test_tube_actual_nominal_separation_work_and_reproducibility():
    options = read_yaml("configs/variants/benchmark.yaml")
    samples = equal_work_samples(options, "tube_mppi")
    one, state = make_tube(samples)
    two, _ = make_tube(samples)
    for ctrl in [one, two]:
        ctrl.reset(state, 51)
        ctrl.config["reset_threshold"] = 0.0
    np.testing.assert_array_equal(one.command(state), two.command(state))
    assert one.model.transitions == 2 * (samples + 1) * 20 + 18 * 20
    assert one.model.transitions <= 7700
    observed = one.nominal_state.copy()
    observed[1] += 10.0
    observed[2] += 0.2
    one.command(observed)
    assert one.diagnostics["tracking_error"] > 0
    assert not one.diagnostics["selected_actual"]
    assert one.diagnostics["tracking_control_norm"] > 0
    assert one.diagnostics["rollout_evaluations"] == 2 * samples
    assert np.isfinite(one.nominal_state).all()


def test_tracking_reduces_local_quadratic_error():
    c, x = make_tube()
    c.reset(x, 19)
    c.command(x)
    gains = c.diagnostics["tracking_gains"]
    assert np.isfinite(gains).all()
    assert np.linalg.norm(gains) > 0
    with pytest.raises(ValueError):
        select_actual(1, 2, -1)
