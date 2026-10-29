"""Independent scalar SDP solution, Monte Carlo covariance, and CC vehicle invariants."""

import numpy as np
import pytest

from analysis.variant_campaign import case_config, equal_work_samples
from controllers.covariance import CovarianceInfeasible, covariance_steering
from evaluation.runner import make_controller
from utils.config import read_yaml


def test_binding_scalar_terminal_constraint_has_analytic_solution():
    a = b = np.ones((2, 1, 1))
    gains, cov, info = covariance_steering(a, b, [[1]], [[1.1]], np.zeros(3), np.ones(2))
    assert gains[1, 0, 0] == pytest.approx(-1 + np.sqrt(0.1), abs=1e-5)
    assert cov[-1, 0, 0] == pytest.approx(1.1, abs=1e-5)
    assert info["covariance_constraint_residual"] <= 1e-5
    rng = np.random.default_rng(991)
    noise = rng.normal(size=(100000, 2))
    endpoints = (1 + gains[1, 0, 0]) * noise[:, 0] + noise[:, 1]
    assert np.var(endpoints) == pytest.approx(cov[-1, 0, 0], abs=0.015)


def test_infeasible_last_noise_floor_is_reported():
    with pytest.raises(CovarianceInfeasible, match="status"):
        covariance_steering(
            np.ones((2, 1, 1)), np.ones((2, 1, 1)), [[1]], [[0.5]], np.zeros(3), np.ones(2)
        )


def test_unconstrained_scalar_quadratic_solution():
    # min (1+k)^2+1 + k^2 -> k=-1/2, terminal variance1.25.
    gains, cov, info = covariance_steering(
        np.ones((2, 1, 1)), np.ones((2, 1, 1)), [[1]], [[2]], np.array([0, 0, 1]), np.ones(2)
    )
    assert gains[1, 0, 0] == pytest.approx(-0.5, abs=1e-8)
    assert cov[-1, 0, 0] == pytest.approx(1.25, abs=1e-8)
    assert info["solver"] == "unconstrained_quadratic_feasible"


def test_vehicle_covariance_sampling_fit_and_accounting():
    options = read_yaml("configs/variants/benchmark.yaml")
    samples = equal_work_samples(options, "cc_mppi")
    c, _ = case_config(options, "open_space", "cc_mppi", samples)
    c["controller"]["snapshot_steps"] = [0]
    one, two = make_controller(c), make_controller(c)
    x = np.asarray(c["scenario"]["initial_state"])
    for ctrl in (one, two):
        ctrl.reset(x, 97)
    np.testing.assert_allclose(one.command(x), two.command(x), atol=1e-12)
    assert one.model.transitions == (samples + 21) * 20 == 7700
    diag = one.get_diagnostics()
    snap = one.get_rollout_snapshot()
    for matrix in [diag["predicted_terminal_covariance"], diag["empirical_terminal_covariance"]]:
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
        assert np.linalg.eigvalsh(matrix).min() > -1e-10
    np.testing.assert_allclose(
        snap["updated_mean"], np.einsum("k,kti->ti", snap["weights"], snap["latent_controls"])
    )
    np.testing.assert_array_equal(one.mean[-1], snap["updated_mean"][-1])
    assert diag["covariance_constraint_residual"] <= 1e-5


def test_invalid_covariance_rejected():
    with pytest.raises(ValueError):
        covariance_steering(
            np.ones((2, 1, 1)), np.ones((2, 1, 1)), [[np.nan]], [[1]], np.zeros(3), np.ones(2)
        )
