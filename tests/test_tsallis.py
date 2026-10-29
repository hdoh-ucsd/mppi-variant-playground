"""Paper-convention support, KL limit and independent constrained optimum."""

import numpy as np
import pytest
from scipy.optimize import minimize

from analysis.dynamic_campaign import configuration, resolved_case
from controllers.tsallis import tsallis_weights
from evaluation.runner import make_controller
from sampling.gaussian import importance_weights


def test_cutoff_and_hand_calculated_mean():
    w = tsallis_weights([0.0, 1.0, 2.0, 3.0], 2.0, 2.0)
    np.testing.assert_allclose(w, [2 / 3, 1 / 3, 0, 0])
    assert w @ np.array([1, 4, 8, 16]) == pytest.approx(2.0)
    with pytest.raises(FloatingPointError, match="support"):
        tsallis_weights([2, 3], 2, 2)


def test_fixed_scale_kl_limit_and_large_costs():
    cost = np.array([0.0, 1.0, 3.0, 100.0])
    ratio = np.array([1.0, -0.2, 0.0, 100.0])
    expected = importance_weights(cost, 2.0, ratio)
    np.testing.assert_allclose(tsallis_weights(cost, 2.0, 1.0, ratio), expected)
    for r in [1 + 1e-8, 1 - 1e-8]:
        np.testing.assert_allclose(tsallis_weights(cost, 2.0, r, ratio), expected, atol=1e-7)
    assert np.isfinite(tsallis_weights([0.0, 1e200], 1, 1.5)).all()


def test_tsallis_density_solves_divergence_problem_independently():
    # r=2: D_r(q||p)=sum(q^2/p)-1. Implied objective lambda is scale*Z/r.
    prior = np.array([0.2, 0.5, 0.3])
    cost = np.array([0.0, 0.8, 3.0])
    scale = 2.0
    target = tsallis_weights(cost, scale, 2.0, np.log(prior))
    z = np.sum(prior * np.maximum(1 - cost / scale, 0))
    lam = scale * z / 2
    result = minimize(
        lambda q: cost @ q / lam + np.sum(q * q / prior) - 1,
        prior,
        bounds=[(0, 1)] * 3,
        constraints={"type": "eq", "fun": lambda q: q.sum() - 1},
        method="SLSQP",
        options={"ftol": 1e-12},
    )
    assert result.success
    np.testing.assert_allclose(target, result.x, atol=1e-6)


def test_seven_state_kl_reference_and_reproducibility():
    config, *_ = resolved_case(configuration(), "open_space", 64)
    config["controller"].update(kind="standard", covariance=config["proposals"]["standard"])
    baseline = make_controller(config)
    config["controller"].update(algorithm="tsallis_vi", tsallis_r=1.0, elite_fraction=None)
    other = make_controller(config)
    state = config["scenario"]["initial_state"]
    for c in (baseline, other):
        c.reset(state, 431)
    np.testing.assert_array_equal(baseline.command(state), other.command(state))
    np.testing.assert_array_equal(baseline.mean, other.mean)
    config["controller"].update(tsallis_r=1.5, elite_fraction=0.8)
    first, second = make_controller(config), make_controller(config)
    for c in (first, second):
        c.reset(state, 432)
    np.testing.assert_array_equal(first.command(state), second.command(state))
    assert first.diagnostics["support_fraction"] <= 0.81


@pytest.mark.parametrize(
    "args", [([], 1, 2), ([1], 0, 2), ([-1], 1, 2), ([np.nan], 1, 2), ([1], 1, 0)]
)
def test_invalid_parameters(args):
    with pytest.raises(ValueError):
        tsallis_weights(*args)


def test_registry_counts_real_model_calls_and_rejects_unknown_names():
    config, *_ = resolved_case(configuration(), "open_space", 16)
    config["controller"].update(
        kind="standard", covariance=config["proposals"]["standard"], algorithm="tsallis_vi"
    )
    controller = make_controller(config)
    state = config["scenario"]["initial_state"]
    controller.reset(state, 18)
    controller.command(state)
    assert controller.model.transitions == (16 + 1) * 20
    config["controller"]["algorithm"] = "renamed_covariance"
    with pytest.raises(ValueError, match="unimplemented"):
        make_controller(config)
