"""DIAL covariance schedule, independent cost-only update, and full work accounting."""

import numpy as np
import pytest

from analysis.dynamic_campaign import configuration, resolved_case
from controllers.dial import noise_scales
from evaluation.runner import make_controller
from sampling.gaussian import importance_weights


def test_equation_seven_variance_schedule():
    scales = noise_scales(5, 4, 2, 0.5, 2.0)
    expected = np.exp(-2 / (0.5 * 4) - (4 - np.arange(5)) / (2 * 4))
    np.testing.assert_allclose(scales**2, expected)
    assert np.all(np.diff(scales) > 0)
    assert np.all(scales < noise_scales(5, 4, 0, 0.5, 2.0))
    np.testing.assert_array_equal(noise_scales(5, 4, 2, 0.5, 2.0, False), np.ones(5))


def test_cost_only_gaussian_update_against_exact_integral():
    # J=.5*a*(z-b)^2 tilted Gaussian N(u,s^2) has exact mean below.
    rng = np.random.default_rng(47901)
    u, s, a, b, temp = -0.2, 0.7, 2.0, 1.0, 0.8
    samples = rng.normal(u, s, 200000)
    cost = 0.5 * a * (samples - b) ** 2
    estimate = importance_weights(cost, temp) @ samples
    exact = (u / s**2 + a * b / temp) / (1 / s**2 + a / temp)
    assert estimate == pytest.approx(exact, abs=0.005)


def make_dial(iterations=4, anneal=True):
    c, *_ = resolved_case(configuration(), "open_space", 16)
    c["controller"].update(
        algorithm="dial_mpc",
        kind="standard",
        covariance=[[0.2, 0], [0, 0.2]],
        iterations=iterations,
        anneal=anneal,
        snapshot_steps=[0],
    )
    return make_controller(c), np.asarray(c["scenario"]["initial_state"])


def test_all_passes_counted_and_repeatable():
    first, state = make_dial()
    second, _ = make_dial()
    for c in [first, second]:
        c.reset(state, 49)
    np.testing.assert_array_equal(first.command(state), second.command(state))
    assert first.model.transitions == 4 * 17 * 20
    assert first.diagnostics["dynamics_evaluations"] == first.model.transitions
    assert first.diagnostics["rollout_evaluations"] == 64
    assert len(first.diagnostics["inner_passes"]) == 4
    np.testing.assert_array_equal(first.mean[-1], np.zeros(2))


def test_one_pass_is_cost_only_not_fixed_base_it_mppi():
    controller, state = make_dial(1, False)
    controller.mean[:] = 0.1
    controller.plan(state)
    snap = controller.get_rollout_snapshot()
    expected = importance_weights(snap["costs"], controller.temperature)
    np.testing.assert_allclose(snap["weights"], expected)
    np.testing.assert_allclose(
        snap["updated_mean"], np.einsum("k,kti->ti", expected, snap["latent_controls"])
    )


@pytest.mark.parametrize("args", [(0, 4, 0, 1, 1), (5, 4, 4, 1, 1), (5, 4, 0, 0, 1)])
def test_schedule_rejects_invalid_args(args):
    with pytest.raises(ValueError):
        noise_scales(*args)
