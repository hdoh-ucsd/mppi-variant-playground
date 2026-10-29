"""AIS mixed objective and feedback likelihood, nominal selection, and work audit."""

import numpy as np
from scipy.stats import multivariate_normal

from analysis.variant_campaign import case_config, equal_work_samples
from controllers.robust import augmented_costs, candidate_index
from evaluation.runner import make_controller
from utils.config import read_yaml


def test_feedback_correction_equals_conditional_gaussian_density_ratio():
    rng = np.random.default_rng(99)
    noise = rng.normal(size=(9, 4, 2)) * 0.1
    mean = rng.normal(size=(4, 2)) * 0.2
    feedback = rng.normal(size=(9, 4, 2)) * 0.3
    cov = np.array([[0.14, 0], [0, 0.26]])
    costs = np.arange(9.0)
    nominal, actual = augmented_costs(costs, costs, mean, noise, feedback, cov, 8.0, 4.0)
    corrected = []
    for i in range(9):
        ratio = 0.0
        for t in range(4):
            mu = mean[t] + feedback[i, t]
            v = mu + noise[i, t]
            ratio += multivariate_normal.logpdf(v, np.zeros(2), cov) - multivariate_normal.logpdf(
                v, mu, cov
            )
        corrected.append(costs[i] - 8 * ratio)
    np.testing.assert_allclose(actual, corrected, atol=1e-12)
    assert np.isfinite(nominal).all()


def test_mixed_cost_clamp_independent_values_and_zero_feedback():
    mean = np.zeros((1, 2))
    noise = np.zeros((3, 1, 2))
    feedback = noise.copy()
    nominal, actual = augmented_costs(
        np.array([2.0, 6.0, 2.0]),
        np.array([8.0, 3.0, 1.0]),
        mean,
        noise,
        feedback,
        np.eye(2),
        1.0,
        4.0,
    )
    np.testing.assert_array_equal(nominal, [3.0, 6.0, 2.0])
    np.testing.assert_array_equal(actual, [8.0, 3.0, 1.0])


def test_candidate_feasibility_and_fallback():
    states = np.zeros((3, 7))
    states[:, 0] = [0, 1, 2]
    assert candidate_index(states, states[2], [1, 2, 9], 3) == (1, False)
    assert candidate_index(states, states[2], [4, 5, 9], 3) == (0, True)


def test_robust_seven_state_two_updates_work_and_reproducibility():
    options = read_yaml("configs/variants/benchmark.yaml")
    samples = equal_work_samples(options, "robust_mppi")
    c, _ = case_config(options, "disturbance", "robust_mppi", samples)
    one, two = make_controller(c), make_controller(c)
    state = np.asarray(c["scenario"]["initial_state"])
    for ctrl in (one, two):
        ctrl.reset(state, 83)
    np.testing.assert_array_equal(one.command(state), two.command(state))
    assert one.model.transitions == 40 * samples + 1340
    assert one.model.transitions <= options["transition_budget"]
    # A tight candidate threshold retains the nominal rather than silently resetting.
    one.config["free_energy_threshold"] = 0.0
    disturbed = one.nominal_state.copy()
    disturbed[1] += 1.0
    one.command(disturbed)
    assert one.diagnostics["nominal_selection_infeasible"]
    assert one.diagnostics["sampled_feedback_rms"] > 0
    assert one.diagnostics["tracking_control_norm"] > 0
    assert np.isclose(one.diagnostics["weights"].sum(), 1.0)
    assert np.isfinite(one.mean).all()
