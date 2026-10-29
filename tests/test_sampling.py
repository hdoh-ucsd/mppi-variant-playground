import numpy as np
import pytest
from scipy.special import softmax
from scipy.stats import multivariate_normal

from mppi_variant_playground.sampling.gaussian import (
    importance_weights,
    log_base_over_proposal,
    sample_gaussian,
    validate_covariance,
    weighted_update,
)


@pytest.mark.parametrize(
    "cov", [np.eye(2) * 0.22, np.diag([0.34, 0.10]), [[0.28, 0.08], [0.08, 0.16]]]
)
def test_gaussian_moments_and_factorization(cov):
    cov = np.array(cov)
    samples = sample_gaussian(np.random.default_rng(42), (160000,), cov)
    np.testing.assert_allclose(samples.mean(0), 0, atol=0.004)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), cov, atol=0.003)
    factor = np.linalg.cholesky(cov)
    np.testing.assert_allclose(factor @ factor.T, cov)


@pytest.mark.parametrize(
    "cov",
    [[[1, 2], [2, 1]], [[1, 0.2], [0, 1]], [[1, 0], [0, 0]], [[1, np.nan], [0, 1]], np.eye(3)],
)
def test_reject_invalid_covariance(cov):
    with pytest.raises(ValueError):
        validate_covariance(cov)


def test_full_gaussian_density_ratio():
    rng = np.random.default_rng(17)
    samples, mean = rng.normal(size=(9, 3, 2)), rng.normal(size=(3, 2))
    base, proposal = np.diag([0.28, 0.16]), np.array([[0.3, 0.06], [0.06, 0.14]])
    expected = sum(
        multivariate_normal.logpdf(samples[:, i], mean=np.zeros(2), cov=base)
        - multivariate_normal.logpdf(samples[:, i], mean=mean[i], cov=proposal)
        for i in range(3)
    )
    np.testing.assert_allclose(log_base_over_proposal(samples, mean, base, proposal), expected)


def test_equal_covariance_cross_term_and_cost_only_difference():
    cov = np.diag([0.28, 0.16])
    mean = np.array([[0.5, -0.2], [0.1, 0.3]])
    eps = np.array([[[0.1, 0.2], [-0.2, 0.1]], [[-0.2, -0.1], [0.3, -0.4]]])
    ratio = log_base_over_proposal(mean + eps, mean, cov, cov)
    tilde = np.einsum("ti,ij,ktj->k", mean, np.linalg.inv(cov), eps)
    actual = importance_weights(np.array([2.0, 3.0]), 5, ratio)
    np.testing.assert_allclose(actual, softmax(-np.array([2.0, 3.0]) / 5 - tilde))
    assert not np.allclose(actual, importance_weights(np.array([2.0, 3.0]), 5))


def test_stable_weights_update_and_degenerate_distribution():
    costs = np.array([1e10, 1e10 + 1, 1e10 + 2])
    weights = importance_weights(costs, 0.5)
    np.testing.assert_allclose(weights, softmax(-np.array([0.0, 1.0, 2.0]) / 0.5))
    assert weights.sum() == pytest.approx(1)
    np.testing.assert_allclose(importance_weights([2, 2, 2], 1), np.ones(3) / 3)
    np.testing.assert_allclose(importance_weights([0, 1e200, 1e200], 0.001), [1, 0, 0])
    mean = np.array([[1.0, 2.0]])
    eps = np.array([[[1.0, 0]], [[-1.0, 4.0]]])
    np.testing.assert_allclose(weighted_update(mean, eps, [0.25, 0.75]), [[0.5, 5.0]])


@pytest.mark.parametrize("costs", [[1, np.nan], [1, np.inf], []])
def test_nonfinite_or_empty_costs(costs):
    with pytest.raises((ValueError, FloatingPointError)):
        importance_weights(costs, 1)


@pytest.mark.parametrize("temperature", [0, -1, np.inf, np.nan])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        importance_weights([1, 2], temperature)


def test_nonfinite_ratio():
    with pytest.raises(FloatingPointError):
        importance_weights([1, 2], 1, [0, np.inf])


def test_quadratic_gibbs_mean_matches_analytic_solution():
    # Independent closed-form target: prior times exp(-quadratic / lambda).
    base = np.diag([0.28, 0.16])
    proposal = np.array([[0.3, 0.06], [0.06, 0.2]])
    precision_cost = np.diag([2.0, 4.0])
    target = np.array([0.3, -0.2])
    temperature = 0.7
    mean = np.array([[0.2, -0.1]])
    eps = sample_gaussian(np.random.default_rng(42), (200000, 1), proposal)
    samples = mean + eps
    delta = samples[:, 0] - target
    cost = 0.5 * np.einsum("ki,ij,kj->k", delta, precision_cost, delta)
    ratio = log_base_over_proposal(samples, mean, base, proposal)
    weights = importance_weights(cost, temperature, ratio)
    estimate = weighted_update(mean, eps, weights)[0]
    posterior_covariance = np.linalg.inv(np.linalg.inv(base) + precision_cost / temperature)
    exact_mean = posterior_covariance @ precision_cost @ target / temperature
    np.testing.assert_allclose(estimate, exact_mean, atol=0.003)
