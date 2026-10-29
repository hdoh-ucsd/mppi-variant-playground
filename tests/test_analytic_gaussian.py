import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.stats import multivariate_normal

from analysis.analytic_gaussian import exact_target, quadratic_cost


def test_diagonal_posterior_and_zero_cost():
    p, h, b, temp = np.diag([2.0, 0.5]), np.diag([3.0, 0.0]), [1.0, -4.0], 2.0
    mean, cov, _ = exact_target(p, h, b, temp)
    np.testing.assert_allclose(cov, np.diag([0.5, 0.5]))
    np.testing.assert_allclose(mean, [0.75, 0.0])
    mu0, c0, logz = exact_target(p, np.zeros((2, 2)), b, temp)
    np.testing.assert_array_equal(mu0, [0, 0])
    np.testing.assert_allclose(c0, p)
    assert logz == pytest.approx(0)


def test_rotated_posterior_against_independent_numerical_integration():
    p = np.array([[1.1, 0.15], [0.15, 0.6]])
    h = np.array([[2.0, -1.0], [-1.0, 3.0]])
    b, temp = np.array([0.6, -0.3]), 0.8
    prior = multivariate_normal(cov=p)

    def integrand(y, x):
        z = np.array([x, y])
        delta = z - b
        weight = prior.pdf(z) * np.exp(-0.5 * delta @ h @ delta / temp)
        return weight * np.array([1, x, y, x * x, x * y, y * y])

    integral, _ = quad_vec(lambda x: quad_vec(lambda y: integrand(y, x), -8, 8)[0], -8, 8)
    exact_mean, exact_cov, logz = exact_target(p, h, b, temp)
    numerical_mean = integral[1:3] / integral[0]
    second = np.array([[integral[3], integral[4]], [integral[4], integral[5]]]) / integral[0]
    np.testing.assert_allclose(exact_mean, numerical_mean, atol=1e-8)
    np.testing.assert_allclose(
        exact_cov, second - np.outer(numerical_mean, numerical_mean), atol=1e-8
    )
    assert np.exp(logz) == pytest.approx(integral[0], abs=1e-9)


@pytest.mark.parametrize("h,temp", [(np.diag([1, -1]), 1), (np.eye(2), 0), ([[1, 2], [0, 1]], 1)])
def test_reject_invalid_quadratic(h, temp):
    with pytest.raises(ValueError):
        exact_target(np.eye(2), h, [0, 0], temp)


def test_vectorized_quadratic_matches_scalar_definition():
    z = np.random.default_rng(2).normal(size=(20, 2))
    h, b = np.array([[2, 0.4], [0.4, 1]]), np.array([0.5, -0.2])
    expected = [0.5 * (x - b) @ h @ (x - b) for x in z]
    np.testing.assert_allclose(quadratic_cost(z, h, b), expected)
