"""Exact Gibbs moments for a Gaussian base and a PSD quadratic cost."""

import numpy as np

from sampling.gaussian import validate_covariance


def quadratic_parameters(base, hessian, center, temperature):
    center = np.asarray(center, dtype=float)
    if center.ndim != 1 or not len(center) or not np.isfinite(center).all():
        raise ValueError("Finite nonempty center vector required")
    d = len(center)
    base = validate_covariance(base, dimension=d)
    hessian = np.asarray(hessian, dtype=float)
    if (
        hessian.shape != (d, d)
        or not np.isfinite(hessian).all()
        or not np.allclose(hessian, hessian.T, atol=1e-12, rtol=1e-12)
        or np.linalg.eigvalsh(hessian).min() < -1e-12
    ):
        raise ValueError("Finite symmetric positive semidefinite Hessian required")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Positive finite temperature required")
    return base, hessian, center


def exact_target(base, hessian, center, temperature):
    base, hessian, center = quadratic_parameters(base, hessian, center, temperature)
    precision = np.linalg.solve(base, np.eye(len(base))) + hessian / temperature
    cov = np.linalg.solve(precision, np.eye(len(base)))
    mean = np.linalg.solve(precision, hessian @ center / temperature)
    # Evidence from completing the square; useful for independent quadrature checks.
    log_evidence = 0.5 * (
        np.linalg.slogdet(cov)[1]
        - np.linalg.slogdet(base)[1]
        - center @ hessian @ center / temperature
        + mean @ precision @ mean
    )
    return mean, cov, float(log_evidence)


def quadratic_cost(samples, hessian, center):
    delta = np.asarray(samples) - center
    return 0.5 * np.einsum("...i,ij,...j->...", delta, hessian, delta)
