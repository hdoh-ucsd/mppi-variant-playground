"""Self-normalized importance sampling; finite-batch estimates may be biased."""

import numpy as np

from sampling.gaussian import importance_weights, validate_covariance, weighted_update


def gaussian_logpdf(samples, mean, cov):
    samples, mean = np.asarray(samples, dtype=float), np.asarray(mean, dtype=float)
    if mean.ndim != 1 or samples.shape[-1] != len(mean):
        raise ValueError("Gaussian dimensions do not match")
    cov = validate_covariance(cov, dimension=len(mean))
    if not np.isfinite(samples).all() or not np.isfinite(mean).all():
        raise ValueError("Gaussian samples/mean must be finite")
    delta = samples - mean
    return -0.5 * (
        len(mean) * np.log(2 * np.pi)
        + np.linalg.slogdet(cov)[1]
        + np.einsum("...i,ij,...j->...", delta, np.linalg.solve(cov, np.eye(len(mean))), delta)
    )


def estimate(samples, costs, mean, base, proposal, temperature):
    samples, mean = np.asarray(samples), np.asarray(mean)
    ratio = gaussian_logpdf(samples, np.zeros_like(mean), base) - gaussian_logpdf(
        samples, mean, proposal
    )
    weights = importance_weights(costs, temperature, ratio)
    updated = weighted_update(mean[None], (samples - mean)[:, None], weights)[0]
    positive = weights > 0
    return (
        updated,
        weights,
        {
            "ess": float(1 / (weights @ weights)),
            "weight_entropy": float(-weights[positive] @ np.log(weights[positive])),
            "max_weight": float(weights.max()),
            "log_ratio_std": float(ratio.std()),
            "scaled_cost_std": float(np.std(costs) / temperature),
        },
    )


def target_proposal_kl(target_mean, target_cov, proposal_mean, proposal_cov):
    delta = np.asarray(target_mean) - proposal_mean
    return float(
        0.5
        * (
            np.trace(np.linalg.solve(proposal_cov, target_cov))
            + delta @ np.linalg.solve(proposal_cov, delta)
            - len(delta)
            + np.linalg.slogdet(proposal_cov)[1]
            - np.linalg.slogdet(target_cov)[1]
        )
    )


def gaussian_log_weight_second_moment(target_mean, target_cov, mean, cov):
    """log integral target**2 / proposal; None when the Gaussian integral diverges.

    A finite second moment is required for the usual importance-weight ESS limit.
    This is a population overlap diagnostic, not a finite-batch accuracy guarantee.
    """
    d = len(mean)
    target_precision = np.linalg.solve(target_cov, np.eye(d))
    precision = np.linalg.solve(cov, np.eye(d))
    a = 2 * target_precision - precision
    if np.linalg.eigvalsh(a).min() <= 0:
        return None
    b = 2 * target_precision @ target_mean - precision @ mean
    return float(
        0.5 * np.linalg.slogdet(cov)[1]
        - np.linalg.slogdet(target_cov)[1]
        - 0.5 * np.linalg.slogdet(a)[1]
        - target_mean @ target_precision @ target_mean
        + 0.5 * mean @ precision @ mean
        + 0.5 * b @ np.linalg.solve(a, b)
    )
