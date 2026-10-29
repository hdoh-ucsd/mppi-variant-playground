"""Gaussian densities are evaluated on latent, unsaturated normalized controls."""

import numpy as np


def validate_covariance(covariance, dimension=2):
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (dimension, dimension) or not np.isfinite(cov).all():
        raise ValueError("Covariance must be a finite square matrix of the control dimension")
    if not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Covariance must be symmetric")
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance must be positive definite") from exc
    return cov.copy()


def sample_gaussian(rng, shape, covariance):
    cov = validate_covariance(covariance)
    return rng.standard_normal((*shape, len(cov))) @ np.linalg.cholesky(cov).T


def log_base_over_proposal(samples, mean, base_covariance, proposal_covariance):
    """log N(V;0,C0)/N(V;U,C), summed over independent horizon steps.

    Supports unequal SPD covariances. Never substitute clipped perturbations.
    """
    base = validate_covariance(base_covariance)
    proposal = validate_covariance(proposal_covariance)
    eps = samples - mean
    base_q = np.einsum("kti,ij,ktj->k", samples, np.linalg.inv(base), samples)
    prop_q = np.einsum("kti,ij,ktj->k", eps, np.linalg.inv(proposal), eps)
    horizon = samples.shape[1]
    return -0.5 * (
        base_q - prop_q + horizon * (np.linalg.slogdet(base)[1] - np.linalg.slogdet(proposal)[1])
    )


def importance_weights(costs, temperature, log_ratio=None):
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or not len(costs):
        raise ValueError("Costs must be a nonempty vector")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive")
    if not np.isfinite(costs).all():
        raise FloatingPointError("Nonfinite rollout costs; refusing a silent fallback")
    ratio = np.zeros_like(costs) if log_ratio is None else np.asarray(log_ratio)
    if ratio.shape != costs.shape or not np.isfinite(ratio).all():
        raise FloatingPointError("Nonfinite or invalid likelihood ratios")
    with np.errstate(over="ignore"):
        logits = -(costs - costs.min()) / temperature + ratio
        logits -= logits.max()
    weights = np.exp(logits)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Importance weights cannot be normalized")
    weights /= total
    if not np.isclose(weights.sum(), 1.0):
        raise FloatingPointError("Weights do not sum to one")
    return weights


def weighted_update(mean, perturbations, weights):
    return mean + np.einsum("k,kti->ti", weights, perturbations)
