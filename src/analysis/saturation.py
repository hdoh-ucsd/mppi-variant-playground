"""Moments and atoms induced by physical actuator clipping; no clipped likelihoods."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.artifacts import begin, finish
from analysis.gaussian_study import named_proposals, seeds


def clipped_marginal_moments(mean, variance, lower=-1.0, upper=1.0):
    mean, variance = np.asarray(mean), np.asarray(variance)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Finite means and positive finite variances required")
    if not lower < upper:
        raise ValueError("Clipping bounds must be ordered")
    sd = np.sqrt(variance)
    a, b = (lower - mean) / sd, (upper - mean) / sd
    left, right = norm.cdf(a), norm.sf(b)
    middle = norm.cdf(b) - norm.cdf(a)
    pa, pb = norm.pdf(a), norm.pdf(b)
    first = lower * left + upper * right + mean * middle + sd * (pa - pb)
    second = (
        lower**2 * left
        + upper**2 * right
        + (mean**2 + variance) * middle
        + 2 * mean * sd * (pa - pb)
        + variance * (a * pa - b * pb)
    )
    return first, second - first**2, left, right


def saturation_statistics(latent, physical_scale):
    latent, scale = np.asarray(latent), np.asarray(physical_scale)
    if latent.ndim != 2 or latent.shape[1] != 2 or len(latent) < 2:
        raise ValueError("At least two two-input latent samples required")
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Positive finite physical control scales required")
    applied = np.clip(latent, -1, 1) * scale
    clipped = applied / scale
    latent_cov = np.cov(latent, rowvar=False)
    applied_cov = np.cov(applied, rowvar=False)
    result = {
        "saturation_fraction": float(np.mean(np.abs(latent) >= 1)),
        "any_input_saturated_fraction": float(np.mean(np.any(np.abs(latent) >= 1, axis=1))),
        "latent_covariance_trace": float(np.trace(latent_cov)),
        "normalized_applied_covariance_trace": float(np.trace(np.cov(clipped, rowvar=False))),
        "normalized_applied_second_moment": float(np.mean(np.sum(clipped**2, axis=1))),
        "applied_cross_covariance": float(applied_cov[0, 1]),
    }
    for i in range(2):
        result.update(
            {
                f"latent_mean_{i}": float(latent[:, i].mean()),
                f"latent_variance_{i}": float(latent_cov[i, i]),
                f"applied_mean_{i}": float(applied[:, i].mean()),
                f"applied_variance_{i}": float(applied_cov[i, i]),
                f"lower_boundary_mass_{i}": float(np.mean(latent[:, i] <= -1)),
                f"upper_boundary_mass_{i}": float(np.mean(latent[:, i] >= 1)),
            }
        )
    return applied, result


def run_saturation(config, output):
    replicate_seeds = seeds(config)
    output, manifest = begin(output, config, "saturation")
    cases, rows, examples = [], [], []
    scale = np.array(config["physical_scale"])
    for mean in config["proposal_means"]:
        for proposal_scale in config["scales"]:
            for prop in named_proposals(config["reference_trace"], proposal_scale):
                case_id = len(cases)
                cov = np.array(prop["proposal_covariance"])
                first, var, left, right = clipped_marginal_moments(mean, np.diag(cov))
                cases.append(
                    {
                        "case_id": case_id,
                        **prop,
                        "proposal_mean": mean,
                        "exact_applied_mean": (first * scale).tolist(),
                        "exact_applied_variance": (var * scale**2).tolist(),
                        "exact_lower_boundary_mass": left.tolist(),
                        "exact_upper_boundary_mass": right.tolist(),
                    }
                )
                for seed in replicate_seeds:
                    normal = np.random.default_rng(seed).normal(size=(config["samples"], 2))
                    latent = np.array(mean) + normal @ np.linalg.cholesky(cov).T
                    applied, metrics = saturation_statistics(latent, scale)
                    rows.append({"case_id": case_id, "seed": seed, **metrics})
                    if seed == replicate_seeds[0] and proposal_scale == config["plot_scale"]:
                        examples.extend(
                            {
                                "case_id": case_id,
                                "sample": i,
                                "latent_0": float(z[0]),
                                "latent_1": float(z[1]),
                                "applied_0": float(u[0]),
                                "applied_1": float(u[1]),
                            }
                            for i, (z, u) in enumerate(
                                zip(latent[:2048], applied[:2048], strict=True)
                            )
                        )
    pd.DataFrame(examples).to_csv(output / "samples.csv", index=False)
    return finish(
        output,
        manifest,
        cases,
        rows,
        "# Saturation study\n\n"
        "Latent samples are mapped through D and actuator clipping. Analytic univariate clipped "
        "moments and boundary masses provide independent marginal references. Cross-covariance "
        "is empirical. No Gaussian likelihood is evaluated on the applied controls. Equal latent "
        "trace need not imply equal applied covariance trace or physical marginal variance. "
        "See results/archive/legacy_index.md for actual numerical comparisons.",
    )
