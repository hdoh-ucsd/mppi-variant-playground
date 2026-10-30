"""Frozen-proposal Gaussian experiments with paired independent replicate seeds."""

from itertools import product

import numpy as np
import pandas as pd

from analysis.analytic_gaussian import exact_target, quadratic_cost
from analysis.artifacts import begin, fingerprint, finish
from analysis.covariance_geometry import alignment, covariance, rotation
from analysis.importance_sampling import (
    estimate,
    gaussian_log_weight_second_moment,
    target_proposal_kl,
)


def seeds(config):
    evaluation = config["evaluation_seeds"]
    tuning = config["tuning_seeds"]
    if config["phase"] not in {"evaluation", "exploratory"}:
        raise ValueError("Declare evaluation or exploratory phase")
    if len(evaluation) < 2 or len(set(evaluation)) != len(evaluation):
        raise ValueError("At least two distinct replicate seeds required")
    if set(evaluation) & set(tuning) or any(s < 0 for s in evaluation):
        raise ValueError("Evaluation seeds must be nonnegative and disjoint from tuning")
    return evaluation


def named_proposals(reference_trace, scale=1, constraint="trace"):
    return [
        proposal(name, ratio, angle, reference_trace, scale, constraint)
        for name, ratio, angle in (
            ("isotropic", 1, 0),
            ("diagonal_r4", 4, 0),
            ("full_r4_a45", 4, 45),
            ("diagonal_r4_a90", 4, 90),
            ("full_r4_a-45", 4, -45),
        )
    ]


def proposal(name, ratio, angle, trace, scale, constraint):
    return {
        "proposal_name": name,
        "family": "isotropic" if ratio == 1 else ("diagonal" if angle % 90 == 0 else "full"),
        "ratio": ratio,
        "angle_deg": angle,
        "scale": scale,
        "constraint": constraint,
        "proposal_covariance": covariance(ratio, angle, scale, trace, constraint).tolist(),
    }


def cases_for(config, suites):
    cases = []
    base = np.asarray(config["base_covariance"])
    center = np.asarray(config["cost_center"])
    trace = config["reference_trace"]

    def add(suite, props, angle, mean, count, temperature=None):
        r = rotation(angle)
        h = (r * config["hessian_eigenvalues"]) @ r.T
        temp = config["temperature"] if temperature is None else temperature
        target_mean, target_cov, evidence = exact_target(base, h, center, temp)
        signature = {
            "base": base.tolist(),
            "hessian": h.tolist(),
            "center": center.tolist(),
            "lambda": temp,
        }
        for prop in props:
            cov = np.asarray(prop["proposal_covariance"])
            comparison = {
                "suite": suite,
                "target": signature,
                "mean": mean,
                "K": count,
                "scale": prop["scale"],
                "constraint": prop["constraint"],
            }
            cases.append(
                {
                    "case_id": len(cases),
                    "suite": suite,
                    **prop,
                    "samples": count,
                    "proposal_mean": mean,
                    "temperature": temp,
                    "cost_angle_deg": angle,
                    "base_covariance": base.tolist(),
                    "hessian": h.tolist(),
                    "cost_center": center.tolist(),
                    "target_mean": target_mean.tolist(),
                    "target_covariance": target_cov.tolist(),
                    "log_evidence": evidence,
                    "target_sha256": fingerprint(signature),
                    "comparison_id": fingerprint(comparison),
                    **alignment(cov, target_cov),
                    "target_proposal_kl": target_proposal_kl(target_mean, target_cov, mean, cov),
                }
            )

    if "budget" in suites:
        for count in config["sample_counts"]:
            add("budget", named_proposals(trace), config["cost_angle_deg"], [0.0, 0.0], count)
    if "geometry" in suites:
        for constraint, scale, mean in product(
            config["constraints"], config["scales"], config["proposal_means"]
        ):
            props = [proposal("isotropic", 1, 0, trace, scale, constraint)]
            for ratio, angle in product(config["anisotropy_ratios"], config["angles_deg"]):
                if ratio != 1:
                    props.append(
                        proposal(f"r{ratio}_a{angle}", ratio, angle, trace, scale, constraint)
                    )
            add("geometry", props, config["cost_angle_deg"], mean, config["geometry_samples"])
    if "cost_rotation" in suites:
        for angle in config["cost_angles_deg"]:
            add(
                "cost_rotation",
                named_proposals(trace),
                angle,
                [0.0, 0.0],
                config["geometry_samples"],
            )
    if "temperature" in suites:
        for temp in config["temperatures"]:
            add(
                "temperature",
                named_proposals(trace),
                config["cost_angle_deg"],
                [0.0, 0.0],
                config["temperature_samples"],
                temp,
            )
    if "dimension" in suites:
        for d, cause, temp in product(
            config["dimensions"],
            ("cost_only", "ratio_only", "combined"),
            config["diagnostic_temperatures"],
        ):
            h = np.zeros((d, d)) if cause == "ratio_only" else np.eye(d) * 2
            p = np.eye(d)
            q = p if cause == "cost_only" else p * 0.6
            b = np.full(d, 0.2)
            mu, cov, evidence = exact_target(p, h, b, temp)
            cases.append(
                {
                    "case_id": len(cases),
                    "suite": "dimension",
                    "cause": cause,
                    "dimension": d,
                    "proposal_name": cause,
                    "proposal_mean": np.zeros(d).tolist(),
                    "proposal_covariance": q.tolist(),
                    "base_covariance": p.tolist(),
                    "hessian": h.tolist(),
                    "cost_center": b.tolist(),
                    "temperature": temp,
                    "samples": config["temperature_samples"],
                    "target_mean": mu.tolist(),
                    "target_covariance": cov.tolist(),
                    "log_evidence": evidence,
                    "target_sha256": fingerprint(
                        {"base": p.tolist(), "H": h.tolist(), "b": b.tolist(), "lambda": temp}
                    ),
                    "comparison_id": f"dimension-{d}-{cause}-{temp}",
                    "target_proposal_kl": target_proposal_kl(mu, cov, np.zeros(d), q),
                }
            )
    for case in cases:
        log_second = gaussian_log_weight_second_moment(
            np.array(case["target_mean"]),
            np.array(case["target_covariance"]),
            np.array(case["proposal_mean"]),
            np.array(case["proposal_covariance"]),
        )
        case["log_weight_second_moment"] = log_second
        case["weight_second_moment_finite"] = log_second is not None
        case["asymptotic_ess_fraction"] = (
            float(np.exp(-log_second)) if log_second is not None else 0.0
        )
    return cases


def run_gaussian(config, output, suites=("budget", "geometry", "cost_rotation")):
    replicate_seeds = seeds(config)
    cases = cases_for(config, suites)
    if not cases or any(not isinstance(c["samples"], int) or c["samples"] < 2 for c in cases):
        raise ValueError("Nonempty study and integer sample counts >= 2 required")
    output, manifest = begin(output, config | {"suites": list(suites)}, "analytic_gaussian")
    rows = []
    representative_count = min(config["sample_counts"], key=lambda k: abs(k - 512))
    representative_ids = {
        c["case_id"]
        for c in cases
        if c["suite"] == "budget" and c["samples"] == representative_count
    } or {c["case_id"] for c in cases[:3] if len(c["target_mean"]) == 2}
    saved_samples = []
    for case in cases:
        cov = np.array(case["proposal_covariance"])
        mean = np.array(case["proposal_mean"])
        target = np.array(case["target_mean"])
        factor = np.linalg.cholesky(cov)
        h, b, p = [np.array(case[key]) for key in ("hessian", "cost_center", "base_covariance")]
        for seed in replicate_seeds:
            # Reset per case: paired standard normals; larger K shares the smaller-K prefix.
            normal = np.random.default_rng(seed).standard_normal((case["samples"], len(mean)))
            samples = mean + normal @ factor.T
            costs = quadratic_cost(samples, h, b)
            mu, weights, diagnostics = estimate(samples, costs, mean, p, cov, case["temperature"])
            error = float(np.linalg.norm(mu - target))
            rows.append(
                {
                    "case_id": case["case_id"],
                    "comparison_id": case["comparison_id"],
                    "proposal_name": case["proposal_name"],
                    "seed": seed,
                    "error": error,
                    "squared_error": error**2,
                    "cost_evaluations": len(samples),
                    **diagnostics,
                    **{f"estimate_{i}": float(value) for i, value in enumerate(mu)},
                }
            )
            if case["case_id"] in representative_ids and seed == replicate_seeds[0]:
                saved_samples.extend(
                    {
                        "case_id": case["case_id"],
                        "seed": seed,
                        "sample": i,
                        "z0": float(z[0]),
                        "z1": float(z[1]),
                        "cost": float(costs[i]),
                        "weight": float(weights[i]),
                    }
                    for i, z in enumerate(samples)
                )
    pd.DataFrame(saved_samples).to_csv(output / "samples.csv", index=False)
    frame = finish(
        output,
        manifest,
        cases,
        rows,
        "# Analytic Gaussian study\n\n"
        "Each case has a fixed proposal mean. Covariance comparisons preserve the base, cost "
        "and temperature, recorded by target_sha256. Temperature and cost-rotation cases "
        "have their own exact targets. All error measures use the corresponding exact mean. "
        "Replicates are independent across seeds and paired across cases.\n\n"
        "summary.csv reports empirical bias, sample variance trace, MSE and 5/50/95% quantiles. "
        "These are descriptive estimates, not claims of an unbiased finite-K estimator or "
        "a statistically significant ordering. Read the determinant-constrained cases separately "
        "from trace-constrained cases. The dimension diagnostic separates zero-cost density "
        "ratios, cost-only concentration (proposal equals base), and their combination.\n\n"
        "See results/historical/archive/legacy_index.md for the executed numerical interpretation.",
    )
    return frame
