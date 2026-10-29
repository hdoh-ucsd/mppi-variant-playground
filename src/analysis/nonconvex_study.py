"""Adaptive proposal means with a fixed Gibbs target; best sample != target mean."""

import numpy as np
import pandas as pd

from analysis.artifacts import begin, fingerprint, finish
from analysis.gaussian_study import named_proposals, seeds
from analysis.importance_sampling import estimate
from analysis.nonconvex_objectives import grid_reference, minimizer, objective


def run_nonconvex(config, output):
    replicate_seeds = seeds(config)
    if config["samples"] < 2 or config["iterations"] < 1:
        raise ValueError("Positive sample/iteration budgets required")
    output, manifest = begin(output, config, "nonconvex")
    cases, rows, history = [], [], []
    base = np.eye(2) * config["base_variance"]
    for name in config["objectives"]:
        reference = grid_reference(
            name,
            config["temperature"],
            config["base_variance"],
            resolution=config["reference_resolution"],
        )
        coarse = grid_reference(
            name,
            config["temperature"],
            config["base_variance"],
            resolution=(config["reference_resolution"] + 1) // 2,
        )
        reference["mean_refinement_difference"] = float(
            np.linalg.norm(np.array(reference["mean"]) - coarse["mean"])
        )
        for start in config["initial_means"]:
            comparison = fingerprint({"objective": name, "start": start})
            for prop in named_proposals(config["reference_trace"]):
                case_id = len(cases)
                cov = np.array(prop["proposal_covariance"])
                cases.append(
                    {
                        "case_id": case_id,
                        **prop,
                        "objective": name,
                        "initial_mean": start,
                        "target_mean": reference["mean"],
                        "quadrature_reference": reference,
                        "global_minimizer": minimizer(name).tolist(),
                        "comparison_id": comparison,
                    }
                )
                for seed in replicate_seeds:
                    rng = np.random.default_rng(seed)
                    mean = np.array(start, dtype=float)
                    best, best_point = float(objective(mean, name)), mean.copy()
                    history.append(
                        {
                            "case_id": case_id,
                            "seed": seed,
                            "iteration": 0,
                            "mean_0": mean[0],
                            "mean_1": mean[1],
                            "mean_cost": best,
                            "best_cost": best,
                            "cost_evaluations": 1,
                        }
                    )
                    for iteration in range(1, config["iterations"] + 1):
                        normal = rng.normal(size=(config["samples"], 2))
                        samples = mean + normal @ np.linalg.cholesky(cov).T
                        costs = objective(samples, name)
                        mean, _, diagnostics = estimate(
                            samples, costs, mean, base, cov, config["temperature"]
                        )
                        mean_cost = float(objective(mean, name))
                        k = int(costs.argmin())
                        if costs[k] < best:
                            best, best_point = float(costs[k]), samples[k].copy()
                        if mean_cost < best:
                            best, best_point = mean_cost, mean.copy()
                        history.append(
                            {
                                "case_id": case_id,
                                "seed": seed,
                                "iteration": iteration,
                                "mean_0": mean[0],
                                "mean_1": mean[1],
                                "mean_cost": mean_cost,
                                "best_cost": best,
                                **diagnostics,
                                "cost_evaluations": 1 + iteration * (config["samples"] + 1),
                            }
                        )
                    error = float(np.linalg.norm(mean - reference["mean"]))
                    rows.append(
                        {
                            "case_id": case_id,
                            "comparison_id": comparison,
                            "proposal_name": prop["proposal_name"],
                            "seed": seed,
                            "error": error,
                            "squared_error": error**2,
                            "estimate_0": mean[0],
                            "estimate_1": mean[1],
                            "mean_cost": mean_cost,
                            "best_cost": best,
                            "best_point_0": best_point[0],
                            "best_point_1": best_point[1],
                            "basin_success": bool(
                                np.linalg.norm(best_point - minimizer(name))
                                <= config["basin_radius"]
                            ),
                            "mean_in_target_basin": bool(
                                np.linalg.norm(mean - minimizer(name)) <= config["basin_radius"]
                            ),
                            "cost_evaluations": 1 + config["iterations"] * (config["samples"] + 1),
                            **diagnostics,
                        }
                    )
    pd.DataFrame(history).to_csv(output / "iterations.csv", index=False)
    return finish(
        output,
        manifest,
        cases,
        rows,
        "# Nonconvex adaptive-mean study\n\n"
        "Every iteration preserves the base/cost/temperature target and refits the proposal mean. "
        "Both objectives have a known zero-cost global minimizer. Their target means are separate "
        "quadrature references, with grid-refinement discrepancy and prior tail bounds saved. "
        "The target mean need not minimize the objective, especially between modes. Best visited "
        "cost, mean cost and mean-estimation error are separate. Visiting the declared "
        "basin counts as discovery, not convergence of the mean. No monotonic or "
        "global-convergence guarantee is assumed for finite-batch adaptive updates.",
    )
