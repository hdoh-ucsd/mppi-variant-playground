"""Parameter sweeps using the unchanged controller, bicycle model and episode runner."""

from itertools import product

import pandas as pd

from analysis.artifacts import begin, fingerprint, finish
from analysis.gaussian_study import proposal, seeds
from evaluation.runner import run_episode
from utils.config import load_experiment


def run_closed_loop(config, output):
    replicate_seeds = seeds(config)
    output, manifest = begin(output, config, "closed_loop")
    cases, rows = [], []
    for scenario, count, scale in product(
        config["scenarios"], config["sample_counts"], config["scales"]
    ):
        props = [proposal("isotropic", 1, 0, config["reference_trace"], scale, "trace")]
        for ratio, angle in product(config["anisotropy_ratios"], config["angles_deg"]):
            if ratio != 1:
                props.append(
                    proposal(
                        f"r{ratio}_a{angle}",
                        ratio,
                        angle,
                        config["reference_trace"],
                        scale,
                        "trace",
                    )
                )
        for prop in props:
            case_id = len(cases)
            case = {
                "case_id": case_id,
                **prop,
                "scenario": scenario,
                "samples": count,
                "comparison_id": fingerprint({"scenario": scenario, "K": count, "scale": scale}),
            }
            cases.append(case)
            for seed in replicate_seeds:
                episode = load_experiment(scenario=scenario)
                episode["controller"].update(
                    kind=prop["family"], covariance=prop["proposal_covariance"], samples=count
                )
                episode["budget"] = {
                    "mode": "covariance_mechanism",
                    "sampling_scale_constraint": "normalized_matched_trace",
                }
                if "max_steps" in config:
                    episode["scenario"]["max_steps"] = config["max_steps"]
                directory = output / "episodes" / f"case_{case_id:03d}" / f"seed_{seed}"
                record = run_episode(episode, seed, directory, mode="covariance_mechanism")
                metrics = record["metrics"]
                diagnostics = pd.read_csv(directory / "diagnostics.csv")
                rows.append(
                    {
                        "case_id": case_id,
                        "comparison_id": case["comparison_id"],
                        "proposal_name": prop["proposal_name"],
                        "seed": seed,
                        **metrics,
                        "ess": metrics["mean_ess"],
                        "saturation_fraction": float(diagnostics.saturation_fraction.mean()),
                        "max_weight": float(diagnostics.max_weight.mean()),
                        "weight_entropy": float(diagnostics.weight_entropy.mean()),
                        "cost_evaluations": metrics["total_rollout_evaluations"],
                        "planning_transition_evaluations": metrics["total_dynamics_evaluations"],
                        "episode_path": str(directory.relative_to(output)),
                    }
                )
            if (case_id + 1) % 10 == 0:
                print(f"Completed {case_id + 1} covariance/scenario/budget cases", flush=True)
    return finish(
        output,
        manifest,
        cases,
        rows,
        "# Closed-loop covariance validation\n\n"
        "These runs reuse the existing MPPI, costs, actuator constraints and prediction/execution "
        "separation. Resolved episode YAML and all trajectories, commands, per-step diagnostics "
        "and rollout archives are saved. Covariance acts in normalized input coordinates; its "
        "angle is not a spatial obstacle orientation. Each episode changes state and warm start, "
        "so this is a separate transfer check, with no analytic reference for its update mean. "
        "The static Gaussian target and vehicle control-sequence target are different problems. "
        "Report all failures, seed variability and timing; do not infer causal mediation from "
        "a correlation between ESS, saturation and success.",
    )
