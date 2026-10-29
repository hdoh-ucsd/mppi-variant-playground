"""Published-formulation orchestration over the existing seven-state runner and artifacts."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from analysis.artifacts import begin, fingerprint
from analysis.dynamic_campaign import (
    assert_validated,
    configuration,
    resolved_case,
    run_episode,
    summary,
)
from analysis.dynamic_plots import publish
from evaluation.runner import make_controller, write_json
from utils.config import read_yaml


def case_config(options, scenario, algorithm, samples):
    config, geometry, *_ = resolved_case(
        configuration(), scenario, samples, options["s_turn_clearance"]
    )
    config["controller"].update(
        algorithm=algorithm,
        kind="standard",
        covariance=config["proposals"]["standard"],
        saved_rollouts=0,
    )
    config["controller"].update(options["algorithms"][algorithm])
    config["planning_deadline_s"] = options["planning_deadline_s"]
    config["execution"]["extra_snapshot_seeds"] = [options["seeds"][0]]
    fraction = options.get("stochastic_input_fraction", 0.0)
    if fraction:
        config["execution_control_noise_covariance"] = (
            fraction * np.array(config["controller"]["base_covariance"])
        ).tolist()
    return config, geometry


def equal_work_samples(options, algorithm):
    # All work is counted again by the model-call instrument during execution.
    if algorithm == "cc_mppi":
        return options["transition_budget"] // 20 - 21
    if algorithm == "robust_mppi":
        probes = options["algorithms"][algorithm].get("candidate_samples", 8)
        overhead = 20 * (5 * (probes + 1) + 18 + 4)
        return (options["transition_budget"] - overhead) // 40
    if algorithm == "tube_mppi":
        return (options["transition_budget"] - 18 * 20) // (2 * 20) - 1
    iterations = options["algorithms"][algorithm].get("iterations", 1)
    return options["transition_budget"] // (20 * iterations) - 1


def calibrate(options, study, root):
    records, settings = [], {}
    for scenario in study["scenarios"]:
        for algorithm in study["algorithms"]:
            for samples in options["calibration_samples"]:
                c, _ = case_config(options, scenario, algorithm, samples)
                controller = make_controller(c)
                state = np.asarray(c["scenario"]["initial_state"])
                controller.reset(state, 47899)
                controller.command(state)  # Warmup excluded; not an evaluation seed.
                for seed in options["calibration_seeds"]:
                    controller.reset(state, seed)
                    count = controller.model.transitions
                    start = perf_counter()
                    controller.command(state)
                    records.append(
                        dict(
                            scenario=scenario,
                            algorithm=algorithm,
                            samples=samples,
                            seed=seed,
                            seconds=perf_counter() - start,
                            transitions=controller.model.transitions - count,
                        )
                    )
                recent = [r["seconds"] for r in records[-len(options["calibration_seeds"]) :]]
                if (
                    options.get("calibration_stop_on_first_overrun", False)
                    and np.quantile(recent, 0.9)
                    > options["calibration_fraction"] * options["planning_deadline_s"]
                ):
                    break
            selected = pd.DataFrame(records)
            selected = selected[(selected.scenario == scenario) & (selected.algorithm == algorithm)]
            times = selected.groupby("samples").seconds.quantile(0.9)
            valid = times[times <= options["calibration_fraction"] * options["planning_deadline_s"]]
            settings[f"{scenario}/{algorithm}"] = (
                int(valid.index.max()) if len(valid) else int(times.index.min())
            )
            print(
                f"calibration {scenario}/{algorithm}: K={settings[f'{scenario}/{algorithm}']}",
                flush=True,
            )
    pd.DataFrame(records).to_csv(root / "calibration.csv", index=False)
    write_json(root / "wall_clock_settings.json", settings)
    return settings


def execute_job(args):
    options, scenario, algorithm, seed, protocol, samples, raw = args
    config, _ = case_config(options, scenario, algorithm, samples)
    path = Path(raw) / protocol / scenario / algorithm / f"seed_{seed}"
    result, trajectory, diag = run_episode(
        config, algorithm, seed, path, capture_steps=[0, 20, 21, 40]
    )
    # Keep one external metric schema and retain optional formulation-specific columns.
    row = {k: v for k, v in result.items() if not isinstance(v, (dict, list))}
    row.update(
        algorithm=algorithm,
        proposal="standard" if algorithm == "it_mppi" else algorithm,
        scenario=scenario,
        protocol=protocol,
        samples=samples,
        seed=seed,
        shared_condition_sha256=fingerprint({k: v for k, v in config.items() if k != "controller"}),
        raw_episode=str(path),
        deadline_violation_fraction=float(diag.deadline_violation.mean()),
        max_plan_transitions=int(diag.dynamics_evaluations.max()),
        transition_budget_violation=bool(
            (diag.dynamics_evaluations > options["transition_budget"]).any()
        ),
    )
    for key in [
        "tracking_error",
        "tracking_control_norm",
        "reset_event",
        "support_fraction",
        "covariance_violation",
        "covariance_prediction_error",
        "solver_seconds",
        "solver_infeasible",
        "sampled_feedback_rms",
        "nominal_ess",
        "nominal_selection_infeasible",
        "covariance_constraint_residual",
        "empirical_covariance_residual",
    ]:
        if key in diag:
            row[f"mean_{key}"] = float(diag[key].mean())
    record = trajectory.to_dict("list") if seed == options["seeds"][0] else None
    return row, record


def plot(root):
    import matplotlib.pyplot as plt

    root = Path(root)
    data = pd.read_csv(root / "summary.csv")
    trials = pd.read_csv(root / "per_seed.csv")
    algorithms = list(data.proposal.unique())
    scenarios = list(data.scenario.unique())
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
    labels = {
        "standard": "IT-MPPI",
        "tsallis_vi": "Tsallis-VI",
        "dial_mpc": "DIAL-MPC",
        "dial_no_anneal": "DIAL fixed noise",
        "tube_mppi": "Tube-MPPI",
        "robust_mppi": "Robust-MPPI",
        "cc_mppi": "CC-MPPI",
    }
    colors = {
        "standard": "#2563eb",
        "tsallis_vi": "#d97706",
        "dial_mpc": "#059669",
        "dial_no_anneal": "#64748b",
        "tube_mppi": "#9333ea",
        "robust_mppi": "#e11d48",
        "cc_mppi": "#0891b2",
    }
    for i, protocol in enumerate(["dynamics_budget", "wall_clock"]):
        sub = data[data.protocol == protocol]
        for j, algorithm in enumerate(algorithms):
            rows = sub[sub.proposal == algorithm].set_index("scenario").loc[scenarios]
            x = np.arange(len(scenarios)) + (j - (len(algorithms) - 1) / 2) * 0.09
            mean = rows.success_mean.to_numpy()
            err = np.maximum(
                0, np.stack([mean - rows.success_ci95_low, rows.success_ci95_high - mean])
            )
            axes[0, i].errorbar(
                x,
                mean,
                yerr=err,
                fmt="o",
                capsize=3,
                label=labels[algorithm],
                color=colors[algorithm],
            )
            axes[1, i].plot(
                x,
                rows.mean_planning_latency_s_mean * 1000,
                "o-",
                label=labels[algorithm],
                color=colors[algorithm],
            )
            axes[2, i].plot(
                x,
                rows.cumulative_evaluation_cost_mean,
                "o-",
                label=labels[algorithm],
                color=colors[algorithm],
            )
            for position, scenario in zip(x, scenarios, strict=True):
                values = trials[
                    (trials.protocol == protocol)
                    & (trials.proposal == algorithm)
                    & (trials.scenario == scenario)
                ].cumulative_evaluation_cost
                offsets = np.linspace(-0.015, 0.015, len(values))
                axes[2, i].scatter(
                    position + offsets, values, s=18, color=colors[algorithm], alpha=0.55
                )
        axes[2, i].set(ylabel="External task cost: seeds + mean", yscale="log")
        axes[0, i].set(
            title=(
                "Work ceiling (timing under concurrent load)"
                if i == 0
                else "Calibrated deadline (serial timing)"
            ),
            ylabel="Completion / Wilson 95% interval",
            ylim=(-0.05, 1.05),
        )
        axes[1, i].axhline(150, color="black", linestyle="--", label="150 ms deadline")
        axes[1, i].set(ylabel="Mean command time [ms]")
        for ax in axes[:, i]:
            ax.set_xticks(range(len(scenarios)), scenarios, rotation=20)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        f"{root.name}: shared seven-state vehicle; disk collision proxy\n"
        "Small paired cohort; calibration and holdout seeds are disjoint"
    )
    publish(fig, root, "primary", [root / "summary.csv", root / "per_seed.csv"])
    (root / "primary.png").write_bytes((root / "figures/primary.png").read_bytes())
    import hashlib
    import json

    provenance_path = root / "primary_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["figure"] = str(root / "primary.png")
    provenance["variant_plot_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    write_json(provenance_path, provenance)


def run(study_id, output, raw, *, options_path="configs/variants/benchmark.yaml"):
    options = read_yaml(options_path)
    study = options["studies"][study_id]
    assert_validated()
    options["stochastic_input_fraction"] = study.get("stochastic_input_fraction", 0.0)
    config = deepcopy(options) | {"study": study_id, "shared_campaign": configuration()}
    root, manifest = begin(output, config, "published_formulation_comparison")
    settings = calibrate(options, study, root)
    rows, paths = [], {}
    for protocol in ["dynamics_budget", "wall_clock"]:
        jobs = [
            (
                options,
                scenario,
                algorithm,
                seed,
                protocol,
                equal_work_samples(options, algorithm)
                if protocol == "dynamics_budget"
                else settings[f"{scenario}/{algorithm}"],
                str(raw),
            )
            for scenario in study["scenarios"]
            for algorithm in study["algorithms"]
            for seed in options["seeds"]
        ]
        # Timing protocol is serial, without concurrent benchmark workers.
        if protocol == "dynamics_budget":
            with ProcessPoolExecutor(max_workers=options["workers"]) as pool:
                results = list(pool.map(execute_job, jobs))
        else:
            results = []
            for job in jobs:
                results.append(execute_job(job))
                print(f"{study_id} wall clock {len(results)}/{len(jobs)}", flush=True)
        for row, path in results:
            rows.append(row)
            if path is not None:
                paths[f"{protocol}/{row['scenario']}/{row['algorithm']}"] = path
        pd.DataFrame(rows).to_csv(root / "per_seed.csv", index=False)
        print(f"{study_id}: {len(rows)} episodes", flush=True)
    frame = pd.DataFrame(rows)
    summary(frame, root, ["protocol", "scenario"])
    write_json(root / "representative_paths.json", paths)
    plot(root)
    manifest.update(
        complete=True,
        episodes=len(frame),
        finished_at_utc=datetime.now(UTC).isoformat(),
        raw_storage=str(raw),
        noise=(
            "Paired SeedSequence([4402,seed]) normalized actuator noise; covariance "
            f"{options['stochastic_input_fraction']} times common base C0. "
            "Existing impulses retained."
        ),
        timing=(
            "Wall-clock protocol serial; work protocol concurrent. "
            "Concurrent timing is not used for cross-method speed claims."
        ),
        budget="7700 predicted transitions per command; every actual model call is instrumented.",
    )
    write_json(root / "manifest.json", manifest)
    return frame
