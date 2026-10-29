"""Plots of saved nonlinear, saturation and closed-loop experiments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.nonconvex_objectives import objective
from analysis.plots import COLORS, ellipse, proposal_color, publish
from utils.config import read_yaml
from visualization.figures import draw_scene


def saturation_figures(root, cases, trials):
    samples = pd.read_csv(root / "samples.csv")
    selected = [
        c
        for c in cases
        if c["proposal_mean"] == [0.0, 0.0]
        and c["scale"] == 2.0
        and c["proposal_name"] in {"isotropic", "diagonal_r4", "full_r4_a45"}
    ]
    if not selected:
        selected = [c for c in cases if c["case_id"] in samples.case_id.unique()][:3]
    fig, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 7), squeeze=False)
    for j, (case, color) in enumerate(zip(selected, COLORS, strict=False)):
        selected_samples = samples[samples.case_id == case["case_id"]]
        axes[0, j].scatter(
            selected_samples.latent_0, selected_samples.latent_1, s=3, alpha=0.2, color=color
        )
        ellipse(axes[0, j], case["proposal_mean"], case["proposal_covariance"], color)
        axes[0, j].set(
            xlabel="Latent acceleration [dimensionless]",
            ylabel="Latent steering [dimensionless]",
            title=case["proposal_name"],
            xlim=(-3, 3),
            ylim=(-3, 3),
        )
        axes[0, j].set_aspect("equal")
        axes[1, j].scatter(
            selected_samples.applied_0, selected_samples.applied_1, s=3, alpha=0.2, color=color
        )
        axes[1, j].set(
            xlabel="Applied acceleration [m/s²]",
            ylabel="Applied steering [rad]",
            xlim=(-2.15, 2.15),
            ylim=(-0.55, 0.55),
        )
    fig.suptitle("7 · Equal latent trace produces different clipped physical distributions")
    publish(fig, root, "07_saturation_distributions", sample_records=samples.to_numpy(dtype=float))
    merged = trials.merge(pd.DataFrame(cases), on="case_id", suffixes=("", "_case"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mean in zip(axes, ([0.0, 0.0], [0.8, -0.8]), strict=True):
        part = merged[merged.proposal_mean.apply(lambda m, selected_mean=mean: m == selected_mean)]
        for (name, group), color in zip(part.groupby("proposal_name"), COLORS, strict=False):
            color = proposal_color(name)
            agg = group.groupby("scale").normalized_applied_covariance_trace.agg(["mean", "std"])
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], color=color, label=name, capsize=2)
        ax.set(
            xlabel="Covariance scale multiplier",
            ylabel="Trace of Cov(applied / D)",
            title=f"Latent mean {mean}",
        )
    axes[0].legend(fontsize=8)
    fig.suptitle("7b · Normalized applied covariance trace; mean ± SD across trials")
    publish(
        fig,
        root,
        "07b_clipped_variance",
        trial_case_ids=trials.case_id.to_numpy(),
        normalized_applied_trace=trials.normalized_applied_covariance_trace.to_numpy(),
    )


def nonconvex_figures(root, config, cases, trials):
    iterations = pd.read_csv(root / "iterations.csv")
    names = list(dict.fromkeys(case["objective"] for case in cases))
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 5 * len(names)), squeeze=False)
    x, y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-2.5, 3, 200))
    z = np.stack([x, y], axis=-1)
    for i, name in enumerate(names):
        selected = [
            c
            for c in cases
            if c["objective"] == name and c["initial_mean"] == config["initial_means"][0]
        ]
        axes[i, 0].contourf(x, y, np.log1p(objective(z, name)), levels=25, cmap="Greys", alpha=0.35)
        for case, color in zip(selected, COLORS, strict=False):
            path = iterations[
                (iterations.case_id == case["case_id"])
                & (iterations.seed == config["evaluation_seeds"][0])
            ]
            axes[i, 0].plot(
                path.mean_0,
                path.mean_1,
                "o-",
                markersize=3,
                color=color,
                label=case["proposal_name"],
            )
            # Marker per iteration, plus labels every fifth update to show direction.
            for row in path.itertuples():
                if row.iteration % 5 == 0:
                    axes[i, 0].annotate(str(row.iteration), (row.mean_0, row.mean_1), fontsize=7)
            part = iterations[iterations.case_id == case["case_id"]]
            stat = part.groupby("iteration").best_cost.agg(["mean", "std"])
            axes[i, 1].plot(stat.index, stat["mean"], color=color, label=case["proposal_name"])
        ref = selected[0]
        axes[i, 0].scatter(
            *ref["global_minimizer"], marker="*", s=140, color="black", label="Cost minimizer"
        )
        axes[i, 0].scatter(
            *ref["target_mean"], marker="x", s=80, color="black", label="Gibbs mean (quadrature)"
        )
        axes[i, 0].set(
            xlabel="z₁ [dimensionless]", ylabel="z₂ [dimensionless]", title=name.replace("_", " ")
        )
        axes[i, 0].legend(fontsize=7)
        axes[i, 1].set(
            xlabel="Optimization iteration",
            ylabel="Mean best visited cost",
            title="All seeds, same initialization",
        )
    fig.suptitle("6 · Adaptive proposal means: a Gibbs mean can differ from the cost minimizer")
    publish(
        fig,
        root,
        "06_nonconvex_iterations",
        grid_x=x,
        grid_y=y,
        cost_grids=np.array([objective(z, name) for name in names]),
        iterations=iterations.drop(columns=[]).to_numpy(dtype=float),
    )
    merged = trials.merge(pd.DataFrame(cases), on="case_id", suffixes=("", "_case"))
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        part = merged[merged.objective == name]
        for (label, group), color in zip(part.groupby("proposal_name"), COLORS, strict=False):
            color = proposal_color(label)
            means = group.groupby(group.initial_mean.apply(str)).basin_success.mean()
            ax.plot(range(len(means)), means, "o-", color=color, label=label)
            ax.set_xticks(range(len(means)), means.index, rotation=15)
        ax.set(
            title=name,
            xlabel="Initial proposal mean",
            ylabel="Best-point basin discovery fraction",
            ylim=(-0.05, 1.05),
        )
    axes[0, 0].legend(fontsize=7)
    publish(
        fig,
        root,
        "06b_initialization",
        final_case_ids=trials.case_id.to_numpy(),
        basin_success=trials.basin_success.to_numpy(),
    )


def closed_loop_figures(root, config, cases, trials):
    cases_frame = pd.DataFrame(cases)
    merged = trials.merge(cases_frame, on="case_id", suffixes=("", "_case"))
    fig, axes = plt.subplots(
        2, len(config["scenarios"]), figsize=(5 * len(config["scenarios"]), 8), squeeze=False
    )
    plotted = {}
    for j, scenario in enumerate(config["scenarios"]):
        selected = [
            c
            for c in cases
            if c["scenario"] == scenario
            and c["scale"] == 1.0
            and c["samples"] == max(config["sample_counts"])
        ]
        if not selected:
            selected = [c for c in cases if c["scenario"] == scenario][:5]
        for case, color in zip(selected, COLORS, strict=False):
            color = proposal_color(case["proposal_name"])
            case_trials = merged[merged.case_id == case["case_id"]]
            for trial in case_trials.itertuples():
                directory = root / trial.episode_path
                states = pd.read_csv(directory / "trajectory.csv")
                primary = trial.seed == config["evaluation_seeds"][0]
                axes[0, j].plot(
                    states.px,
                    states.py,
                    color=color,
                    alpha=0.9 if primary else 0.1,
                    label=case["proposal_name"] if primary else None,
                )
                if primary:
                    plotted[f"case_{case['case_id']}_trajectory"] = states.to_numpy()
            if case == selected[0]:
                episode = read_yaml(root / case_trials.episode_path.iloc[0] / "config.yaml")
                draw_scene(axes[0, j], episode["scenario"])
                # Retained rollouts are unweighted coverage examples, not a posterior cloud.
                archive = root / case_trials.episode_path.iloc[0] / "rollouts.npz"
                if archive.exists():
                    with np.load(archive) as data:
                        index = min(10, len(data["rollouts"]) - 1)
                        subset = data["rollouts"][index, :8, :, :2]
                        for line in subset:
                            axes[0, j].plot(
                                line[:, 0], line[:, 1], color="#64748b", alpha=0.15, lw=0.6
                            )
                        plotted[f"{scenario}_unweighted_rollouts"] = subset
            sample_trial = case_trials.iloc[0]
            diagnostics = pd.read_csv(root / sample_trial.episode_path / "diagnostics.csv")
            axes[1, j].plot(
                diagnostics.time_s, diagnostics.ess, color=color, label=case["proposal_name"]
            )
        axes[0, j].set_title(scenario.replace("_", " "))
        axes[0, j].legend(fontsize=7)
        axes[1, j].set(xlabel="Simulation time [s]", ylabel="ESS [samples]", yscale="log")
    fig.suptitle(
        "8 · Closed-loop validation; bold trajectories and ESS use the first evaluation seed"
    )
    publish(fig, root, "08_closed_loop_trajectories", **plotted)
    fig, axes = plt.subplots(
        1, len(config["scenarios"]), figsize=(5 * len(config["scenarios"]), 4), squeeze=False
    )
    for ax, scenario in zip(axes[0], config["scenarios"], strict=True):
        part = merged[
            (merged.scenario == scenario) & (merged.samples == max(config["sample_counts"]))
        ]
        for (name, group), color in zip(part.groupby("proposal_name"), COLORS, strict=False):
            color = proposal_color(name)
            agg = group.groupby("scale").success.mean()
            ax.plot(agg.index, agg.to_numpy(), "o-", label=name, color=color)
        ax.set(
            xlabel="Covariance scale multiplier",
            ylabel="Goal-pose success fraction",
            title=scenario,
            ylim=(-0.05, 1.05),
        )
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("8b · Separate empirical transfer check; eight seeds per cell")
    publish(
        fig,
        root,
        "08b_closed_loop_scale",
        case_ids=trials.case_id.to_numpy(),
        success=trials.success.to_numpy(),
    )


def extension_figures(root, config, cases, trials, summary, level):
    if level == "saturation":
        saturation_figures(root, cases, trials)
    elif level == "nonconvex":
        nonconvex_figures(root, config, cases, trials)
    elif level == "closed_loop":
        closed_loop_figures(root, config, cases, trials)
    else:
        raise ValueError(f"Unknown study level: {level}")
