"""Executed distributions and conditional rollout clouds from saved records only."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Polygon

from analysis.plots import publish as save_figure
from analysis.rollout_snapshots import load_snapshot
from analysis.trajectory_density import load_runs
from evaluation.runner import git_info, write_json
from utils.config import read_yaml
from visualization.figures import COLORS, draw_scene, style


def color(variant):
    return COLORS.get(variant, "#475569")


def publish(fig, root, name, **data):
    # Keep two-line seminar captions above equal-aspect axes and shared colorbars.
    if fig._suptitle is not None:
        title = fig._suptitle
        fig.suptitle(
            title.get_text(), fontsize=title.get_fontsize(), y=1 + 0.6 / fig.get_figheight()
        )
    save_figure(fig, root, name, tight_layout=False, **data)


def scene(ax, setup, bounds=None):
    draw_scene(ax, setup)
    if bounds is not None:
        ax.set(xlim=bounds[0], ylim=bounds[1])
    ax.set_aspect("equal", adjustable="box")


def distribution(root, index, scenario, variant):
    entry = next(x for x in index if x["scenario"] == scenario and x["variant"] == variant)
    with np.load(root / "distributions" / entry["path"], allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    return entry, arrays


def representative_snapshot(records, config, variant):
    record = next(
        r for r in records if r["variant"] == variant and r["seed"] == config["representative_seed"]
    )
    selected = record["snapshot_index"]
    selected = selected[selected.step == config["representative_step"]]
    if selected.empty:
        raise ValueError(
            f"Requested snapshot absent for {variant}; available steps: "
            f"{record['snapshot_index'].step.to_list()}"
        )
    return record, load_snapshot(record["directory"] / selected.iloc[0].path)


def bundle_figure(root, scenario, records, setup, config, index, bounds):
    ratio = (bounds[1][1] - bounds[1][0]) / (bounds[0][1] - bounds[0][0])
    fig, axes = plt.subplots(
        1,
        len(config["variants"]),
        figsize=(4.1 * len(config["variants"]), max(2.6, 1.2 + 3.3 * ratio)),
        squeeze=False,
        layout="constrained",
    )
    data = {}
    for ax, variant in zip(axes[0], config["variants"], strict=True):
        selected = [r for r in records if r["variant"] == variant]
        entry, _ = distribution(root, index, scenario, variant)
        scene(ax, setup, bounds)
        for record in selected:
            path = record["trajectory"]
            primary = record["seed"] == entry["representative_seed"]
            ax.plot(
                path.px,
                path.py,
                color=color(variant),
                alpha=1 if primary else 0.25,
                lw=2.6 if primary else 0.8,
                ls="-" if record["success"] else "--",
                zorder=4 if primary else 2,
                label=f"Median-cost run, seed {record['seed']}" if primary else None,
            )
            ax.scatter(
                path.px.iloc[-1],
                path.py.iloc[-1],
                marker="o" if record["success"] else "x",
                color=color(variant),
                s=10,
                alpha=0.6,
            )
            data[f"{variant}_seed_{record['seed']}_xy"] = path[["px", "py"]].to_numpy()
        successes = sum(r["success"] for r in selected)
        ax.set_title(f"{variant} · goal {successes}/{len(selected)}")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"A · {scenario.replace('_', ' ')} · {len(config['seeds'])} paired executed runs\n"
        "Dashed paths and × endpoints denote failure; bold is an actual median-cost run",
        fontsize=13,
    )
    publish(fig, root, f"{scenario}_A_executed_bundle", **data)


def density_panel(ax, density, setup, norm):
    edges_x, edges_y = density["occupancy_x_edges"], density["occupancy_y_edges"]
    image = ax.pcolormesh(
        edges_x,
        edges_y,
        density["occupancy_density"],
        cmap="magma_r",
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    scene(ax, setup, ((edges_x[0], edges_x[-1]), (edges_y[0], edges_y[-1])))
    return image


def occupancy_figure(root, scenario, records, setup, config, index):
    distributions = [
        distribution(root, index, scenario, variant)[1] for variant in config["variants"]
    ]
    maximum = max(float(x["occupancy_density"].max()) for x in distributions)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=maximum)
    fig, axes = plt.subplots(
        1,
        len(distributions),
        figsize=(4.2 * len(distributions), 4.5),
        squeeze=False,
        layout="constrained",
    )
    data = {}
    for ax, variant, density in zip(axes[0], config["variants"], distributions, strict=True):
        image = density_panel(ax, density, setup, norm)
        ax.set_title(variant)
        data.update(
            {
                f"{variant}_{key}": density[f"occupancy_{key}"]
                for key in ("density", "mass", "x_edges", "y_edges")
            }
        )
    fig.colorbar(
        image,
        ax=list(axes[0]),
        location="bottom",
        fraction=0.06,
        pad=0.22,
        label="Empirical executed-state occupancy [m⁻²] · common square-root color scale",
    )
    fig.suptitle(
        f"B · {scenario.replace('_', ' ')} · {config['occupancy_weighting']} occupancy\n"
        "Uniform time within runs; equal total mass per seed with equal_seed weighting",
        fontsize=13,
    )
    publish(fig, root, f"{scenario}_B_occupancy_density", **data)


def cloud_panel(ax, snapshot, setup, bounds, title):
    states, weights = snapshot["sampled_states"], snapshot["weights"]
    scene(ax, setup, bounds)
    lines = LineCollection(
        states[:, :, :2],
        array=weights,
        norm=Normalize(0, 1),
        cmap="viridis",
        linewidths=0.65,
        alpha=0.42,
    )
    ax.add_collection(lines)
    best = states[snapshot["best_rollout_index"]]
    dominant = states[snapshot["highest_weight_index"]]
    ax.plot(best[:, 0], best[:, 1], color="#111827", ls="--", lw=1.7, label="Lowest raw cost")
    ax.plot(dominant[:, 0], dominant[:, 1], color="#f59e0b", lw=1.5, label="Highest weight")
    if "nominal_states" in snapshot:
        plan = snapshot["nominal_states"]
        ax.plot(plan[:, 0], plan[:, 1], color="#db2777", lw=1.8, label="Updated mean plan")
    ax.scatter(
        *snapshot["current_state"][:2],
        color="#111827",
        marker="s",
        s=35,
        zorder=8,
        label="Current state",
    )
    ax.scatter(
        *snapshot["executed_next_state"][:2],
        color="#ef4444",
        marker="x",
        s=50,
        zorder=9,
        label="Executed next state",
    )
    ess = 1 / (weights @ weights)
    ax.set_title(f"{title} · K={len(weights)}, ESS={ess:.2f}")
    return lines


def local_snapshots(records, config):
    snapshots = {
        variant: representative_snapshot(records, config, variant) for variant in config["variants"]
    }
    points = np.concatenate(
        [snapshot["sampled_states"][:, :, :2].reshape(-1, 2) for _, snapshot in snapshots.values()]
    )
    low, high = points.min(axis=0) - 0.3, points.max(axis=0) + 0.3
    return snapshots, ((low[0], high[0]), (low[1], high[1]))


def cloud_figure(root, scenario, records, setup, config, *, suffix=""):
    snapshots, bounds = local_snapshots(records, config)
    fig, axes = plt.subplots(
        1, len(snapshots), figsize=(4.4 * len(snapshots), 5), squeeze=False, layout="constrained"
    )
    data = {}
    for ax, variant in zip(axes[0], config["variants"], strict=True):
        _, snapshot = snapshots[variant]
        lines = cloud_panel(ax, snapshot, setup, bounds, variant)
        data.update(
            {
                f"{variant}_{key}": snapshot[key]
                for key in (
                    "sampled_states",
                    "costs",
                    "weights",
                    "current_state",
                    "executed_next_state",
                    "chosen_control",
                )
            }
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.colorbar(
        lines,
        ax=list(axes[0]),
        location="bottom",
        fraction=0.04,
        pad=0.25,
        label="Normalized optimizer weight · common [0,1] color scale",
    )
    fig.suptitle(
        f"C · {scenario.replace('_', ' ')} · seed {config['representative_seed']}, "
        f"step {config['representative_step']}\n"
        "All sampled paths; paired time index, different current states and warm starts",
        fontsize=13,
    )
    publish(fig, root, f"{scenario}_C_rollout_cloud{suffix}", **data)


def dispersion_figure(root, scenario, setup, config, index, bounds):
    fig, axes = plt.subplots(
        2,
        len(config["variants"]),
        figsize=(4.2 * len(config["variants"]), 8),
        squeeze=False,
        layout="constrained",
    )
    data = {}
    all_arrays = [distribution(root, index, scenario, variant)[1] for variant in config["variants"]]
    ymin = min(float(a["q05_xy"][:, 1].min()) for a in all_arrays)
    ymax = max(float(a["q95_xy"][:, 1].max()) for a in all_arrays)
    for j, variant in enumerate(config["variants"]):
        _, arrays = distribution(root, index, scenario, variant)
        center, normal = arrays["mean_xy"], arrays["normal_xy"]
        scene(axes[0, j], setup, bounds)
        for lower, upper, alpha in (
            ("offset_q05", "offset_q95", 0.12),
            ("offset_q25", "offset_q75", 0.28),
        ):
            left = center + normal * arrays[lower][:, None]
            right = center + normal * arrays[upper][:, None]
            axes[0, j].add_patch(
                Polygon(np.vstack([left, right[::-1]]), color=color(variant), alpha=alpha, lw=0)
            )
        axes[0, j].plot(center[:, 0], center[:, 1], color=color(variant), label="Mean path")
        axes[0, j].set_title(variant)
        progress = arrays["progress"]
        axes[1, j].fill_between(
            progress,
            arrays["q05_xy"][:, 1],
            arrays["q95_xy"][:, 1],
            color=color(variant),
            alpha=0.12,
            label="5–95%",
        )
        axes[1, j].fill_between(
            progress,
            arrays["q25_xy"][:, 1],
            arrays["q75_xy"][:, 1],
            color=color(variant),
            alpha=0.28,
            label="25–75%",
        )
        axes[1, j].plot(progress, center[:, 1], color=color(variant), label="Mean y")
        axes[1, j].set(
            xlabel="Fraction of each run's traveled arc length",
            ylabel="Lateral position y [m]",
            ylim=(ymin - 0.15, ymax + 0.15),
        )
        data.update(
            {
                f"{variant}_{key}": value
                for key, value in arrays.items()
                if not key.startswith("occupancy_")
            }
        )
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        f"D · {scenario.replace('_', ' ')} · pointwise inter-seed 50% / 90% dispersion\n"
        "Pointwise quantiles; failures included; the mean need not be a feasible path",
        fontsize=13,
    )
    publish(fig, root, f"{scenario}_D_progress_dispersion", **data)


def dashboard_figure(root, scenario, records, setup, config, index):
    variants = config["variants"]
    fig, axes = plt.subplots(
        4,
        len(variants),
        figsize=(4.6 * len(variants), 13),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1, 0.45]},
        layout="constrained",
    )
    densities = [distribution(root, index, scenario, v)[1] for v in variants]
    norm = PowerNorm(0.5, vmin=0, vmax=max(d["occupancy_density"].max() for d in densities))
    snapshots, cloud_bounds = local_snapshots(records, config)
    data = {}
    max_duration = max(r["executed_duration_s"] for r in records)
    for j, variant in enumerate(variants):
        group = [r for r in records if r["variant"] == variant]
        density_image = density_panel(axes[0, j], densities[j], setup, norm)
        axes[0, j].set_title(f"{variant} · empirical occupancy")
        _, snapshot = snapshots[variant]
        cloud_lines = cloud_panel(
            axes[1, j], snapshot, setup, cloud_bounds, "Local rollout population"
        )
        for record in group:
            diag = record["diagnostics"]
            primary = record["seed"] == config["representative_seed"]
            axes[2, j].plot(
                diag.time_s,
                diag.ess,
                color=color(variant),
                alpha=1 if primary else 0.14,
                lw=1.8 if primary else 0.6,
            )
            data[f"{variant}_seed_{record['seed']}_diagnostics"] = diag[
                ["time_s", "ess", "weight_entropy", "max_weight", "saturation_fraction"]
            ].to_numpy()
        impulse = setup.get("disturbance")
        if impulse:
            axes[2, j].axvline(
                (impulse["step"] + 1)
                * read_yaml(group[0]["directory"] / "config.yaml")["dynamics"]["dt"],
                ls=":",
                color="#64748b",
            )
        axes[2, j].set(
            xlabel="Simulation time [s]",
            ylabel="ESS [samples]",
            yscale="log",
            ylim=(0.9, config["samples"] * 1.1),
            xlim=(0, max_duration),
        )
        axes[3, j].axis("off")
        successes, collisions = sum(r["success"] for r in group), sum(r["collision"] for r in group)
        costs = np.array([r["cumulative_evaluation_cost"] for r in group])
        axes[3, j].text(
            0.5,
            0.5,
            f"Goal {successes}/{len(group)} · collision {collisions}/{len(group)}\n"
            f"Evaluation cost {costs.mean():.1f} ± {costs.std(ddof=1):.1f} SD",
            ha="center",
            va="center",
            fontsize=10,
        )
        data[f"{variant}_costs"] = costs
        data[f"{variant}_snapshot_weights"] = snapshot["weights"]
    fig.colorbar(
        density_image,
        ax=list(axes[0]),
        location="bottom",
        fraction=0.04,
        pad=0.1,
        label="Executed occupancy [m⁻²] · common square-root color scale",
    )
    fig.colorbar(
        cloud_lines,
        ax=list(axes[1]),
        location="bottom",
        fraction=0.04,
        pad=0.1,
        label="Local normalized optimizer weight · common [0,1] color scale",
    )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.025)
    )
    fig.suptitle(
        f"E · {scenario.replace('_', ' ')} · behavior dashboard\n"
        f"All-seed occupancy; cloud and bold ESS: seed {config['representative_seed']}, "
        f"step {config['representative_step']}",
        fontsize=14,
    )
    publish(fig, root, f"{scenario}_E_behavior_dashboard", **data)


def generate(input_path):
    root = Path(input_path)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Cannot plot an incomplete behavior study")
    config = read_yaml(root / "config.yaml")
    records = load_runs(root)
    index = json.loads((root / "distributions/index.json").read_text())
    style()
    additional_snapshots = []
    for scenario in config["scenarios"]:
        selected = [r for r in records if r["scenario"] == scenario]
        setup = read_yaml(selected[0]["directory"] / "config.yaml")["scenario"]
        _, first_density = distribution(root, index, scenario, config["variants"][0])
        bounds = (
            (first_density["occupancy_x_edges"][0], first_density["occupancy_x_edges"][-1]),
            (first_density["occupancy_y_edges"][0], first_density["occupancy_y_edges"][-1]),
        )
        bundle_figure(root, scenario, selected, setup, config, index, bounds)
        occupancy_figure(root, scenario, selected, setup, config, index)
        cloud_figure(root, scenario, selected, setup, config)
        dispersion_figure(root, scenario, setup, config, index, bounds)
        dashboard_figure(root, scenario, selected, setup, config, index)
        if "disturbance" in setup:
            post_step = setup["disturbance"]["step"] + 1
            if post_step in config["snapshot_steps"]:
                post_config = config | {"representative_step": post_step}
                cloud_figure(root, scenario, selected, setup, post_config, suffix="_post_impulse")
                additional_snapshots.append(
                    {
                        "scenario": scenario,
                        "seed": config["representative_seed"],
                        "step": post_step,
                        "reason": "first planning state after impulse",
                    }
                )
    write_json(
        root / "figure_provenance.json",
        {
            "git": git_info(),
            "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": ["config.yaml", "cases.json", "trials.csv", "distributions/", "episodes/"],
            "sampling_during_plotting": False,
            "main_snapshot_seed": config["representative_seed"],
            "main_snapshot_step": config["representative_step"],
            "additional_snapshots": additional_snapshots,
            "figures": sorted(path.name for path in (root / "figures").glob("*.svg")),
        },
    )
    print(f"Behavior figures reconstructed from saved data: {root / 'figures'}")
