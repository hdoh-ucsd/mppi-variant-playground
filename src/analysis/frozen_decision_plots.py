"""Saved-data-only figures: same inputs -> proposals -> weights -> means -> predictions."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.stats import t as student_t

from analysis.frozen_decision import PROPOSALS, load_decision
from analysis.plots import publish
from evaluation.runner import git_info, write_json

COLORS = ["#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626", "#475569"]
LABELS = ["Standard", "Isotropic", "Diagonal", "Full", "Aligned", "Misaligned"]


def interval(values):
    values = np.asarray(values)
    return values.mean(), student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(
        len(values)
    )


def decorate(ax, frozen, bounds):
    scenario = frozen["config"]["scenario"]
    for x, y, radius in scenario["obstacles"]:
        ax.add_patch(
            Circle((x, y), radius + scenario["vehicle_radius"], color="#94a3b8", alpha=0.2)
        )
        ax.add_patch(Circle((x, y), radius, color="#475569", alpha=0.6))
    ax.scatter(*frozen["state"][:2], color="black", s=20, marker="o", zorder=5)
    ax.set(xlim=bounds[0], ylim=bounds[1], xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)


def cloud(ax, states, weights=None, norm=None):
    lines = LineCollection(
        states[:, :, :2],
        linewidths=0.5,
        alpha=0.15 if weights is None else 0.6,
        colors="#64748b" if weights is None else None,
        cmap="viridis" if weights is not None else None,
        norm=norm,
    )
    if weights is not None:
        # Paint low weights first; all trajectories remain represented.
        order = np.argsort(weights)
        lines.set_segments(states[order, :, :2])
        lines.set_array(weights[order])
    ax.add_collection(lines)
    return lines


def figure_data(snaps):
    return {
        f"{proposal}_{key}": snap[key]
        for proposal, snap in zip(PROPOSALS, snaps, strict=True)
        for key in (
            "sampled_states",
            "costs",
            "weights",
            "nominal_states",
            "updated_mean",
            "command",
        )
    }


def table_data(frame):
    columns = list(frame.select_dtypes(include=[np.number, bool]).columns)
    return dict(
        table=frame[columns].to_numpy(dtype=float),
        columns=np.array(columns),
        proposal=np.array(frame.proposal),
        state=np.array(frame.state),
    )


def generate(root):
    root = Path(root)
    config = yaml.safe_load((root / "config.yaml").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Cannot render incomplete evidence")
    frame = pd.read_csv(root / "per_seed.csv")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "figure.dpi": 120,
        }
    )
    for state in frame.state.unique():
        frozen = json.loads((root / "frozen_states" / f"{state}.json").read_text())
        primary = frame[(frame.state == state) & (frame.samples == config["primary_samples"])]
        representative = primary[primary.seed == config["representative_seed"]].set_index(
            "proposal"
        )
        snaps = [load_decision(root / representative.loc[p, "snapshot_path"]) for p in PROPOSALS]
        points = np.concatenate([s["sampled_states"][:, :, :2].reshape(-1, 2) for s in snaps])
        low, high = points.min(axis=0) - 0.2, points.max(axis=0) + 0.2
        bounds = ((low[0], high[0]), (low[1], high[1]))
        common = dict(
            bounds=np.array(bounds), representative_seed=np.array(config["representative_seed"])
        )
        ref_status = primary.reference_status.iloc[0]
        title = state.replace("_", " ").title()
        norm = Normalize(0, max(s["weights"].max() for s in snaps))
        fig, axes = plt.subplots(
            3, 6, figsize=(20, 10), layout="constrained", sharex=True, sharey=True
        )
        for column, (proposal, label, snap) in enumerate(
            zip(PROPOSALS, LABELS, snaps, strict=True)
        ):
            for row in range(3):
                decorate(axes[row, column], frozen, bounds)
            cloud(axes[0, column], snap["sampled_states"])
            artist = cloud(axes[1, column], snap["sampled_states"], snap["weights"], norm)
            axes[2, column].plot(*snap["nominal_states"][:, :2].T, color=COLORS[column], lw=2)
            for row in (0, 1):
                ax = axes[row, column]
                ax.plot(
                    *snap["sampled_states"][int(snap["best_rollout_index"]), :, :2].T,
                    color="#e11d48",
                    ls="--",
                    lw=1,
                    label="Min raw cost",
                )
                ax.plot(
                    *snap["sampled_states"][int(snap["highest_weight_index"]), :, :2].T,
                    color="#0f172a",
                    ls=":",
                    lw=1.2,
                    label="Max weight",
                )
            axes[0, column].set_title(label)
            row = representative.loc[proposal]
            axes[2, column].set_title(
                f"J(mean)={row.nominal_objective:.1f}; "
                f"ESS={row.ess:.1f}/{config['primary_samples']}",
                fontsize=8,
            )
        axes[0, 0].set_ylabel("Unweighted rollouts\ny [m]")
        axes[1, 0].set_ylabel("Weighted rollouts\ny [m]")
        axes[2, 0].set_ylabel("Updated mean prediction\ny [m]")
        axes[0, 0].legend(fontsize=7, loc="upper left")
        fig.colorbar(
            artist, ax=axes[1, :].tolist(), label="Normalized optimizer weight", shrink=0.6
        )
        fig.suptitle(
            f"A · {title}: how does proposal covariance change one frozen decision?\n"
            "Same observed state, warm start and target; "
            f"seed {config['representative_seed']}, K={config['primary_samples']}"
        )
        publish(
            fig,
            root,
            f"{state}_A_decision_microscope",
            tight_layout=False,
            **common,
            **figure_data(snaps),
        )
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
        for axis, (field, label, scale) in enumerate(
            (("acceleration", "acceleration", 2.0), ("steering", "steering", 0.5))
        ):
            ax = axes[axis]
            for i, proposal in enumerate(PROPOSALS):
                values = primary[primary.proposal == proposal][
                    f"delta_normalized_{field}"
                ].to_numpy()
                jitter = np.linspace(-0.18, 0.18, len(values))
                ax.scatter(i + jitter, values, color=COLORS[i], s=10, alpha=0.5)
                mean, half = interval(values)
                ax.errorbar(i, mean, yerr=half, color="black", fmt="D", capsize=4, ms=4)
            ax.axhline(0, color="#94a3b8", lw=0.8)
            ax.set(xticks=range(6), xticklabels=LABELS, ylabel=f"Paired Δ normalized {label}")
            ax.tick_params(axis="x", rotation=25)
            sec = ax.secondary_yaxis(
                "right", functions=(lambda x, s=scale: x * s, lambda x, s=scale: x / s)
            )
            sec.set_ylabel("Δ acceleration [m/s²]" if axis == 0 else "Δ steering [rad]")
        fig.suptitle(
            f"B · {title}: how much does the first command change from standard?\n"
            "Dots: paired seeds; diamonds: mean ± pointwise 95% t interval; clipped commands"
        )
        publish(
            fig,
            root,
            f"{state}_B_first_command_difference",
            tight_layout=False,
            **table_data(primary),
        )
        plt.close(fig)

        edges = [np.linspace(*bound, 61) for bound in bounds]
        densities = []
        for snap in snaps:
            xy = snap["sampled_states"][:, -1, :2]
            mass = np.histogram2d(xy[:, 0], xy[:, 1], bins=edges, weights=snap["weights"])[0]
            densities.append(mass / (np.diff(edges[0])[:, None] * np.diff(edges[1])[None, :]))
        fig, axes = plt.subplots(
            2, 3, figsize=(13, 8), layout="constrained", sharex=True, sharey=True
        )
        for ax, label, density in zip(axes.flat, LABELS, densities, strict=True):
            image = ax.pcolormesh(
                *edges, density.T, cmap="magma", vmin=0, vmax=np.max(densities), shading="flat"
            )
            decorate(ax, frozen, bounds)
            ax.set_title(label)
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Weighted endpoint mass / bin area [m⁻²]",
            shrink=0.7,
        )
        fig.suptitle(
            f"C · {title}: where is the weighted endpoint mass?\n"
            "Empirical weighted rollout distribution; common bins and linear scale; no smoothing"
        )
        publish(
            fig,
            root,
            f"{state}_C_weighted_endpoint_density",
            tight_layout=False,
            densities=np.array(densities),
            x_edges=edges[0],
            y_edges=edges[1],
            **common,
        )
        plt.close(fig)

        subset = frame[frame.state == state]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
        for ax, metric, ylabel in zip(
            axes.flat,
            (
                "reference_distance_rms",
                "first_command_reference_distance",
                "ess",
                "nominal_objective",
                "paired_mean_distance_rms",
                "paired_first_command_distance",
            ),
            (
                "RMS latent mean distance to reference",
                "Normalized first-command distance to reference",
                "ESS [samples]",
                "Objective J(updated mean)",
                "RMS latent mean distance to paired standard",
                "Normalized first-command distance to paired standard",
            ),
            strict=True,
        ):
            for i, proposal in enumerate(PROPOSALS):
                x, y, lo, hi = [], [], [], []
                for count, group in subset[subset.proposal == proposal].groupby("samples"):
                    values = group[metric].to_numpy()
                    m, half = interval(values)
                    x.append(count)
                    y.append(m)
                    lo.append(m - half)
                    hi.append(m + half)
                ax.plot(x, y, color=COLORS[i], marker=".", label=LABELS[i])
                ax.fill_between(x, lo, hi, color=COLORS[i], alpha=0.1)
            ax.set(xscale="log", xlabel="Rollout budget K", ylabel=ylabel)
            ax.grid(alpha=0.15)
        axes[0, 0].legend(fontsize=8, ncols=2)
        fig.suptitle(
            f"D · {title}: do decision differences diminish as K grows?\n"
            f"Reference: {ref_status}; unresolved distances do not establish estimation error. "
            "Mean ± 95% t interval.",
            fontsize=11,
        )
        publish(fig, root, f"{state}_D_budget_estimation", tight_layout=False, **table_data(subset))
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, metric, ylabel in zip(
            axes.flat,
            (
                "ess",
                "update_rms",
                "delta_normalized_steering",
                "nominal_objective",
                "planning_latency_s",
                "dynamics_evaluations",
            ),
            (
                "ESS",
                "Latent update RMS",
                "Paired Δ normalized steering",
                "Objective J(updated mean)",
                "Planning latency [s]",
                "Scalar dynamics transitions",
            ),
            strict=True,
        ):
            for i, proposal in enumerate(PROPOSALS):
                values = primary[primary.proposal == proposal][metric].to_numpy()
                m, half = interval(values)
                ax.scatter(np.full(len(values), i), values, color=COLORS[i], alpha=0.3, s=8)
                ax.errorbar(i, m, yerr=half, color=COLORS[i], fmt="D", capsize=3)
            ax.set(xticks=range(6), xticklabels=LABELS, ylabel=ylabel)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(
            f"E · {title}: which mechanisms accompany the changed decision?\n"
            "Same frozen inputs and K; dots are paired seeds; diamonds are means ± 95% t interval"
        )
        publish(
            fig, root, f"{state}_E_decision_dashboard", tight_layout=False, **table_data(primary)
        )
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
        nominal_points = np.concatenate([s["nominal_states"][:, :2] for s in snaps])
        nl, nh = nominal_points.min(axis=0) - 0.2, nominal_points.max(axis=0) + 0.2
        nominal_bounds = ((nl[0], nh[0]), (nl[1], nh[1]))
        for ax in axes[:2]:
            decorate(ax, frozen, nominal_bounds)
        obstacles = frozen["config"]["scenario"]["obstacles"]
        if obstacles:
            x, y, radius = obstacles[0]
            r = radius + frozen["config"]["scenario"]["vehicle_radius"] + 0.7
            zoom_bounds = ((x - r, x + r), (y - r, y + r))
        else:
            zoom_bounds = (
                (nominal_points[-1, 0] - 1, nominal_points[-1, 0] + 1),
                (nominal_points[-1, 1] - 1, nominal_points[-1, 1] + 1),
            )
        axes[1].set(xlim=zoom_bounds[0], ylim=zoom_bounds[1])
        clearances = []
        for i, (label, snap) in enumerate(zip(LABELS, snaps, strict=True)):
            for ax in axes[:2]:
                ax.plot(*snap["nominal_states"][:, :2].T, color=COLORS[i], lw=1.8, label=label)
            plan = snap["nominal_states"]
            if obstacles:
                from scenarios.environment import swept_clearance

                gaps = swept_clearance(
                    plan[:-1], plan[1:], obstacles, frozen["config"]["scenario"]["vehicle_radius"]
                )
                axes[2].plot(
                    np.arange(len(gaps)) * frozen["config"]["dynamics"]["dt"],
                    gaps,
                    color=COLORS[i],
                    label=label,
                )
                clearances.append(gaps)
            else:
                axes[2].plot(
                    np.arange(len(plan)) * frozen["config"]["dynamics"]["dt"],
                    plan[:, 1],
                    color=COLORS[i],
                    label=label,
                )
        axes[0].legend(fontsize=8)
        axes[0].set_title("Updated nominal predictions")
        axes[1].set_title("Declared common zoom")
        axes[2].set(
            xlabel="Prediction time [s]",
            ylabel="Swept obstacle clearance [m]"
            if obstacles
            else "Predicted lateral position [m]",
        )
        if obstacles:
            axes[2].axhline(0, color="black", ls="--", lw=0.8)
        fig.suptitle(
            f"F · {title}: what would each updated plan predict from this state?\n"
            "Single paired seed; open-loop prediction without new disturbances; "
            "no closed-loop guarantee"
        )
        publish(
            fig,
            root,
            f"{state}_F_counterfactual_consequence",
            tight_layout=False,
            **figure_data(snaps),
            zoom_bounds=np.array(zoom_bounds),
            clearance=np.array(clearances),
            **common,
        )
        plt.close(fig)
        print(f"Rendered {state}: figures A–F from saved evidence", flush=True)
    files = [
        root / "per_seed.csv",
        root / "config.yaml",
        *sorted((root / "snapshots").rglob("*.npz")),
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    write_json(
        root / "figure_provenance.json",
        dict(
            git=git_info(),
            inputs_sha256=digest.hexdigest(),
            plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            figures=sorted(p.name for p in (root / "figures").glob("*.png")),
            regeneration="No optimization, environment execution or random draws",
            weighted_density="Empirical weighted endpoints; not physical uncertainty density",
        ),
    )
