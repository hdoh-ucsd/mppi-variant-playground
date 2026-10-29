"""Headless presentation figures reconstructed exclusively from saved runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, Polygon

from utils.config import read_yaml

COLORS = {"standard": "#2563eb", "isotropic": "#db7b14", "diagonal": "#16806a", "full": "#a344ae"}


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "lines.linewidth": 2,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def draw_scene(ax, scenario):
    for x, y, radius in scenario["obstacles"]:
        ax.add_patch(
            Circle((x, y), radius + scenario["vehicle_radius"], color="#d4d9df", alpha=0.5)
        )
        ax.add_patch(Circle((x, y), radius, color="#526071", alpha=0.85))
    start, goal = scenario["initial_state"], scenario["goal_state"]
    ax.scatter(*start[:2], marker="o", s=45, color="#172331", zorder=6)
    ax.scatter(*goal[:2], marker="*", s=160, color="#e9aa26", edgecolor="#172331", zorder=6)
    ax.set(xlabel="Position x [m]", ylabel="Position y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def covariance_figure(records, output):
    by_name = {r["meta"]["controller"]: r for r in records}
    fig, axes = plt.subplots(1, len(by_name), figsize=(3.6 * len(by_name), 3.6), squeeze=False)
    for ax, (name, record) in zip(axes[0], by_name.items(), strict=True):
        cov = np.array(record["config"]["controller"]["covariance"])
        eigen, vectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
        # Density of the configured normalized Gaussian; this is analytic, not fitted data.
        xx, yy = np.meshgrid(np.linspace(-1.4, 1.4, 120), np.linspace(-1.4, 1.4, 120))
        points = np.stack([xx, yy], axis=-1)
        density = np.exp(-0.5 * np.einsum("...i,ij,...j->...", points, np.linalg.inv(cov), points))
        ax.contourf(xx, yy, density, levels=10, cmap="Blues", alpha=0.65)
        ax.add_patch(
            Ellipse(
                (0, 0),
                4 * np.sqrt(eigen[-1]),
                4 * np.sqrt(eigen[0]),
                angle=angle,
                fill=False,
                edgecolor=COLORS[name],
                linewidth=2.5,
            )
        )
        ax.set(
            title=f"{name.capitalize()}\ntrace={np.trace(cov):.2f}, det={np.linalg.det(cov):.4f}",
            xlabel="Normalized acceleration perturbation",
            ylabel="Normalized steering perturbation",
            xlim=(-1.4, 1.4),
            ylim=(-1.4, 1.4),
        )
        ax.set_aspect("equal")
    fig.suptitle("Sampling geometry · matched normalized perturbation energy", y=1.04, fontsize=15)
    fig.tight_layout()
    save(fig, output, "sampling_covariances")


def overview(records, output):
    scenarios = {r["meta"]["scenario"]: r["config"]["scenario"] for r in records}
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 3.7), squeeze=False)
    for ax, (name, scenario) in zip(axes[0], scenarios.items(), strict=True):
        draw_scene(ax, scenario)
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlim(-0.6, 8.6)
        ax.set_ylim(-2.8, 2.8)
        if "disturbance" in scenario:
            ax.text(
                0.03, 0.95, "Execution-only impulse at t = 3.15 s", transform=ax.transAxes, va="top"
            )
    fig.suptitle("Shared toy environments · disk footprint and inflated obstacles", fontsize=15)
    fig.tight_layout()
    save(fig, output, "environment_overview")


def trajectories_and_diagnostics(records, output):
    scenarios = sorted({r["meta"]["scenario"] for r in records})
    for scenario in scenarios:
        selected = [r for r in records if r["meta"]["scenario"] == scenario]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        draw_scene(ax, selected[0]["config"]["scenario"])
        for r in selected:
            name, states = r["meta"]["controller"], r["states"]
            primary = r["meta"]["seed"] == 42
            ax.plot(
                states.px,
                states.py,
                color=COLORS[name],
                alpha=1 if primary else 0.18,
                linewidth=2.2 if primary else 1,
                label=name if primary else None,
            )
            if primary:
                ax.scatter(states.px.iloc[-1], states.py.iloc[-1], color=COLORS[name], s=20)
        ax.set_title(
            f"{scenario.replace('_', ' ').title()} · executed trajectories (bold: seed 42)"
        )
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
        fig.tight_layout()
        save(fig, output, f"{scenario}_trajectories")
        primary = [r for r in selected if r["meta"]["seed"] == 42]
        if not primary:
            primary = selected[:4]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fields = [
            ("minimum_sample_cost", "Minimum sampled objective [cost]"),
            ("ess", "Effective sample size"),
            ("weight_entropy", "Weight entropy [nats]"),
            ("trajectory_spread", "Terminal position spread [m²]"),
            ("planning_latency_s", "Planning latency [ms]"),
            ("saturation_fraction", "Latent saturation fraction"),
        ]
        for r in primary:
            name, diag = r["meta"]["controller"], r["diagnostics"]
            for ax, (field, label) in zip(axes.flat, fields, strict=True):
                scale = 1000 if field == "planning_latency_s" else 1
                ax.plot(diag.time_s, diag[field] * scale, color=COLORS[name], label=name, alpha=0.9)
                ax.set(xlabel="Simulation time [s]", ylabel=label)
                impulse = r["config"]["scenario"].get("disturbance")
                if impulse:
                    ax.axvline(
                        (impulse["step"] + 1) * r["config"]["dynamics"]["dt"],
                        color="grey",
                        ls=":",
                        lw=1,
                    )
        axes[0, 0].legend(frameon=False)
        fig.suptitle(f"{scenario.replace('_', ' ').title()} · seed 42 MPC diagnostics", fontsize=15)
        fig.tight_layout()
        save(fig, output, f"{scenario}_diagnostics")
        fig, axes = plt.subplots(1, len(primary), figsize=(4 * len(primary), 4), squeeze=False)
        for ax, r in zip(axes[0], primary, strict=True):
            draw_scene(ax, r["config"]["scenario"])
            snapshots = r["snapshots"]
            if snapshots is not None:
                idx = min(15, len(snapshots["plans"]) - 1)
                segments = snapshots["rollouts"][idx, :, :, :2]
                ax.add_collection(
                    LineCollection(
                        segments, color=COLORS[r["meta"]["controller"]], alpha=0.15, linewidth=0.7
                    )
                )
                plan = snapshots["plans"][idx]
                ax.plot(plan[:, 0], plan[:, 1], color="#192534", ls="--", label="Updated plan")
                executed = r["states"].iloc[: idx + 1]
                ax.plot(executed.px, executed.py, color=COLORS[r["meta"]["controller"]])
                ax.autoscale_view()
                ax.set_title(f"{r['meta']['controller']} · MPC step {idx}")
            else:
                ax.text(0.05, 0.9, "Rollout archive unavailable", transform=ax.transAxes)
        fig.suptitle(
            f"{scenario.replace('_', ' ').title()} · first 32 unweighted samples", fontsize=15
        )
        fig.tight_layout()
        save(fig, output, f"{scenario}_rollouts")


def summary_figures(root, output):
    path = root / "per_seed.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    scenarios = list(frame.scenario.unique())
    kinds = list(COLORS)
    fig, axes = plt.subplots(3, len(scenarios), figsize=(5 * len(scenarios), 10), squeeze=False)
    for column, scenario in enumerate(scenarios):
        group = frame[frame.scenario == scenario]
        for row, (metric, label) in enumerate(
            [
                ("success", "Success fraction"),
                ("cumulative_evaluation_cost", "Cumulative evaluation cost"),
                ("mean_ess", "Mean effective sample size"),
            ]
        ):
            ax = axes[row, column]
            for i, name in enumerate(kinds):
                values = group[group.controller == name][metric].astype(float).to_numpy()
                if not len(values):
                    continue
                ax.bar(i, values.mean(), width=0.6, color=COLORS[name], alpha=0.7)
                if metric != "success":
                    ax.scatter(
                        i + np.linspace(-0.12, 0.12, len(values)),
                        values,
                        s=20,
                        color="#1c2938",
                        zorder=5,
                    )
                    if len(values) > 1:
                        ax.errorbar(
                            i, values.mean(), yerr=values.std(ddof=1), color="#1c2938", capsize=4
                        )
                else:
                    ax.text(
                        i, values.mean() + 0.035, f"{int(values.sum())}/{len(values)}", ha="center"
                    )
                    ax.set_ylim(0, 1.2)
            ax.set_xticks(range(len(kinds)), kinds, rotation=15)
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(scenario.replace("_", " ").title())
    fig.suptitle(
        f"{frame['mode'].iloc[0]} budget · independent seed values with mean ± SD", fontsize=16
    )
    fig.tight_layout()
    save(fig, output, "benchmark_summary")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    values = [
        frame[frame.controller == kind].mean_planning_latency_s.to_numpy() * 1000 for kind in kinds
    ]
    boxes = ax.boxplot(values, tick_labels=kinds, patch_artist=True, showmeans=True)
    for patch, kind in zip(boxes["boxes"], kinds, strict=True):
        patch.set_facecolor(COLORS[kind])
        patch.set_alpha(0.6)
    ax.set(
        ylabel="Episode mean planning latency [ms]",
        title="CPU planning latency across scenarios and seeds",
    )
    fig.tight_layout()
    save(fig, output, "latency_comparison")


def animate_record(record, output):
    snapshots = record["snapshots"]
    if snapshots is None:
        raise ValueError("Animation requires saved rollouts.npz; rerun this episode to regenerate")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    scenario = record["config"]["scenario"]
    draw_scene(ax, scenario)
    states = record["states"]
    ax.set_xlim(min(-0.6, states.px.min() - 0.6), max(8.6, states.px.max() + 0.6))
    ax.set_ylim(min(-2.6, states.py.min() - 0.6), max(2.6, states.py.max() + 0.6))
    ax.set_aspect("equal", adjustable="box")
    (executed,) = ax.plot([], [], color=COLORS[record["meta"]["controller"]], label="Executed")
    (planned,) = ax.plot([], [], color="#192534", ls="--", label="Plan")
    cloud = LineCollection([], color="#93b9f0", alpha=0.3, linewidth=0.7)
    ax.add_collection(cloud)
    vehicle = Polygon(np.zeros((3, 2)), color=COLORS[record["meta"]["controller"]], zorder=7)
    ax.add_patch(vehicle)
    title = ax.set_title("")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    path = output / (
        f"{record['meta']['scenario']}_{record['meta']['controller']}"
        f"_seed_{record['meta']['seed']}.gif"
    )
    frames = []
    for i in range(len(states)):
        index = min(i, len(snapshots["plans"]) - 1)
        executed.set_data(states.px.iloc[: i + 1], states.py.iloc[: i + 1])
        plan = snapshots["plans"][index]
        planned.set_data(plan[:, 0], plan[:, 1])
        cloud.set_segments(snapshots["rollouts"][index, :, :, :2])
        state = states.iloc[i]
        triangle = np.array([[0.3, 0], [-0.2, 0.16], [-0.2, -0.16]])
        angle = state.theta
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        vehicle.set_xy(triangle @ rotation.T + np.array([state.px, state.py]))
        title.set_text(
            f"{scenario['name'].replace('_', ' ').title()} · "
            f"{record['meta']['controller']} · t={state.time_s:.2f} s"
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    imageio.mimsave(path, frames, duration=record["config"]["dynamics"]["dt"] * 1000, loop=0)
    plt.close(fig)
    return path


def generate(input_path, animate=False):
    style()
    root = Path(input_path)
    paths = sorted(root.rglob("result.json"))
    if not paths:
        raise ValueError(f"No experiment results under {root}")
    records = []
    for path in paths:
        directory = path.parent
        snapshots = directory / "rollouts.npz"
        with np.load(snapshots) if snapshots.exists() else _empty_archive() as data:
            archived = {key: data[key] for key in data} if data is not None else None
        records.append(
            {
                "meta": json.loads(path.read_text()),
                "config": read_yaml(directory / "config.yaml"),
                "states": pd.read_csv(directory / "trajectory.csv"),
                "diagnostics": pd.read_csv(directory / "diagnostics.csv"),
                "snapshots": archived,
            }
        )
    scenario_order = [
        "open_space",
        "narrow_passage",
        "disturbance",
        "multiple_obstacles",
        "model_mismatch",
    ]
    records.sort(
        key=lambda r: (
            scenario_order.index(r["meta"]["scenario"]),
            list(COLORS).index(r["meta"]["controller"]),
            r["meta"]["seed"],
        )
    )
    modes = {r["meta"]["mode"] for r in records}
    if len(modes) > 1:
        raise ValueError("Generate figures for one budget mode at a time")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    overview(records, output)
    covariance_figure(records, output)
    trajectories_and_diagnostics(records, output)
    summary_figures(root, output)
    if animate:
        candidates = [
            r
            for r in records
            if r["meta"]["scenario"] == "disturbance"
            and r["meta"]["controller"] == "standard"
            and r["meta"]["seed"] == 42
        ]
        animate_record(candidates[0] if candidates else records[0], output)
    print(f"Figures saved in {output}")


class _empty_archive:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
