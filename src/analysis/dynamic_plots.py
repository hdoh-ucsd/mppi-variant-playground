"""Saved-data-only figures for the seven-state campaign; never executes a controller."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from evaluation.runner import write_json

PROPOSALS = ["standard", "isotropic", "diagonal", "full", "aligned", "misaligned"]
COLORS = dict(
    zip(PROPOSALS, ["#2563eb", "#d97706", "#059669", "#9333ea", "#e11d48", "#64748b"], strict=True)
)
ANALYTIC = ["isotropic", "diagonal_r4", "diagonal_r4_a90", "full_r4_a45", "full_r4_a-45"]
LABELS = ["isotropic", "diagonal 0°", "diagonal 90°", "full +45°", "full −45°"]


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def publish(fig, root, name, sources, *, key=None, support=False):
    root = Path(root)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    hashes = {str(Path(p)): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    target = None
    if key:
        target = Path(f"results/showcase/{key}_key.png")
    elif support:
        target = Path("docs/figures") / f"seven_state_{name}.png"
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((out / f"{name}.png").read_bytes())
    record = dict(
        figure=str(target or out / f"{name}.png"),
        inputs_sha256=hashes,
        reconstruction="Saved compact records only; no simulation or sampling",
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figure_sha256=hashlib.sha256((out / f"{name}.png").read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((root / "summary.csv").read_bytes()).hexdigest(),
    )
    write_json(root / ("figure_provenance.json" if key else f"{name}_provenance.json"), record)


def analytic(root, question):
    root = Path(root)
    d = pd.read_csv(root / "summary.csv").merge(
        pd.DataFrame(read(root, "cases.json")), on="case_id"
    )
    fig, axes = plt.subplots(
        1, 2 if question == "Q01" else 3, figsize=(12, 3.7), layout="constrained"
    )
    colors = ["#d97706", "#059669", "#64748b", "#9333ea", "#e11d48"]
    if question == "Q01":
        g = d[(d.suite == "geometry") & (d.constraint == "trace") & (d.scale == 1) & (d.ratio == 4)]
        g = g[g.proposal_mean.map(lambda value: value == [0.0, 0.0])].sort_values("angle_deg")
        if g.angle_deg.duplicated().any():
            raise ValueError("Orientation figure mixes distinct conditions")
        axes[0].semilogy(
            g.angle_deg, g.squared_error_mean, "o-", color="#9333ea", label="anisotropy 4:1"
        )
        iso = d[(d.suite == "budget") & (d.proposal_name == "isotropic") & (d.samples == 256)]
        axes[0].axhline(iso.squared_error_mean.iloc[0], ls="--", color="#d97706", label="isotropic")
        axes[0].set(
            xlabel="Covariance angle [degrees]",
            ylabel="Mean squared mean-estimation error",
            title="Orientation helps or hurts · K=256",
        )
        axes[0].legend(fontsize=8)
        for name, label, color in zip(ANALYTIC, LABELS, colors, strict=True):
            g = d[(d.suite == "budget") & (d.proposal_name == name)].sort_values("samples")
            axes[1].loglog(g.samples, g.squared_error_mean, "o-", label=label, color=color)
        axes[1].set(
            xlabel="Samples K",
            ylabel="Mean squared error",
            title="Fixed analytic target · 64 independent seeds",
        )
        axes[1].legend(fontsize=8)
    else:
        for name, label, color in zip(ANALYTIC, LABELS, colors, strict=True):
            g = d[(d.suite == "temperature") & (d.proposal_name == name)].sort_values("temperature")
            axes[0].semilogx(g.temperature, g.ess_mean, "o-", label=label, color=color)
            axes[1].loglog(g.temperature, g.error_mean, "o-", color=color)
        axes[0].set(
            xlabel="Temperature λ", ylabel="Mean ESS [of 512]", title="Weight concentration"
        )
        axes[1].set(
            xlabel="Temperature λ",
            ylabel="Mean error to each exact target",
            title="Temperature changes the target",
        )
        axes[0].legend(fontsize=7)
        for cause, color in zip(["cost_only", "ratio_only", "combined"], colors, strict=False):
            g = d[(d.suite == "dimension") & (d.temperature == 1) & (d.cause == cause)].sort_values(
                "dimension"
            )
            axes[2].semilogy(
                g.dimension, g.ess_mean, "o-", label=cause.replace("_", " "), color=color
            )
        axes[2].set(
            xlabel="Sampling dimension", ylabel="Mean ESS [of 512]", title="Separate causes · λ=1"
        )
        axes[2].legend(fontsize=8)
    fig.suptitle(
        f"{question} · Analytic Gaussian replication · seeds 34000–34063 (vehicle independent)"
    )
    publish(
        fig, root, f"{question}_analytic", [root / "summary.csv", root / "cases.json"], key=question
    )


def geometry(ax, data):
    for x, y, radius in data.get("obstacles", []):
        ax.add_patch(Circle((x, y), radius, facecolor="#cbd5e1", edgecolor="#64748b"))
    for name in ["upper", "lower"]:
        p = np.asarray(data.get(name, []))
        if p.size:
            ax.plot(*p.T, color="#475569", lw=1)
    p = np.asarray(data.get("centerline", []))
    if p.size:
        ax.plot(*p.T, "--", color="#94a3b8", lw=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="x [m]", ylabel="y [m]")


def frozen(root):
    root = Path(root)
    inputs = read(root, "figure_inputs.json")
    state = read(root, "frozen_states/pre_obstacle.json")
    # Geometry is the exact saved scenario config, not reconstructed dynamics.
    geo = state["config"]["scenario"]
    fig, axes = plt.subplots(3, 6, figsize=(17, 8), sharex=True, sharey=True, layout="constrained")
    data = inputs["pre_obstacle"]
    bounds = np.array([data[p]["population_bounds"] for p in PROPOSALS])
    # Limits include every saved population path, not a favorable cropped subset.
    xmin, ymin = bounds[:, 0, :].min(axis=0) - 0.4
    xmax, ymax = bounds[:, 1, :].max(axis=0) + 0.4
    xmin = min(xmin, state["state"][0] - 0.4)
    for j, p in enumerate(PROPOSALS):
        item = data[p]
        paths = np.asarray(item["paths_xy"])
        weights = np.asarray(item["weights"])
        indices = item["indices"]
        for i in range(3):
            ax = axes[i, j]
            geometry(ax, geo)
            ax.scatter(*state["state"][:2], color="black", marker="x", s=20)
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", adjustable="box")
            if i == 0:
                for index in item["unweighted_indices"]:
                    xy = paths[indices.index(index)]
                    ax.plot(*xy.T, color=COLORS[p], alpha=0.18, lw=0.6)
                ax.set_title(p)
            elif i == 1:
                for index, xy in zip(indices, paths, strict=True):
                    ax.plot(
                        *xy.T,
                        color=COLORS[p],
                        alpha=0.10 + 0.8 * weights[index] / weights.max(),
                        lw=0.4 + 2 * weights[index] / weights.max(),
                    )
                endpoints = np.asarray(item["endpoint_xy"])
                ax.scatter(
                    *endpoints.T, s=2 + 75 * weights / weights.max(), color=COLORS[p], alpha=0.4
                )
                for field, marker in [("best_raw_index", "v"), ("max_weight_index", "*")]:
                    ax.scatter(
                        *endpoints[item[field]],
                        marker=marker,
                        s=70,
                        facecolor="none",
                        edgecolor="black",
                    )
                ax.text(
                    0.02,
                    0.98,
                    f"ESS={1 / np.sum(weights**2):.1f}\n"
                    f"min J #{item['best_raw_index']}\nmax w #{item['max_weight_index']}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                )
            else:
                xy = np.asarray(item["nominal_states"])[:, :2]
                ax.plot(*xy.T, color=COLORS[p], lw=2)
                ax.text(
                    0.02,
                    0.98,
                    f"δ={item['command'][0]:+.3f} rad\nτ={item['command'][1]:+.3f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
            if j:
                ax.set_ylabel("")
            if i < 2:
                ax.set_xlabel("")
    for i, label in enumerate(
        [
            "Unweighted\n48 paths",
            "Corrected weights\n≤64 paths + all endpoints",
            "Updated nominal\n7-state prediction",
        ]
    ):
        axes[i, 0].set_ylabel(label + "\ny [m]")
    fig.suptitle(
        "Q03 · One frozen obstacle decision → samples → corrected weights → updated "
        "plan\nPaired seed 33000 · K=384 · T=20 · same axes · "
        "▼ lowest raw cost; ☆ highest corrected weight"
    )
    publish(
        fig,
        root,
        "Q03_microscope",
        [root / "figure_inputs.json", root / "frozen_states/pre_obstacle.json"],
        key="Q03",
    )
    d = pd.read_csv(root / "per_seed.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), layout="constrained")
    for j, name in enumerate(inputs):
        for p in PROPOSALS:
            g = d[(d.state == name) & (d.proposal == p)].groupby("samples")
            x = np.array(sorted(g.groups))
            for i, field in enumerate(["reference_distance_rms", "paired_mean_distance_rms"]):
                mean = g[field].mean()
                se = g[field].std() / np.sqrt(50)
                axes[i, j].plot(x, mean.loc[x], label=p, color=COLORS[p])
                axes[i, j].fill_between(
                    x,
                    (mean - 1.96 * se).loc[x],
                    (mean + 1.96 * se).loc[x],
                    color=COLORS[p],
                    alpha=0.08,
                )
        status = d[d.state == name].reference_status.iloc[0]
        axes[0, j].set_title(f"{name}\nreference: {status}", fontsize=9)
        for i in range(2):
            axes[i, j].set(xscale="log", xlabel="K", ylabel="Normalized plan RMS distance")
    axes[0, 0].set_ylabel("Distance to independent reference\n(error only when reference usable)")
    axes[1, 0].set_ylabel("Paired distance to standard update")
    axes[1, 2].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Q03 · 50 paired seeds · bands: pointwise 95% mean intervals\nUnresolved references do "
        "not establish target-mean accuracy"
    )
    publish(fig, root, "Q03_convergence", [root / "per_seed.csv"], support=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for j, name in enumerate(inputs):
        g = d[(d.state == name) & (d.samples == 384)]
        for i, field in enumerate(["first_steering", "first_throttle"]):
            arrays = [g[g.proposal == p][field].values for p in PROPOSALS]
            boxes = axes[i, j].boxplot(
                arrays, patch_artist=True, tick_labels=PROPOSALS, showfliers=True
            )
            for box, p in zip(boxes["boxes"], PROPOSALS, strict=True):
                box.set_facecolor(COLORS[p])
                box.set_alpha(0.5)
            axes[i, j].tick_params(axis="x", labelrotation=55, labelsize=7)
            axes[i, j].set_ylabel("Steering [rad]" if i == 0 else "Normalized throttle")
            if i == 0:
                axes[i, j].set_title(name)
    fig.suptitle("Q03 · First executed command distributions · K=384 · 50 paired decisions")
    publish(fig, root, "Q03_commands", [root / "per_seed.csv"], support=True)


def intervals(ax, d, metric, xfield, ylabel):
    for p in PROPOSALS[:4]:
        g = d[d.proposal == p].sort_values(xfield)
        x = g[xfield].to_numpy()
        y = g[f"{metric}_mean"].to_numpy()
        ax.plot(x, y, "o-", label=p, color=COLORS[p], ms=4)
        ax.fill_between(
            x, g[f"{metric}_ci95_low"], g[f"{metric}_ci95_high"], color=COLORS[p], alpha=0.10
        )
    ax.set(xlabel=xfield.replace("_", " "), ylabel=ylabel)


def driving(root, question):
    root = Path(root)
    d = pd.read_csv(root / "summary.csv")
    geos = read(root, "geometry.json")
    paths = read(root, "representative_paths.json")
    if question == "Q04":
        scenes = list(geos)
        fig, axes = plt.subplots(3, 5, figsize=(16, 8), layout="constrained")
        for j, scene in enumerate(scenes):
            axes[0, j].set_title(scene.replace("_", " "))
            for i, cohort in enumerate(["compatibility", "holdout"]):
                g = d[(d.scenario == scene) & (d.cohort == cohort)]
                for k, p in enumerate(PROPOSALS[:4]):
                    row = g[g.proposal == p].iloc[0]
                    mean = row.success_mean
                    axes[i, j].errorbar(
                        k,
                        mean,
                        yerr=[[mean - row.success_ci95_low], [row.success_ci95_high - mean]],
                        fmt="o",
                        color=COLORS[p],
                        capsize=3,
                    )
                    axes[i, j].text(
                        k,
                        mean + 0.04,
                        f"{int(row.completions)}/{int(row.n)}",
                        ha="center",
                        fontsize=7,
                    )
                axes[i, j].set(ylim=(-0.02, 1.2), xticks=range(4), xticklabels=PROPOSALS[:4])
                axes[i, j].tick_params(axis="x", rotation=45, labelsize=7)
                axes[i, j].set_ylabel(cohort + "\nsuccess / Wilson 95%")
            ax = axes[2, j]
            geometry(ax, geos[scene])
            for p in PROPOSALS[:4]:
                row = paths[f"{scene}__{p}__holdout"]
                ax.plot(row["px"], row["py"], color=COLORS[p], label=p)
            ax.set_title("Predeclared holdout seed 31000", fontsize=8)
        fig.suptitle(
            "Q04 · Seven-state closed-loop outcomes · K=384 · T=20 · all failures "
            "retained\nCompatibility and independent holdout seeds shown separately; "
            "representative paths are not selected by outcome"
        )
    else:
        fig = plt.figure(figsize=(13, 7), layout="constrained")
        gs = fig.add_gridspec(2, 3)
        ax = fig.add_subplot(gs[0, :2])
        intervals(ax, d, "success", "clearance", "Success rate / Wilson 95%")
        ax.set(
            xlabel="Corridor clearance [m]",
            ylim=(-0.02, 1.02),
            title=f"Q05 · Seven-state S-turn · {int(d.n.iloc[0])} paired holdout seeds / setting",
        )
        ax.legend(ncol=4, fontsize=8)
        ax = fig.add_subplot(gs[0, 2])
        geometry(ax, geos["clearance_0.35"])
        for p in PROPOSALS[:4]:
            row = paths[f"clearance_0.35__{p}__holdout"]
            ax.plot(row["px"], row["py"], color=COLORS[p])
        ax.set_title("Predeclared seed 32000 · clearance .35 m", fontsize=9)
        ax.set(aspect="equal", adjustable="box")
        trace_end = max(paths[f"clearance_0.35__{p}__holdout"]["time_s"][-1] for p in PROPOSALS[:4])
        for j, (field, label) in enumerate(
            [
                ("vy", "Lateral velocity [m/s]"),
                ("yaw_rate", "Yaw rate [rad/s]"),
                ("roll", "Roll [rad]"),
            ]
        ):
            ax = fig.add_subplot(gs[1, j])
            for p in PROPOSALS[:4]:
                row = paths[f"clearance_0.35__{p}__holdout"]
                ax.plot(row["time_s"], row[field], color=COLORS[p])
            ax.set(
                xlabel="Time [s]",
                ylabel=label,
                title="Same representative executions",
                xlim=(0, trace_end),
            )
        fig.suptitle(
            "K=384 · T=20 · feasible reference at every clearance · intervals include timeouts "
            "and collisions"
        )
    publish(
        fig,
        root,
        f"{question}_outcomes",
        [root / "summary.csv", root / "geometry.json", root / "representative_paths.json"],
        key=question,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    if question == "Q04":
        g = d[d.cohort == "holdout"].copy()
        g["scene_index"] = g.scenario.map({name: i for i, name in enumerate(geos)})
        xfield = "scene_index"
    else:
        g = d
        xfield = "clearance"
    for ax, metric, label in zip(
        axes,
        ["mean_ess", "max_abs_vy", "max_abs_yaw_rate"],
        ["Episode mean ESS", "Peak |vy| [m/s]", "Peak |yaw rate| [rad/s]"],
        strict=True,
    ):
        intervals(ax, g, metric, xfield, label)
        if question == "Q04":
            ax.set(
                xticks=range(5),
                xticklabels=[n.replace("_", "\n") for n in geos],
                xlabel="Holdout scenario",
            )
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"{question} · Seven-state diagnostics · means with pointwise 95% intervals")
    publish(fig, root, f"{question}_diagnostics", [root / "summary.csv"], support=True)


def budget(root):
    root = Path(root)
    d = pd.read_csv(root / "summary.csv")
    metrics = [
        "success",
        "cumulative_evaluation_cost",
        "mean_ess",
        "mean_planning_latency_s",
        "model_transitions",
    ]
    # Metric names are recorded in summaries; counters are exact, not wall-clock matching.
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), layout="constrained")
    for ax, metric, label in zip(
        axes,
        metrics,
        [
            "Success / Wilson 95%",
            "Evaluation cost",
            "Mean ESS",
            "Campaign planning time [s]",
            "Predicted transitions / episode",
        ],
        strict=True,
    ):
        intervals(ax, d, metric, "samples", label)
        ax.set(xscale="log", xlabel="K")
    axes[-1].legend(fontsize=7)
    fig.suptitle(
        f"{read(root, 'manifest.json')['run_id']} · Equal K, not equal wall-clock budget · "
        "K=384 reuses primary records\n"
        "Campaign timings span different CPU load; isolated timing probes are reported separately"
    )
    publish(
        fig,
        root,
        "budget_" + read(root, "manifest.json")["run_id"],
        [root / "summary.csv"],
        support=True,
    )


def secondary(root):
    root = Path(root)
    d = pd.read_csv(root / "summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for i, prefix in enumerate(["speed_", "impulse_"]):
        g = d[d["case"].str.startswith(prefix)].copy()
        g["level"] = g["case"].str.removeprefix(prefix).astype(float)
        if i == 1:
            baseline = d[d["case"] == "speed_1.4"].copy()
            baseline["level"] = 0.0
            g = pd.concat([baseline, g], ignore_index=True)
        for j, (metric, label) in enumerate(
            [
                ("success", "Success / Wilson 95%"),
                ("cumulative_evaluation_cost", "Mean evaluation cost"),
                ("mean_ess", "Episode mean ESS"),
            ]
        ):
            intervals(axes[i, j], g, metric, "level", label)
            axes[i, j].set_xlabel(
                "Preferred cruising speed [m/s]" if i == 0 else "Lateral impulse magnitude [m]"
            )
        axes[i, 0].set_ylim(-0.02, 1.02)
    axes[0, 2].legend(fontsize=7)
    fig.suptitle(
        "Q05 · Separate secondary checks · new paired seeds 35000–35019\n"
        "Fixed geometry, K=384 and terminal speed 1.4 m/s; baseline records reused in plot"
    )
    publish(fig, root, "Q05_secondary", [root / "summary.csv"], support=True)


def rate_study(root):
    root = Path(root)
    d = pd.read_csv(root / "summary.csv")
    d["rate_limit"] = d["case"].str.removeprefix("rate_").astype(float)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for ax, metric, label in zip(
        axes.flat,
        [
            "success",
            "cumulative_evaluation_cost",
            "mean_ess",
            "max_abs_vy",
            "max_abs_yaw_rate",
            "max_abs_roll",
        ],
        [
            "Success / Wilson 95%",
            "Evaluation cost",
            "Episode mean ESS",
            "Peak |vy| [m/s]",
            "Peak |yaw rate| [rad/s]",
            "Peak |roll| [rad]",
        ],
        strict=True,
    ):
        intervals(ax, d, metric, "rate_limit", label)
        ax.set_xlabel("Discrete steering command slew limit [rad/s]")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 2].legend(fontsize=7)
    fig.suptitle(
        "Q05 · Command slew constraints using previous-command memory\n"
        "Seven plant states · K=384 · clearance .35 m · paired seeds 37000–37019"
    )
    publish(fig, root, "Q05_rate", [root / "summary.csv"], support=True)


def generate(root, stage):
    if not read(root, "manifest.json").get("complete"):
        raise ValueError("Cannot plot an incomplete run")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    if stage in ["Q01", "Q02"]:
        analytic(root, stage)
    elif stage == "Q03":
        frozen(root)
    elif stage in ["Q04", "Q05"]:
        driving(root, stage)
    elif stage == "rate":
        rate_study(root)
    elif stage == "secondary":
        secondary(root)
    else:
        budget(root)
