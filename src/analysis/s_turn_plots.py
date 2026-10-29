"""One question-driven composite and an interpretation reconstructed from saved records."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis.artifacts import fingerprint
from evaluation.runner import git_info, write_json
from visualization.figures import save

COLORS = ["#2563eb", "#d97706", "#059669", "#9333ea"]


def generate(root, showcase=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Cannot plot incomplete evidence")
    config = yaml.safe_load((root / "config.yaml").read_text())
    frame = pd.read_csv(root / "per_seed.csv")
    summary = pd.read_csv(root / "summary.csv")
    conditions = json.loads((root / "conditions.json").read_text())
    paths = json.loads((root / "representative_paths.json").read_text())
    primary_samples = config["controller"]["samples"]
    rate = config["vehicle"]["steering_rate_limit"]
    primary = summary[(summary.samples == primary_samples) & (summary.steering_rate_limit == rate)]
    selected = next(
        c
        for c in conditions
        if c["suite"] == "primary"
        and c["config"]["corridor"]["clearance"] == config["execution"]["representative_clearance"]
    )
    geometry = json.loads((root / "geometry.json").read_text())[selected["cell"]]
    upper, lower = np.array(geometry["upper"]), np.array(geometry["lower"])
    corridor = SimpleNamespace(
        upper=upper,
        lower=lower,
        boundary=SimpleNamespace(points=np.vstack([upper, lower[::-1], upper[:1]])),
        center=SimpleNamespace(points=np.array(geometry["centerline"])),
        goal=np.array(geometry["goal"]),
    )
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(14, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]
    data = dict(
        primary_summary=primary.astype(object).where(pd.notna(primary), None).to_dict("records"),
        representative_seed=config["execution"]["representative_seed"],
        representative_clearance=config["execution"]["representative_clearance"],
        paths={},
        upper_boundary=corridor.upper.tolist(),
        lower_boundary=corridor.lower.tolist(),
    )
    ax = axes[0]
    for proposal, color in zip(config["proposals"], COLORS, strict=True):
        g = primary[primary.proposal == proposal].sort_values("clearance")
        x, y = g.clearance.to_numpy(), g.success_mean.to_numpy()
        ax.plot(x, y, color=color, marker="o", label=proposal)
        ax.fill_between(x, g.success_ci95_low, g.success_ci95_high, color=color, alpha=0.10)
    ax.set(
        xlabel="Nominal corridor clearance [m] → easier",
        ylabel="Finish-gate completion fraction",
        ylim=(-0.04, 1.04),
        xticks=sorted(primary.clearance.unique()),
    )
    ax.grid(alpha=0.15)
    ax.legend(fontsize=9, ncols=2)
    ax.set_title("Does reducing clearance separate the proposals?")
    for ax in axes[1:]:
        ax.fill(corridor.boundary.points[:, 0], corridor.boundary.points[:, 1], color="#f1f5f9")
        for boundary in (corridor.upper, corridor.lower):
            ax.plot(*boundary.T, color="#64748b", lw=1)
        ax.plot(*corridor.center.points.T, color="#cbd5e1", lw=1, ls=":")
        ax.set(xlabel="x [m]", ylabel="y [m]")
        ax.set_aspect("equal", adjustable="box")
    for proposal, color in zip(config["proposals"], COLORS, strict=True):
        record = paths[f"{selected['cell']}__{proposal}"]
        row = frame[
            (frame.cell == selected["cell"])
            & (frame.proposal == proposal)
            & (frame.seed == config["execution"]["representative_seed"])
        ].iloc[0]
        for ax in axes[1:]:
            ax.plot(
                record["px"], record["py"], color=color, lw=1.8, ls="-" if row.success else "--"
            )
            ax.scatter(
                record["px"][-1],
                record["py"][-1],
                color=color,
                marker="o" if row.success else "x",
                s=25,
            )
        data["paths"][proposal] = dict(x=record["px"], y=record["py"], termination=row.termination)
    allx = np.concatenate([np.array(v["x"]) for v in data["paths"].values()])
    bounds = [
        float(min(allx.min(), 0) - 0.5),
        float(max(allx.max(), corridor.goal[0]) + 0.5),
        float(lower[:, 1].min() - 0.2),
        float(upper[:, 1].max() + 0.2),
    ]
    axes[1].set(
        xlim=bounds[:2],
        ylim=bounds[2:],
        title=f"Same map, matched seed {config['execution']['representative_seed']}; "
        f"clearance={config['execution']['representative_clearance']} m",
    )
    axes[1].scatter(*corridor.goal[:2], marker="*", color="black", s=65)
    apex = corridor.center.points[np.argmax(corridor.center.points[:, 1])]
    zoom = [float(apex[0] - 4), float(apex[0] + 4), float(apex[1] - 1.2), float(apex[1] + 0.8)]
    axes[2].set(
        xlim=zoom[:2], ylim=zoom[2:], title="Shared turn zoom · same axes for all proposals"
    )
    axes[2].text(
        0.01,
        -0.32,
        "Solid: completed; dashed / ×: failed; ★: finish gate",
        transform=axes[2].transAxes,
        fontsize=8,
    )
    data["map_bounds"], data["zoom_bounds"] = bounds, zoom
    fig.suptitle(
        (
            f"Q05 · S-turn difficulty: {config['seeds']['count']} paired seeds, "
            f"K={primary_samples}, T={config['controller']['horizon']}\n"
            f"v target={config['corridor']['speed_target']} m/s; "
            f"|steering|≤{config['vehicle']['steering_limit']} rad; |rate|≤{rate} "
            f"rad/s. Shading: marginal Wilson 95% intervals."
        ),
        fontsize=11,
    )
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    save(fig, folder, "Q05_key")
    plt.close(fig)
    svg = folder / "Q05_key.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    write_json(root / "key_figure_data.json", data)
    target = folder / "Q05_key.png"
    if showcase is not None:
        target = Path(showcase)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((folder / "Q05_key.png").read_bytes())
    write_json(
        root / "figure_provenance.json",
        dict(
            git=git_info(),
            figure_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            summary_sha256=hashlib.sha256((root / "summary.csv").read_bytes()).hexdigest(),
            plotting_data_sha256=fingerprint(data),
            plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            representative_rule=(
                "Predeclared first evaluation seed at clearance .35; failures retained"
            ),
            regeneration=(
                "Saved summaries, paths and polygon geometry only; no model execution "
                "or random draws"
            ),
        ),
    )
    figure_link = os.path.relpath(target, root)
    lines = [
        "# Q05 — Executed S-turn findings",
        "",
        (
            "One common objective and fixed Gaussian base were used within every "
            "condition. The primary axis is clearance; separate sample-budget and "
            "rate-limit checks use only the predeclared middle clearance."
        ),
        "",
        (
            "| Clearance [m] | K | Rate limit [rad/s] | Proposal | Completions | "
            "Collisions | Censored | Mean max progress | Mean ESS |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.clearance:g} | {r.samples} | {r.steering_rate_limit:g} | "
            f"{r.proposal} | {r.completions}/{r.n} | {r.collisions}/{r.n} | "
            f"{r.censored}/{r.n} | {r.max_progress_fraction_mean:.3f} | "
            f"{r.mean_ess_mean:.2f} |"
        )
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        (
            "All feasibility checks passed before evaluation. This rules out "
            "mechanical impossibility of the recorded reference, not every "
            "alternative command sequence. Covariance-specific failures cannot be "
            "inferred from collision counts alone. Inspect feasible-rollout "
            "fractions, ESS/entropy, steering-angle/rate saturation, progress and "
            "the finite horizon together."
        ),
        "",
        (
            "Completion time is summarized only among completions; collisions and "
            "timeout-censored runs remain separate. Success curves have marginal "
            "Wilson intervals; paired differences against standard and their "
            "descriptive t intervals are in paired_differences.csv. Intervals are "
            "pointwise, unadjusted for multiple comparisons, and twenty seeds may "
            "give broad uncertainty."
        ),
        "",
        (
            "The matrices are specific trace-.40 proposals in normalized "
            "acceleration/rate coordinates. They do not represent all covariances "
            "or published feedback architectures. Latency comes from shared CPU "
            "execution and includes geometry checks; it is not a real-time "
            "guarantee."
        ),
        "",
        (
            f"[Primary composite]({figure_link}) · [Numeric plot "
            "inputs](key_figure_data.json) · [Per-seed metrics](per_seed.csv) · "
            "[Feasibility checks](feasibility.json)"
        ),
        "",
        (
            "Raw episode CSVs, controls, ESS traces and selected rollout "
            "snapshots are kept locally under episodes/ and ignored. "
            "Summary/provenance, representative paths and numeric plotting inputs "
            "are versioned. Regeneration needs no controller execution."
        ),
        "",
        "Source: " + manifest["git"]["commit"] + "; phase: " + manifest["phase"] + ".",
        "",
    ]
    (root / "interpretation.md").write_text("\n".join(lines))
    print(f"Regenerated one Q05 composite from saved data: {target}", flush=True)
