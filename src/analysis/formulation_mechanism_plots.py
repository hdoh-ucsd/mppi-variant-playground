"""Saved-record mechanism figures; no controllers, sampling or simulation imports."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse

from analysis.dynamic_plots import publish
from evaluation.runner import write_json

COLORS = ["#2563eb", "#d97706", "#059669", "#9333ea", "#e11d48", "#64748b"]
NAMES = {
    "V01": "V01_weighting_mechanism",
    "V02": "V02_annealing_mechanism",
    "V03": "V03_tracking_mechanism",
    "V04": "V04_robust_mechanism",
    "V05": "V05_covariance_mechanism",
}


def decorate(axes):
    for ax in np.asarray(axes).ravel():
        ax.grid(alpha=0.16)
        ax.tick_params(labelsize=8)


def weighting(root, data, summary):
    fig, ax = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    for column, case in enumerate(data["vehicle"]):
        cost = np.array(case["costs"])
        order = np.argsort(cost)
        for label, key, color in [
            ("KL", "kl_weights", COLORS[0]),
            ("Tsallis", "tsallis_weights", COLORS[1]),
        ]:
            w = np.array(case[key])
            ess = 1 / np.sum(w**2)
            ax[0, column].plot(
                np.arange(len(w)), w[order], color=color, label=f"{label}: ESS {ess:.2f}"
            )
        ax[0, column].set(
            title=case["scene"].replace("_", " ") + " · identical samples",
            xlabel="Sample rank by shared raw cost (low → high)",
            ylabel="Normalized optimizer weight",
        )
        ax[0, column].legend(fontsize=8)
        commands = np.array(case["commands"])
        for i, label in enumerate(["KL", "Tsallis"]):
            ax[1, column].bar(
                np.arange(2) + (i - 0.5) * 0.3, commands[i], width=0.3, color=COLORS[i], label=label
            )
        ax[1, column].set(
            xticks=[0, 1],
            xticklabels=["steering / 0.5 rad", "throttle / 1"],
            ylabel="First command, normalized",
            title="Different target → different update",
        )
        ax[1, column].axhline(0, color="gray", lw=0.5)
    reference = pd.read_csv(root / "reference_trials.csv")
    for i, (label, record) in enumerate(data["reference"].items()):
        ax[2, 0].plot(
            data["grid"],
            record["density"],
            color=COLORS[i],
            label=f"{label}, mean {record['mean']:.3f}",
        )
        selected = reference[reference.target == label]
        means = selected.groupby("samples").absolute_error.mean()
        ax[2, 1].loglog(means.index, means, "o-", color=COLORS[i], label=label)
        ax[2, 1].scatter(selected.samples, selected.absolute_error, color=COLORS[i], alpha=0.35)
    ax[2, 0].set(
        title="Separate scalar objective: independent quadrature",
        xlabel="Latent scalar z",
        ylabel="Normalized target density",
    )
    ax[2, 1].set(
        title="Each estimator compared to its own target",
        xlabel="Samples (three seeds)",
        ylabel="Absolute error in target mean",
    )
    for a in ax[2]:
        a.legend(fontsize=8)
    fig.suptitle(
        "V01 · Weighting intervention, fixed vehicle state / warm start / costs / samples\n"
        "First predeclared seed shown; KL and Tsallis targets differ. Commands are not executed.",
        fontsize=12,
    )
    decorate(ax)
    return fig


def annealing(root, data, summary):
    fig, ax = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    d = data["dial_mpc"]
    maximum = max(np.max(x) for x in d["histograms"])
    for i in range(4):
        hist = np.array(d["histograms"][i])
        mesh = ax[0, i].pcolormesh(
            d["xedges"],
            d["yedges"],
            hist.T,
            cmap="Blues",
            norm=LogNorm(vmin=1, vmax=maximum),
            shading="auto",
        )
        plan = np.array(d["plans"][i])
        ax[0, i].plot(plan[:, 0], plan[:, 1], color=COLORS[2], lw=2)
        for x, y, r in data["obstacles"]:
            ax[0, i].add_patch(Circle((x, y), r, color="gray", alpha=0.65))
        ax[0, i].set(
            title=f"DIAL pass {i + 1}: unweighted occupancy",
            xlabel="x [m]",
            ylabel="y [m]",
            aspect="equal",
        )
        ax[1, 0].plot(0.2 * np.array(d["variance_scales"][i]), label=f"pass {i + 1}")
    fig.colorbar(mesh, ax=ax[0, :], shrink=0.7, label="Sample-time points per bin (log)")
    ax[1, 0].axhline(0.2, color="gray", ls="--", label="fixed-noise ablation")
    ax[1, 0].set(
        title="Actual horizon / iteration noise schedule",
        xlabel="Horizon control index",
        ylabel="Latent variance (both channels)",
    )
    ax[1, 0].legend(fontsize=7)
    for alg, label, color in [
        ("it_mppi", "IT-MPPI", COLORS[0]),
        ("dial_mpc", "DIAL", COLORS[2]),
        ("dial_no_anneal", "DIAL fixed noise", COLORS[5]),
    ]:
        sub = summary[summary.algorithm == alg]
        grouped = sub.groupby("transitions")[["planned_cost", "update_rms"]].mean()
        for index, metric in [(1, "planned_cost"), (3, "update_rms")]:
            ax[1, index].plot(grouped.index, grouped[metric], "o-", color=color, label=label)
            for _, seed in sub.groupby("seed"):
                ax[1, index].plot(seed.transitions, seed[metric], ".-", color=color, alpha=0.25)
        plan = np.array(data[alg]["plans"][-1])
        ax[1, 2].plot(plan[:, 0], plan[:, 1], color=color, label=label)
    for i, plan in enumerate(d["plans"][:-1]):
        plan = np.array(plan)
        ax[1, 2].plot(plan[:, 0], plan[:, 1], "--", color=COLORS[2], alpha=0.25 + 0.15 * i)
    for x, y, r in data["obstacles"]:
        ax[1, 2].add_patch(Circle((x, y), r, color="gray", alpha=0.5))
    ax[1, 1].set(
        title="Progress · every inner pass charged",
        xlabel="Cumulative model transitions",
        ylabel="Updated-plan task cost",
        yscale="log",
    )
    ax[1, 2].set(title="Predicted mean plans, first seed", xlabel="x [m]", ylabel="y [m]")
    ax[1, 3].set(
        title="Change from common initial mean",
        xlabel="Cumulative model transitions",
        ylabel="Latent-sequence RMS change",
    )
    for a in ax[1, 1:]:
        a.legend(fontsize=7)
    fig.suptitle(
        "V02 · Annealing and refinement at a fixed two-route observation\n"
        "7,680 transitions each; three seeds retained. Unweighted time-pooled occupancy; "
        "not physical uncertainty.",
        fontsize=12,
    )
    decorate(ax)
    return fig


def tracking(root, data, summary):
    steps = pd.read_csv(root / "steps.csv")
    fig, ax = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    representative = steps[(steps.seed == data["representative_seed"]) & (steps.impulse == 1)]
    labels = [
        ("it_mppi", "IT-MPPI", COLORS[0]),
        ("tube_450", "Tube reset 450", COLORS[3]),
        ("tube_0", "Tube reset 0", COLORS[2]),
    ]
    for algorithm, label, color in labels:
        g = representative[representative.algorithm == algorithm]
        time = g.step * data["dt"]
        ax[0, 0].plot(g.px, g.py, color=color, label=label)
        if algorithm != "it_mppi":
            ax[0, 0].plot(g.nominal_state_0, g.nominal_state_1, "--", color=color, alpha=0.55)
            ax[0, 1].plot(time, g.py - g.nominal_before_reset_1, "--", color=color)
            ax[0, 1].plot(time, g.tracking_error_vector_1, color=color, label=label)
            ax[0, 2].plot(time, 0.5 * g.tracking_correction_0, "--", color=color)
            ax[0, 2].plot(time, g.applied_correction_0, color=color, label=label)
            ax[1, 2].plot(time, g.applied_correction_1, color=color, label=label)
        sub = summary[summary.algorithm == algorithm]
        means = sub.groupby("impulse")["tracking_active_commands"].mean().reindex([-1, 0, 1])
        offset = {"it_mppi": -0.22, "tube_450": 0, "tube_0": 0.22}[algorithm]
        ax[1, 0].bar(
            np.arange(3) + offset, means / data["steps"], width=0.22, color=color, label=label
        )
        # All signed/control cases; total absolute lateral position is an external response measure.
        sd = steps[steps.algorithm == algorithm].copy()
        sd["abs_y"] = sd.py.abs() * data["dt"]
        totals = sd.groupby(["impulse", "seed"]).abs_y.sum().reset_index()
        aggregate = totals.groupby("impulse").abs_y.mean().reindex([-1, 0, 1])
        ax[1, 1].bar(np.arange(3) + offset, aggregate, width=0.22, color=color)
        for j, sign in enumerate([-1, 0, 1]):
            ax[1, 1].scatter(
                np.full(3, j + offset),
                totals[totals.impulse == sign].abs_y,
                color="black",
                alpha=0.35,
                s=12,
            )
    ax[0, 0].set(title="Executed paths; dashed = selected nominal", xlabel="x [m]", ylabel="y [m]")
    ax[0, 1].set(
        title="Lateral error: dashed before reset, solid after",
        xlabel="Time [s]",
        ylabel="Actual − nominal y [m]",
    )
    ax[0, 2].set(
        title="Steering: dashed requested, solid applied",
        xlabel="Time [s]",
        ylabel="Ancillary steering [rad]",
    )
    ax[1, 0].set(
        title="All declared disturbances and seeds",
        ylabel="Fraction of commands with active tracker",
    )
    ax[1, 1].set(title="External response · seeds + mean", ylabel="Integrated |y| [m·s]")
    ax[1, 2].set(
        title="Applied ancillary throttle",
        xlabel="Time [s]",
        ylabel="Throttle correction [normalized]",
    )
    for a in ax[1, :2]:
        a.set_xticks(range(3), ["negative", "zero", "positive"])
    for a in [ax[0, 1], ax[0, 2], ax[1, 2]]:
        a.axvline(
            (data["event_step"] + 1) * data["dt"],
            color="gray",
            ls=":",
            label="first post-impulse observation",
        )
    for a in [ax[0, 0], ax[1, 0]]:
        a.legend(fontsize=8)
    fig.suptitle(
        "V03 · Nominal reset determines whether the tracker acts\n"
        "Signed/control impulses and paired seeds; positive first-seed time series. "
        "Reset thresholds: original 450 and declared 0.",
        fontsize=12,
    )
    decorate(ax)
    return fig


def robust(root, data, summary):
    fig, ax = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    chosen = next(c for c in data["cases"] if c["impulse"] == 1 and c["threshold"] == 100)
    for sign, color in zip([-1, 0, 1], COLORS, strict=False):
        case = next(c for c in data["cases"] if c["impulse"] == sign and c["threshold"] == 100)
        ax[0, 0].plot(range(5), case["energies"], "o-", color=color, label=f"impulse {sign:+d}")
    for alpha in [100, 5000]:
        ax[0, 0].axhline(alpha, color="gray", ls="--")
    ax[0, 0].set(
        title="Candidate free energy; thresholds 100 / 5000",
        xlabel="Nominal candidate index",
        ylabel="Estimated free energy",
        yscale="log",
    )
    ax[0, 0].legend(fontsize=8)
    for key, label, color in [
        ("nominal_mean", "Nominal", COLORS[0]),
        ("actual_mean", "Actual with feedback", COLORS[4]),
        ("it_mean", "Same-noise IT", COLORS[5]),
    ]:
        points = np.array(chosen[key])
        ax[0, 1].plot(points[:, 0], points[:, 1], color=color, label=label)
    for key, color in [("nominal_paths", COLORS[0]), ("actual_paths", COLORS[4])]:
        for path in chosen[key]:
            path = np.array(path)
            ax[0, 1].plot(path[:, 0], path[:, 1], color=color, alpha=0.07)
    ax[0, 1].set(
        title="α=100, + impulse: infeasible-search fallback", xlabel="x [m]", ylabel="y [m]"
    )
    ax[0, 1].legend(fontsize=7)
    for case in data["cases"]:
        ax[0, 2].plot(
            case["feedback_rms"],
            ls="--" if case["threshold"] == 5000 else "-",
            color=COLORS[[-1, 0, 1].index(case["impulse"])],
            label=f"impulse {case['impulse']:+d}, α={case['threshold']:g}",
        )
    ax[0, 2].set(
        title="Feedback inside sampled rollouts",
        xlabel="Horizon index",
        ylabel="RMS correction [latent units]",
    )
    ax[0, 2].legend(fontsize=7)
    representative = summary[summary.seed == summary.seed.min()]
    names, planner, tracker, command, baseline = [], [], [], [], []
    for _, row in representative.iterrows():
        names.append(f"{row.impulse:+.0f}\nα={row.threshold:g}")
        planner.append(row.planner_steering)
        tracker.append(row.applied_tracking_steering)
        command.append(row.command_steering)
        baseline.append(row.it_command_steering)
    x = np.arange(len(names))
    ax[1, 0].bar(x - 0.15, planner, width=0.3, label="Planner before tracker", color=COLORS[0])
    ax[1, 0].bar(x + 0.15, tracker, width=0.3, label="Applied tracker increment", color=COLORS[1])
    ax[1, 0].scatter(x, command, marker="x", color="black", label="Total applied")
    ax[1, 0].scatter(x, baseline, marker="_", color=COLORS[5], label="Same-noise IT")
    ax[1, 0].set(
        xticks=x,
        xticklabels=names,
        title="Planner vs tracker actions, first seed",
        ylabel="Steering [rad]",
    )
    ax[1, 0].legend(fontsize=7)
    for alpha, color in [(5000, COLORS[3]), (100, COLORS[4])]:
        group = summary[summary.threshold == alpha]
        g = group.groupby("impulse").plan_difference_rms.mean()
        ax[1, 1].plot(g.index, g, "o-", color=color, label=f"α={alpha}")
        ax[1, 1].scatter(group.impulse, group.plan_difference_rms, color=color, alpha=0.3)
    ax[1, 1].set(
        title="Actual fitted plan vs same-noise IT",
        xlabel="Signed impulse multiplier",
        ylabel="Latent-plan RMS difference",
    )
    ax[1, 1].legend(fontsize=8)
    order = np.argsort(chosen["actual_weights"])[::-1]
    for key, label, color in [
        ("nominal_weights", "Mixed nominal weights", COLORS[0]),
        ("actual_weights", "Actual corrected weights", COLORS[4]),
    ]:
        ax[1, 2].plot(np.array(chosen[key])[order], color=color, label=label)
    ax[1, 2].set(
        title="Two fits inside the fallback decision",
        xlabel="Sample rank by actual weight",
        ylabel="Normalized optimization weight",
    )
    ax[1, 2].legend(fontsize=7)
    fig.suptitle(
        "V04 · Coupled nominal/actual planning on identical perturbations\n"
        "Active feedback here occurs only with infeasible-search fallback at α=100; "
        "no executed robustness claim.",
        fontsize=12,
    )
    decorate(ax)
    return fig


def ellipse(ax, center, covariance, color, linestyle, label):
    values, vectors = np.linalg.eigh(np.asarray(covariance)[:2, :2])
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * np.sqrt(5.991464547 * np.maximum(values, 0))
    ax.add_patch(
        Ellipse(
            np.asarray(center)[:2],
            width,
            height,
            angle=angle,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            lw=1.5,
            label=label,
        )
    )


def covariance(root, data, summary):
    fig, ax = plt.subplots(3, 3, figsize=(15, 13), layout="constrained")
    scenes = ["open_space", "narrow_passage"]
    frozen = json.loads((root / "frozen_inputs.json").read_text())
    for column, scene in enumerate(scenes):
        for fraction, row in [(1.0, 0), (0.01, 2)]:
            for mode, color in [("steered", COLORS[2]), ("zero_gain", COLORS[5])]:
                c = next(
                    c
                    for c in data["cases"]
                    if c["scene"] == scene and c["fraction"] == fraction and c["mode"] == mode
                )
                for t in [10, 20] if row == 0 else [20]:
                    ellipse(
                        ax[row, column],
                        c["reference"][t] if row == 0 else np.zeros(2),
                        c["predicted_covariances"][t],
                        color,
                        "--",
                        f"{mode} linear" if t == 20 else None,
                    )
                    ellipse(
                        ax[row, column],
                        c["empirical_mean"][t] if row == 0 else np.zeros(2),
                        c["empirical_covariances"][t],
                        color,
                        "-",
                        f"{mode} executed" if t == 20 else None,
                    )
                path = np.array(c["empirical_mean"])
                if row == 0:
                    ax[row, column].plot(path[:, 0], path[:, 1], color=color, alpha=0.3)
                if row == 0:
                    for path in c["paths"]:
                        path = np.array(path)
                        ax[row, column].plot(path[:, 0], path[:, 1], color=color, alpha=0.035)
            ax[row, column].autoscale_view()
            ax[row, column].set(
                title=(
                    f"{scene.replace('_', ' ')} · noise {fraction:g}C₀\n"
                    + ("95% contours at t=10,20" if row == 0 else "Centered terminal 95% contours")
                ),
                xlabel="x [m]" if row == 0 else "Centered x [m]",
                ylabel="y [m]" if row == 0 else "Centered y [m]",
                aspect="equal",
            )
            if row == 0:
                for x, y, radius in frozen[scene]["config"]["scenario"]["obstacles"]:
                    ax[row, column].add_patch(Circle((x, y), radius, color="gray", alpha=0.35))
                counts = summary[(summary.scene == scene) & (summary.noise_fraction == fraction)]
                counts = counts.set_index("mode")
                ax[row, column].text(
                    0.03,
                    0.03,
                    f"Proxy contacts: steered {int(counts.loc['steered', 'collisions'])}/256; "
                    f"zero {int(counts.loc['zero_gain', 'collisions'])}/256",
                    transform=ax[row, column].transAxes,
                    fontsize=7,
                )
            ax[row, column].legend(fontsize=7)
        for case in [c for c in data["cases"] if c["scene"] == scene]:
            color = COLORS[2] if case["mode"] == "steered" else COLORS[5]
            shade = 1 if case["fraction"] == 1 else 0.45
            label = f"{case['mode']}, {case['fraction']:g}C₀"
            ax[1, column].semilogy(
                range(1, 21), case["empirical_trace"][1:], color=color, alpha=shade, label=label
            )
            ax[1, column].semilogy(
                range(1, 21), case["predicted_trace"][1:], "--", color=color, alpha=shade
            )
        ax[1, column].set(
            title="Normalized covariance trace over execution\nsolid = empirical, dashed = linear",
            xlabel="Horizon timestep",
            ylabel="Trace in bound-normalized coordinates",
        )
        ax[1, column].legend(fontsize=7)
        case = next(c for c in data["cases"] if c["scene"] == scene and c["mode"] == "steered")
        gain = np.array(case["gains"])
        # Dimensional gain norm is avoided: multiply by state normalization first.
        scale = np.sqrt([1, 1, 0.25, 0.04, 1, 0.5, 0.5])
        ax[0, 2].plot(np.linalg.norm(gain * scale, axis=(1, 2)), label=scene.replace("_", " "))
    ax[0, 2].set(
        title="Feedback gains in normalized coordinates",
        xlabel="Horizon timestep",
        ylabel="Frobenius norm of K·sqrt(bound)",
    )
    ax[0, 2].legend(fontsize=8)
    labels = [
        f"{'open' if r.scene == 'open_space' else 'narrow'}\n{r.noise_fraction:g}/{r.mode[:1]}"
        for r in summary.itertuples()
    ]
    x = np.arange(len(summary))
    ax[1, 2].bar(x, summary.normalized_prediction_error, color=COLORS[1])
    ax[1, 2].set(
        title="Terminal prediction discrepancy",
        ylabel="Normalized Frobenius error",
        xticks=x,
        xticklabels=labels,
    )
    ax[2, 2].plot(
        x,
        summary.predicted_normalized_psd_residual,
        "o",
        label="Linear prediction",
        color=COLORS[0],
    )
    ax[2, 2].plot(
        x,
        summary.empirical_normalized_psd_residual,
        "x",
        label="Independent executions",
        color=COLORS[1],
    )
    ax[2, 2].axhline(0, color="black", ls="--")
    ax[2, 2].set(
        title="Terminal covariance constraint: >0 violates",
        ylabel="Largest eigenvalue of normalized Σ − I",
        xticks=x,
        xticklabels=labels,
    )
    ax[2, 2].legend(fontsize=7)
    fig.suptitle(
        "V05 · Fixed-policy independent stochastic executions, 256 replicates per condition\n"
        "Paired noise; fixed mean/gains; no weights or replanning. "
        "Ellipses summarize covariance, not safety.",
        fontsize=12,
    )
    decorate(ax)
    return fig


def plot_all(output):
    for study, function in [
        ("V01", weighting),
        ("V02", annealing),
        ("V03", tracking),
        ("V04", robust),
        ("V05", covariance),
    ]:
        root = Path(output) / study
        data = json.loads((root / "plot_data.json").read_text())
        summary = pd.read_csv(root / "summary.csv")
        if study == "V03":
            data["steps"] = int(summary.executed_steps.max())
        figure = function(root, data, summary)
        sources = [root / "plot_data.json", root / "summary.csv"]
        for extra in ["steps.csv", "reference_trials.csv", "frozen_inputs.json"]:
            if (root / extra).exists():
                sources.append(root / extra)
        name = NAMES[study]
        publish(figure, root, name, sources)
        target = root / (name + ".png")
        target.write_bytes((root / "figures" / (name + ".png")).read_bytes())
        provenance_path = root / (name + "_provenance.json")
        provenance = json.loads(provenance_path.read_text())
        provenance.update(
            figure=str(target),
            mechanism_plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        write_json(provenance_path, provenance)
