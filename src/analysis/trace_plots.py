"""Six numerical views of saved DecisionTrace reductions; no controller execution."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Patch

from analysis.decision_trace import STAGES
from visualization.figures import draw_scene, style

NAMES = {
    "it_mppi": "IT",
    "tsallis_vi": "Tsallis",
    "dial_mpc": "DIAL",
    "tube_mppi": "Tube",
    "robust_mppi": "Robust",
    "cc_mppi": "CC",
}
COLORS = dict(
    zip(NAMES, ["#17648d", "#c05825", "#7863a2", "#168375", "#aa4c78", "#79702e"], strict=True)
)
STATUS_COLORS = {
    "available": "#d8e6f0",
    "missing": "#e6e6e6",
    "not_applicable": "#f5f2e6",
    "mechanism_inactive": "#e8dcf0",
    "numerical_failure": "#f5cfcf",
}


def plot_all(output, destination=None):
    """Read only compact records. Full populations and historical raw files are unnecessary."""
    output = Path(output)
    destination = Path(destination or output)
    destination.mkdir(parents=True, exist_ok=True)
    data = json.loads((output / "plot_data.json").read_text())
    traces = data["traces"]
    frozen = {t["algorithm"]: t for t in traces if t["experiment_id"].endswith("-frozen")}
    if set(frozen) != set(NAMES):
        raise ValueError("The six stage comparisons require the selected live frozen decisions")
    closed = [t for t in traces if t["experiment_id"].endswith("-closed")]
    retro = [t for t in traces if t["experiment_id"].startswith("retro-")]
    sources = {}
    style()

    def finish(fig, filename, title, note, records):
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.text(0.02, 0.012, note, fontsize=9, va="bottom")
        fig.tight_layout(rect=(0, 0.075, 1, 0.94))
        fig.savefig(destination / filename, dpi=170, bbox_inches="tight")
        plt.close(fig)
        sources[filename] = {
            "decision_ids": sorted({t["decision_id"] for t in records}),
            "sha256": hashlib.sha256((destination / filename).read_bytes()).hexdigest(),
        }

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    scene = data["scenes"][frozen["it_mppi"]["configuration_hash"]]
    draw_scene(ax[0, 0], scene)
    for alg, t in frozen.items():
        plan = np.asarray(t["updates"][-1]["nominal_prediction"])
        ax[0, 0].plot(plan[:, 0], plan[:, 1], color=COLORS[alg], label=NAMES[alg])
    x = frozen["it_mppi"]["observed"]
    ax[0, 0].scatter(*x[:2], c="black", marker="x", s=75, label="shared observed state", zorder=8)
    ax[0, 0].set_title("Same task; updated predicted paths")
    ax[0, 0].legend(fontsize=8, ncol=2)
    for alg in NAMES:
        rows = [t for t in closed if t["algorithm"] == alg]
        times = [t["feedback"]["next_time_s"] for t in rows]
        actual = [t["feedback"]["next_executed_state"][1] for t in rows]
        predicted = [t["feedback"]["predicted_next_state"][1] for t in rows]
        ax[0, 1].plot(times, actual, color=COLORS[alg], label=NAMES[alg])
        ax[0, 1].plot(times, predicted, ":", color=COLORS[alg], alpha=0.7)
    ax[0, 1].axvline(1.35, color="gray", linestyle="--")
    ax[0, 1].set(
        title="Short executions: actual (solid), one-step prediction (dotted)",
        xlabel="Next-state time [s]",
        ylabel="Lateral position [m]",
    )
    ax[0, 1].legend(fontsize=8, ncol=3)
    terms = frozen["it_mppi"]["updates"][0]["cost_components"]
    keys = [k for k, v in terms.items() if abs(v) > 1e-7]
    positions = np.arange(len(keys))
    for shift, alg in [(-0.18, "it_mppi"), (0.18, "tsallis_vi")]:
        values = frozen[alg]["updates"][0]["cost_components"]
        ax[1, 0].barh(
            positions + shift,
            [values[k] for k in keys],
            height=0.34,
            label=NAMES[alg],
            color=COLORS[alg],
        )
    ax[1, 0].set_yticks(positions, [k.replace("_", " ") for k in keys], fontsize=8)
    ax[1, 0].set(
        xlabel="Unweighted mean candidate cost contribution",
        title="Identical IT/Tsallis task costs",
    )
    ax[1, 0].legend(fontsize=8)
    for alg in NAMES:
        rows = [t for t in closed if t["algorithm"] == alg]
        ax[1, 1].plot(
            [t["step"] for t in rows],
            [t["feedback"]["prediction_error"][1] for t in rows],
            label=NAMES[alg],
            color=COLORS[alg],
            alpha=0.8,
        )
    ax[1, 1].set(
        title="Recorded model error at the injected impulse",
        xlabel="Decision step (impulse after step 8)",
        ylabel="Executed minus predicted y [m]",
    )
    finish(
        fig,
        "stage_01_dynamics.png",
        "1 · Dynamics + Task",
        "Frozen inputs shared; closed-loop inputs diverge. One-step prediction uses the "
        "actual commanded control.\n"
        "Cost bars are unweighted task components; no claim of controller ranking or task "
        "completion.",
        list(frozen.values()) + closed,
    )

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))
    for alg in ["it_mppi", "tsallis_vi"]:
        t = frozen[alg]
        u = t["updates"][0]
        cost = np.asarray(u["raw_costs"])
        weights = np.asarray(u["weights"])
        order = np.argsort(cost)
        ax[0].plot(
            cost[order],
            weights[order],
            ".",
            label=f"{NAMES[alg]} ESS={u['ess']:.2f}",
            color=COLORS[alg],
        )
        scores = np.asarray([np.nan if v is None else v for v in u["log_scores"]])
        finite = np.isfinite(scores)
        ax[1].plot(
            scores[finite] - np.nanmax(scores),
            weights[finite],
            ".",
            color=COLORS[alg],
            label=NAMES[alg],
        )
    ts = frozen["tsallis_vi"]["objective"]["tsallis"]
    ax[0].set(
        title=f"384 identical samples; support {sum(ts['support'])}/384",
        xlabel="Raw task cost",
        ylabel="Normalized optimizer weight",
    )
    ax[0].legend(fontsize=8)
    ax[1].set(
        title=f"Corrected log scores; Tsallis scale={ts['transformed_scale']:.3f}",
        xlabel="Log score minus maximum",
        ylabel="Normalized optimizer weight",
        xlim=(-25, 1),
    )
    ref = next(t for t in retro if "related_scalar_reference" in t["objective"])
    r = ref["objective"]["related_scalar_reference"]
    for name, v in r["targets"].items():
        ax[2].plot(r["grid"], v["density"], label=f"{name}: mean {v['mean']:.3f}")
    ax[2].set(
        title="Independent scalar quadrature references", xlabel="Scalar z", ylabel="Target density"
    )
    ax[2].legend(fontsize=8)
    finish(
        fig,
        "stage_02_objective.png",
        "2 · Distributional Objective",
        "Vehicle: IT temperature 8; Tsallis r=1.5, "
        f"normalized cutoff={ts['normalized_cutoff']:.3f}; "
        "same density correction.\n"
        "Right: separate J(z)=0.5(z−1)², prior N(0,1), scale 0.8. Different targets, not "
        "errors against one common optimum.",
        [frozen["it_mppi"], frozen["tsallis_vi"], ref],
    )

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for alg in ["it_mppi", "tsallis_vi"]:
        p = frozen[alg]["populations"][0]
        v = np.asarray(p["first_latent"])
        ax[0, 0].scatter(
            v[:, 0],
            v[:, 1],
            s=10 if alg == "it_mppi" else 30,
            facecolors=COLORS[alg] if alg == "it_mppi" else "none",
            edgecolors=COLORS[alg],
            alpha=0.5,
            label=NAMES[alg],
        )
    ax[0, 0].set(
        title="Identical latent population (unweighted)",
        xlabel="First latent steering",
        ylabel="First latent throttle",
    )
    ax[0, 0].legend(fontsize=8)
    dial = frozen["dial_mpc"]
    for p, u in zip(dial["populations"], dial["updates"], strict=True):
        i = p["iteration"]
        color = plt.cm.viridis(i / 3)
        for path in np.asarray(p["selected_xy"]):
            ax[0, 1].plot(path[:, 0], path[:, 1], color=color, alpha=0.15, lw=1)
        plan = np.asarray(u["nominal_prediction"])
        ax[0, 1].plot(plan[:, 0], plan[:, 1], color=color, label=f"Pass {i + 1}")
        v = np.asarray(p["first_latent"])
        ax[1, 0].scatter(v[:, 0], v[:, 1], color=color, s=10, alpha=0.4, label=f"Pass {i + 1}")
    ax[0, 1].set(
        title="DIAL: first 8 unweighted paths/pass; updated plans", xlabel="x [m]", ylabel="y [m]"
    )
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].set(
        title="DIAL: all first-control samples per pass",
        xlabel="First latent steering",
        ylabel="First latent throttle",
    )
    ax[1, 0].legend(fontsize=8)
    ax[1, 2].bar(
        list(NAMES.values()),
        [t["model_transitions"] for t in frozen.values()],
        color=list(COLORS.values()),
    )
    ax[1, 2].axhline(7700, color="black", linestyle="--", lw=1)
    ax[1, 2].set(
        title="Total work, including internal passes / linearization", ylabel="Model transitions"
    )
    ax[1, 2].text(
        0.02,
        0.08,
        "DIAL: 4 × 1,920 = 7,680 transitions\nIT: 1 × 7,700 transitions",
        transform=ax[1, 2].transAxes,
        fontsize=9,
    )
    for p in dial["populations"]:
        covariance = np.asarray(p["proposal_covariance"])
        ax[0, 2].plot(
            np.arange(len(covariance)), covariance[:, 0, 0], label=f"Pass {p['iteration'] + 1}"
        )
    ax[0, 2].set(
        title="Recorded annealing schedule",
        xlabel="Horizon index",
        ylabel="Latent steering variance",
    )
    ax[0, 2].legend(fontsize=8)
    work = np.cumsum([p["model_transitions"] for p in dial["populations"]])
    ax[1, 1].plot(
        work,
        [u["updated_plan_external_cost"] for u in dial["updates"]],
        "o-",
        label="DIAL updated plan",
    )
    baseline = frozen["it_mppi"]
    ax[1, 1].scatter(
        [baseline["model_transitions"]],
        [baseline["updates"][0]["updated_plan_external_cost"]],
        label="IT updated plan",
    )
    ax[1, 1].set(
        title="External task cost after each update",
        xlabel="Cumulative model transitions",
        ylabel="Updated-plan cost",
    )
    ax[1, 1].legend(fontsize=8)
    finish(
        fig,
        "stage_03_sampling.png",
        "3 · Sample + Refine",
        "Same frozen state, task and warm start; distinct proposal/noise schedules are preserved.\n"
        "Unweighted planning populations are not executed physical uncertainty. Fixed "
        "first-eight path selection; no outcome filtering.",
        list(frozen.values()),
    )

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for alg, t in frozen.items():
        u = t["updates"][-1]
        ax[0, 0].plot(
            np.linspace(0, 1, len(u["raw_costs"])),
            np.sort(u["raw_costs"]),
            label=NAMES[alg],
            color=COLORS[alg],
        )
    ax[0, 0].set(
        title="Raw candidate costs (final pass/population)",
        xlabel="Empirical quantile",
        ylabel="Raw cost",
        yscale="log",
    )
    ax[0, 0].legend(fontsize=8, ncol=3)
    for u in frozen["robust_mppi"]["updates"]:
        order = np.argsort(u["weights"])[::-1]
        ax[0, 1].plot(
            np.asarray(u["weights"])[order],
            marker="." if u["role"] == "nominal" else None,
            linestyle="-" if u["role"] == "nominal" else "--",
            label=u["role"],
        )
    ax[0, 1].set(
        title="Robust dual weight populations (overlap here)",
        xlabel="Weight rank",
        ylabel="Normalized optimizer mass",
        xlim=(-1, 15),
    )
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].bar(
        list(NAMES.values()),
        [t["updates"][-1]["ess"] for t in frozen.values()],
        color=list(COLORS.values()),
    )
    ax[1, 0].set(title="Optimizer ESS, not physical covariance", ylabel="1 / sum(w²)")
    pos = np.arange(6)
    ax[1, 1].bar(
        pos - 0.18, [t["command"][0] for t in frozen.values()], width=0.36, label="Steering [rad]"
    )
    ax[1, 1].bar(
        pos + 0.18,
        [t["command"][1] for t in frozen.values()],
        width=0.36,
        label="Signed throttle [normalized]",
    )
    ax[1, 1].set_xticks(pos, NAMES.values())
    ax[1, 1].set(
        title="First physical command candidate (not executed)", ylabel="Command in labeled units"
    )
    ax[1, 1].legend(fontsize=8)
    finish(
        fig,
        "stage_04_update.png",
        "4 · Weight + Update",
        "Raw-cost minima, largest weights and updated means are distinct quantities. DIAL "
        "uses its last pass; Tube/Robust show actual population ESS.\n"
        "Counts and objectives differ under a common transition cap. These one-decision "
        "measurements establish no universal ranking.",
        list(frozen.values()),
    )

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    tube = []
    for threshold in (0, 450):
        rows = [t for t in retro if t["experiment_id"] == f"retro-V03-tube{threshold}"]
        tube += rows
        times = np.array([t["step"] for t in rows]) * 0.15
        observed = np.asarray([t["observed"] for t in rows])
        nominal = np.asarray([t["feedback"]["nominal_state"] for t in rows])
        color = COLORS["tube_mppi"] if threshold == 0 else "gray"
        ax[0, 0].plot(
            observed[:, 0], observed[:, 1], color=color, label=f"Actual threshold {threshold}"
        )
        ax[0, 0].plot(nominal[:, 0], nominal[:, 1], ":", color=color, label=f"Nominal {threshold}")
        corrections = [t["feedback"]["applied_correction"][0] for t in rows]
        ax[0, 1].plot(times, corrections, color=color, label=f"Threshold {threshold}")
        resets = [t["feedback"]["reset_event"] for t in rows]
        ax[0, 1].scatter(
            times[resets], np.asarray(corrections)[resets], color=color, marker="x", s=35
        )
    ax[0, 0].set(title="Historical paired Tube disturbance", xlabel="x [m]", ylabel="y [m]")
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].set(
        title="Applied correction; crosses = nominal reset",
        xlabel="Decision time [s]",
        ylabel="Steering correction [rad]",
    )
    ax[0, 1].axvline(1.35, color="gray", linestyle="--")
    ax[0, 1].legend(fontsize=8)
    robust = [t for t in retro if t["experiment_id"].startswith("retro-V04")]
    for t in robust:
        threshold = t["objective"]["selection"]["free_energy_threshold"]
        energies = t["robust"]["free_energies"]
        ax[0, 2].plot(
            energies, "o-", label=f"α={threshold}, selected {t['feedback']['selected_candidate']}"
        )
        ax[0, 2].axhline(threshold, linestyle=":", color="gray")
    ax[0, 2].set(
        title="Robust nominal candidates; low α infeasible",
        xlabel="Candidate index",
        ylabel="Free energy",
        yscale="log",
    )
    ax[0, 2].legend(fontsize=7)
    cc = next(t for t in retro if "related_fixed_policy_experiment" in t["feedback"])
    ensemble = cc["feedback"]["related_fixed_policy_experiment"]
    pred = np.asarray(ensemble["predicted_covariances"])
    emp = np.asarray(ensemble["empirical_covariances"])
    ref = np.asarray(ensemble["reference"])
    mean = np.asarray(ensemble["mean_states"])
    ax[1, 0].plot(ref[:, 0], ref[:, 1], color="black", label="Reference")
    for path in ensemble["selected_paths"]:
        ax[1, 0].plot(*np.asarray(path).T, color="gray", alpha=0.22, lw=1)
    for index in [5, 10, 20]:
        for covariance, center, color in [
            (pred[index, :2, :2], ref[index, :2], COLORS["cc_mppi"]),
            (emp[index, :2, :2], mean[index, :2], "#c05825"),
        ]:
            values, vectors = np.linalg.eigh(covariance)
            angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
            ax[1, 0].add_patch(
                Ellipse(
                    center,
                    2 * np.sqrt(values[-1]),
                    2 * np.sqrt(values[0]),
                    angle=angle,
                    fill=False,
                    edgecolor=color,
                    lw=2,
                )
            )
    ax[1, 0].set(title="CC: 1σ spatial ellipses, t=0.75/1.5/3 s", xlabel="x [m]", ylabel="y [m]")
    ax[1, 0].set_aspect("equal", adjustable="datalim")
    ax[1, 0].legend(
        handles=[
            Patch(color=COLORS["cc_mppi"], label="Predicted"),
            Patch(color="#c05825", label="Empirical"),
        ],
        fontsize=8,
    )
    bound = np.asarray(ensemble["terminal_bound"])
    norm = np.diag(bound)
    ax[1, 1].plot(
        ensemble["time"], np.trace(pred / norm[None, :, None], axis1=1, axis2=2), label="Predicted"
    )
    ax[1, 1].plot(
        ensemble["time"], np.trace(emp / norm[None, :, None], axis1=1, axis2=2), label="Empirical"
    )
    ax[1, 1].set(
        title=f"Independent fixed-policy ensemble: n={ensemble['replicates']}",
        xlabel="Prediction time [s]",
        ylabel="Sum of variances / terminal diagonal bounds",
    )
    ax[1, 1].legend(fontsize=8)
    inv = np.linalg.inv(np.linalg.cholesky(bound))
    ratios = [np.linalg.eigvalsh(inv @ c[-1] @ inv.T).max() for c in [pred, emp]]
    ax[1, 2].bar(["Predicted", "Empirical"], ratios, color=[COLORS["cc_mppi"], "#c05825"])
    ax[1, 2].axhline(1, color="black", linestyle="--")
    ax[1, 2].set(
        title=f"Contacts: {ensemble['collisions']}/{ensemble['replicates']}; no impact model",
        ylabel="Largest eigenvalue of bound-normalized terminal covariance",
    )
    finish(
        fig,
        "stage_05_feedback.png",
        "5 · Feedback + Execute",
        "Tube threshold 450 resets and suppresses tracking here; low-threshold Robust has "
        "no feasible candidate. Neither is a universal performance result.\n"
        "CC: separate fixed warm-start/gain policy, NOT execution of the optimized mean; "
        "full noise, 256 trials including contacts. Linear predictions need not match.",
        tube + robust + [cc],
    )

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis("off")
    cells = []
    audit = []
    for alg, t in frozen.items():
        execution = next(c for c in closed if c["algorithm"] == alg and c["step"] == 9)
        rows = []
        for i, stage in enumerate(STAGES):
            record = execution if i == 4 else t
            status = record["stage_status"][stage]
            values = [
                f"x={t['observed'][0]:.2f}, y={t['observed'][1]:.2f} m",
                t["objective"]["regularizer"].split(";")[0],
                f"{len(t['populations'])} populations\n{t['model_transitions']} transitions",
                f"last/actual ESS={t['updates'][-1]['ess']:.2f}",
                f"next y={execution['feedback']['next_executed_state'][1]:.3f} m",
            ]
            measurement = values[i]
            if i == 1:
                objective = t["objective"]
                if alg == "it_mppi":
                    measurement = f"KL; temperature={objective['temperature']:g}"
                elif alg == "tsallis_vi":
                    ts = objective["tsallis"]
                    measurement = (
                        f"r={ts['parameter']}; scale={ts['transformed_scale']:.3f}"
                        f"\nsupport={sum(ts['support'])}/{len(ts['support'])}"
                    )
                elif alg == "dial_mpc":
                    measurement = (
                        f"{objective['dial']['iterations']} smoothing passes"
                        f"\nanneal={objective['dial']['anneal']}"
                    )
                elif alg == "tube_mppi":
                    measurement = (
                        f"Two KL plans"
                        f"\nreset threshold={objective['selection']['reset_threshold']:g}"
                    )
                elif alg == "robust_mppi":
                    measurement = (
                        f"Augmented objective pair"
                        f"\nselection α={objective['selection']['free_energy_threshold']:g}"
                    )
                else:
                    measurement = f"Covariance steering\nresidual={objective['cc']['residual']:.2g}"

            rows.append(
                status.replace("_", " ") + "\n" + measurement + "\n" + record["decision_id"][-8:]
            )
            audit.append(
                dict(
                    algorithm=alg,
                    stage=stage,
                    status=status,
                    decision_id=record["decision_id"],
                    measurement=measurement,
                )
            )
        cells.append(rows)
    table = ax.table(
        cellText=cells,
        rowLabels=list(NAMES.values()),
        colLabels=[
            "1 Dynamics + Task",
            "2 Distributional Objective",
            "3 Sample + Refine",
            "4 Weight + Update",
            "5 Feedback + Execute",
        ],
        cellLoc="center",
        loc="center",
        bbox=[0.03, 0.09, 0.97, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for row in range(6):
        for col in range(5):
            table[row + 1, col].set_facecolor(STATUS_COLORS[audit[row * 5 + col]["status"]])
    ax.legend(
        handles=[
            Patch(color=color, label=status.replace("_", " "))
            for status, color in STATUS_COLORS.items()
        ],
        loc="lower center",
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.025),
    )
    finish(
        fig,
        "pipeline_overview.png",
        "Five-stage evidence coverage · not a leaderboard",
        "Stages 1–4: shared frozen decision. Stage 5: step 9 of each short execution "
        "(inputs have diverged). Cell suffix identifies its saved trace.\n"
        "Statuses concern evidence/mechanism activation, not quality. Historical traces "
        "remain partial; missing-field lists are retained in index.json.",
        list(frozen.values()) + closed,
    )
    provenance = {
        "schema_version": "1.0",
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plot_data_sha256": hashlib.sha256((output / "plot_data.json").read_bytes()).hexdigest(),
        "figures": sources,
        "coverage_cells": audit,
    }
    (destination / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return sources
