"""Evidence map with an explicit boundary between static estimation and MPC transfer."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.artifacts import begin
from analysis.plots import publish
from evaluation.runner import write_json
from visualization.figures import style


def collect(study_root, output):
    study_root = Path(study_root)
    manifests = {}
    for name in ("gaussian", "temperature", "saturation", "nonconvex", "closed_loop"):
        manifests[name] = json.loads((study_root / f"{name}_v1/manifest.json").read_text())
        if not manifests[name]["complete"]:
            raise ValueError(f"Incomplete input: {name}")
    output, manifest = begin(
        output, {"phase": "evaluation", "study_root": str(study_root)}, "evidence_summary"
    )
    cases = pd.DataFrame(json.loads((study_root / "gaussian_v1/cases.json").read_text()))
    summary = pd.read_csv(study_root / "gaussian_v1/summary.csv").merge(cases, on="case_id")
    selected = summary[
        (summary.suite == "budget")
        & (summary.samples == 256)
        & summary.proposal_name.isin(["isotropic", "full_r4_a45", "full_r4_a-45"])
    ]
    analytic = selected[
        ["proposal_name", "target_proposal_kl", "ess_mean", "squared_error_mean"]
    ].to_dict("records")
    loop_cases = pd.DataFrame(json.loads((study_root / "closed_loop_v1/cases.json").read_text()))
    loop = pd.read_csv(study_root / "closed_loop_v1/trials.csv").merge(
        loop_cases, on="case_id", suffixes=("", "_case")
    )
    loop = loop[
        (loop.samples == 384) & (loop.scale == 1) & loop.proposal_name.isin(["isotropic", "r4_a45"])
    ]
    control = [
        {
            "scenario": scenario,
            "proposal_name": name,
            "success_count": int(group.success.sum()),
            "n": len(group),
        }
        for (scenario, name), group in loop.groupby(["scenario", "proposal_name"])
    ]
    evidence = {
        "analytic_K": 256,
        "analytic": analytic,
        "closed_loop": control,
        "source_manifests": manifests,
        "boundary": "Static and bicycle targets differ; causal mediation remains unestablished.",
    }
    write_json(output / "evidence.json", evidence)
    pd.DataFrame(analytic).to_csv(output / "analytic_evidence.csv", index=False)
    pd.DataFrame(control).to_csv(output / "control_evidence.csv", index=False)
    manifest["complete"] = True
    write_json(output / "manifest.json", manifest)
    (output / "interpretation.md").write_text(
        "# Evidence boundary\n\nThe analytic study connects proposal geometry, target overlap, "
        "weights and target-mean estimation on a controlled problem. The control experiment "
        "tests separately whether an input covariance gives useful behavior under nonlinear "
        "dynamics, clipping and warm starts. The displayed numbers support only those "
        "specific problems and settings; arrows are an evidence map, not a causal proof.\n"
    )
    return evidence


def render(output):
    output = Path(output)
    evidence = json.loads((output / "evidence.json").read_text())
    analytic = {row["proposal_name"]: row for row in evidence["analytic"]}
    iso, aligned, opposite = [
        analytic[name] for name in ("isotropic", "full_r4_a45", "full_r4_a-45")
    ]
    style()
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    rows = [
        (
            "Geometry and overlap",
            f"Equal trace 1.2; ratio 4 at +45° / −45°\n"
            f"KL(target || proposal): {aligned['target_proposal_kl']:.2f} / "
            f"{opposite['target_proposal_kl']:.2f}",
        ),
        (
            "Importance weights",
            f"K=256, mean ESS: isotropic {iso['ess_mean']:.1f}; "
            f"+45° {aligned['ess_mean']:.1f}; −45° {opposite['ess_mean']:.1f}",
        ),
        (
            "Mean-update accuracy",
            f"Observed MSE: isotropic {iso['squared_error_mean']:.4f}; "
            f"+45° {aligned['squared_error_mean']:.4f}; −45° {opposite['squared_error_mean']:.4f}\n"
            "64 paired trials; fixed Gaussian target and proposal mean",
        ),
    ]
    for i, (title, text) in enumerate(rows):
        y = 0.86 - i * 0.2
        ax.text(
            0.5,
            y,
            title + "\n" + text,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.6", "fc": "#eff6ff", "ec": "#2563eb"},
        )
        if i < 2:
            ax.annotate(
                "",
                xy=(0.5, y - 0.12),
                xytext=(0.5, y - 0.075),
                arrowprops={"arrowstyle": "->", "color": "#2563eb"},
            )
    ax.axhline(0.33, color="#64748b", ls="--")
    ax.text(
        0.5,
        0.30,
        "Separate transfer test: nonlinear dynamics + clipping + warm start",
        ha="center",
        fontsize=11,
    )
    pairs = {}
    for row in evidence["closed_loop"]:
        pairs.setdefault(row["scenario"], {})[row["proposal_name"]] = (
            f"{row['success_count']}/{row['n']}"
        )
    text = "  ·  ".join(
        f"{name}: iso {values['isotropic']}, +45° {values['r4_a45']}"
        for name, values in pairs.items()
    )
    ax.text(
        0.5,
        0.19,
        "Goal-pose completion, K=384, trace=0.44\n" + text,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.6", "fc": "#fff7ed", "ec": "#c2410c"},
    )
    ax.text(0.5, 0.06, evidence["boundary"], ha="center", fontsize=10, color="#475569")
    fig.suptitle("9 · Which links are supported, and where transfer remains a separate question")
    publish(
        fig,
        output,
        "09_mechanism_evidence",
        analytic_values=np.array(
            [
                [r["target_proposal_kl"], r["ess_mean"], r["squared_error_mean"]]
                for r in evidence["analytic"]
            ]
        ),
        control_counts=np.array([[r["success_count"], r["n"]] for r in evidence["closed_loop"]]),
    )
