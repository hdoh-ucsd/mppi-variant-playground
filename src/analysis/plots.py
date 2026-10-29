"""Seminar figures reconstructed from saved numerical records and declared models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from analysis.analytic_gaussian import quadratic_cost
from analysis.covariance_geometry import covariance
from evaluation.runner import git_info, write_json
from visualization.figures import save, style

COLORS = ["#2563eb", "#d97706", "#059669", "#9333ea", "#e11d48"]


def publish(fig, root, name, **data):
    fig.tight_layout()
    out = Path(root) / "figures"
    out.mkdir(exist_ok=True)
    save(fig, out, name)
    np.savez_compressed(out / f"{name}_data.npz", **data)


def ellipse(ax, mean, cov, color, **kwargs):
    values, vectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
    ax.add_patch(
        Ellipse(
            mean,
            4 * np.sqrt(values[-1]),
            4 * np.sqrt(values[0]),
            angle=angle,
            fill=False,
            edgecolor=color,
            **kwargs,
        )
    )


def band(ax, x, group, metric, label, color):
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = group[f"{metric}_mean"].to_numpy()[order]
    lo, hi = [group[f"{metric}_{q}"].to_numpy()[order] for q in ("q05", "q95")]
    ax.plot(x, y, label=label, color=color)
    ax.fill_between(x, lo, hi, color=color, alpha=0.1)


def gaussian_figures(root, config, cases, trials, summary):
    merged = summary.merge(pd.DataFrame(cases), on="case_id")
    if (merged.suite == "geometry").any() or (merged.suite == "budget").any():
        covs = [
            covariance(r, a, reference_trace=config["reference_trace"])
            for r, a in ((1, 0), (4, 0), (4, 45))
        ]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, cov, name, color in zip(
            axes, covs, ("Isotropic", "Diagonal", "Rotated full"), COLORS, strict=False
        ):
            ellipse(ax, [0, 0], cov, color, linewidth=2)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            ax.set(
                xlim=(-2.5, 2.5),
                ylim=(-2.5, 2.5),
                title=name,
                xlabel="Latent input 1 [dimensionless]",
                ylabel="Latent input 2 [dimensionless]",
            )
            ax.set_aspect("equal")
        fig.suptitle("1 · Covariance geometry: equal trace, two-standard-deviation contours")
        publish(fig, root, "01_covariance_geometry", covariances=covs)
        sample_path = root / "samples.csv"
        if sample_path.stat().st_size > 1:
            samples = pd.read_csv(sample_path)
            ids = samples.case_id.unique()
            ids = [
                i
                for i in ids
                if cases[int(i)]["proposal_name"] in {"isotropic", "diagonal_r4", "full_r4_a45"}
            ] or list(ids[:3])
            fig, axes = plt.subplots(1, len(ids), figsize=(4.5 * len(ids), 4), squeeze=False)
            xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 130), np.linspace(-2.5, 2.5, 130))
            points = np.stack([xx, yy], axis=-1)
            for ax, case_id, color in zip(axes[0], ids, COLORS, strict=False):
                c = cases[int(case_id)]
                selected = samples[samples.case_id == case_id]
                costs = quadratic_cost(points, c["hessian"], c["cost_center"])
                ax.contour(
                    xx, yy, costs, levels=[0.1, 0.5, 1, 2, 4, 8], colors="#94a3b8", linewidths=0.6
                )
                ax.scatter(selected.z0, selected.z1, s=5, alpha=0.2, color=color)
                ellipse(ax, c["proposal_mean"], c["proposal_covariance"], color)
                ax.scatter(
                    *c["target_mean"], marker="*", s=120, color="black", label="Exact target mean"
                )
                estimate_row = trials[
                    (trials.case_id == case_id) & (trials.seed == selected.seed.iloc[0])
                ].iloc[0]
                ax.scatter(
                    estimate_row.estimate_0,
                    estimate_row.estimate_1,
                    marker="x",
                    s=65,
                    color="#dc2626",
                    label="SNIS estimate",
                )
                ax.set(
                    xlabel="z₁ [dimensionless]",
                    ylabel="z₂ [dimensionless]",
                    title=c["proposal_name"],
                    xlim=(-2.5, 2.5),
                    ylim=(-2.5, 2.5),
                )
                ax.set_aspect("equal")
            axes[0, 0].legend(fontsize=8, loc="lower left")
            fig.suptitle("2 · Frozen proposal, rotated quadratic cost, exact and sampled means")
            publish(
                fig,
                root,
                "02_rotated_quadratic",
                grid_x=xx,
                grid_y=yy,
                plotted_case_ids=ids,
                sample_records=samples.to_numpy(dtype=float),
            )
    geometry = merged[
        (merged.suite == "geometry")
        & (merged.constraint == "trace")
        & (merged.scale == 1)
        & merged.proposal_mean.apply(lambda m: m == [0.0, 0.0])
    ]
    if not geometry.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        iso = geometry[geometry.ratio == 1].iloc[0]
        axes[0].axhline(iso.error_mean, color=COLORS[0], ls="--", label="Isotropic")
        plot_angles = []
        for ratio, color in zip(sorted(set(geometry.ratio) - {1}), COLORS[1:], strict=False):
            group = geometry[geometry.ratio == ratio]
            # Signed relative major-axis angle, modulo 180 degrees.
            target = np.array(group.iloc[0].target_covariance)
            _, vectors = np.linalg.eigh(target)
            theta = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
            x = (group.angle_deg.to_numpy() - theta + 90) % 180 - 90
            band(axes[0], x, group, "error", f"Eigenvalue ratio {ratio}:1", color)
            plot_angles.extend(zip(x, group.error_mean, strict=True))
        axes[0].set(
            xlabel="Proposal minus target major-axis angle [degrees]",
            ylabel="Mean estimation error ‖μ̂ − μ*‖",
            title="B · Fixed cost, rotated proposal",
        )
        rotated = merged[merged.suite == "cost_rotation"]
        for (name, group), color in zip(rotated.groupby("proposal_name"), COLORS, strict=False):
            band(axes[1], group.cost_angle_deg, group, "error", name, color)
        axes[1].set(
            xlabel="Cost Hessian rotation [degrees]",
            ylabel="Mean estimation error ‖μ̂ − μ*‖",
            title="A · Fixed proposals, rotated cost",
        )
        for ax in axes:
            ax.legend(fontsize=8)
        fig.suptitle("3 · Alignment: curves are means; shading is the 5–95% trial range")
        publish(
            fig,
            root,
            "03_alignment_sweep",
            relative_angles_and_error=plot_angles,
            geometry_case_ids=geometry.case_id.to_numpy(),
        )
        fig, axes = plt.subplots(2, len(config["scales"]), figsize=(13, 7), squeeze=False)
        geometry_all = merged[
            (merged.suite == "geometry") & merged.proposal_mean.apply(lambda m: m == [0.0, 0.0])
        ]
        for i, constraint in enumerate(config["constraints"]):
            for j, scale in enumerate(config["scales"]):
                ax = axes[i, j]
                group = geometry_all[
                    (geometry_all.constraint == constraint) & (geometry_all.scale == scale)
                ]
                for ratio, color in zip(config["anisotropy_ratios"], COLORS, strict=False):
                    part = group[group.ratio == ratio]
                    if ratio == 1:
                        ax.axhline(part.error_mean.iloc[0], color=color, ls="--", label="Isotropic")
                    else:
                        order = part.sort_values("angle_deg")
                        ax.plot(
                            order.angle_deg, order.error_mean, color=color, label=f"Ratio {ratio}"
                        )
                ax.set(
                    title=f"Matched {constraint} · scale {scale}",
                    xlabel="Proposal angle [degrees]",
                    ylabel="Mean estimation error",
                )
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("3b · Scale constraints are separate experiments")
        publish(fig, root, "03b_scale_constraints", case_ids=geometry_all.case_id.to_numpy())
    budget = merged[merged.suite == "budget"]
    if not budget.empty:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for (name, group), color in zip(budget.groupby("proposal_name"), COLORS, strict=False):
            band(axes[0, 0], group.samples, group, "error", name, color)
            band(axes[0, 1], group.samples, group, "ess", name, color)
            group = group.sort_values("samples")
            axes[1, 0].plot(group.samples, group.estimator_variance_trace, color=color, label=name)
        count = min(config["sample_counts"], key=lambda k: abs(k - 256))
        selected = budget[budget.samples == count]
        values = [trials[trials.case_id == c].error.to_numpy() for c in selected.case_id]
        axes[1, 1].boxplot(values, tick_labels=selected.proposal_name, showfliers=True)
        axes[1, 1].tick_params(axis="x", labelrotation=35, labelsize=7)
        for ax, label in zip(
            axes.flat[:3],
            ("Mean estimation error", "ESS [samples]", "Estimator variance trace"),
            strict=True,
        ):
            ax.set(xscale="log", xlabel="Sample budget K [cost evaluations]", ylabel=label)
        axes[1, 1].set(title=f"Error distribution · K={count}", ylabel="‖μ̂ − μ*‖")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("4 · Finite-sample accuracy and variability; shading: 5–95% trial range")
        publish(
            fig,
            root,
            "04_sample_efficiency",
            summary_case_ids=budget.case_id.to_numpy(),
            error_distributions=np.array(values),
        )
    temp = merged[merged.suite == "temperature"]
    if not temp.empty:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for (name, group), color in zip(temp.groupby("proposal_name"), COLORS, strict=False):
            for ax, metric in zip(
                axes.flat, ("ess", "max_weight", "error", "weight_entropy"), strict=True
            ):
                band(ax, group.temperature, group, metric, name, color)
        for ax, ylabel in zip(
            axes.flat,
            ("ESS [samples]", "Maximum normalized weight", "‖μ̂ − μ*(λ)‖", "Weight entropy [nats]"),
            strict=True,
        ):
            ax.set(xscale="log", xlabel="Temperature λ [cost units]", ylabel=ylabel)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("5 · Each temperature defines a different analytic target")
        publish(fig, root, "05_temperature_weights", case_ids=temp.case_id.to_numpy())
        dimension = merged[merged.suite == "dimension"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, cause in zip(axes, ("cost_only", "ratio_only", "combined"), strict=True):
            for (temperature, group), color in zip(
                dimension[dimension.cause == cause].groupby("temperature"), COLORS, strict=False
            ):
                group = group.sort_values("dimension")
                ax.plot(
                    group.dimension,
                    group.ess_mean / group.samples,
                    label=f"λ={temperature}",
                    color=color,
                )
            ax.set(
                xlabel="Control-sequence dimension d",
                ylabel="Mean ESS / K",
                title=cause.replace("_", " "),
            )
            ax.legend(fontsize=8)
        fig.suptitle("5b · Separate cost concentration from high-dimensional density ratios")
        publish(fig, root, "05b_dimension_diagnostics", case_ids=dimension.case_id.to_numpy())


def generate(input_path):
    root = Path(input_path)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Cannot plot an incomplete study")
    import yaml

    config = yaml.safe_load((root / "config.yaml").read_text())
    cases = json.loads((root / "cases.json").read_text())
    trials = pd.read_csv(root / "trials.csv")
    summary = pd.read_csv(root / "summary.csv")
    style()
    if manifest["level"] == "analytic_gaussian":
        gaussian_figures(root, config, cases, trials, summary)
    else:
        from analysis.extension_plots import extension_figures

        extension_figures(root, config, cases, trials, summary, manifest["level"])
    write_json(
        root / "figure_provenance.json",
        {
            "git": git_info(),
            "source_sha256": manifest["source_sha256"],
            "inputs": ["config.yaml", "cases.json", "trials.csv", "summary.csv"],
            "figures": sorted(path.name for path in (root / "figures").glob("*.svg")),
        },
    )
    print(f"Figures reconstructed from saved data: {root / 'figures'}")
