"""Numerical interpretation generated only from recorded frozen-decision evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as student_t

from analysis.frozen_decision import PROPOSALS
from evaluation.runner import git_info, write_json


def mean_interval(values):
    values = np.asarray(values)
    half = student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()), float(half)


def markdown_table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def generate_report(root):
    root = Path(root)
    frame = pd.read_csv(root / "per_seed.csv")
    config = yaml.safe_load((root / "config.yaml").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    references = json.loads((root / "reference_status.json").read_text())
    states = list(frame.state.unique())
    primary = frame[frame.samples == config["primary_samples"]]
    orientation = []
    for (state, count), group in frame.groupby(["state", "samples"], sort=False):
        aligned = group[group.proposal == "aligned"].set_index("seed")
        misaligned = group[group.proposal == "misaligned"].set_index("seed")
        delta = aligned.nominal_objective - misaligned.nominal_objective
        m, half = mean_interval(delta)
        orientation.append(
            dict(
                state=state,
                samples=int(count),
                mean_objective_difference=m,
                ci95_halfwidth=half,
                aligned_lower_n=int((delta < 0).sum()),
                misaligned_lower_n=int((delta > 0).sum()),
            )
        )
    orientation = pd.DataFrame(orientation)
    orientation.to_csv(root / "orientation_comparison.csv", index=False)
    convergence = []
    for (state, proposal), group in frame[frame.proposal != "standard"].groupby(
        ["state", "proposal"]
    ):
        low = group[group.samples == min(config["sample_counts"])].set_index("seed")
        high = group[group.samples == max(config["sample_counts"])].set_index("seed")
        for metric in ("paired_mean_distance_rms", "paired_first_command_distance"):
            delta = high[metric] - low[metric]
            m, half = mean_interval(delta)
            convergence.append(
                dict(
                    state=state,
                    proposal=proposal,
                    metric=metric,
                    low_K=int(low.samples.iloc[0]),
                    high_K=int(high.samples.iloc[0]),
                    low_mean=float(low[metric].mean()),
                    high_mean=float(high[metric].mean()),
                    paired_change_mean=m,
                    ci95_halfwidth=half,
                )
            )
    pd.DataFrame(convergence).to_csv(root / "convergence_summary.csv", index=False)
    different_raw_weight = int(primary.best_is_highest_weight.eq(False).sum())
    orientation_counts = []
    for n in config["repetition_counts"]:
        for state, group in primary.groupby("state", sort=False):
            a = group[group.proposal == "aligned"].sort_values("seed").head(n)
            m = group[group.proposal == "misaligned"].sort_values("seed").head(n)
            delta = a.nominal_objective.to_numpy() - m.nominal_objective.to_numpy()
            mean, half = mean_interval(delta)
            orientation_counts.append(
                dict(
                    state=state,
                    repetitions=len(delta),
                    mean_objective_difference=mean,
                    ci95_halfwidth=half,
                    aligned_lower_n=int((delta < 0).sum()),
                    misaligned_lower_n=int((delta > 0).sum()),
                )
            )
    pd.DataFrame(orientation_counts).to_csv(
        root / "orientation_repetition_sensitivity.csv", index=False
    )
    lines = [
        "# Frozen-state MPPI decision findings",
        "",
        "**Measured conclusion:** changing only the covariance proposal changes finite-sample "
        "controls and predicted paths at the same state and warm start. The high-sample references "
        "remain weight-degenerate, so these runs do not resolve target-mean estimation accuracy "
        "or demonstrate convergence to the Gibbs mean. Local covariance orientation can lower "
        "the deterministic updated-plan objective in aggregate while reversing on individual "
        "seeds.",
        "",
        "## Executed protocol and common target",
        "",
        f"The study executed **{manifest['decisions']:,} frozen decisions**, six trace-matched "
        f"proposals, {config['seeds']['count']} independent paired evaluation seeds, three states "
        f"and K={config['sample_counts']}. Evaluation used {manifest['evaluation_rollouts']:,} "
        f"sampled trajectory objectives and {manifest['evaluation_dynamics_evaluations']:,} scalar "
        f"dynamics transitions, including updated nominal predictions. Reference computation used "
        f"{manifest['reference_rollouts']:,} additional trajectories. There are "
        f"{manifest['snapshots']:,} full-population snapshots, including every primary-budget "
        "seed.",
        "",
        "All proposals use the same observed state, latent warm start, previous physical control, "
        "vehicle, cost, temperature 5, horizon 28 and base diag(.28,.16). The six proposal traces "
        "equal .44 in normalized latent coordinates. The correction is exp(-J/λ)p/q, evaluated "
        "in log space. Shared PCG64 normals are transformed by each proposal Cholesky factor. "
        "Evaluation namespace 101 and reference namespace 202 are disjoint; "
        "K uses nested prefixes. "
        "`audit.json` verifies saved inputs, transforms, ratios, weights, updates, objectives and "
        "endpoint mass. This verifies a common mathematical target; it does not certify its "
        "numerical mean has been estimated accurately.",
        "",
        "## Frozen states selected before proposal evaluation",
        "",
    ]
    rows = []
    for state in states:
        frozen = json.loads((root / "frozen_states" / f"{state}.json").read_text())
        rows.append(
            [
                state,
                frozen["step"],
                ", ".join(f"{v:.6f}" for v in frozen["state"]),
                frozen["source_seed"],
            ]
        )
    lines.extend(
        [
            markdown_table(
                ["State", "Planning step", "x, y, heading, speed [m,m,rad,m/s]", "Reference seed"],
                rows,
            ),
            "",
            "Open space uses step 8 while moving. Pre-obstacle uses the first state in x∈[.7,1.0] "
            "with two strictly collision-free deterministic grid witnesses crossing x=4 m within "
            "the 4.2-second horizon. The chosen state is about 3.3 m before the obstacle center. "
            "These geometric checks preceded covariance evaluation; the initial later-state pilot "
            "could not verify both routes and was rejected. Post-disturbance uses step 21, "
            "the first observation after the fixed step-20 lateral/heading/speed impulse. "
            "The exact warm starts, previous commands, resolved configurations, selection rules, "
            "disturbance metadata and source prefixes are in `frozen_states/`.",
            "",
        ]
    )
    pre = root / "frozen_states/pre_obstacle.json"
    if pre.exists():
        witnesses = json.loads(pre.read_text())["route_witnesses"]
        lines.extend(
            [
                markdown_table(
                    ["Route witness", "Station crossing y [m]", "Minimum swept clearance [m]"],
                    [
                        [name, f"{v['station_y']:.6f}", f"{v['clearance']:.6f}"]
                        for name, v in witnesses.items()
                    ],
                ),
                "",
                "Route availability does not imply equal route probability, equal cost, "
                "or coverage "
                "by finite random samples around this warm start.",
                "",
            ]
        )
    rows = [
        [
            state,
            f"{v['pooled_ess']:.3f}",
            f"{100 * v['pooled_max_weight']:.2f}%",
            f"{v['rms_ci95_halfwidth']:.3f}",
            v["status"],
        ]
        for state, v in references.items()
    ]
    lines.extend(
        [
            "## The independent high-sample mean is unresolved",
            "",
            f"Each state has {config['reference_repetitions']} independent batches of "
            f"{config['reference_samples']:,} rollouts. Pooled means use unnormalized log weight "
            "masses, rather than equally averaging batch SNIS estimates. Delete-one-batch "
            "jackknife intervals describe block variability only, not unknown importance bias.",
            "",
            markdown_table(
                ["State", "Pooled ESS", "Largest weight", "RMS CI95 halfwidth [latent]", "Status"],
                rows,
            ),
            "",
            "Every reference fails the predeclared reliability screens. Consequently, "
            "`reference_distance_rms` and the first-command distance are descriptive distances "
            "to an uncertain estimate, **not validated estimation errors**. Increasing a nominal "
            "budget cannot be declared accurate by proximity to this reference. This negative "
            "result is retained rather than hidden by a fallback or a changed temperature.",
            "",
            "## First-command changes, weights and updated-plan objective",
            "",
            "The table gives paired physical command changes from standard, with pointwise 95% "
            "t interval halfwidths across optimizer seeds. Commands are clipped before comparison. "
            "Figures B also show normalized units, complete paired-seed distributions and physical "
            "secondary axes. A small signed average can conceal large seed-wise differences.",
            "",
        ]
    )
    rows = []
    for state in states:
        for proposal in PROPOSALS:
            group = primary[(primary.state == state) & (primary.proposal == proposal)]
            a, ah = mean_interval(group.delta_acceleration)
            s, sh = mean_interval(group.delta_steering)
            rows.append(
                [
                    state,
                    proposal,
                    f"{a:+.4f} ± {ah:.4f}",
                    f"{s:+.4f} ± {sh:.4f}",
                    f"{group.ess.mean():.3f}",
                    f"{group.nominal_objective.mean():.2f}",
                    f"{group.reference_distance_rms.mean():.4f}",
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "State",
                    "Proposal",
                    "Δ acceleration [m/s²]",
                    "Δ steering [rad]",
                    "Mean ESS",
                    "Mean J(mean)",
                    "RMS reference distance",
                ],
                rows,
            ),
            "",
            f"At K={config['primary_samples']}, minimum raw cost and maximum corrected weight "
            f"identify different rollouts in **{different_raw_weight}"
            f"/{len(primary)} decisions**. The weighted cloud can collapse while unweighted "
            "trajectories remain broad. Gibbs mean proximity and deterministic mean-plan cost "
            "are different objectives. Saturation, entropy, actual perturbation moments, raw "
            "minimum cost and corrected weights remain recorded separately.",
            "",
            "## Do differences shrink with more samples?",
            "",
            "These reference-independent distances compare each nonstandard proposal with its "
            "paired standard decision. Average across the five alternatives within each seed, "
            "then compare the smallest and largest K across paired seeds. This avoids treating "
            "proposals as independent experimental repetitions.",
            "",
        ]
    )
    rows = []
    for state in states:
        group = frame[(frame.state == state) & (frame.proposal != "standard")]
        g = group.groupby(["seed", "samples"])["paired_mean_distance_rms"].mean().unstack()
        low, high = min(config["sample_counts"]), max(config["sample_counts"])
        m, h = mean_interval(g[high] - g[low])
        rows.append([state, f"{g[low].mean():.4f}", f"{g[high].mean():.4f}", f"{m:+.4f} ± {h:.4f}"])
    lines.extend(
        [
            markdown_table(
                [
                    "State",
                    "Mean paired RMS distance at min K",
                    "At max K",
                    "Paired change ± CI95 halfwidth",
                ],
                rows,
            ),
            "",
            "There is no observed systematic collapse of covariance-induced mean differences "
            "over this K range. Figure D shows nonmonotonic paired distances, ESS and mean-plan "
            "objectives. This finite-budget observation neither disproves asymptotic consistency "
            "nor establishes a convergence rate. `convergence_summary.csv` retains each proposal "
            "and first-command distances; `repetition_sensitivity.csv` reports the first 10,25,50 "
            "preordered seeds and their uncertainty.",
            "",
            "## Does favorable local orientation help, and where does it reverse?",
            "",
            "Alignment is the cost-descent direction for a constant two-channel latent offset, "
            "estimated by local vehicle/cost sensitivity. The perpendicular intervention uses the "
            "same eigenvalues (.34,.10), trace and determinant. `proposals.json` saves both finite "
            "difference resolutions, stability and eigenvectors. It is not an optimal importance "
            "covariance, and local gradient alignment does not identify the target's "
            "56-dimensional "
            "covariance or its mean-estimation variance.",
            "",
        ]
    )
    op = orientation[orientation.samples == config["primary_samples"]]
    lines.extend(
        [
            markdown_table(
                [
                    "State",
                    "Mean J(aligned)−J(misaligned) ± CI95 halfwidth",
                    "Aligned lower",
                    "Misaligned lower",
                ],
                [
                    [
                        r.state,
                        f"{r.mean_objective_difference:+.3f} ± {r.ci95_halfwidth:.3f}",
                        int(r.aligned_lower_n),
                        int(r.misaligned_lower_n),
                    ]
                    for r in op.itertuples()
                ],
            ),
            "",
            "The aligned proposal has lower average primary-budget predicted-plan cost in these "
            "three states. Individual paired seeds reverse that effect. In open space and "
            "pre-obstacle the primary-budget cost-difference intervals include zero; the obstacle "
            "mean is sensitive to a rare collision penalty. Post-disturbance provides the clearer "
            "primary-budget favorable example, while seed reversals are negative examples. "
            "The per-K table and `orientation_repetition_sensitivity.csv` retain all "
            "predeclared subsets, without selecting "
            "a favorable K or discarding reversals. These comparisons do not establish better "
            "target-mean accuracy, global superiority of anisotropy or closed-loop robustness.",
            "",
        ]
    )
    lines.extend(
        [
            "Seed-count sensitivity for the same paired orientation contrast:",
            "",
            markdown_table(
                ["State", "First n seeds", "Mean ΔJ ± CI95 halfwidth"],
                [
                    [
                        r["state"],
                        r["repetitions"],
                        f"{r['mean_objective_difference']:+.3f} ± {r['ci95_halfwidth']:.3f}",
                    ]
                    for r in orientation_counts
                ],
            ),
            "",
        ]
    )
    collisions = primary[primary.min_predicted_clearance <= 0]
    if len(collisions):
        lines.extend(
            [
                "The updated nominal prediction can intersect an obstacle. This is an "
                "open-loop mathematical consequence, not an executed collision:",
                "",
                markdown_table(
                    ["Proposal", "Seed", "Minimum predicted swept clearance [m]"],
                    [
                        [r.proposal, r.seed, f"{r.min_predicted_clearance:.8f}"]
                        for r in collisions.itertuples()
                    ],
                ),
                "",
            ]
        )
    latency = primary.planning_latency_s.mean() * 1000
    lines.extend(
        [
            f"Primary-budget mean recorded planning latency is {latency:.3f} ms on the "
            "manifest's single-thread CPU environment. This includes snapshot observation overhead "
            "and excludes construction, extra mean-objective scoring, serialization and plotting. "
            "Every proposal uses the same (K+1)T dynamics budget; references, capture prefixes, "
            "route witnesses and local sensitivity are separate work.",
            "",
            "## Figure paths and regeneration",
            "",
            "Each figure has `.png`, `.svg` and `_data.npz` companions. All spatial/color domains "
            "within comparisons are shared. Weighted endpoints are an empirical optimizer-weighted "
            "distribution, not physical uncertainty. F is a disturbance-free open-loop prediction.",
            "",
        ]
    )
    suffixes = (
        "A_decision_microscope",
        "B_first_command_difference",
        "C_weighted_endpoint_density",
        "D_budget_estimation",
        "E_decision_dashboard",
        "F_counterfactual_consequence",
    )
    lines.append(
        markdown_table(
            ["State", "Figure", "PNG path"],
            [
                [state, suffix[0], f"[figures/{state}_{suffix}.png](figures/{state}_{suffix}.png)"]
                for state in states
                for suffix in suffixes
            ],
        )
    )
    lines.extend(
        [
            "",
            "```bash",
            "uv sync --frozen",
            "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg",
            "uv run python experiments/legacy/frozen/frozen_state_decision.py \\\n"
            "  --output runs/legacy/investigations/frozen_decision_reproduction",
            "uv run python experiments/legacy/frozen/frozen_state_decision.py \\\n"
            "  --output runs/legacy/investigations/frozen_decision_v1 --plots-only",
            "uv run python experiments/legacy/frozen/verify_frozen_decision.py \\\n"
            "  --input runs/legacy/investigations/frozen_decision_v1",
            "uv run ruff check .",
            "uv run ruff format --check .",
            "uv run pytest",
            "```",
            "",
            "## Provenance and limits",
            "",
            f"Evaluated clean source commit: `{manifest['git']['commit']}`. Branch: "
            "`frozen-state-decision-analysis`; it is not merged into main. Source/config hashes, "
            "versions, platform and RNG namespaces are in `manifest.json`. Plot and report "
            "provenance separately record postprocessing sources. Prior experiments and "
            "baseline configurations are unchanged.",
            "",
            "Only three deterministic reference states and one reference warm start per state "
            "were studied. The local two-channel covariance is repeated independently over time; "
            "it does not span arbitrary temporal covariance. Trace matching does not match "
            "determinant or clipped perturbation energy. Reference uncertainty remains unresolved. "
            "The t intervals "
            "are pointwise descriptive comparisons with no multiple-comparison adjustment. Seeds "
            "describe optimizer variation, not physical disturbance uncertainty. State-space means "
            "and control-space means differ under nonlinear dynamics. Other non-primary budgets "
            "retain all updated means and scalar records, with full clouds for the representative "
            "seed only; every primary-budget population is archived. "
            "Timing has observation overhead. "
            "No complete evaluation episode is executed and no safety, navigation or general "
            "covariance-family superiority claim follows.",
            "",
        ]
    )
    (root / "interpretation.md").write_text("\n".join(lines))
    write_json(
        root / "report_provenance.json",
        dict(
            git=git_info(),
            report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            inputs=[
                "per_seed.csv",
                "manifest.json",
                "config.yaml",
                "reference_status.json",
                "frozen_states/",
            ],
            mode="Saved-data interpretation; no optimization or sampling",
        ),
    )
