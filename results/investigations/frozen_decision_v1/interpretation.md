# Frozen-state MPPI decision findings

**Measured conclusion:** changing only the covariance proposal changes finite-sample controls and predicted paths at the same state and warm start. The high-sample references remain weight-degenerate, so these runs do not resolve target-mean estimation accuracy or demonstrate convergence to the Gibbs mean. Local covariance orientation can lower the deterministic updated-plan objective in aggregate while reversing on individual seeds.

## Executed protocol and common target

The study executed **7,200 frozen decisions**, six trace-matched proposals, 50 independent paired evaluation seeds, three states and K=[64, 128, 256, 384, 512, 1024, 2048, 4096]. Evaluation used 7,660,800 sampled trajectory objectives and 214,704,000 scalar dynamics transitions, including updated nominal predictions. Reference computation used 1,572,864 additional trajectories. There are 1,026 full-population snapshots, including every primary-budget seed.

All proposals use the same observed state, latent warm start, previous physical control, vehicle, cost, temperature 5, horizon 28 and base diag(.28,.16). The six proposal traces equal .44 in normalized latent coordinates. The correction is exp(-J/λ)p/q, evaluated in log space. Shared PCG64 normals are transformed by each proposal Cholesky factor. Evaluation namespace 101 and reference namespace 202 are disjoint; K uses nested prefixes. `audit.json` verifies saved inputs, transforms, ratios, weights, updates, objectives and endpoint mass. This verifies a common mathematical target; it does not certify its numerical mean has been estimated accurately.

## Frozen states selected before proposal evaluation

| State | Planning step | x, y, heading, speed [m,m,rad,m/s] | Reference seed |
| --- | --- | --- | --- |
| open_space | 8 | 0.879331, -0.380471, 0.125446, 1.416284 | 31415 |
| pre_obstacle | 7 | 0.706866, -0.427944, -0.172562, 1.688686 | 31415 |
| post_disturbance | 21 | 4.942900, 1.154699, 0.270468, 1.029675 | 31415 |

Open space uses step 8 while moving. Pre-obstacle uses the first state in x∈[.7,1.0] with two strictly collision-free deterministic grid witnesses crossing x=4 m within the 4.2-second horizon. The chosen state is about 3.3 m before the obstacle center. These geometric checks preceded covariance evaluation; the initial later-state pilot could not verify both routes and was rejected. Post-disturbance uses step 21, the first observation after the fixed step-20 lateral/heading/speed impulse. The exact warm starts, previous commands, resolved configurations, selection rules, disturbance metadata and source prefixes are in `frozen_states/`.

| Route witness | Station crossing y [m] | Minimum swept clearance [m] |
| --- | --- | --- |
| lower | -1.270695 | 0.150478 |
| upper | 1.410576 | 0.063145 |

Route availability does not imply equal route probability, equal cost, or coverage by finite random samples around this warm start.

## The independent high-sample mean is unresolved

Each state has 8 independent batches of 65,536 rollouts. Pooled means use unnormalized log weight masses, rather than equally averaging batch SNIS estimates. Delete-one-batch jackknife intervals describe block variability only, not unknown importance bias.

| State | Pooled ESS | Largest weight | RMS CI95 halfwidth [latent] | Status |
| --- | --- | --- | --- | --- |
| open_space | 1.284 | 87.88% | 0.963 | unresolved |
| pre_obstacle | 1.099 | 95.31% | 0.983 | unresolved |
| post_disturbance | 2.186 | 65.77% | 0.735 | unresolved |

Every reference fails the predeclared reliability screens. Consequently, `reference_distance_rms` and the first-command distance are descriptive distances to an uncertain estimate, **not validated estimation errors**. Increasing a nominal budget cannot be declared accurate by proximity to this reference. This negative result is retained rather than hidden by a fallback or a changed temperature.

## First-command changes, weights and updated-plan objective

The table gives paired physical command changes from standard, with pointwise 95% t interval halfwidths across optimizer seeds. Commands are clipped before comparison. Figures B also show normalized units, complete paired-seed distributions and physical secondary axes. A small signed average can conceal large seed-wise differences.

| State | Proposal | Δ acceleration [m/s²] | Δ steering [rad] | Mean ESS | Mean J(mean) | RMS reference distance |
| --- | --- | --- | --- | --- | --- | --- |
| open_space | standard | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | 1.572 | 226.13 | 0.6079 |
| open_space | isotropic | -0.0308 ± 0.0831 | +0.0044 ± 0.0219 | 1.518 | 225.65 | 0.6123 |
| open_space | diagonal | -0.1119 ± 0.1151 | -0.0018 ± 0.0263 | 1.695 | 226.22 | 0.6111 |
| open_space | full | -0.1244 ± 0.1632 | +0.0078 ± 0.0337 | 1.725 | 228.06 | 0.5929 |
| open_space | aligned | -0.0409 ± 0.1517 | -0.0056 ± 0.0525 | 1.854 | 221.98 | 0.5983 |
| open_space | misaligned | -0.0816 ± 0.1200 | -0.0039 ± 0.0259 | 1.737 | 225.75 | 0.6157 |
| pre_obstacle | standard | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | 1.562 | 269.33 | 0.6076 |
| pre_obstacle | isotropic | -0.1301 ± 0.1353 | +0.0125 ± 0.0208 | 1.580 | 281.30 | 0.6056 |
| pre_obstacle | diagonal | +0.1162 ± 0.1836 | +0.0008 ± 0.0272 | 1.493 | 262.48 | 0.6079 |
| pre_obstacle | full | -0.0397 ± 0.2013 | +0.0192 ± 0.0330 | 1.600 | 262.25 | 0.6039 |
| pre_obstacle | aligned | -0.0288 ± 0.2035 | +0.0237 ± 0.0305 | 1.545 | 262.67 | 0.6081 |
| pre_obstacle | misaligned | -0.1611 ± 0.2609 | -0.0161 ± 0.0568 | 1.424 | 288.53 | 0.6016 |
| post_disturbance | standard | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | 1.821 | 93.89 | 0.4883 |
| post_disturbance | isotropic | +0.0199 ± 0.1194 | -0.0512 ± 0.0231 | 1.785 | 92.22 | 0.4874 |
| post_disturbance | diagonal | -0.0030 ± 0.1206 | +0.0279 ± 0.0285 | 1.836 | 92.44 | 0.4972 |
| post_disturbance | full | -0.0551 ± 0.1712 | -0.0127 ± 0.0438 | 1.597 | 93.07 | 0.5044 |
| post_disturbance | aligned | +0.0568 ± 0.1568 | +0.0170 ± 0.0285 | 1.705 | 90.35 | 0.4962 |
| post_disturbance | misaligned | -0.1642 ± 0.2464 | -0.0523 ± 0.0604 | 1.538 | 100.33 | 0.4941 |

At K=384, minimum raw cost and maximum corrected weight identify different rollouts in **874/900 decisions**. The weighted cloud can collapse while unweighted trajectories remain broad. Gibbs mean proximity and deterministic mean-plan cost are different objectives. Saturation, entropy, actual perturbation moments, raw minimum cost and corrected weights remain recorded separately.

## Do differences shrink with more samples?

These reference-independent distances compare each nonstandard proposal with its paired standard decision. Average across the five alternatives within each seed, then compare the smallest and largest K across paired seeds. This avoids treating proposals as independent experimental repetitions.

| State | Mean paired RMS distance at min K | At max K | Paired change ± CI95 halfwidth |
| --- | --- | --- | --- |
| open_space | 0.2310 | 0.2579 | +0.0268 ± 0.0316 |
| pre_obstacle | 0.2553 | 0.2651 | +0.0098 ± 0.0404 |
| post_disturbance | 0.2206 | 0.2544 | +0.0338 ± 0.0283 |

There is no observed systematic collapse of covariance-induced mean differences over this K range. Figure D shows nonmonotonic paired distances, ESS and mean-plan objectives. This finite-budget observation neither disproves asymptotic consistency nor establishes a convergence rate. `convergence_summary.csv` retains each proposal and first-command distances; `repetition_sensitivity.csv` reports the first 10,25,50 preordered seeds and their uncertainty.

## Does favorable local orientation help, and where does it reverse?

Alignment is the cost-descent direction for a constant two-channel latent offset, estimated by local vehicle/cost sensitivity. The perpendicular intervention uses the same eigenvalues (.34,.10), trace and determinant. `proposals.json` saves both finite difference resolutions, stability and eigenvectors. It is not an optimal importance covariance, and local gradient alignment does not identify the target's 56-dimensional covariance or its mean-estimation variance.

| State | Mean J(aligned)−J(misaligned) ± CI95 halfwidth | Aligned lower | Misaligned lower |
| --- | --- | --- | --- |
| open_space | -3.765 ± 4.664 | 30 | 20 |
| pre_obstacle | -25.863 ± 37.222 | 34 | 16 |
| post_disturbance | -9.980 ± 8.310 | 34 | 16 |

The aligned proposal has lower average primary-budget predicted-plan cost in these three states. Individual paired seeds reverse that effect. In open space and pre-obstacle the primary-budget cost-difference intervals include zero; the obstacle mean is sensitive to a rare collision penalty. Post-disturbance provides the clearer primary-budget favorable example, while seed reversals are negative examples. The per-K table and `orientation_repetition_sensitivity.csv` retain all predeclared subsets, without selecting a favorable K or discarding reversals. These comparisons do not establish better target-mean accuracy, global superiority of anisotropy or closed-loop robustness.

Seed-count sensitivity for the same paired orientation contrast:

| State | First n seeds | Mean ΔJ ± CI95 halfwidth |
| --- | --- | --- |
| open_space | 10 | -10.348 ± 13.817 |
| pre_obstacle | 10 | -26.284 ± 25.525 |
| post_disturbance | 10 | -1.757 ± 26.417 |
| open_space | 25 | -5.532 ± 8.172 |
| pre_obstacle | 25 | -15.432 ± 11.152 |
| post_disturbance | 25 | -13.630 ± 13.526 |
| open_space | 50 | -3.765 ± 4.664 |
| pre_obstacle | 50 | -25.863 ± 37.222 |
| post_disturbance | 50 | -9.980 ± 8.310 |

The updated nominal prediction can intersect an obstacle. This is an open-loop mathematical consequence, not an executed collision:

| Proposal | Seed | Minimum predicted swept clearance [m] |
| --- | --- | --- |
| standard | 10006 | -0.00007392 |
| isotropic | 10006 | -0.03214372 |
| misaligned | 10030 | -0.00190503 |

Primary-budget mean recorded planning latency is 5.224 ms on the manifest's single-thread CPU environment. This includes snapshot observation overhead and excludes construction, extra mean-objective scoring, serialization and plotting. Every proposal uses the same (K+1)T dynamics budget; references, capture prefixes, route witnesses and local sensitivity are separate work.

## Figure paths and regeneration

Each figure has `.png`, `.svg` and `_data.npz` companions. All spatial/color domains within comparisons are shared. Weighted endpoints are an empirical optimizer-weighted distribution, not physical uncertainty. F is a disturbance-free open-loop prediction.

| State | Figure | PNG path |
| --- | --- | --- |
| open_space | A | [figures/open_space_A_decision_microscope.png](figures/open_space_A_decision_microscope.png) |
| open_space | B | [figures/open_space_B_first_command_difference.png](figures/open_space_B_first_command_difference.png) |
| open_space | C | [figures/open_space_C_weighted_endpoint_density.png](figures/open_space_C_weighted_endpoint_density.png) |
| open_space | D | [figures/open_space_D_budget_estimation.png](figures/open_space_D_budget_estimation.png) |
| open_space | E | [figures/open_space_E_decision_dashboard.png](figures/open_space_E_decision_dashboard.png) |
| open_space | F | [figures/open_space_F_counterfactual_consequence.png](figures/open_space_F_counterfactual_consequence.png) |
| pre_obstacle | A | [figures/pre_obstacle_A_decision_microscope.png](figures/pre_obstacle_A_decision_microscope.png) |
| pre_obstacle | B | [figures/pre_obstacle_B_first_command_difference.png](figures/pre_obstacle_B_first_command_difference.png) |
| pre_obstacle | C | [figures/pre_obstacle_C_weighted_endpoint_density.png](figures/pre_obstacle_C_weighted_endpoint_density.png) |
| pre_obstacle | D | [figures/pre_obstacle_D_budget_estimation.png](figures/pre_obstacle_D_budget_estimation.png) |
| pre_obstacle | E | [figures/pre_obstacle_E_decision_dashboard.png](figures/pre_obstacle_E_decision_dashboard.png) |
| pre_obstacle | F | [figures/pre_obstacle_F_counterfactual_consequence.png](figures/pre_obstacle_F_counterfactual_consequence.png) |
| post_disturbance | A | [figures/post_disturbance_A_decision_microscope.png](figures/post_disturbance_A_decision_microscope.png) |
| post_disturbance | B | [figures/post_disturbance_B_first_command_difference.png](figures/post_disturbance_B_first_command_difference.png) |
| post_disturbance | C | [figures/post_disturbance_C_weighted_endpoint_density.png](figures/post_disturbance_C_weighted_endpoint_density.png) |
| post_disturbance | D | [figures/post_disturbance_D_budget_estimation.png](figures/post_disturbance_D_budget_estimation.png) |
| post_disturbance | E | [figures/post_disturbance_E_decision_dashboard.png](figures/post_disturbance_E_decision_dashboard.png) |
| post_disturbance | F | [figures/post_disturbance_F_counterfactual_consequence.png](figures/post_disturbance_F_counterfactual_consequence.png) |

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run python experiments/frozen_state_decision.py \
  --output results/investigations/frozen_decision_reproduction
uv run python experiments/frozen_state_decision.py \
  --output results/investigations/frozen_decision_v1 --plots-only
uv run python experiments/verify_frozen_decision.py \
  --input results/investigations/frozen_decision_v1
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Provenance and limits

Evaluated clean source commit: `7aaf76eaed90588b9404b2fbed483ae9efce1156`. Branch: `frozen-state-decision-analysis`; it is not merged into main. Source/config hashes, versions, platform and RNG namespaces are in `manifest.json`. Plot and report provenance separately record postprocessing sources. Prior experiments and baseline configurations are unchanged.

Only three deterministic reference states and one reference warm start per state were studied. The local two-channel covariance is repeated independently over time; it does not span arbitrary temporal covariance. Trace matching does not match determinant or clipped perturbation energy. Reference uncertainty remains unresolved. The t intervals are pointwise descriptive comparisons with no multiple-comparison adjustment. Seeds describe optimizer variation, not physical disturbance uncertainty. State-space means and control-space means differ under nonlinear dynamics. Other non-primary budgets retain all updated means and scalar records, with full clouds for the representative seed only; every primary-budget population is archived. Timing has observation overhead. No complete evaluation episode is executed and no safety, navigation or general covariance-family superiority claim follows.
