# Trajectory behavior: distributions in state space

This suite makes the current MPPI covariance settings visible through the trajectories they generate. Standard, isotropic, diagonal and full are four fixed proposal settings of the same information-theoretic MPPI controller. Tsallis, DIAL, Tube, Robust and CC-MPPI remain unimplemented; this suite provides a saved-data interface that their future implementations can use.

The existing analytic experiments are preserved. [Figure 1: covariance geometry](../results/investigations/gaussian_v1/figures/01_covariance_geometry.png) and [Figure 2: the quadratic target](../results/investigations/gaussian_v1/figures/02_rotated_quadratic.png) remain methodological/appendix figures. The new A–E figures address behavior in physical XY space.

## Protocol

The default configuration is `configs/investigations/trajectory_behavior.yaml`: variants `standard`, `isotropic`, `diagonal`, `full`; seeds **42–61**; K=384 samples; T=28 steps; control period 0.15 s. Dynamics, actuator limits, start, goal, cost, obstacle geometry, horizon and per-command rollout budget are shared within each scenario and checked by a setup hash. Prediction and execution remain separate. No scenario or covariance is retuned from these outcomes. These are descriptive experiments; seeds 42–46 have been inspected in earlier work and are not a new holdout set.

Default scenarios are narrow passage, open space and `sudden_disturbance` (an explicit alias of the original `disturbance` configuration). The optional `two_route_obstacle` adds one circle centered at (4,0), radius 0.9 m, under the same start/goal and disk footprint. Its start at y=−0.4 m is not symmetric around the obstacle, so equal route frequencies are not expected. The obstacle is not changed after evaluation.

The representative local cloud is **seed 42, planning step 10**, selected in advance. At steps 0, 5, 10, 15, 20, 21 and 25, the controller records **all K rollout trajectories**, raw costs, normalized optimization weights, lowest-raw-cost index, highest-weight index, current state, updated-mean plan, applied sampled controls and log base/proposal ratio. Steps 21 and 25 are after the disturbance applied during execution step 20. Events after episode termination are absent, never fabricated.

A supplemental disturbance cloud uses step 21, the first observed planning state after the impulse at 3.15 s. This rendering uses the already recorded population and leaves the main seed-42/step-10 comparison intact. Its selection follows the known event time, and changes neither the experiment nor its recorded configuration.

## Four different distributions

1. **Proposal controls:** the configured latent Gaussian. This is the input distribution used for sampling.
2. **Local rollout population:** all state trajectories predicted from one observed state and warm start. Nonlinear dynamics, clipping and horizon turn the control distribution into this state-space cloud. Costs and importance correction define its optimizer weights.
3. **Executed occupancy:** a histogram of physically executed XY states, pooled over recorded times and seeds. Closed-loop decisions, disturbances, termination and dwell time all affect it. It is an empirical occupation measure, not the proposal covariance or a predictive state PDF.
4. **Inter-seed dispersion:** variation among executed paths after an explicit alignment. This describes repeated closed-loop outcomes, not the within-step rollout cloud.

Low ESS can coexist with a broad **unweighted** rollout cloud while the weighted empirical cloud collapses onto a few paths. It does not alone establish narrow or unstable executed trajectories. Snapshot tables record both unweighted endpoint covariance trace and the normalized-weight endpoint covariance trace in m². These are descriptive empirical moments, not Gaussian confidence ellipses.

## Figure guide and inference limits

All variants within a figure use the same spatial bounds, histogram grid, density color normalization, or weight color normalization as applicable. Numeric plot inputs are stored alongside PNG and SVG exports.

| Figure | Fixed setup and variation | Distribution being shown | Legitimate interpretation and limits |
|---|---|---|---|
| **A — executed bundle** | Scenario, budget and paired seed set fixed; proposal setting varies between panels. | Inter-seed executed path variability. | Every run remains visible. Solid/dashed paths indicate goal completion/failure; endpoints use circles/crosses. The bold path is an actual run nearest median evaluation cost, with ties broken by seed. It is not a synthetic median path or a selected successful example. |
| **B — occupancy density** | Common XY bins, smoothing and color scale; variant changes. | Recorded-time occupancy aggregated across seeds. | By default, each seed contributes total mass 1/n and divides it uniformly among its own recorded states. Long failures do not receive more total mass, but stationary dwell within an episode still concentrates occupancy. Density integrates to one over the displayed grid. It does not estimate a probability of successful route choice. |
| **C — local rollout cloud** | Same scenario, seed and planning step; variant changes. | Full intra-step sampled population and optimizer weights. | All 384 paths are shown and colored on a common normalized-weight scale. Lowest raw cost, highest weight, updated mean plan, current state and actual next state are marked separately. Current states and warm starts generally differ across variants, so this is not a frozen-state covariance ablation. A nominal nonlinear trajectory is not the weighted average of state trajectories. |
| **D — dispersion** | Same recorded seed set and alignment rule; variant changes. | Inter-seed paths aligned by fraction of each run's traveled arc length. | Each run spans its own traveled distance, including failures and stationary runs. There is no padding after termination. The upper ribbon projects pointwise 5/95 and 25/75% quantiles onto normals of the mean path; the lower panels show corresponding y-position quantiles. These are descriptive pointwise dispersion bands, not simultaneous confidence or safety regions. Averaging multiple routes can put the mean inside an obstacle. Progress alignment removes timing differences and is not progress toward the goal. |
| **E — dashboard** | Same scenario and seed set, with the declared local snapshot. | Occupancy, one local cloud, per-seed ESS traces, and task outcomes. | Each variant occupies one column. ESS traces end when the episode ends. The bold trace is seed 42; all others remain visible. Success and collision counts use all seeds; cost shows mean and sample SD. The dashboard juxtaposes these quantities but does not establish that ESS or occupancy caused success or failure. |

Histogram smoothing uses a Gaussian kernel with sigma=1 bin on a 120×100 grid and is renormalized afterward. The square-root color mapping is declared on the colorbar; values retain density units m⁻². The alternate `pooled_time` mode gives equal mass to every recorded state, so longer runs contribute more; it is deliberately distinct from default `equal_seed` weighting. Smoothing can color unvisited cells and must not be read as an executed path.

For the optional two-route case, `route_class` is the sign of y at the first forward crossing of x=4 m. Non-crossers remain censored, with all counts reported out of 20. This is a descriptive station-side classification, not a proof of homotopy, route feasibility or successful goal completion. Other scenarios retain crossing values but are not assigned upper/lower route classes.

## Recording and future-controller interface

The existing episode runner accepts optional `observer` and `controller_factory` arguments. Without them the original workflow remains valid. Snapshot capture is opt-in through `snapshot_steps`; it draws no additional randomness and does not modify commands, costs, weights or termination. Selected snapshots include the whole population; the old `rollouts.npz` continues to hold a compact first-32 archive. Snapshot array collection contributes to measured planning latency; file writes and observer serialization are outside that timer.

The new versioned snapshot format uses compressed NumPy arrays with `allow_pickle=False`. States, sampled controls and the updated plan are stored as float32 for presentation; costs, weights and the current/next states remain float64. The schema checks dimensions, finite values, normalized weights and index semantics. Full snapshots are named `snapshots/step_0010.npz`, so the baseline ignore rule for `rollouts.npz` does not discard them.

A future controller is registered via `run_study(..., controller_factories={name: factory})` plus its resolved `variant_overrides`. It supplies the existing controller API and a `get_rollout_snapshot()` adapter with sampled states in the shared state ordering, costs, normalized weights and optional nominal states. The nominal role must be labeled honestly. Plotting consumes saved records and has no dependency on Gaussian sampling or on the controller implementation. An unregistered algorithm raises an error rather than silently being relabeled as standard MPPI. This version displays one recorded population per MPC step; future multi-pass algorithms must declare which pass is shown and count all their optimization work.

Each episode includes original `trajectory.csv`, `controls.csv`, `diagnostics.csv`, `config.yaml` and `result.json`, plus:

- `executed_trajectory.csv`: state time/index, XY, heading, speed, outgoing applied control, terminal success/failure, cumulative collision flag and cumulative evaluation cost. The terminal row has no outgoing command, and its cost includes the terminal penalty.
- `mpc_diagnostics.csv`: per-step ESS, entropy, maximum weight, planning latency, minimum raw sampled cost, saturation and chosen control.
- `snapshot_index.csv` and full-population `snapshots/*.npz`.

Study-level artifacts include manifest/source/configuration hashes, package versions, exact YAML, cases, per-run trials, descriptive summaries and paired differences, `behavior_summary.csv`, `snapshot_summary.csv`, `distributions/*.npz` with exact occupancy/alignment arrays, and `distributions/index.json` recording representative seeds. Figures and their numeric arrays are reconstructed from those files without reoptimizing or resampling.

## Reproduce

From the repository root, use a fresh output directory:

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run ruff check .
uv run ruff format --check .
uv run pytest

# First presentation set: the requested narrow passage only.
uv run python experiments/trajectory_density_study.py --scenarios narrow_passage --output results/investigations/trajectory_narrow_reproduction

# Full executed study, including the optional route-choice case.
uv run python experiments/trajectory_density_study.py --scenarios narrow_passage open_space sudden_disturbance two_route_obstacle --output results/investigations/trajectory_behavior_reproduction

# Rebuild figures from a completed dataset, with no controller execution.
uv run python experiments/trajectory_density_study.py --output results/investigations/trajectory_behavior_v1 --plots-only
uv run python experiments/verify_trajectory_behavior.py --input results/investigations/trajectory_behavior_v1
```

## Executed findings

Executed 320 episodes: four covariance settings × four scenarios × 20 paired seeds. All episodes record clean source revision `077e7d90eb2f2d3fd302a284daf5d9868bf28135`. The dataset contains 2,240 selected-step snapshots, each with all 384 rollouts and their costs and weights. All 320 episodes are retained: 174 goal completions, 146 failures, zero observed collisions and zero actuator/speed violations. Outcomes below are descriptions of these configurations and seeds, not general covariance-family rankings.

### Route mixture can differ when completion counts are similar

In the optional obstacle task, the first forward crossing of x=4 m was:

| Setting | Upper crossing | Lower crossing | No crossing | Goal completion |
|---|---:|---:|---:|---:|
| standard | 0/20 | 20/20 | 0/20 | 4/20 |
| isotropic | 0/20 | 20/20 | 0/20 | 6/20 |
| diagonal | 0/20 | 20/20 | 0/20 | 6/20 |
| full | 3/20 | 17/20 | 0/20 | 5/20 |

The [executed bundle](../results/investigations/trajectory_behavior_v1/figures/two_route_obstacle_A_executed_bundle.png) and [occupancy map](../results/investigations/trajectory_behavior_v1/figures/two_route_obstacle_B_occupancy_density.png) reveal the upper/lower mixture for full covariance. Its mean pointwise y-quantile 90% width across traveled progress is 1.684 m, compared with 0.416–0.463 m for the other three settings. This supports an observed distributional difference even though completion counts range only from 4/20 to 6/20. Twenty seeds do not establish true route probabilities; the asymmetric initial position and finite horizon also matter.

The full setting's arc-length-aligned mean path has minimum swept clearance **−0.206 m** against the inflated disk/circle geometry, despite zero collisions in the individual executed runs. The [dispersion figure](../results/investigations/trajectory_behavior_v1/figures/two_route_obstacle_D_progress_dispersion.png) illustrates why an averaged path or a band spanning multiple routes must not be treated as a feasible plan. This is a property of the descriptive mean, not a controller collision.

### Broad sampling and concentrated weights coexist

At the declared narrow-passage snapshot, seed 42 / step 10:

| Setting | ESS / 384 | Unweighted endpoint variance trace [m²] | Weighted endpoint variance trace [m²] |
|---|---:|---:|---:|
| standard | 1.374 | 2.726 | 0.335 |
| isotropic | 1.437 | 3.380 | 0.247 |
| diagonal | 1.185 | 2.760 | 0.101 |
| full | 1.708 | 3.235 | 0.341 |

The [full local cloud](../results/investigations/trajectory_behavior_v1/figures/narrow_passage_C_rollout_cloud.png) shows broad candidate coverage, including curved alternatives, with weights concentrated on a small portion. The lowest-raw-cost path and highest-weight path differ for all four settings in this snapshot. The importance correction is part of the weight, so raw cost alone does not identify which path dominates the update. Curved paths being present is not proof that they are useful or will be executed; their optimizer weights, costs and obstacle context must be inspected.

The full setting has a broader unweighted endpoint trace than standard at this one snapshot, while its inter-seed mean 90% y-width in the passage is smaller: 0.285 versus 0.418 m. These are distinct populations under different conditioning. One cannot infer closed-loop route spread from proposal variance, one rollout cloud, or ESS alone.

In the five failed standard passage episodes, observations at t≥12 s have pooled median ESS **110.84** and median speed **0 m/s**. The [dashboard](../results/investigations/trajectory_behavior_v1/figures/narrow_passage_E_behavior_dashboard.png) retains those late high-ESS traces. High episode-average ESS can accompany stalled execution; it does not establish useful optimization or goal convergence. These late frames are not independent experimental trials.

### Post-disturbance behavior is separately visible

Goal-pose completion after the shared impulse is 6/20 standard, 5/20 isotropic, 1/20 diagonal and 1/20 full. Mean pointwise 90% y-widths across traveled progress are 0.722, 0.575, 0.587 and 0.717 m, respectively. The [executed bundle](../results/investigations/trajectory_behavior_v1/figures/sudden_disturbance_A_executed_bundle.png), [progress bands](../results/investigations/trajectory_behavior_v1/figures/sudden_disturbance_D_progress_dispersion.png), and [post-impulse rollout cloud](../results/investigations/trajectory_behavior_v1/figures/sudden_disturbance_C_rollout_cloud_post_impulse.png) show the path variations around that event. Reference-line recovery and full goal-pose completion remain different outcomes in the per-run records.

The open-space and disturbance rollouts at step 10 are identical within each variant/seed before the unannounced impulse, as expected. The extra step-21 panel provides the post-event view. Its current states and warm starts have evolved differently across variants, so it supports behavior-level comparison rather than a pure frozen-state covariance attribution.

### Figure files and validation

The output root is `results/investigations/trajectory_behavior_v1/`. For each of `narrow_passage`, `open_space`, `sudden_disturbance`, and `two_route_obstacle`, the five primary stems are:

- `{scenario}_A_executed_bundle`
- `{scenario}_B_occupancy_density`
- `{scenario}_C_rollout_cloud`
- `{scenario}_D_progress_dispersion`
- `{scenario}_E_behavior_dashboard`

Each has `.png`, `.svg` and `_data.npz` files under `figures/`. The supplemental stem is `sudden_disturbance_C_rollout_cloud_post_impulse`. There are **21 PNG/SVG/data sets**. All plots come from executed records and saved arrays, with no controller calls during reconstruction.

The tests cover snapshot shape/weight validation, loss-aware archive round trips, unmodified commands and trajectories with recording enabled, observer-copy isolation, occupancy mass/integral and seed/time weighting, progress alignment without terminal padding, fixed-seed reproduction, raw-record schemas and reconstruction with random sampling disabled. The post-impulse test checks that step 21 starts at executed state index 21 and marks state 22 as the actual next state. Full dataset audit results and the current test count are saved alongside the study.

The optional route case provides an observed multimodal example, but the suite does not isolate covariance as the sole cause of a route switch, establish statistical superiority, or provide a state-distribution safety guarantee. A frozen-state/warm-start experiment would be required for that causal attribution. Occupancy smoothing, finite seed counts, chosen time/progress alignment and the original toy bicycle model remain material limits.
