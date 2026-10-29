# Covariance mechanism findings

This report follows **claim → derivation → hypothesis → isolated test → observation → interpretation → limitation**. Mathematical identities are distinguished from observations throughout. Evaluation configurations were committed before the corresponding runs; no scenario or covariance setting was retuned using these outcomes. The runs are exploratory scientific evidence with predeclared configurations, not a preregistered confirmatory test.

## Executed evidence and provenance

| study | cases | trials | source | clean |
| --- | --- | --- | --- | --- |
| gaussian | 447 | 28608 | dd50f534 | True |
| temperature | 71 | 4544 | dd50f534 | True |
| saturation | 40 | 640 | daba0ba6 | True |
| nonconvex | 30 | 960 | daba0ba6 | True |
| closed_loop | 90 | 720 | daba0ba6 | True |

The analytic and temperature studies comprise 33,152 replicate/case records (64 seeds per case). Covariances within a seed share standard-normal draws and are therefore paired, not independent of each other. Nonconvex optimization uses 32 seeds per case, saturation 16 and closed-loop validation eight. All source states at study start were clean. The per-episode Git records additionally check the control run's provenance. Full manifests contain source/configuration hashes, package versions, environment/thread settings and UTC timestamps. Original baseline results remain unchanged in [results.md](results.md).

These trials cost 9,279,488 analytic sampled objectives, 2,326,528 temperature/diagnostic sampled objectives and 4,935,360 nonconvex objectives. The saturation study draws 10,485,760 latent samples without evaluating a control objective. The controller records trajectory-objective counts and scalar planning transitions separately; these work units should not be pooled with the static objective counts.

## 1. Proposal geometry can help or hurt finite-sample estimation

**Mathematical claim and derivation.** With fixed $P,H,b,\lambda$, the Gibbs target has $C_*=(P^{-1}+H/\lambda)^{-1}$ and $\mu_*=C_*Hb/\lambda$. Proposal covariance does not occur in these moments. The derivation, finite-moment conditions and independent numerical checks are in [covariance_mechanisms.md](covariance_mechanisms.md).

**Hypothesis and isolated test.** A proposal that adequately covers the target's useful directions may reduce estimation error; a narrow or misaligned proposal may increase it. The main experiment keeps target, frozen proposal mean, sample count and trace fixed. It varies orientation, eigenvalue ratio and scale, with determinant matching in a separate comparison. The sample-budget cases below use the fixed 45° cost, proposal mean zero, trace 1.2, temperature 1 and K=256.

**Observed result.** MSE and variance use repeated self-normalized estimates, rather than comparing one realization with an exact answer:

| proposal_name | squared_error_mean | bias_norm | estimator_variance_trace | ess_mean |
| --- | --- | --- | --- | --- |
| isotropic | 0.00765727 | 0.00275329 | 0.00777111 | 122.421 |
| diagonal_r4 | 0.0131405 | 0.00552452 | 0.013318 | 111.064 |
| full_r4_a45 | 0.00369914 | 0.00576918 | 0.00372404 | 156.532 |
| diagonal_r4_a90 | 0.0177046 | 0.0601339 | 0.0143121 | 76.2846 |
| full_r4_a-45 | 0.0492543 | 0.0286923 | 0.0491998 | 61.768 |

The +45° ratio-4 proposal has lower observed MSE than the isotropic proposal, whereas its −45° counterpart has higher MSE. In this cost case the diagonal proposal also has higher observed MSE than isotropic. The −45° proposal violates the finite importance-weight second-moment condition, even though every finite trial has a numerical ESS and a finite error. This cautions against applying the usual finite-variance asymptotic approximation to it.

**Interpretation.** Orientation matters through the distribution being estimated, including prior precision and proposal mean displacement. A covariance family is not represented by one successful or unsuccessful matrix. At K=4096 the recorded MSEs are 0.00035468 (isotropic), 0.000214705 (+45°), and 0.00916012 (−45°). Empirical MSE decreases across these endpoints; per-seed errors need not decrease monotonically.

**Limitation.** Quantiles and paired differences are descriptive, not significance tests. The optimal proposal for mean-estimation variance need not be Gaussian or have covariance equal to the target covariance. A low KL or a high ESS is an overlap diagnostic, not an integrand-specific error guarantee.

The complementary experiment rotates the cost while keeping each proposal fixed. Each cost rotation gets its own analytic target:

| cost_angle_deg | proposal_name | squared_error_mean | ess_mean |
| --- | --- | --- | --- |
| 0 | isotropic | 0.00842456 | 115.064 |
| 0 | diagonal_r4 | 0.00484979 | 140.6 |
| 0 | full_r4_a45 | 0.022033 | 96.442 |
| 45 | isotropic | 0.00765727 | 122.421 |
| 45 | diagonal_r4 | 0.0131405 | 111.064 |
| 45 | full_r4_a45 | 0.00369914 | 156.532 |
| 90 | isotropic | 0.00571848 | 92.5332 |
| 90 | diagonal_r4 | 0.0340794 | 61.008 |
| 90 | full_r4_a45 | 0.0232138 | 59.8827 |

This includes the negative case in which a fixed anisotropic orientation loses its benefit when the landscape rotates. Scales 0.5/1/2, ratios 1/2/4/8, signed correlations, two proposal means and both scale constraints remain available in the full case/trial tables; none was selected after inspecting outcomes.

## 2. Temperature, dimension and ESS require separate interpretations

**Claim and derivation.** Changing $\lambda$ changes the target itself. Finite-K ESS depends on weight concentration; mean-estimation error also depends on the weighted integrand. The study recomputes exact moments at every temperature. An additional factorial diagnostic separates cost-only weights (proposal equals base), density-ratio-only weights (zero cost) and their combination in dimensions 2/8/32/56.

**Hypothesis and isolated test.** Lower temperature can concentrate cost weights; high-dimensional density ratios can concentrate weights even with zero cost. Both are tested rather than treated as interchangeable explanations of the vehicle's ESS.

**Observed result**, isotropic proposal, K=512:

| temperature | ess_mean | max_weight_mean | error_mean |
| --- | --- | --- | --- |
| 0.1 | 41.9465 | 0.0472323 | 0.0341618 |
| 0.25 | 94.4837 | 0.0212827 | 0.0370623 |
| 0.5 | 160.291 | 0.0132206 | 0.0426292 |
| 1 | 244.009 | 0.0105596 | 0.0439662 |
| 2 | 331.09 | 0.00924299 | 0.0434757 |
| 5 | 421.85 | 0.00827745 | 0.045145 |
| 10 | 453.29 | 0.00869509 | 0.0496706 |

**Interpretation.** ESS rises substantially across these temperatures while absolute errors do not track it monotonically. Because the target changes, this is not a fixed-target accuracy comparison or evidence that increasing ESS causes increased error. The controlled dimension panels identify mathematical sources of concentration; they do not diagnose the causal source of concentration in every vehicle episode.

**Limitation.** Population second-moment diagnostics can differ from a finite batch's apparent overlap. The finite-dimensional experiment does not reproduce all dependencies in an MPC trajectory cost.

## 3. Iterative means and objective minimizers are different quantities

**Claim and derivation.** Each iteration estimates a fixed Gibbs mean from an adaptively recentered proposal. A finite-temperature distribution mean need not be a global minimizer of a nonconvex objective. Both test objectives have known zero-cost global minima, and their target means are independently approximated by quadrature:

- `curved_valley`: target mean `[0.027931296843370777, 0.8366131417001813]`; 401/801-grid difference 3.95e-13; prior mass outside the box ≤3.95e-09.
- `two_wells`: target mean `[0.687207076505428, 0.0]`; 401/801-grid difference 9.62e-06; prior mass outside the box ≤3.95e-09.

**Hypothesis and isolated test.** Changing orientation or initialization may affect mean paths, best visited cost and basin discovery. Three initializations and five proposal geometries use identical 20-iteration/256-sample budgets. Every iteration, including the start and evaluation count, is retained.

**Observed result.** All 960 trials discover a point within radius 0.35 of the specified global-minimum basin. Aggregated over three initializations:

| objective | proposal_name | best_cost | mean_cost | error | basin_success |
| --- | --- | --- | --- | --- | --- |
| curved_valley | diagonal_r4 | 0.000396355 | 0.276842 | 0.361306 | 1 |
| curved_valley | diagonal_r4_a90 | 8.844e-05 | 0.439886 | 0.217753 | 1 |
| curved_valley | full_r4_a-45 | 0.000254816 | 0.327102 | 0.305019 | 1 |
| curved_valley | full_r4_a45 | 0.000127434 | 0.345756 | 0.247845 | 1 |
| curved_valley | isotropic | 0.000139776 | 0.411267 | 0.187911 | 1 |
| two_wells | diagonal_r4 | 0.000293876 | 0.0957456 | 0.0607277 | 1 |
| two_wells | diagonal_r4_a90 | 0.000208536 | 0.0452028 | 0.266143 | 1 |
| two_wells | full_r4_a-45 | 0.00026751 | 0.0964929 | 0.195901 | 1 |
| two_wells | full_r4_a45 | 0.000256869 | 0.157553 | 0.25042 | 1 |
| two_wells | isotropic | 0.000401233 | 0.109946 | 0.124515 | 1 |

**Interpretation.** Basin discovery is saturated at this budget and cannot discriminate the settings. Best-point cost and fitted-mean cost differ substantially, especially on the curved valley. The path panels illustrate why a good visited point does not imply a mean update at the objective minimizer.

**Limitation.** The quadrature mean is a finite-grid reference, not a proof of global convergence of adaptive finite-sample updates. Its refinement discrepancy is empirical; the prior tail-probability bound does not alone bound all moment errors. These objectives are inexpensive diagnostic constructions, not robot tasks.

## 4. Clipping changes physical exploration

**Claim and derivation.** For $u=D\,\operatorname{clip}(z,-1,1)$, latent Gaussian moments become clipped moments with boundary atoms. Matching latent trace fixes expected squared latent perturbation magnitude about the proposal mean, not applied covariance. Marginal clipped means, variances and boundary masses are checked against independent integrals. All likelihoods remain in latent space.

**Hypothesis and isolated test.** Equal-trace covariances can yield different applied variance. The table fixes mean zero and scale 2, so the specified latent trace is 0.88 for every proposal. It reports sample means across 16 trials:

| proposal_name | saturation_fraction | normalized_applied_covariance_trace | applied_variance_0 | applied_variance_1 |
| --- | --- | --- | --- | --- |
| diagonal_r4 | 0.125082 | 0.614317 | 1.77679 | 0.0425298 |
| diagonal_r4_a90 | 0.124605 | 0.613363 | 0.680679 | 0.110798 |
| full_r4_a-45 | 0.130703 | 0.687357 | 1.37522 | 0.0858882 |
| full_r4_a45 | 0.130638 | 0.686728 | 1.37522 | 0.085731 |
| isotropic | 0.13098 | 0.686933 | 1.37522 | 0.0857822 |

Physical variance columns have units $(\mathrm{m/s^2})^2$ and $\mathrm{rad}^2$, respectively. They should not be summed as a physical-energy metric. The normalized applied trace is dimensionless.

**Interpretation.** Axis-aligned anisotropy clips differently from isotropic sampling at the same latent trace. The ±45° proposals have the same marginal latent variances as the isotropic proposal here, so their marginal clipping statistics agree up to sampling variability; correlation still changes the joint distribution.

**Limitation.** Marginal analytic formulas do not supply a Gaussian density for the applied joint distribution. Near-boundary mean cases and all scales are retained in the raw records.

## 5. Closed-loop transfer is task- and setting-dependent

**Claim and derivation.** The bicycle uses the same importance-sampling law but adds nonlinear prediction, actuator clipping, shifting controls and changing observed state. Its target is not the static quadratic target. A favorable static proposal cannot be declared favorable for navigation without an additional experiment.

**Hypothesis and test.** Vary the input covariance under unchanged costs, limits, horizon, prediction/execution models and scenarios. Use 15 distinct matrices, two K values, three tasks and eight paired seeds per cell. The main trace-0.44, K=384 slice is:

| scenario | proposal_name | successes | n | cost | ess | saturation |
| --- | --- | --- | --- | --- | --- | --- |
| disturbance | isotropic | 0 | 8 | 206.287 | 2.77589 | 0.127258 |
| disturbance | r4_a-45 | 0 | 8 | 196.573 | 1.9735 | 0.132604 |
| disturbance | r4_a0 | 1 | 8 | 177.475 | 2.32955 | 0.120441 |
| disturbance | r4_a45 | 2 | 8 | 183.925 | 1.93995 | 0.135818 |
| disturbance | r4_a90 | 3 | 8 | 188.845 | 1.89681 | 0.137128 |
| narrow_passage | isotropic | 8 | 8 | 155.401 | 2.1371 | 0.147696 |
| narrow_passage | r4_a-45 | 6 | 8 | 163.629 | 1.74568 | 0.150216 |
| narrow_passage | r4_a0 | 6 | 8 | 159.822 | 1.84432 | 0.138223 |
| narrow_passage | r4_a45 | 6 | 8 | 166.928 | 1.84407 | 0.149326 |
| narrow_passage | r4_a90 | 7 | 8 | 169.511 | 1.78904 | 0.143704 |
| open_space | isotropic | 7 | 8 | 153.894 | 1.96974 | 0.151392 |
| open_space | r4_a-45 | 7 | 8 | 155.061 | 2.02109 | 0.136139 |
| open_space | r4_a0 | 7 | 8 | 152.158 | 2.04069 | 0.128463 |
| open_space | r4_a45 | 7 | 8 | 163.722 | 1.95699 | 0.142556 |
| open_space | r4_a90 | 7 | 8 | 164.835 | 1.74635 | 0.146899 |

The complete sweep, aggregated over all three scales and five geometries within each task/budget, records:

| scenario | samples | successes | n |
| --- | --- | --- | --- |
| disturbance | 128 | 21 | 120 |
| disturbance | 384 | 19 | 120 |
| narrow_passage | 128 | 88 | 120 |
| narrow_passage | 384 | 99 | 120 |
| open_space | 128 | 86 | 120 |
| open_space | 384 | 102 | 120 |

**Observed result.** All 720 episodes are retained, including 305 failures, 0 collisions and 0 actuator/speed violations. Each episode saves resolved YAML, commands, executed trajectory, per-step ESS/saturation/weights summary, planning latency and a compact unweighted rollout subset. Success summaries include Wilson intervals; timing and paired evaluation-cost differences are retained separately.

**Interpretation.** The +45° proposal reduced static Gaussian MSE, but at the declared control slice it completes the passage in 6/8 seeds versus 8/8 for isotropic. Both complete open space in 7/8; after disturbance the counts are 2/8 and 0/8. These small samples are observations, not evidence of a universal ordering. Compare this control slice with the static MSE table as a transfer check, not a common-target ranking. The tables expose whether a static advantage survives in a particular task and configuration. A covariance angle refers to normalized acceleration/steering directions, not a global direction through an obstacle passage.

**Limitation.** There is no exact target-mean oracle for the 56-dimensional bicycle planning update, no mediation analysis, and no proof that ESS or saturation caused a particular failure. Eight seeds per cell yield broad uncertainty. Latency comes from an unpinned shared host. No robustness or safety guarantee is established.

## Figure-by-figure seminar guide

All figures are exported as PNG and SVG, with source numerical records and per-figure arrays. Main figure links point to the PNG; the same directory contains its SVG and `_data.npz`. In the table, “observed” refers to the numerical sections above, not an unexecuted prediction.

| Figure | Equation / mathematical claim | Held fixed → changed; predicted effect | Observation, interpretation and remaining limit |
|---|---|---|---|
| [1 · Covariance geometry](../results/investigations/gaussian_v1/figures/01_covariance_geometry.png) | $\Sigma=R\operatorname{diag}(e_1,e_2)R^T$ | Trace and mean → eigenvalue ratio and rotation; circular versus elongated/rotated contours. | Exact configured contours show geometry, not estimation performance. Physical clipping is absent. |
| [2 · Rotated quadratic](../results/investigations/gaussian_v1/figures/02_rotated_quadratic.png) | Exact Gaussian target and SNIS update | Cost/base/temperature/K → proposal; coverage and mean errors may differ. | Actual saved samples, exact mean and sampled mean illustrate the mechanism. One displayed seed cannot establish an ordering. |
| [3 · Alignment](../results/investigations/gaussian_v1/figures/03_alignment_sweep.png) | Target covariance depends on base and Hessian | B: target fixed → proposal angle; A: proposals fixed → cost angle. | Error depends on relative orientation; the negative case is retained. Shading is 5–95% of trials, not confidence in a mean. |
| [3b · Scale constraints](../results/investigations/gaussian_v1/figures/03b_scale_constraints.png) | Trace and determinant are distinct invariants | Each target fixed → ratio/angle/scale, in separate constraint panels. | Error curves vary with the declared constraint. No single scale represents a covariance family. |
| [4 · Sample efficiency](../results/investigations/gaussian_v1/figures/04_sample_efficiency.png) | Empirical MSE decomposition; finite-K bias | Target and proposal parameters → K. | Error, ESS, variance and seed distributions expose variability and endpoint improvement. Rare-tail behavior limits finite-variance inference. |
| [5 · Weight concentration](../results/investigations/temperature_v1/figures/05_temperature_weights.png) | $C_*(\lambda)$ and $\mu_*(\lambda)$ | Base/cost/proposal/K → temperature. | ESS and errors use different temperature-specific targets. This is not a fixed-target comparison. |
| [5b · Dimension diagnostic](../results/investigations/temperature_v1/figures/05b_dimension_diagnostics.png) | Costs and log-density ratios contribute separately | Zero-ratio / zero-cost controls → dimension and temperature. | Concentration appears through distinct sources; these controls do not fully reproduce MPC dependencies. |
| [6 · Nonconvex iterations](../results/investigations/nonconvex_v1/figures/06_nonconvex_iterations.png) | Weighted target mean need not minimize cost | Target and covariance fixed → adaptive proposal mean. | Every update is shown; best visited cost and fitted mean differ. Paths show one declared seed/initialization. |
| [6b · Initialization](../results/investigations/nonconvex_v1/figures/06b_initialization.png) | Adaptive finite batches can depend on initialization | Target/K/iterations → start and proposal. | All tested basins were found, so this metric has no discriminating power here. |
| [7 · Saturation](../results/investigations/saturation_v1/figures/07_saturation_distributions.png) | $u=D\operatorname{clip}(z)$ has boundary atoms | Latent trace and mean → geometry, then clipping. | Saved before/after samples show boundary accumulation. Applied actions are not assigned latent Gaussian likelihoods. |
| [7b · Applied variance](../results/investigations/saturation_v1/figures/07b_clipped_variance.png) | Applied covariance differs from $D\Sigma D^T$ | Within each mean/scale slice, latent trace fixed → geometry. | Axis-aligned and equal-marginal proposals show different or matching marginal effects. Cross-correlation still matters jointly. |
| [8 · Bicycle transfer](../results/investigations/closed_loop_v1/figures/08_closed_loop_trajectories.png) | The control update is repeatedly replanned through nonlinear dynamics | Scenario/cost/base/limits/K → proposal. | Executed paths and per-step ESS show task behavior; retained rollouts are unweighted examples. No exact update oracle exists here. |
| [8b · Control scale](../results/investigations/closed_loop_v1/figures/08b_closed_loop_scale.png) | Latent scale affects exploration and clipping | Task/base/K → scale and orientation. | Eight-seed success fractions are descriptive; complete costs and failures remain in the CSVs. |
| [9 · Evidence map](../results/investigations/summary/figures/09_mechanism_evidence.png) | Geometry → overlap → weights → update; control is a separate transfer test | Juxtapose declared analytic and control slices. | The static links have equations and experiments; the dashed boundary marks the unestablished causal bridge to control performance. |

## Reproduction, files and unresolved questions

Exact commands are in [mathematical_investigations.md](mathematical_investigations.md). `--plots-only` reconstructs the existing studies. Fresh runs must use new output directories. Numerical replay checks compare stored trials with regenerated fixed-seed calculations; elapsed time and figure file metadata may differ.

New implementation lives under `src/analysis/`, with independent commands under `experiments/`, fixed YAML under `configs/investigations/`, and numerical/figure tests under `tests/`. README and the explicit Hatch package list link and install the analysis package. No heavy dependencies or baseline controller/model changes are introduced; `uv.lock` is unchanged.

Open questions include integrand-aware proposal design, repeated-batch uncertainty for heavy-tailed weights, temporal control correlations, more difficult nonconvex basin-discovery regimes, and an independently validated reference for nonlinear control-sequence mean updates. Resolving those is necessary before a causal or general claim about closed-loop benefit.
