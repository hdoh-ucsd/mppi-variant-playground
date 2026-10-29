# Covariance investigation protocol

The analytic foundation is defined in [covariance_mechanisms.md](covariance_mechanisms.md). Existing baseline experiments and results are preserved. New output directories refuse overwrite.

## Frozen Gaussian study

`configs/investigations/gaussian_target.yaml` declares the base, quadratic cost, proposal means, sample counts, temperatures, scale constraints and replicate seeds. Seeds 0–7 are reserved for exploratory checks; evaluation uses 64 distinct seeds 1000–1063. Tests use small configurations, not a tuning search on the evaluation outcomes.

The main geometry sweep holds the target fixed and varies eigenvalue ratio 1, 2, 4, 8, proposal angle including 0–90° in 15° increments and negative-correlation controls, covariance scale 0.5, 1, 2 and two frozen means. Trace and determinant constraints are separate. Ratios use large/small eigenvalues; angles identify the major axis, modulo 180°. Axis-aligned members at 0° and 90° are diagonal covariances.

Experiment A rotates the cost at fixed proposals. Experiment B rotates proposals at fixed cost. Each cost rotation has a newly computed exact target. The budget study uses K=32–4096, all doubling steps, with five declared proposals. Per seed, standard-normal draws are paired across covariances, and smaller sample counts share prefixes with larger ones. This supplies independent replicate estimates across seeds and paired comparisons across proposals; observations within a seed are dependent.

The temperature study changes the target deliberately. The diagnostic dimension experiment isolates costs and density ratios and uses dimensions 2, 8, 32 and 56. Every trial records cost-evaluation count, ESS, entropy, maximum weight, cost/ratio dispersion and mean-estimation error. Cases also record analytic overlap diagnostics.

Summary statistics include empirical bias, variance trace, MSE, standard deviations and 5/50/95% quantiles across independent trials. Quantile bands describe the trial distribution, not confidence intervals for its mean. Paired differences against the corresponding isotropic case are descriptive. No significance conclusion is drawn from overlapping or nonoverlapping bands.

## Nonlinear and control extensions

These extensions run after the analytic numerical checks pass. Their configuration files declare evaluation settings before the final runs; the original scenarios, baseline configurations and controller code are unchanged.

`nonconvex.yaml` specifies two nonnegative objectives. The rotated curved valley is $0.08(1-x)^2+2(y-x^2)^2$, expressed in coordinates rotated by 30°. Its exact global minimizer maps $(1,1)$ back to the input coordinates. The two-well cost is $\min\{(x-1)^2+4y^2,\;0.35+(x+1)^2+4y^2\}$ and has global minimizer $(1,0)$. A fixed $\mathcal N(0,4I)$ base and temperature 0.2 define Gibbs targets. The mean reference is finite-box quadrature on $[-12,12]^2$, with 401/801-grid refinement and a Gaussian prior tail-probability bound; the global cost minima are known exactly. Best visited cost, final mean cost, and error relative to the quadrature mean are separate outcomes. The experiment uses three initial means, five proposals, 20 iterations, 256 samples per update, and 32 independent seeds. Each initialization costs one evaluation, and every update costs K sampled objectives plus one updated-mean objective. All iterations are saved. Basin discovery means the best visited point lies within 0.35 of the known minimizer; it does not imply mean convergence.

`saturation.yaml` specifies 16 replicate seeds, 16,384 samples per case, four covariance scales and two latent means. Physical control scales are 2 m/s² and 0.5 rad. Analytic clipped marginal moments and left/right boundary masses are saved alongside empirical moments and cross-covariance. Saturation fraction counts input coordinates outside the actuator interval; an additional statistic counts actions with any saturated input. The applied normalized covariance trace and second moment are distinguished from each other and from raw physical marginal variances.

`closed_loop_sweep.yaml` specifies three original scenarios, sample counts 128/384, scales 0.5/1/2 and five geometries: isotropic and ratio-4 proposals at −45°, 0°, 45°, 90°. This is 15 distinct covariance matrices, 90 scenario/budget/covariance cases, and 720 episodes using seeds 2000–2007. The common base remains `diag(0.28, 0.16)`, temperature remains 5, horizon remains 28 and episode limits remain 120 steps. No evaluation outcome is used to retune a scenario. Resolved configs and per-step diagnostics are saved by the existing episode runner. Paired differences include evaluation cost, success, ESS and saturation. Success summaries include Wilson 95% intervals; eight seeds still give broad uncertainty. Latency uses the existing planning timer on a shared CPU without affinity and has no real-time guarantee. Run this timing-sensitive level separately from other computation.

The static Gaussian and bicycle targets are different. No analytic oracle for the bicycle update is introduced, and the experiments do not establish a universal spatial interpretation of input-covariance angle. Closed-loop results test the limits of transferring a mathematical mechanism to the tested controller; they cannot establish a causal mediation chain from ESS to success.

## Artifacts and reproduction commands

Each study saves exact YAML, `manifest.json`, `cases.json`, `trials.csv`, `summary.csv`, `paired_differences.csv`, representative samples where applicable, plots, per-figure numerical arrays, and a concise interpretation. The manifest records Git state, a content hash of source/experiment/configuration files, configuration hash, package versions, threads and RNG convention. Incomplete executions remain marked incomplete.

Plots are reconstructed from saved records and declared analytic models. Figure data include arrays used in the panels; the trial/case files hold the full underlying numerical evidence. Figure reconstruction never resamples an experiment.

From the repository root, use new output directories:

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python experiments/analytic_gaussian.py --output results/investigations/gaussian_v1
uv run python experiments/temperature_sweep.py --output results/investigations/temperature_v1
uv run python experiments/actuator_saturation.py --output results/investigations/saturation_v1
uv run python experiments/nonconvex_optimization.py --output results/investigations/nonconvex_v1
uv run python experiments/covariance_closed_loop.py --output results/investigations/closed_loop_v1
# Independently execute just the alignment experiments if needed:
uv run python experiments/covariance_alignment.py --output results/investigations/alignment_reproduction
# Reconstruct a completed study without executing trials:
uv run python experiments/analytic_gaussian.py --output results/investigations/gaussian_v1 --plots-only
```

Source commit hashes and actual numerical interpretations are recorded in [experimental_findings.md](experimental_findings.md). Machine-dependent timings are not expected to reproduce exactly.
