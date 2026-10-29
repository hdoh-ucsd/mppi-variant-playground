# Covariance investigation protocol

The analytic foundation is defined in [covariance_mechanisms.md](covariance_mechanisms.md). Existing baseline experiments and results are preserved. New output directories refuse overwrite.

## Frozen Gaussian study

`configs/investigations/gaussian_target.yaml` declares the base, quadratic cost, proposal means, sample counts, temperatures, scale constraints and replicate seeds. Seeds 0–7 are reserved for exploratory checks; evaluation uses 64 distinct seeds 1000–1063. Tests use small configurations, not a tuning search on the evaluation outcomes.

The main geometry sweep holds the target fixed and varies eigenvalue ratio 1, 2, 4, 8, proposal angle including 0–90° in 15° increments and negative-correlation controls, covariance scale 0.5, 1, 2 and two frozen means. Trace and determinant constraints are separate. Ratios use large/small eigenvalues; angles identify the major axis, modulo 180°. Axis-aligned members at 0° and 90° are diagonal covariances.

Experiment A rotates the cost at fixed proposals. Experiment B rotates proposals at fixed cost. Each cost rotation has a newly computed exact target. The budget study uses K=32–4096, all doubling steps, with five declared proposals. Per seed, standard-normal draws are paired across covariances, and smaller sample counts share prefixes with larger ones. This supplies independent replicate estimates across seeds and paired comparisons across proposals; observations within a seed are dependent.

The temperature study changes the target deliberately. The diagnostic dimension experiment isolates costs and density ratios and uses dimensions 2, 8, 32 and 56. Every trial records cost-evaluation count, ESS, entropy, maximum weight, cost/ratio dispersion and mean-estimation error. Cases also record analytic overlap diagnostics.

Summary statistics include empirical bias, variance trace, MSE, standard deviations and 5/50/95% quantiles across independent trials. Quantile bands describe the trial distribution, not confidence intervals for its mean. Paired differences against the corresponding isotropic case are descriptive. No significance conclusion is drawn from overlapping or nonoverlapping bands.

## Artifacts and reproduction

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
# Independently execute just the alignment experiments if needed:
uv run python experiments/covariance_alignment.py --output results/investigations/alignment_reproduction
# Reconstruct a completed study without executing trials:
uv run python experiments/analytic_gaussian.py --output results/investigations/gaussian_v1 --plots-only
```

Source commit hashes and actual numerical interpretations are recorded in [experimental_findings.md](experimental_findings.md). Machine-dependent timings are not expected to reproduce exactly.
