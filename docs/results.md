# Executed milestone results

This report preserves the original four-proposal vehicle milestone. Later covariance parameter sweeps and isolated mathematical experiments are reported separately in [covariance mechanism findings](experimental_findings.md).

## What the existing results establish

The research goal is to understand the mathematical characteristics of MPPI variants. This report records **closed-loop observations for four fixed Gaussian proposals within one IT-MPPI implementation**. It is an initial source of behavioral examples and diagnostics. The other algorithm families have no executed results here.

The mathematical interpretation of the current scope is:

1. **Proposal geometry changes while the target stays fixed.** Isotropic, diagonal and correlated proposals redistribute exploration. The retained density ratio compensates for their unequal covariances. Task-dependent outcomes are compatible with finite-sample effects, but success counts alone cannot identify their mechanism.
2. **ESS measures weight concentration.** The recorded low ESS establishes concentration in these sampled batches. It does not separate cost sharpness, proposal/target mismatch, horizon dimension or warm-start effects. Those contributions need isolated probes.
3. **Latent sampling and applied controls differ under saturation.** The code and tests preserve Gaussian likelihoods before clipping. The logged saturation fraction helps interpret how proposal geometry translates into applied-control exploration; matched latent trace alone cannot establish matched physical exploration.
4. **Vehicle scenarios show resulting behavior.** Open space, passage and disturbance runs supply examples of completion, obstacle interaction and post-impulse behavior. Disturbance failure does not characterize Tube-MPPI or RMPPI, since neither architecture is implemented.
5. **Budget and replay checks have limited roles.** Wall-clock mode records behavior with calibrated sample allocations. Native mode reproduces the primary trajectories. Neither supplies a separate mathematical variant.

The tables below preserve the measured outcomes. Read them as observations to explain through the [mechanism studies](research_questions.md), with derivations and numerical controls supplying the explanation.

The four matrices are individual settings, with only one correlated proposal. Additional covariance values are needed to investigate a structural property. The two additional environments each have one standard-controller run and provide behavioral examples only.

## Evidence and scope

Executed 2026-10-28 on Linux x86_64, an AMD Ryzen 7 7800X3D (8 cores / 16 logical CPUs), Python 3.12.2, NumPy 2.5.4, SciPy 1.18.1, with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and headless Matplotlib. First recorded benchmark start: `2026-10-29T02:38:47.165129+00:00`. Timings come from a shared host without CPU affinity or a controlled load guarantee.

There are **180 recorded benchmark episodes**: four configurations × three scenarios × five seeds × three separately reported budget modes. Two additional seed-42 full episodes check the obstacle-field and model-mismatch environments. Pilot runs, unit tests, warmups and timing-calibration batches are outside this count. No results below are fabricated or inferred from unexecuted controllers.

All 180 benchmark episodes have clean source provenance in `result.json`. Primary episode execution used source commit `1f70c5a73be802dab33271b429d4225639f1b392`. Summary statistics and both secondary modes use `5ec715c671e58d6be22731ec090b39288875e9b3`; the intervening change adds Wilson intervals, analysis provenance and a wider wall-clock calibration grid, with no change to controller, dynamics, cost, runner or numerical trajectory behavior. `analysis_provenance.json` records the aggregation revision separately.

## A. Matched dynamics-evaluation budget

Each command uses K=384, T=28 and one optimization pass: **10,780 planning model transitions per step**, including the updated-mean plan. Each sampling covariance has trace 0.44 in normalized acceleration/steering coordinates, and the fixed Gaussian base distribution is shared. Seeds are 42–46.

Success requires all of position, heading, speed and collision conditions. Counts out of five:

| Controller | Open space | Narrow passage | Disturbance |
|---|---:|---:|---:|
| standard | 5/5 | 4/5 | 1/5 |
| isotropic | 3/5 | 5/5 | 0/5 |
| diagonal | 5/5 | 5/5 | 0/5 |
| full | 3/5 | 4/5 | 1/5 |

| Scenario | Controller | Evaluation cost, mean ± SD | Final position error [m], mean ± SD | Mean ESS | Latency [ms], mean ± SD |
|---|---|---:|---:|---:|---:|
| open_space | standard | 150.50 ± 14.95 | 0.36 ± 0.11 | 1.86 | 4.45 ± 0.09 |
| open_space | isotropic | 160.24 ± 15.07 | 0.62 ± 0.32 | 1.99 | 4.30 ± 0.04 |
| open_space | diagonal | 150.43 ± 6.80 | 0.27 ± 0.12 | 2.12 | 4.26 ± 0.05 |
| open_space | full | 165.17 ± 26.61 | 0.65 ± 0.56 | 1.92 | 4.37 ± 0.09 |
| narrow_passage | standard | 158.53 ± 9.12 | 0.48 ± 0.16 | 6.90 | 8.13 ± 0.12 |
| narrow_passage | isotropic | 154.37 ± 9.62 | 0.43 ± 0.03 | 2.05 | 8.09 ± 0.16 |
| narrow_passage | diagonal | 146.27 ± 11.27 | 0.37 ± 0.08 | 1.93 | 7.89 ± 0.14 |
| narrow_passage | full | 156.60 ± 21.15 | 0.36 ± 0.28 | 1.95 | 7.99 ± 0.09 |
| disturbance | standard | 172.29 ± 20.18 | 0.79 ± 0.27 | 18.02 | 4.43 ± 0.11 |
| disturbance | isotropic | 181.65 ± 17.82 | 0.95 ± 0.24 | 3.43 | 4.28 ± 0.05 |
| disturbance | diagonal | 183.33 ± 18.53 | 0.94 ± 0.23 | 2.21 | 4.23 ± 0.04 |
| disturbance | full | 190.19 ± 41.02 | 1.00 ± 0.56 | 2.15 | 4.38 ± 0.05 |

All **60 runs had zero collisions and zero actuator/speed constraint violations**. They consumed **49,275,380 planning transitions**. Mean episode planning latency ranged from 4.18 to 8.34 ms. The obstacle scenario costs more computation because each sampled transition also evaluates obstacle geometry.

The covariance structure affects finite-budget outcomes in these runs, but it does not produce one consistent ranking. Diagonal sampling reaches every open-space and narrow-passage target, while its disturbance runs all time out. Full covariance also times out in several open-space runs. We do not claim statistical superiority from five seeds.

Per-step ESS is strongly concentrated: the median of each episode's median ESS, pooled across the three tasks, is approximately 1.5–1.6 for all configurations. Episode-averaged ESS can be much higher after a controller stops away from the goal (for example standard disturbance seed 45). High stationary ESS does not imply task convergence. The first 32 stored unweighted paths illustrate sampling coverage, not the complete weighted distribution.

Under the declared recovery definition, standard MPPI has one observed disturbance recovery (1.95 s) and diagonal has one (12.0 s); the other disturbance runs are censored/unrecovered. These denominators are **1/5**, and reference-line recovery is not equivalent to successful goal pose completion. Every missing recovery value remains null.

Binary proportions use Wilson 95% intervals: for example 5/5 has interval approximately [0.566, 1.000], while 0/5 has [0.000, 0.434]. Continuous means and paired differences use descriptive Student-t intervals, saved with observation counts. Full per-seed observations, medians, SDs and intervals are in the CSV/JSON files; plots show seed points and SD rather than hiding failures in a single average.

- [Per-seed values](../results/covariance_comparison/per_seed.csv)
- [All summary metrics](../results/covariance_comparison/summary.csv)
- [Paired differences against standard](../results/covariance_comparison/paired_differences.json)

## B. Calibrated 15 ms wall-clock target

This mode is **offline median-latency matching**, with actual deadline misses reported. It does not enforce a hard real-time deadline. Calibration selected 1,536 samples in open space and disturbance, and 768 in the narrow passage. The larger batches change finite-sample behavior and must not be pooled with the K=384 experiment.

| Controller | Open space | Narrow passage | Disturbance |
|---|---:|---:|---:|
| standard | 5/5 | 5/5 | 0/5 |
| isotropic | 5/5 | 5/5 | 0/5 |
| diagonal | 4/5 | 5/5 | 1/5 |
| full | 4/5 | 5/5 | 0/5 |

| Scenario | Controller | Samples | Mean episode latency [ms] | Mean deadline miss fraction |
|---|---|---:|---:|---:|
| open_space | standard | 1536 | 12.38 | 0.011 |
| open_space | isotropic | 1536 | 12.51 | 0.006 |
| open_space | diagonal | 1536 | 12.69 | 0.031 |
| open_space | full | 1536 | 11.59 | 0.000 |
| narrow_passage | standard | 768 | 13.14 | 0.028 |
| narrow_passage | isotropic | 768 | 13.70 | 0.134 |
| narrow_passage | diagonal | 768 | 13.42 | 0.054 |
| narrow_passage | full | 768 | 13.09 | 0.036 |
| disturbance | standard | 1536 | 11.76 | 0.003 |
| disturbance | isotropic | 1536 | 12.90 | 0.135 |
| disturbance | diagonal | 1536 | 12.08 | 0.005 |
| disturbance | full | 1536 | 11.71 | 0.002 |

These 60 runs also have zero observed collisions and zero constraint violations. Some runs exceed the target despite calibration; the individual timings and miss flags are retained. Even larger batches do not resolve the default final-pose disturbance failure. [Per-seed values](../results/wall_clock/per_seed.csv) and [summaries](../results/wall_clock/summary.csv) provide the remaining metrics.

## C. Native configuration

The supplied native configurations also use K=384, T=28. All **60 native trajectories and command sequences were compared with the matched-evaluation runs and are exactly equal** in the recorded environment. Native success counts therefore match table A. This verifies deterministic reproduction under these settings; it is not 60 additional independent samples of task performance. Timing is measured again and reported independently in [native results](../results/native/per_seed.csv).

Across the three budget modes, reported planning transitions total **259,258,944**. The runner also performs 13,296 one-step **evaluation-only model predictions** to calculate tracking error and 13,296 execution-environment transitions; both are outside the optimizer's planning budget. Timing-calibration work is separately described by the saved calibration grid and five warmed measurements per candidate. It is not included in episode optimization counts.

## Additional environment checks

Standard MPPI, seed 42, native settings:

| Environment | Outcome | Final position error | Collision | Interpretation |
|---|---|---:|---|---|
| Multiple obstacles | Timeout at 18 s | 6.094 m | No | Stalls near the first obstacle; minimum swept clearance 0.080 m. A working environment can reveal a controller failure. |
| Model mismatch | Success at 5.85 s | 0.300 m | No | One execution-only parameter mismatch point; not a sensitivity curve or robustness guarantee. |

Exact resolved configs and output records are under `results/additional/`.

## Figures and animation

Each budget directory contains a `figures/` folder with 13 PNG/SVG pairs: environment overview, analytic covariance distributions/ellipses, executed trajectories, rollout panels and time-series diagnostics for all three tasks, multi-seed summary and latency comparison. Diagnostics include minimum sampled cost, ESS, weight entropy, terminal rollout spread, latency and saturation. A primary-run GIF shows vehicle pose, the updated plan, the retained sample paths, obstacles, goal and MPC updates.

- [Primary benchmark summary](../results/covariance_comparison/figures/benchmark_summary.png)
- [Narrow-passage rollouts](../results/covariance_comparison/figures/narrow_passage_rollouts.png)
- [Covariance distributions](../results/covariance_comparison/figures/sampling_covariances.svg)
- [Disturbance trajectories](../results/covariance_comparison/figures/disturbance_trajectories.png)
- Local animation: `results/covariance_comparison/figures/disturbance_standard_seed_42.gif`

The animation and rollout archives are intentionally ignored by Git. They exist locally and are regenerated by the commands below. Numerical CSV/JSON/YAML evidence and PNG/SVG figures are tracked. MP4 export is not implemented.

## Validation and reproduction

Local validation: **49 pytest tests passed**, Ruff lint passed, Ruff formatting passed, a four-step/64-sample headless smoke episode and figure/GIF reconstruction succeeded, and full seed-42 regressions ran in open space, narrow passage and disturbance. Mathematical tests include an independent Gaussian-density oracle and a quadratic Gibbs target with an analytic posterior mean. Native/primary exact replay and clean-source provenance were verified for all recorded episodes. The Python 3.11/3.12 GitHub Actions workflow is configured; it has not run remotely because the repository has not been published.

From the repository root, with uv installed, use fresh output directories:

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python experiments/run_benchmark.py --mode dynamics --output results/reproduce_dynamics
uv run python experiments/generate_figures.py --input results/reproduce_dynamics --animate
uv run python experiments/run_benchmark.py --mode wall_clock --output results/reproduce_wall_clock
uv run python experiments/generate_figures.py --input results/reproduce_wall_clock
uv run python experiments/run_benchmark.py --mode native --output results/reproduce_native
uv run python experiments/generate_figures.py --input results/reproduce_native
uv run python experiments/run_single.py --scenario multiple_obstacles --seed 42 --output results/reproduce_obstacle_field
uv run python experiments/run_single.py --scenario model_mismatch --seed 42 --output results/reproduce_mismatch
```

Wall-clock calibration can select different sample counts on a different computer or under different load. For exact numerical replay of a recorded wall-clock run, use its saved `config.yaml` through `run_episode` and the recorded seed instead of recalibrating. Latencies are measurements and will differ on rerun.

## Delivery and next milestone

The package uses a src layout with independent dynamics, controllers, scenarios, costs, sampling, evaluation, visualization and configuration modules; [README](../README.md) maps the structure. Only the four Phase A covariance configurations are implemented. CEM MPC, Tsallis-VI, DIAL, Tube, Robust and CC-MPPI remain planned with their mathematical distinctions in [formulations](formulations.md), [paper map](paper_map.md) and [fidelity](implementation_fidelity.md).

Known limitations include concentrated importance weights, local forward-only final-pose behavior, a kinematic disk model, toy obstacle geometry, limited seed counts, and soft timing calibration. These observations do not transfer to tire dynamics, contact-rich locomotion or hardware safety. See [limitations](limitations.md).

Recommended next milestone: preregister sample-count/horizon/covariance-scale sweeps, then implement actual elite-refitting CEM under the same evaluation protocol. A deliberate tracking architecture must precede Tube/RMPPI, and an explicit stochastic state-covariance model must precede CC-MPPI.

No Git remote existed. GitHub CLI authentication was checked and its token is invalid. Local implementation and results are committed; **no GitHub repository was created and no push is claimed**. To publish a private repository after authenticating:

```bash
gh auth login --hostname github.com
gh repo create mppi-variant-playground --private --source=. --remote=origin --push
gh repo view --json url --jq .url
git ls-remote --heads origin main
```

The final delivery commit is available with `git rev-parse HEAD`; the run and analysis source revisions above are preserved inside every artifact. No software license was added.
