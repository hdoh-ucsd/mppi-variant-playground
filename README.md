# MPPI Variant Playground

**[Formulations](docs/formulations.md) · [Paper Map](docs/paper_map.md)**

**A reproducible research playground for understanding *why* Model Predictive Path Integral (MPPI) controllers behave differently.**

MPPI plans by sampling control sequences, predicting their trajectories, and updating a nominal plan using cost-dependent weights. This repository studies how changes to the **sampling proposal** affect finite-sample estimates, selected controls, and closed-loop behavior. It connects mathematical derivations to controlled toy problems and a seven-state vehicle benchmark.

> **Scope.** The implemented controller is Gaussian information-theoretic MPPI (IT-MPPI) with several **fixed proposal covariance settings**. These are *not* implementations of separate published MPPI variants. Tsallis-VI, DIAL-MPC, Tube-MPPI, Robust-MPPI, and CC-MPPI remain future comparison targets.

## Start here

- **Understand the math:** [Formulations](docs/formulations.md) · [Paper map](docs/paper_map.md)
- **Browse the evidence:** [Research question index](docs/research_index.md) · [Machine-readable registry](results/registry.yaml)
- **Reproduce an experiment:** [Q03 frozen-state decisions](docs/questions/Q03_frozen_decision.md) · [Q05 S-turn difficulty](docs/questions/Q05_s_turn_difficulty.md)
- **Audit the model and findings:** [Seven-state scientific report](docs/seven_state_benchmark_report.md) · [Implementation fidelity](docs/implementation_fidelity.md)

## The central question

**If several controllers seek the same MPPI target distribution, when—and why—does the way they sample controls change their decisions?**

A proposal covariance can alter which trajectories are sampled and how concentrated the importance weights become, without changing the ideal target distribution. We test that distinction at three scales:

1. **Analytic:** compare an estimator to a known Gaussian target.
2. **Decision-level:** freeze the physical state and nominal control sequence, then compare MPPI updates.
3. **Closed-loop:** execute the same vehicle tasks across paired random seeds and increasing difficulty.

## Mathematical baseline

Let $Z=(z_0,\ldots,z_{T-1})$ be a sequence of normalized latent control inputs, $J(Z)$ its rollout cost, and $\lambda>0$ a temperature. The implemented controller uses a fixed Gaussian base distribution $p$, and samples from a possibly different Gaussian proposal $q_{U,C}$:

$$
\pi^*(Z)\;\propto\;\exp[-J(Z)/\lambda]p(Z),
\qquad
\tilde w_k = \exp[-J(Z_k)/\lambda]\frac{p(Z_k)}{q_{U,C}(Z_k)}.
$$

After normalizing $w_k=\tilde w_k/\sum_j\tilde w_j$, it computes

$$
U^+ = \sum_{k=1}^{K}w_k Z_k.
$$

Changing proposal covariance $C$ does not change $\pi^*$ when the base, objective, and temperature are held fixed. It **does** change the finite-sample estimate, including the density correction $p/q$. Gaussian likelihoods are evaluated in latent coordinates before physical controls are clipped. Details and implementation assumptions: [Formulations](docs/formulations.md).

## Research questions and key figures

Each completed question has **one primary figure** below. Follow its link for complete methods, uncertainty, negative results, configurations, and reproduction commands. The figures summarize evidence; they do not replace the run records.

### Q01 · Does covariance geometry improve finite-sample estimation?

![Q01: sampling covariance and estimation accuracy](results/showcase/Q01_key.png)

For a **fixed, analytically solvable Gaussian target**, proposal orientation can help *or* hurt. At $K=256$ with 64 independent replication seeds, observed mean squared error was **0.003646** for the +45° full proposal, **0.006505** for isotropic, and **0.065524** for the −45° full proposal. These results concern estimation, **not vehicle performance**. [Experiment Q01](docs/questions/Q01_sampling_accuracy.md).

### Q02 · What makes importance weights concentrate?

![Q02: importance-weight concentration](results/showcase/Q02_key.png)

Temperature and high-dimensional proposal mismatch can concentrate weights by different mechanisms. Effective sample size (ESS) is an overlap diagnostic, **not a guarantee of estimator accuracy**; changing temperature also changes the target. [Experiment Q02](docs/questions/Q02_weight_degeneracy.md).

### Q03 · How does sampling geometry change one MPPI decision?

![Q03: frozen-state sampling, weighting and updated plan](results/showcase/Q03_key.png)

At an identical seven-state vehicle observation and warm start, different proposal covariances produce different sampled trajectories, corrected weights, and updated controls. The high-sample target reference is **unresolved** for the open-space and pre-obstacle decisions; only the post-disturbance reference passes the stated diagnostic screens. The pre-obstacle reference's pooled ESS is just **1.51 despite 524,288 samples**. [Experiment Q03](docs/questions/Q03_frozen_decision.md).

### Q04 · Do decision differences persist in closed-loop driving?

![Q04: seven-state vehicle closed-loop comparisons](results/showcase/Q04_key.png)

The seven-state study recorded **391 completions in 400 independent holdout episodes** across five scenarios. Completion is often nearly saturated, but route selection can differ: in the two-route task, the full proposal selected the upper route in **17/20** holdout runs, versus **5/20** for standard and isotropic and **2/20** for diagonal. These are observations about specific covariance matrices, **not general rankings of MPPI families**. [Experiment Q04](docs/questions/Q04_closed_loop.md).

### Q05 · When does a more demanding S-turn expose limitations?

![Q05: S-turn success, representative paths and lateral dynamics](results/showcase/Q05_key.png)

In the primary seven-state S-turn study ($K=384$, 20 paired holdout seeds per setting), **313/320 episodes completed**. At the tightest 0.20 m clearance, completions were 20/20 standard, 16/20 isotropic, 20/20 diagonal, and 19/20 full. All seven primary failures were terminal-state timeouts; there were zero disk-proxy collisions. Separate studies investigate command-slew limits, speed preference, state-jump disturbances and sample budgets; they are **not pooled** into the primary clearance results. [Experiment Q05](docs/questions/Q05_s_turn_difficulty.md).

### Q06 · What changes in the published MPPI variants?

**Planned; no comparative performance results yet.** The [paper map](docs/paper_map.md) organizes work on divergence/weighting (Tsallis), annealing (DIAL), tracking and robustness (Tube/RMPPI), and state covariance/feedback (CC-MPPI). These require faithful, distinct implementations before cross-method performance claims. [Experiment plan Q06](docs/questions/Q06_variant_formulations.md).

## Dynamics and controls

| Model | States | Physical control input | Role |
| --- | --- | --- | --- |
| Kinematic bicycle | 4 | acceleration, steering angle | Historical baseline |
| Steering-rate bicycle | 5 | acceleration, steering rate | Historical S-turn study |
| Dynamic single-track vehicle | 7 | steering angle, signed normalized throttle | Main Q03–Q05 campaign |

The primary vehicle state is $[p_x,p_y,\psi,\phi,v_x,v_y,r]$: global position, heading, roll, body-frame longitudinal/lateral velocity and yaw rate. Its physics model includes tire-force saturation, planar body dynamics and an overdamped roll state. It is **not** the learned AutoRally model from Williams et al. The control-sequence sampling dimension is $2T$, **not seven**; the main vehicle campaign uses $T=20$ and therefore a 40-dimensional latent control sequence. [Model equations and validation](docs/seven_state_benchmark_report.md).

## Run and reproduce

Requirements: Linux, Python 3.11+, [`uv`](https://docs.astral.sh/uv/). The benchmark is CPU-oriented; no GPU is required.

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python experiments/research_index.py
```

Rebuild the Q03 visualization from **already saved** numerical records, without running the controller again:

```bash
uv run python experiments/plot_seven_state.py \
  --stage Q03 \
  --input runs/seven_state/Q03/seven_state_q03_v1
```

For fresh simulation runs, use each question's **Reproduce and regenerate** section rather than overwriting completed run IDs. See the [research index](docs/research_index.md) for all experiments.

### Repository map

```text
configs/                 Controllers, costs, dynamics and scenarios
src/                     MPPI, dynamics, sampling, evaluation and analysis
experiments/             Simulation and saved-data reconstruction entry points
docs/questions/          One research question per document
results/showcase/        Exactly one curated key image per executed question
results/registry.yaml    Experiments, claims, run IDs and provenance
runs/seven_state/Qxx/     Question-specific summaries/configs and local raw data
tests/                   Numerical and reproducibility checks
```

Full episode trajectories and rollout arrays are not all version-controlled. **A fresh clone includes curated figures and compact evidence, but not every raw rollout.** Rerun the question-specific commands to regenerate omitted data. See [storage and provenance](docs/research_index.md#storage-and-provenance).

## How to interpret the results

These experiments are **diagnostic, not a leaderboard**.

- The compared covariance matrices are fixed choices within one IT-MPPI implementation. Their performance does not establish universal superiority of isotropic, diagonal, or full covariance.
- Q01–Q02's analytic studies are **vehicle-independent**; Q03–Q05 use the seven-state physics approximation.
- A high ESS, low sampled cost, or visually broad rollout cloud does not by itself establish a good or safe executed control decision. Two Q03 reference estimates remain unresolved.
- Reported collision counts use a **centered disk proxy**, not a full-vehicle collision certificate. The S-turn reference has an additional capsule check, but the complete executed campaigns do not establish full-body safety.
- Q05's primary success criterion allows terminal runout/recovery; it is not a stopped-point parking test. Secondary exogenous state jumps can directly cause boundary violations and are analyzed separately.
- Equal-$K$ experiments are **not equal-wall-clock comparisons**. In isolated S-turn timing probes, $K=384$ took roughly 191–198 ms per plan, longer than the 150 ms simulation control interval. This implementation has not demonstrated a real-time deadline guarantee.

For equations, seed protocols, detailed failure cases, uncertainty, and limitations, read the [full seven-state report](docs/seven_state_benchmark_report.md).

## Reading and attribution

The starting point is Williams et al., [*Information Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving*](https://arxiv.org/abs/1707.02342). The [paper map](docs/paper_map.md) records the other MPPI families and how their formulations differ. This repository is an independent educational/research implementation, not an official implementation of those papers.
