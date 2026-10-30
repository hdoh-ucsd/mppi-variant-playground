# Historical research evidence index

This index retains Q01–Q05 configurations, findings and provenance. The [README](../README.md) contains the current formulations, V01–V06 comparisons and mechanism figures. Q06 below is the superseded historical plan.

| Question | Status | Primary numerical evidence |
| --- | --- | --- |
| [Q01 — Sampling covariance and estimation accuracy](#q01) | EXECUTED | [Summary](../runs/seven_state/Q01/seven_state_q01_v1/summary.csv) |
| [Q02 — Importance-weight concentration](#q02) | EXECUTED | [Summary](../runs/seven_state/Q02/seven_state_q02_v1/summary.csv) |
| [Q03 — Frozen-state MPPI decision analysis](#q03) | EXECUTED | [Summary](../runs/seven_state/Q03/seven_state_q03_v1/summary.csv) |
| [Q04 — Closed-loop trajectory differences](#q04) | EXECUTED | [Summary](../runs/seven_state/Q04/seven_state_q04_v1/summary.csv) |
| [Q05 — S-turn difficulty and controller limitations](#q05) | EXECUTED | [Summary](../runs/seven_state/Q05/seven_state_q05_v1/summary.csv) |
| [Q06 — Original formulation plan (superseded by V01–V06)](#q06) | PLANNED | Historical plan only |

## Q01

**Sampling covariance and estimation accuracy** — EXECUTED

**Research question:** How do proposal orientation, anisotropy and budget affect finite-sample error for a fixed Gibbs target?

**Mathematical motivation:** The Gaussian Gibbs mean is (P⁻¹+H/λ)⁻¹Hb/λ, independent of the proposal covariance.

**Hypothesis:** A suitably oriented proposal can reduce finite-sample MSE; misalignment can reverse the effect.

**Main finding:** An independent 64-seed replication preserves positive and negative orientation examples: at K=256, MSE is 0.006505 isotropic, 0.003646 at +45° and 0.065524 at −45°. These are estimates of one exact Gaussian target, not vehicle results.

**Interpretation:** A common Gaussian Gibbs target separates proposal-induced finite-sample error from a changed optimization problem. ESS alone does not determine mean-estimation variance.

**Limitations:** An exact quadratic target is not a vehicle task. Trace matching does not match determinant; empirical 64-seed variance and bias are finite-sample quantities.

**Independent variables:** Orientation, anisotropy, sample count, scale constraint; cost rotation in a separate factorial slice.

**Controlled variables:** Base distribution, quadratic cost and temperature within each comparison; common random numbers per paired seed.

**Metrics:** Bias, variance, MSE, ESS and moment error across 64 evaluation seeds.

**Dynamics model:** analytic_gaussian

**Control inputs:** Gaussian vector (no vehicle)

**Controller configuration:** Fixed Gaussian base/proposal-corrected SNIS; exact target moments

**Seeds:** 34000–34063 (64 independent replication seeds)

**Sampling budget:** K=32…4096; d=2; 28,608 trials

[Exact configuration](../runs/seven_state/Q01/seven_state_q01_v1/config.yaml)

[Primary figure](../results/historical/showcase/Q01_key.png) · [Numerical summary](../runs/seven_state/Q01/seven_state_q01_v1/summary.csv)

- **seven_state_q01_v1**: [manifest](../runs/seven_state/Q01/seven_state_q01_v1/manifest.json), [summary](../runs/seven_state/Q01/seven_state_q01_v1/summary.csv), source `0006f840f99ab6be155b66ea7833e9210fe40367`.

```bash
uv run python experiments/legacy/seven_state/seven_state_campaign.py --stage q01 --output runs/seven_state/Q01/q01_reproduction
uv run python experiments/legacy/seven_state/plot_seven_state.py --stage Q01 --input runs/seven_state/Q01/seven_state_q01_v1
```


## Q02

**Importance-weight concentration** — EXECUTED

**Research question:** When do cost weighting and Gaussian density corrections concentrate importance weights, and how does ESS relate to error?

**Mathematical motivation:** ESS=1/Σw² measures weight concentration, whereas estimator variance also depends on the integrand.

**Hypothesis:** Sharper Boltzmann factors and high-dimensional proposal mismatch can each concentrate weights.

**Main finding:** At K=512 the isotropic mean ESS grows from 40.56 at λ=.1 to 450.61 at λ=10, while mean error changes from .04284 to .05207 against each changed exact target. Cost-only, ratio-only and combined dimension probes separate concentration mechanisms.

**Interpretation:** ESS summarizes normalized weights. Temperature changes the target; density ratios alone can concentrate in high dimensions.

**Limitations:** Temperature changes the target. ESS does not certify estimation accuracy or target coverage. Seven states correspond to a 40-dimensional sampled control sequence here.

**Independent variables:** Temperature and dimension; cost-only, ratio-only and combined controls.

**Controlled variables:** Analytic target recomputed for each temperature; paired seeds and sample budgets within each comparison.

**Metrics:** ESS, entropy, largest weight, exact mean error and Gaussian overlap diagnostics.

**Dynamics model:** analytic_gaussian

**Control inputs:** Gaussian vector (no vehicle)

**Controller configuration:** Fixed Gaussian base/proposal-corrected SNIS; exact target moments

**Seeds:** 34000–34063 (64 independent replication seeds)

**Sampling budget:** K=512; d=2…56; 4,544 trials

[Exact configuration](../runs/seven_state/Q02/seven_state_q02_v1/config.yaml)

[Primary figure](../results/historical/showcase/Q02_key.png) · [Numerical summary](../runs/seven_state/Q02/seven_state_q02_v1/summary.csv)

- **seven_state_q02_v1**: [manifest](../runs/seven_state/Q02/seven_state_q02_v1/manifest.json), [summary](../runs/seven_state/Q02/seven_state_q02_v1/summary.csv), source `0006f840f99ab6be155b66ea7833e9210fe40367`.

```bash
uv run python experiments/legacy/seven_state/seven_state_campaign.py --stage q02 --output runs/seven_state/Q02/q02_reproduction
uv run python experiments/legacy/seven_state/plot_seven_state.py --stage Q02 --input runs/seven_state/Q02/seven_state_q02_v1
```


## Q03

**Frozen-state MPPI decision analysis** — EXECUTED

**Research question:** How does changing only proposal covariance alter one MPPI decision at a frozen state and warm start?

**Mathematical motivation:** Fixing p, J, λ and the proposal mean isolates the covariance intervention in self-normalized importance sampling.

**Hypothesis:** Covariance changes finite-budget decisions; differences may diminish with K if sampling adequately covers the target.

**Main finding:** 7,200 seven-state frozen decisions show finite-budget differences. Only the post-disturbance high-sample reference passes the declared diagnostics: standard RMS error drops .1085→.0140 from K=64→4096; sensitivity-aligned .2471→.1597. Open-space and obstacle references remain unresolved.

**Interpretation:** Proposal geometry changes finite sampled rollouts, corrected weights, controls and open-loop predictions, even with the same target.

**Limitations:** Two reference means fail overlap/precision diagnostics. Distances to those estimates are not validated target errors. Local sensitivity alignment is not target covariance matching; three states and one warm start each limit inference.

**Independent variables:** Six trace-matched proposals, K=64…4096 and 50 paired evaluation seeds.

**Controlled variables:** Identical seven-state observation, latent warm start, previous control, model, cost, horizon, base, temperature and paired standard normals; only proposal covariance differs.

**Metrics:** Paired command differences, ESS, entropy, mean-plan objective, reference diagnostics and latency.

**Dynamics model:** dynamic_single_track_7

**Control inputs:** steering [rad], signed normalized throttle [−1,1]

**Controller configuration:** Gaussian IT-MPPI; T=20, λ=8, base diag(.14,.26); standard/isotropic/diagonal/full normalized trace .40; Q03 adds aligned/misaligned

**Seeds:** 33000–33049, source 31415; reference namespaces [202,state,rep]

**Sampling budget:** K=64,128,256,384,512,1024,2048,4096; 7,200 decisions + 3×8×65,536 independent reference rollouts

[Exact configuration](../runs/seven_state/Q03/seven_state_q03_v1/config.yaml)

[Primary figure](../results/historical/showcase/Q03_key.png) · [Numerical summary](../runs/seven_state/Q03/seven_state_q03_v1/summary.csv)

- **seven_state_q03_v1**: [manifest](../runs/seven_state/Q03/seven_state_q03_v1/manifest.json), [summary](../runs/seven_state/Q03/seven_state_q03_v1/summary.csv), source `0006f840f99ab6be155b66ea7833e9210fe40367`.

[Supporting figure](../docs/figures/seven_state_Q03_convergence.png)

[Supporting figure](../docs/figures/seven_state_Q03_commands.png)

```bash
uv run python experiments/legacy/seven_state/seven_state_campaign.py --stage q03 --output runs/seven_state/Q03/q03_reproduction
uv run python experiments/legacy/seven_state/plot_seven_state.py --stage Q03 --input runs/seven_state/Q03/seven_state_q03_v1
```


## Q04

**Closed-loop trajectory differences** — EXECUTED

**Research question:** How do covariance settings change executed route distributions, and which distribution does each plot describe?

**Mathematical motivation:** Closed-loop feedback changes observed states and warm starts, so static estimation benefits need not transfer.

**Hypothesis:** Proposal changes may accumulate into visibly different route and recovery distributions.

**Main finding:** Across five seven-state scenes, compatibility completion is 397/400 and independent holdout completion 391/400, with zero disk-proxy collisions. In holdout two-route trials, full takes 17 upper routes out of 20; standard/isotropic take 5 and diagonal 2. Route frequencies characterize these matrices, not published algorithm families.

**Interpretation:** Executed bundles, occupancy, local rollout clouds and inter-seed dispersion describe different distributions. A mean over routes need not be feasible.

**Limitations:** Twenty seeds per scenario and cohort give wide intervals. Feedback changes state and warm start. Failures are terminal-region timeouts; one calibration and selected matrices cannot establish general superiority. Collision counts use a centered disk proxy; full-body avoidance is not established.

**Independent variables:** Four covariance settings; five scenarios; separate compatibility and holdout cohorts; K in a controlled obstacle-task sweep.

**Controlled variables:** Costs, dynamics, limits, budgets and exogenous disturbances within each scenario.

**Metrics:** Completion, collision, clearance, route station crossing, occupancy, ESS and progress-aligned dispersion.

**Dynamics model:** dynamic_single_track_7

**Control inputs:** steering [rad], signed normalized throttle [−1,1]

**Controller configuration:** Gaussian IT-MPPI; T=20, λ=8, base diag(.14,.26); standard/isotropic/diagonal/full normalized trace .40; Q03 adds aligned/misaligned

**Seeds:** 42–61 compatibility; 31000–31019 independent holdout; 36000–36009 isolated timing

**Sampling budget:** Primary K=384; T=20; 800 episodes; separate K=128/384/768/1536 sweep

[Exact configuration](../runs/seven_state/Q04/seven_state_q04_v1/config.yaml)

[Primary figure](../results/historical/showcase/Q04_key.png) · [Numerical summary](../runs/seven_state/Q04/seven_state_q04_v1/summary.csv)

- **seven_state_q04_v1**: [manifest](../runs/seven_state/Q04/seven_state_q04_v1/manifest.json), [summary](../runs/seven_state/Q04/seven_state_q04_v1/summary.csv), source `fe98accd84213647c0ff7c006226df6ef1dd3d1f`.

- **seven_state_q04_pilot_v1**: [manifest](../runs/seven_state/Q04/seven_state_q04_pilot_v1/manifest.json), [summary](../runs/seven_state/Q04/seven_state_q04_pilot_v1/summary.csv), source `0006f840f99ab6be155b66ea7833e9210fe40367`.

- **seven_state_q04_budget_v1**: [manifest](../runs/seven_state/Q04/seven_state_q04_budget_v1/manifest.json), [summary](../runs/seven_state/Q04/seven_state_q04_budget_v1/summary.csv), source `3ced0faafe0390e6377dfe85498787a33a5b2d39`.

- **seven_state_q04_timing_v1**: [manifest](../runs/seven_state/Q04/seven_state_q04_timing_v1/manifest.json), [summary](../runs/seven_state/Q04/seven_state_q04_timing_v1/summary.csv), source `79d626f2ba41e88416a52993d98599e4ce9acae5`.

```bash
uv run python experiments/legacy/seven_state/seven_state_campaign.py --stage q04 --output runs/seven_state/Q04/q04_reproduction
uv run python experiments/legacy/seven_state/plot_seven_state.py --stage Q04 --input runs/seven_state/Q04/seven_state_q04_v1
```


## Q05

**S-turn difficulty and controller limitations** — EXECUTED

**Research question:** At what corridor clearance do specific proposal covariances produce measurable differences in controlled S-turn traversal?

**Mathematical motivation:** Finite-budget importance sampling, actuator limits and a restricted horizon interact with feasible-set geometry.

**Hypothesis:** Sequential steering reversals with less clearance may expose proposal inefficiency; no proposal is expected to win by construction.

**Main finding:** Seven-state S-turn completion: 0.9 m: standard 20/20, isotropic 19/20, diagonal 20/20, full 20/20; 0.6 m: standard 20/20, isotropic 20/20, diagonal 20/20, full 20/20; 0.35 m: standard 20/20, isotropic 19/20, diagonal 20/20, full 20/20; 0.2 m: standard 20/20, isotropic 16/20, diagonal 20/20, full 19/20. Across 320 primary episodes, 0 disk-proxy collisions; all failed trajectories are retained. The separate state-jump checks have 44 collisions, reported separately.

**Interpretation:** A larger observed completion gap at the tightest clearance coexists with terminal speed/heading failures; this does not isolate covariance-driven route discovery.

**Limitations:** One handedness, one horizon and specific matrices limit generalization. Completion allows straight-runout recovery. Intervals are pointwise. Discrete command slew is separate; continuous actuator dynamics are omitted. Executed collision counts use a centered disk proxy that does not enclose the axles; full-body collision avoidance is not established.

**Independent variables:** Primary: clearance .90,.60,.35,.20 m. Separate K=128/384/768/1536 at .35 m; separate cruising-speed and disturbance checks; discrete command slew studied separately; continuous actuator dynamics omitted.

**Controlled variables:** Vehicle, objective, speed target, steering limit, horizon and samples within each clearance condition.

**Metrics:** Success/collision, censored progress, cross-track error, clearance, completion time, evaluation cost, ESS, entropy, effort, rate saturation, latency and transitions.

**Dynamics model:** dynamic_single_track_7

**Control inputs:** steering [rad], signed normalized throttle [−1,1]

**Controller configuration:** Gaussian IT-MPPI; T=20, λ=8, base diag(.14,.26); standard/isotropic/diagonal/full normalized trace .40; Q03 adds aligned/misaligned

**Seeds:** 42 pilot; 32000–32019 primary holdout (20 per setting); 35000–35019 cruise/impulse; 37000–37019 slew; 36000–36009 timing

**Sampling budget:** Primary K=384; T=20; 320 episodes; separate K sweep

[Exact configuration](../runs/seven_state/Q05/seven_state_q05_v1/config.yaml)

[Primary figure](../results/historical/showcase/Q05_key.png) · [Numerical summary](../runs/seven_state/Q05/seven_state_q05_v1/summary.csv)

- **seven_state_q05_v1**: [manifest](../runs/seven_state/Q05/seven_state_q05_v1/manifest.json), [summary](../runs/seven_state/Q05/seven_state_q05_v1/summary.csv), source `7bea369e3455bf62785fcf8fc0db59c8bcbfc6a7`.

- **seven_state_q05_pilot_v1**: [manifest](../runs/seven_state/Q05/seven_state_q05_pilot_v1/manifest.json), [summary](../runs/seven_state/Q05/seven_state_q05_pilot_v1/summary.csv), source `fe98accd84213647c0ff7c006226df6ef1dd3d1f`.

- **seven_state_q05_budget_v1**: [manifest](../runs/seven_state/Q05/seven_state_q05_budget_v1/manifest.json), [summary](../runs/seven_state/Q05/seven_state_q05_budget_v1/summary.csv), source `3ced0faafe0390e6377dfe85498787a33a5b2d39`.

- **seven_state_q05_secondary_v1**: [manifest](../runs/seven_state/Q05/seven_state_q05_secondary_v1/manifest.json), [summary](../runs/seven_state/Q05/seven_state_q05_secondary_v1/summary.csv), source `3ced0faafe0390e6377dfe85498787a33a5b2d39`.

- **seven_state_q05_rate_v1**: [manifest](../runs/seven_state/Q05/seven_state_q05_rate_v1/manifest.json), [summary](../runs/seven_state/Q05/seven_state_q05_rate_v1/summary.csv), source `afc63ebc9495cff3c4469b80a86e170684af2996`.

- **seven_state_q05_timing_v1**: [manifest](../runs/seven_state/Q05/seven_state_q05_timing_v1/manifest.json), [summary](../runs/seven_state/Q05/seven_state_q05_timing_v1/summary.csv), source `79d626f2ba41e88416a52993d98599e4ce9acae5`.

- **seven_state_reference_footprint_v1**: [manifest](../runs/seven_state/validation/footprint_v1/manifest.json), [summary](../runs/seven_state/validation/footprint_v1/validation.json), source `79d626f2ba41e88416a52993d98599e4ce9acae5`.

```bash
uv run python experiments/legacy/seven_state/seven_state_campaign.py --stage q05 --output runs/seven_state/Q05/q05_reproduction
uv run python experiments/legacy/seven_state/plot_seven_state.py --stage Q05 --input runs/seven_state/Q05/seven_state_q05_v1
```


## Q06

**Original formulation plan (superseded by V01–V06)** — PLANNED

**Research question:** Which defining mathematical changes in published MPPI formulations produce observable mechanisms under their stated assumptions?

**Mathematical motivation:** Different objectives and feedback architectures cannot be compared merely by renaming one Gaussian sampler.

**Hypothesis:** Each defining mathematical change should produce a measurable mechanism under its own assumptions.

**Main finding:** This historical experiment ID was never executed. Its plan was superseded by the implemented and executed V01–V06 formulation studies and mechanism experiments in README.md; CEM remains planned.

**Interpretation:** PLANNED describes this archived question ID, not the implementation status of the current controllers.

**Limitations:** Use the current README for formulation fidelity, experiments and limits; do not interpret this historical plan as current implementation status.

**Independent variables:** Future divergence/weighting law, search schedule, feedback architecture and state covariance objectives.

**Controlled variables:** To be declared per formulation, including target, tracker, noise assumptions and all inner-loop work.

**Metrics:** To be declared: update identities, oracle checks, task outcomes, actual computational work and assumptions.

**Dynamics model:** Not specified in the historical plan

**Control inputs:** Not specified in the historical plan

**Controller configuration:** Not specified for this historical ID; current implementations are documented in README.md.

**Seeds:** Not specified in the historical plan

**Sampling budget:** Not specified in the historical plan

[Exact configuration](../configs/investigations/published_formulations_plan.yaml)

```bash
uv run python experiments/benchmarks/variant_benchmark.py --study V06_unified --output results/variants/benchmarks/V06_reproduction --raw runs/variants/V06_reproduction
uv run python experiments/mechanisms/formulation_mechanisms.py --plot-only --output results/variants/mechanisms
```

## Storage and recovery

- `results/historical/showcase/` retains the five primary Q01–Q05 figures.
- `runs/seven_state/` retains exact configurations, compact summaries and provenance. Full episodes and rollout arrays are ignored; reproduce them before raw audits.
- `results/variants/` retains current formulation comparisons and mechanism evidence.
- [Historical recovery index](../results/historical/archive/legacy_index.md) links superseded documents and bulky earlier results at their immutable pre-cleanup revision.

[Seven-state model equations, validation and campaign report](seven_state_benchmark_report.md)

Validate or regenerate this single index (the README is maintained separately):

```bash
uv run python experiments/legacy/research_index.py --render
uv run python experiments/legacy/research_index.py
```
