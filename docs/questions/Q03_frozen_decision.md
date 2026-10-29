# Q03 — Frozen-state MPPI decision analysis

**Status: EXECUTED**

## Research question

How does changing only proposal covariance alter one MPPI decision at a frozen state and warm start?

## Mathematical motivation

Fixing p, J, λ and the proposal mean isolates the covariance intervention in self-normalized importance sampling.

## Hypothesis

Covariance changes finite-budget decisions; differences may diminish with K if sampling adequately covers the target.

## Independent variables

Six trace-matched proposals, K=64…4096 and 50 paired evaluation seeds.

## Controlled variables

Identical seven-state observation, latent warm start, previous control, model, cost, horizon, base, temperature and paired standard normals; only proposal covariance differs.

## Metrics

Paired command differences, ESS, entropy, mean-plan objective, reference diagnostics and latency.

## Main finding

7,200 seven-state frozen decisions show finite-budget differences. Only the post-disturbance high-sample reference passes the declared diagnostics: standard RMS error drops .1085→.0140 from K=64→4096; sensitivity-aligned .2471→.1597. Open-space and obstacle references remain unresolved.

## Interpretation

Proposal geometry changes finite sampled rollouts, corrected weights, controls and open-loop predictions, even with the same target.

## Limitations

Two reference means fail overlap/precision diagnostics. Distances to those estimates are not validated target errors. Local sensitivity alignment is not target covariance matching; three states and one warm start each limit inference.

## Registered experiment

| Field | Value |
| --- | --- |
| experiment name | Seven-state frozen MPPI decisions at three common inputs |
| dynamics model | dynamic_single_track_7 |
| control inputs | steering [rad], signed normalized throttle [−1,1] |
| controller configuration | Gaussian IT-MPPI; T=20, λ=8, base diag(.14,.26); standard/isotropic/diagonal/full normalized trace .40; Q03 adds aligned/misaligned |
| seeds | 33000–33049, source 31415; reference namespaces [202,state,rep] |
| sampling budget | K=64,128,256,384,512,1024,2048,4096; 7,200 decisions + 3×8×65,536 independent reference rollouts |
| source commit | 0006f840f99ab6be155b66ea7833e9210fe40367 |

## Configuration and provenance

[Original configuration](../../runs/seven_state/Q03/seven_state_q03_v1/config.yaml)

Source run IDs: seven_state_q03_v1, frozen_decision_v1

- [Run frozen_decision_v1](../../results/investigations/frozen_decision_v1/): [manifest](../../results/investigations/frozen_decision_v1/manifest.json), source `7aaf76eaed90588b9404b2fbed483ae9efce1156`; model: kinematic_bicycle_4; inputs: acceleration [m/s²], steering [rad].

- [Run seven_state_q03_v1](../../runs/seven_state/Q03/seven_state_q03_v1/): [manifest](../../runs/seven_state/Q03/seven_state_q03_v1/manifest.json), source `0006f840f99ab6be155b66ea7833e9210fe40367`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle [−1,1].

## Selected figure

![Frozen-state MPPI decision analysis](../../results/showcase/Q03_key.png)

[Numeric summary](../../runs/seven_state/Q03/seven_state_q03_v1/summary.csv)

## Reproduce and regenerate

```bash
uv run python experiments/seven_state_campaign.py --stage q03 --output runs/seven_state/Q03/q03_reproduction
uv run python experiments/plot_seven_state.py --stage Q03 --input runs/seven_state/Q03/seven_state_q03_v1
```

## Supporting numerical records

- [per_seed.csv](../../runs/seven_state/Q03/seven_state_q03_v1/per_seed.csv)

## Supporting figures

- [Saved-data figure](../../docs/figures/seven_state_Q03_convergence.png)
- [Saved-data figure](../../docs/figures/seven_state_Q03_commands.png)

[Seven-state campaign report and methods](../seven_state_benchmark_report.md)

[Historical methods (earlier model / original analytic seeds)](../frozen_decision_analysis.md)
