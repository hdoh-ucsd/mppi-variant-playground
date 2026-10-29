# Q04 — Closed-loop trajectory differences

**Status: EXECUTED**

## Research question

How do covariance settings change executed route distributions, and which distribution does each plot describe?

## Mathematical motivation

Closed-loop feedback changes observed states and warm starts, so static estimation benefits need not transfer.

## Hypothesis

Proposal changes may accumulate into visibly different route and recovery distributions.

## Independent variables

Four covariance settings; five scenarios; separate compatibility and holdout cohorts; K in a controlled obstacle-task sweep.

## Controlled variables

Costs, dynamics, limits, budgets and exogenous disturbances within each scenario.

## Metrics

Completion, collision, clearance, route station crossing, occupancy, ESS and progress-aligned dispersion.

## Main finding

Across five seven-state scenes, compatibility completion is 397/400 and independent holdout completion 391/400, with zero disk-proxy collisions. In holdout two-route trials, full takes 17 upper routes out of 20; standard/isotropic take 5 and diagonal 2. Route frequencies characterize these matrices, not published algorithm families.

## Interpretation

Executed bundles, occupancy, local rollout clouds and inter-seed dispersion describe different distributions. A mean over routes need not be feasible.

## Limitations

Twenty seeds per scenario and cohort give wide intervals. Feedback changes state and warm start. Failures are terminal-region timeouts; one calibration and selected matrices cannot establish general superiority. Collision counts use a centered disk proxy; full-body avoidance is not established.

## Registered experiment

| Field | Value |
| --- | --- |
| experiment name | Seven-state five-scenario paired closed-loop campaign |
| dynamics model | dynamic_single_track_7 |
| control inputs | steering [rad], signed normalized throttle [−1,1] |
| controller configuration | Gaussian IT-MPPI; T=20, λ=8, base diag(.14,.26); standard/isotropic/diagonal/full normalized trace .40; Q03 adds aligned/misaligned |
| seeds | 42–61 compatibility; 31000–31019 independent holdout; 36000–36009 isolated timing |
| sampling budget | Primary K=384; T=20; 800 episodes; separate K=128/384/768/1536 sweep |
| source commit | fe98accd84213647c0ff7c006226df6ef1dd3d1f |

## Configuration and provenance

[Original configuration](../../runs/seven_state/Q04/seven_state_q04_v1/config.yaml)

Source run IDs: seven_state_q04_v1, seven_state_q04_timing_v1, seven_state_q04_budget_v1, seven_state_q04_pilot_v1, trajectory_behavior_v1

- [Run trajectory_behavior_v1](../../results/investigations/trajectory_behavior_v1/): [manifest](../../results/investigations/trajectory_behavior_v1/manifest.json), source `077e7d90eb2f2d3fd302a284daf5d9868bf28135`; model: kinematic_bicycle_4; inputs: acceleration [m/s²], steering [rad].

- [Run seven_state_q04_v1](../../runs/seven_state/Q04/seven_state_q04_v1/): [manifest](../../runs/seven_state/Q04/seven_state_q04_v1/manifest.json), source `fe98accd84213647c0ff7c006226df6ef1dd3d1f`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle [−1,1].

- [Run seven_state_q04_pilot_v1](../../runs/seven_state/Q04/seven_state_q04_pilot_v1/): [manifest](../../runs/seven_state/Q04/seven_state_q04_pilot_v1/manifest.json), source `0006f840f99ab6be155b66ea7833e9210fe40367`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle [−1,1].

- [Run seven_state_q04_budget_v1](../../runs/seven_state/Q04/seven_state_q04_budget_v1/): [manifest](../../runs/seven_state/Q04/seven_state_q04_budget_v1/manifest.json), source `3ced0faafe0390e6377dfe85498787a33a5b2d39`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle [−1,1].

- [Run seven_state_q04_timing_v1](../../runs/seven_state/Q04/seven_state_q04_timing_v1/): [manifest](../../runs/seven_state/Q04/seven_state_q04_timing_v1/manifest.json), source `79d626f2ba41e88416a52993d98599e4ce9acae5`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle.

## Selected figure

![Closed-loop trajectory differences](../../results/showcase/Q04_key.png)

[Numeric summary](../../runs/seven_state/Q04/seven_state_q04_v1/summary.csv)

## Reproduce and regenerate

```bash
uv run python experiments/seven_state_campaign.py --stage q04 --output runs/seven_state/Q04/q04_reproduction
uv run python experiments/plot_seven_state.py --stage Q04 --input runs/seven_state/Q04/seven_state_q04_v1
```

## Supporting numerical records

- [per_seed.csv](../../runs/seven_state/Q04/seven_state_q04_v1/per_seed.csv)
- [paired_differences.csv](../../runs/seven_state/Q04/seven_state_q04_v1/paired_differences.csv)
- [goal_summary.csv](../../runs/seven_state/Q04/seven_state_q04_v1/goal_summary.csv)
- [failure_modes.csv](../../runs/seven_state/Q04/seven_state_q04_v1/failure_modes.csv)
- [selected_weight_components.csv](../../runs/seven_state/Q04/seven_state_q04_v1/selected_weight_components.csv)

## Supporting figures

- [Saved-data figure](../../docs/figures/seven_state_Q04_diagnostics.png)
- [Saved-data figure](../../docs/figures/seven_state_budget_seven_state_q04_budget_v1.png)

[Seven-state campaign report and methods](../seven_state_benchmark_report.md)

[Historical methods (earlier model / original analytic seeds)](../trajectory_behavior_findings.md)
