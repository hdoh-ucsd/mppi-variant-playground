# Q05 — S-turn difficulty and controller limitations

**Status: EXECUTED**

## Research question

At what corridor clearance do specific proposal covariances produce measurable differences in controlled S-turn traversal?

## Mathematical motivation

Finite-budget importance sampling, actuator limits and a restricted horizon interact with feasible-set geometry.

## Hypothesis

Sequential steering reversals with less clearance may expose proposal inefficiency; no proposal is expected to win by construction.

## Independent variables

Primary: clearance .90,.60,.35,.20 m. Separate K=128/384/768/1536 at .35 m; separate cruising-speed and disturbance checks; discrete command slew studied separately; continuous actuator dynamics omitted.

## Controlled variables

Vehicle, objective, speed target, steering limit, horizon and samples within each clearance condition.

## Metrics

Success/collision, censored progress, cross-track error, clearance, completion time, evaluation cost, ESS, entropy, effort, rate saturation, latency and transitions.

## Main finding

Seven-state S-turn completion: 0.9 m: standard 20/20, isotropic 19/20, diagonal 20/20, full 20/20; 0.6 m: standard 20/20, isotropic 20/20, diagonal 20/20, full 20/20; 0.35 m: standard 20/20, isotropic 19/20, diagonal 20/20, full 20/20; 0.2 m: standard 20/20, isotropic 16/20, diagonal 20/20, full 19/20. Across 320 primary episodes, 0 disk-proxy collisions; all failed trajectories are retained. The separate state-jump checks have 44 collisions, reported separately.

## Interpretation

A larger observed completion gap at the tightest clearance coexists with terminal speed/heading failures; this does not isolate covariance-driven route discovery.

## Limitations

One handedness, one horizon and specific matrices limit generalization. Completion allows straight-runout recovery. Intervals are pointwise. Discrete command slew is separate; continuous actuator dynamics are omitted. Executed collision counts use a centered disk proxy that does not enclose the axles; full-body collision avoidance is not established.

## Registered experiment

| Field | Value |
| --- | --- |
| experiment name | Seven-state feasible S-turn clearance campaign |
| dynamics model | dynamic_single_track_7 |
| control inputs | steering [rad], signed normalized throttle [−1,1] |
| controller configuration | Gaussian IT-MPPI; T=20, λ=8, base diag(.14,.26); standard/isotropic/diagonal/full normalized trace .40; Q03 adds aligned/misaligned |
| seeds | 42 pilot; 32000–32019 primary holdout (20 per setting); 35000–35019 cruise/impulse; 37000–37019 slew; 36000–36009 timing |
| sampling budget | Primary K=384; T=20; 320 episodes; separate K sweep |
| source commit | 7bea369e3455bf62785fcf8fc0db59c8bcbfc6a7 |

## Configuration and provenance

[Original configuration](../../runs/seven_state/Q05/seven_state_q05_v1/config.yaml)

Source run IDs: seven_state_q05_v1, seven_state_q05_rate_v1, seven_state_q05_secondary_v1, seven_state_q05_budget_v1, seven_state_q05_pilot_v1, s_turn_clearance_v1

- [Run s_turn_clearance_v1](../../runs/Q05/s_turn_clearance_v1/): [manifest](../../runs/Q05/s_turn_clearance_v1/manifest.json), source `ff3df5824856e8252a221bc6a4fdce8105ec298c`; model: steering_rate_bicycle_5; inputs: acceleration [m/s²], steering rate [rad/s].

- [Run seven_state_q05_v1](../../runs/seven_state/Q05/seven_state_q05_v1/): [manifest](../../runs/seven_state/Q05/seven_state_q05_v1/manifest.json), source `7bea369e3455bf62785fcf8fc0db59c8bcbfc6a7`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle [−1,1].

- [Run seven_state_q05_pilot_v1](../../runs/seven_state/Q05/seven_state_q05_pilot_v1/): [manifest](../../runs/seven_state/Q05/seven_state_q05_pilot_v1/manifest.json), source `fe98accd84213647c0ff7c006226df6ef1dd3d1f`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle [−1,1].

- [Run seven_state_q05_budget_v1](../../runs/seven_state/Q05/seven_state_q05_budget_v1/): [manifest](../../runs/seven_state/Q05/seven_state_q05_budget_v1/manifest.json), source `3ced0faafe0390e6377dfe85498787a33a5b2d39`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle [−1,1].

- [Run seven_state_q05_secondary_v1](../../runs/seven_state/Q05/seven_state_q05_secondary_v1/): [manifest](../../runs/seven_state/Q05/seven_state_q05_secondary_v1/manifest.json), source `3ced0faafe0390e6377dfe85498787a33a5b2d39`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle.

- [Run seven_state_q05_rate_v1](../../runs/seven_state/Q05/seven_state_q05_rate_v1/): [manifest](../../runs/seven_state/Q05/seven_state_q05_rate_v1/manifest.json), source `afc63ebc9495cff3c4469b80a86e170684af2996`; model: dynamic_single_track_7; inputs: steering [rad], signed normalized throttle.

## Selected figure

![S-turn difficulty and controller limitations](../../results/showcase/Q05_key.png)

[Numeric summary](../../runs/seven_state/Q05/seven_state_q05_v1/summary.csv)

## Reproduce and regenerate

```bash
uv run python experiments/seven_state_campaign.py --stage q05 --output runs/seven_state/Q05/q05_reproduction
uv run python experiments/plot_seven_state.py --stage Q05 --input runs/seven_state/Q05/seven_state_q05_v1
```

## Supporting numerical records

- [per_seed.csv](../../runs/seven_state/Q05/seven_state_q05_v1/per_seed.csv)
- [paired_differences.csv](../../runs/seven_state/Q05/seven_state_q05_v1/paired_differences.csv)
- [goal_summary.csv](../../runs/seven_state/Q05/seven_state_q05_v1/goal_summary.csv)
- [failure_modes.csv](../../runs/seven_state/Q05/seven_state_q05_v1/failure_modes.csv)
- [selected_weight_components.csv](../../runs/seven_state/Q05/seven_state_q05_v1/selected_weight_components.csv)

## Supporting figures

- [Saved-data figure](../../docs/figures/seven_state_Q05_diagnostics.png)
- [Saved-data figure](../../docs/figures/seven_state_Q05_secondary.png)
- [Saved-data figure](../../docs/figures/seven_state_Q05_rate.png)
- [Saved-data figure](../../docs/figures/seven_state_budget_seven_state_q05_budget_v1.png)

[Seven-state campaign report and methods](../seven_state_benchmark_report.md)

[Historical methods (earlier model / original analytic seeds)](../Q05_methods.md)
