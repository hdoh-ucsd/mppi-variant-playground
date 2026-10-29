# Q05 — S-turn difficulty and controller limitations

**Status: EXECUTED**

## Research question

At what corridor clearance do specific proposal covariances produce measurable differences in controlled S-turn traversal?

## Mathematical motivation

Finite-budget importance sampling, actuator limits and a restricted horizon interact with feasible-set geometry.

## Hypothesis

Sequential steering reversals with less clearance may expose proposal inefficiency; no proposal is expected to win by construction.

## Independent variables

Primary: corridor clearance. Secondary planned checks: rollout count and steering-rate limit.

## Controlled variables

Vehicle, objective, speed target, steering limit, horizon and samples within each clearance condition.

## Metrics

Success/collision, censored progress, cross-track error, clearance, completion time, evaluation cost, ESS, entropy, effort, rate saturation, latency and transitions.

## Main finding

At .20 m clearance, completion is 17/20 standard, 18/20 isotropic and 20/20 diagonal/full. All 17 failures among 480 episodes are timeouts near the finish, with zero collisions.

## Interpretation

A larger observed completion gap at the tightest clearance coexists with terminal speed/heading failures; this does not isolate covariance-driven route discovery.

## Limitations

Twenty seeds give broad paired intervals. The difficulty threshold is unresolved; finish-region recovery, one corridor handedness, one horizon and specific matrices limit generalization.

## Configuration and provenance

[Original configuration](../../runs/Q05/s_turn_clearance_v1/config.yaml)

Source run IDs: s_turn_clearance_v1

- [Run s_turn_clearance_v1](../../runs/Q05/s_turn_clearance_v1/): [manifest](../../runs/Q05/s_turn_clearance_v1/manifest.json), source `ff3df5824856e8252a221bc6a4fdce8105ec298c`.

## Selected figure

![S-turn difficulty and controller limitations](../../results/showcase/Q05_key.png)

[Numeric summary](../../runs/Q05/s_turn_clearance_v1/summary.csv)

## Reproduce and regenerate

```bash
uv run python experiments/s_turn_study.py --config runs/Q05/s_turn_clearance_v1/config.yaml --phase evaluation --output runs/Q05/s_turn_reproduction
uv run python experiments/s_turn_study.py --plots-only --output runs/Q05/s_turn_clearance_v1 --showcase results/showcase/Q05_key.png
```

[Full methods, interpretation and additional figures](../Q05_methods.md)
