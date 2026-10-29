# Q05 — S-turn difficulty and controller limitations

**Status: PLANNED**

## Research question

S-turn difficulty and controller limitations?

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

PLANNED — implementation and mechanical feasibility are validated; the final finish-gate seed-42 pilot and multi-seed evaluation are pending.

## Interpretation

The study separates physical feasibility from finite-horizon, finite-sampling and objective limitations.

## Limitations

Do not attribute failure to covariance before geometry and reference feasibility are validated.

## Configuration and provenance

[Original configuration](../../configs/investigations/s_turn.yaml)

Source run IDs: none — PLANNED

## Reproduce and regenerate

```bash
uv run python experiments/s_turn_study.py --phase evaluation --output runs/Q05/s_turn_clearance_v1
uv run python experiments/s_turn_study.py --plots-only --output runs/Q05/s_turn_clearance_v1
```

[Full methods, interpretation and additional figures](../Q05_methods.md)
