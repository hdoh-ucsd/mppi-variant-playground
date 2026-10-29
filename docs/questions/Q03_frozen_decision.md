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

Observed state, warm start, previous control, bicycle model, cost, base, temperature and horizon.

## Metrics

Paired command differences, ESS, entropy, mean-plan objective, reference diagnostics and latency.

## Main finding

7,200 decisions isolate covariance at common frozen inputs. High-sample reference ESS is only 1.10–2.19 from 524,288 rollouts per state; all three reference means remain unresolved.

## Interpretation

Proposal geometry changes finite sampled rollouts, corrected weights, controls and open-loop predictions, even with the same target.

## Limitations

Distances to unresolved references are not validated target-mean errors. Three states, one warm start per state and pointwise intervals do not establish general superiority.

## Configuration and provenance

[Original configuration](../../results/investigations/frozen_decision_v1/config.yaml)

Source run IDs: frozen_decision_v1

- [Run frozen_decision_v1](../../results/investigations/frozen_decision_v1/): [manifest](../../results/investigations/frozen_decision_v1/manifest.json), source `7aaf76eaed90588b9404b2fbed483ae9efce1156`.

## Selected figure

![Frozen-state MPPI decision analysis](../../results/showcase/Q03_key.png)

[Numeric summary](../../results/investigations/frozen_decision_v1/summary.csv)

## Reproduce and regenerate

```bash
uv run python experiments/frozen_state_decision.py --output runs/Q03/frozen_decision_reproduction
uv run python experiments/frozen_state_decision.py --output results/investigations/frozen_decision_v1 --plots-only
```

[Full methods, interpretation and additional figures](../frozen_decision_analysis.md)
