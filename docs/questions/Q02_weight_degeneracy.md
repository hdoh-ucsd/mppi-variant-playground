# Q02 — Importance-weight concentration

**Status: EXECUTED**

## Research question

When do cost weighting and Gaussian density corrections concentrate importance weights, and how does ESS relate to error?

## Mathematical motivation

ESS=1/Σw² measures weight concentration, whereas estimator variance also depends on the integrand.

## Hypothesis

Sharper Boltzmann factors and high-dimensional proposal mismatch can each concentrate weights.

## Independent variables

Temperature and dimension; cost-only, ratio-only and combined controls.

## Controlled variables

Analytic target recomputed for each temperature; paired seeds and sample budgets within each comparison.

## Metrics

ESS, entropy, largest weight, exact mean error and Gaussian overlap diagnostics.

## Main finding

At K=512 the isotropic mean ESS grows from 40.56 at λ=.1 to 450.61 at λ=10, while mean error changes from .04284 to .05207 against each changed exact target. Cost-only, ratio-only and combined dimension probes separate concentration mechanisms.

## Interpretation

ESS summarizes normalized weights. Temperature changes the target; density ratios alone can concentrate in high dimensions.

## Limitations

Temperature changes the target. ESS does not certify estimation accuracy or target coverage. Seven states correspond to a 40-dimensional sampled control sequence here.

## Registered experiment

| Field | Value |
| --- | --- |
| experiment name | Independent-seed temperature and dimension concentration replication |
| dynamics model | analytic_gaussian |
| control inputs | Gaussian vector (no vehicle) |
| controller configuration | Fixed Gaussian base/proposal-corrected SNIS; exact target moments |
| seeds | 34000–34063 (64 independent replication seeds) |
| sampling budget | K=512; d=2…56; 4,544 trials |
| source commit | 0006f840f99ab6be155b66ea7833e9210fe40367 |

## Configuration and provenance

[Original configuration](../../runs/seven_state/Q02/seven_state_q02_v1/config.yaml)

Source run IDs: seven_state_q02_v1, temperature_v1

- [Run temperature_v1](../../results/investigations/temperature_v1/): [manifest](../../results/investigations/temperature_v1/manifest.json), source `dd50f534ad47e765bec39715f6f801b790a251b2`; model: analytic_gaussian; inputs: not applicable (Gaussian latent vectors).

- [Run seven_state_q02_v1](../../runs/seven_state/Q02/seven_state_q02_v1/): [manifest](../../runs/seven_state/Q02/seven_state_q02_v1/manifest.json), source `0006f840f99ab6be155b66ea7833e9210fe40367`; model: analytic_gaussian; inputs: Gaussian vector (no vehicle).

## Selected figure

![Importance-weight concentration](../../results/showcase/Q02_key.png)

[Numeric summary](../../runs/seven_state/Q02/seven_state_q02_v1/summary.csv)

## Reproduce and regenerate

```bash
uv run python experiments/seven_state_campaign.py --stage q02 --output runs/seven_state/Q02/q02_reproduction
uv run python experiments/plot_seven_state.py --stage Q02 --input runs/seven_state/Q02/seven_state_q02_v1
```

## Supporting numerical records

- [paired_differences.csv](../../runs/seven_state/Q02/seven_state_q02_v1/paired_differences.csv)

[Seven-state campaign report and methods](../seven_state_benchmark_report.md)

[Historical methods (earlier model / original analytic seeds)](../covariance_mechanisms.md)
