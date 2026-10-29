# Q01 — Sampling covariance and estimation accuracy

**Status: EXECUTED**

## Research question

How do proposal orientation, anisotropy and budget affect finite-sample error for a fixed Gibbs target?

## Mathematical motivation

The Gaussian Gibbs mean is (P⁻¹+H/λ)⁻¹Hb/λ, independent of the proposal covariance.

## Hypothesis

A suitably oriented proposal can reduce finite-sample MSE; misalignment can reverse the effect.

## Independent variables

Orientation, anisotropy, sample count, scale constraint; cost rotation in a separate factorial slice.

## Controlled variables

Base distribution, quadratic cost and temperature within each comparison; common random numbers per paired seed.

## Metrics

Bias, variance, MSE, ESS and moment error across 64 evaluation seeds.

## Main finding

An independent 64-seed replication preserves positive and negative orientation examples: at K=256, MSE is 0.006505 isotropic, 0.003646 at +45° and 0.065524 at −45°. These are estimates of one exact Gaussian target, not vehicle results.

## Interpretation

A common Gaussian Gibbs target separates proposal-induced finite-sample error from a changed optimization problem. ESS alone does not determine mean-estimation variance.

## Limitations

An exact quadratic target is not a vehicle task. Trace matching does not match determinant; empirical 64-seed variance and bias are finite-sample quantities.

## Registered experiment

| Field | Value |
| --- | --- |
| experiment name | Independent-seed exact Gaussian covariance replication |
| dynamics model | analytic_gaussian |
| control inputs | Gaussian vector (no vehicle) |
| controller configuration | Fixed Gaussian base/proposal-corrected SNIS; exact target moments |
| seeds | 34000–34063 (64 independent replication seeds) |
| sampling budget | K=32…4096; d=2; 28,608 trials |
| source commit | 0006f840f99ab6be155b66ea7833e9210fe40367 |

## Configuration and provenance

[Original configuration](../../runs/seven_state/Q01/seven_state_q01_v1/config.yaml)

Source run IDs: seven_state_q01_v1, gaussian_v1

- [Run gaussian_v1](../../results/investigations/gaussian_v1/): [manifest](../../results/investigations/gaussian_v1/manifest.json), source `dd50f534ad47e765bec39715f6f801b790a251b2`; model: analytic_gaussian; inputs: not applicable (Gaussian latent vectors).

- [Run seven_state_q01_v1](../../runs/seven_state/Q01/seven_state_q01_v1/): [manifest](../../runs/seven_state/Q01/seven_state_q01_v1/manifest.json), source `0006f840f99ab6be155b66ea7833e9210fe40367`; model: analytic_gaussian; inputs: Gaussian vector (no vehicle).

## Selected figure

![Sampling covariance and estimation accuracy](../../results/showcase/Q01_key.png)

[Numeric summary](../../runs/seven_state/Q01/seven_state_q01_v1/summary.csv)

## Reproduce and regenerate

```bash
uv run python experiments/seven_state_campaign.py --stage q01 --output runs/seven_state/Q01/q01_reproduction
uv run python experiments/plot_seven_state.py --stage Q01 --input runs/seven_state/Q01/seven_state_q01_v1
```

## Supporting numerical records

- [paired_differences.csv](../../runs/seven_state/Q01/seven_state_q01_v1/paired_differences.csv)

[Seven-state campaign report and methods](../seven_state_benchmark_report.md)

[Historical methods (earlier model / original analytic seeds)](../experimental_findings.md)
