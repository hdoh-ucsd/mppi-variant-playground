# Q01 — Sampling covariance and estimation accuracy

**Status: EXECUTED**

## Research question

Sampling covariance and estimation accuracy?

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

At K=256 in the declared quadratic case, MSE is 0.007657 for isotropic, 0.003699 for +45° anisotropy and 0.049254 for −45°. Geometry helps or hurts depending on alignment.

## Interpretation

A common Gaussian Gibbs target separates proposal-induced finite-sample error from a changed optimization problem. ESS alone does not determine mean-estimation variance.

## Limitations

An exact quadratic target is not a vehicle task. Trace matching does not match determinant; one matrix cannot represent a covariance family.

## Configuration and provenance

[Original configuration](../../results/investigations/gaussian_v1/config.yaml)

Source run IDs: gaussian_v1

- [Run gaussian_v1](../../results/investigations/gaussian_v1/): [manifest](../../results/investigations/gaussian_v1/manifest.json), source `dd50f534ad47e765bec39715f6f801b790a251b2`.

## Selected figure

![Sampling covariance and estimation accuracy](../../results/showcase/Q01_key.png)

[Numeric summary](../../results/investigations/gaussian_v1/summary.csv)

## Reproduce and regenerate

```bash
uv run python experiments/analytic_gaussian.py --output runs/Q01/gaussian_reproduction
uv run python experiments/analytic_gaussian.py --output results/investigations/gaussian_v1 --plots-only
```

[Full methods, interpretation and additional figures](../experimental_findings.md)
