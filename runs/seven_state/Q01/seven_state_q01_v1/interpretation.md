# Analytic Gaussian study

Each case has a fixed proposal mean. Covariance comparisons preserve the base, cost and temperature, recorded by target_sha256. Temperature and cost-rotation cases have their own exact targets. All error measures use the corresponding exact mean. Replicates are independent across seeds and paired across cases.

summary.csv reports empirical bias, sample variance trace, MSE and 5/50/95% quantiles. These are descriptive estimates, not claims of an unbiased finite-K estimator or a statistically significant ordering. Read the determinant-constrained cases separately from trace-constrained cases. The dimension diagnostic separates zero-cost density ratios, cost-only concentration (proposal equals base), and their combination.

See docs/experimental_findings.md for the executed numerical interpretation.
