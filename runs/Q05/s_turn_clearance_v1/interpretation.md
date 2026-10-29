# Q05 — Executed S-turn findings

One common objective and fixed Gaussian base were used within every condition. The primary axis is clearance; separate sample-budget and rate-limit checks use only the predeclared middle clearance.

| Clearance [m] | K | Rate limit [rad/s] | Proposal | Completions | Collisions | Censored | Mean max progress | Mean ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9 | 256 | 0.8 | standard | 20/20 | 0/20 | 0/20 | 0.988 | 3.29 |
| 0.9 | 256 | 0.8 | isotropic | 18/20 | 0/20 | 2/20 | 0.990 | 1.88 |
| 0.9 | 256 | 0.8 | diagonal | 19/20 | 0/20 | 1/20 | 0.992 | 2.04 |
| 0.9 | 256 | 0.8 | full | 20/20 | 0/20 | 0/20 | 0.990 | 1.84 |
| 0.6 | 256 | 0.8 | standard | 20/20 | 0/20 | 0/20 | 0.987 | 5.70 |
| 0.6 | 256 | 0.8 | isotropic | 18/20 | 0/20 | 2/20 | 0.991 | 1.66 |
| 0.6 | 256 | 0.8 | diagonal | 20/20 | 0/20 | 0/20 | 0.990 | 1.73 |
| 0.6 | 256 | 0.8 | full | 20/20 | 0/20 | 0/20 | 0.991 | 1.68 |
| 0.35 | 256 | 0.8 | standard | 19/20 | 0/20 | 1/20 | 0.987 | 6.46 |
| 0.35 | 256 | 0.8 | isotropic | 20/20 | 0/20 | 0/20 | 0.985 | 1.57 |
| 0.35 | 256 | 0.8 | diagonal | 20/20 | 0/20 | 0/20 | 0.988 | 1.83 |
| 0.35 | 256 | 0.8 | full | 19/20 | 0/20 | 1/20 | 0.989 | 1.78 |
| 0.2 | 256 | 0.8 | standard | 17/20 | 0/20 | 3/20 | 0.988 | 9.25 |
| 0.2 | 256 | 0.8 | isotropic | 18/20 | 0/20 | 2/20 | 0.985 | 1.60 |
| 0.2 | 256 | 0.8 | diagonal | 20/20 | 0/20 | 0/20 | 0.989 | 1.66 |
| 0.2 | 256 | 0.8 | full | 20/20 | 0/20 | 0/20 | 0.988 | 1.59 |
| 0.35 | 128 | 0.8 | standard | 19/20 | 0/20 | 1/20 | 0.986 | 1.48 |
| 0.35 | 128 | 0.8 | isotropic | 19/20 | 0/20 | 1/20 | 0.987 | 1.37 |
| 0.35 | 128 | 0.8 | diagonal | 20/20 | 0/20 | 0/20 | 0.987 | 1.48 |
| 0.35 | 128 | 0.8 | full | 19/20 | 0/20 | 1/20 | 0.987 | 1.54 |
| 0.35 | 256 | 0.4 | standard | 20/20 | 0/20 | 0/20 | 0.988 | 2.54 |
| 0.35 | 256 | 0.4 | isotropic | 19/20 | 0/20 | 1/20 | 0.989 | 1.70 |
| 0.35 | 256 | 0.4 | diagonal | 19/20 | 0/20 | 1/20 | 0.989 | 1.86 |
| 0.35 | 256 | 0.4 | full | 20/20 | 0/20 | 0/20 | 0.987 | 1.76 |

## Interpretation boundaries

All feasibility checks passed before evaluation. This rules out mechanical impossibility of the recorded reference, not every alternative command sequence. Covariance-specific failures cannot be inferred from collision counts alone. Inspect feasible-rollout fractions, ESS/entropy, steering-angle/rate saturation, progress and the finite horizon together.

Completion time is summarized only among completions; collisions and timeout-censored runs remain separate. Success curves have marginal Wilson intervals; paired differences against standard and their descriptive t intervals are in paired_differences.csv. Intervals are pointwise, unadjusted for multiple comparisons, and twenty seeds may give broad uncertainty.

The matrices are specific trace-.40 proposals in normalized acceleration/rate coordinates. They do not represent all covariances or published feedback architectures. Latency comes from shared CPU execution and includes geometry checks; it is not a real-time guarantee.

[Primary composite](../../../results/showcase/Q05_key.png) · [Numeric plot inputs](key_figure_data.json) · [Per-seed metrics](per_seed.csv) · [Feasibility checks](feasibility.json)

Raw episode CSVs, controls, ESS traces and selected rollout snapshots are kept locally under episodes/ and ignored. Summary/provenance, representative paths and numeric plotting inputs are versioned. Regeneration needs no controller execution.

Source: ff3df5824856e8252a221bc6a4fdce8105ec298c; phase: evaluation.
