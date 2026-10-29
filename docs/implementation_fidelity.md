# Implementation fidelity

The shared IT-MPPI now also supports a validated seven-state physics approximation with steering/throttle inputs. It is not the AutoRally system-identified dynamics. Historical four-state and five-state rate models remain separate; no additional published MPPI algorithm has been implemented. [Seven-state report](seven_state_benchmark_report.md) · [Current research index](research_index.md).

| Component | Delivered behavior | Difference / scope |
|---|---|---|
| Standard MPPI | Finite-dimensional Gaussian IT-MPPI, full base/proposal ratio, fixed SPD covariance, exponential normalized weights, latent weighted mean | Follows the accessible 2017 information-theoretic formulation; not an equation-by-equation reimplementation of inaccessible 2016 publisher full text |
| Covariance ablations | Isotropic, diagonal and full proposals with normalized trace matched | Common base held fixed; unequal-proposal quadratic correction retained; no covariance learning |
| Dynamics | Vectorized Euler kinematic bicycle, normalized-control scaling, physical limits | No tire slip, suspension, contact, stochastic process diffusion in prediction, or learned dynamics |
| Constraints | Saturation in rollout dynamics and output; unconstrained latent mean | Gaussian densities remain latent; no incorrect Gaussian likelihood assigned to clipped controls |
| Robustness | Standard closed-loop replanning under execution-only disturbances and mismatch | Not Tube-MPPI or RMPPI; no nominal/actual tracking architecture or guarantee |
| Costs | Shared goal, heading, desired speed, clearance, collision, control and variation terms, terminal pose/speed error | Toy goal-reaching cost, additional to base prior regularization; evaluation weights independently configurable |
| Collision | Disk/circle geometry with swept straight-segment checks | Disk footprint, Euler interpolation and impulse sweep are approximations, not a realistic vehicle body/contact model |
| Budget | All scalar prediction transitions counted including nominal-plan visualization | Planning latency and process peak RSS have explicitly limited measurement scopes |
| Diagnostics | Full-population weights/statistics; first 32 unweighted paths retained | Saved paths are a visualization subset, not the full sampling distribution |
| Precision | Float64 controller, dynamics and CSVs; float32 visual archive | Byte-identical behavior across arbitrary BLAS/hardware is not promised |

The six additional algorithms (CEM, Tsallis-VI, DIAL, Tube, Robust and CC-MPPI) are planned. Unsupported controller kinds raise an error. None is implemented by relabeling standard MPPI. The [mathematical comparison](formulations.md) and [paper map](paper_map.md) document their mechanisms and required next work.

The [covariance mechanism study](covariance_mechanisms.md) adds isolated Gaussian estimation, nonconvex adaptive-mean updates, saturation analysis and parameter sweeps through the unchanged bicycle runner. These investigate the implemented baseline; they do not implement any of the six planned algorithms. Analytic targets have independent quadrature checks; nonconvex target means use finite-box grid references; bicycle mean updates have no exact oracle.

Independent numerical checks cover the Gaussian density ratio (SciPy oracle), equal-covariance cross-term reduction, cost-only difference, shifted exponentials, exact weighted mean examples, Cholesky validity, Gaussian moments, actuator saturation, swept collisions, dynamics, configuration, metrics and deterministic replay. Passing them establishes consistency of this implementation, not formal equivalence to continuous-time stochastic optimal control or a published robustness proof.

## S-turn study extension

The original-model table above remains the scope of historical runs. Q05 adds a separate five-state explicit-Euler bicycle with acceleration/steering-rate inputs and angle/rate limits; it wraps the original four-state model rather than replacing it. New normalized covariances are declared for the new input semantics. A common progress/clearance cost and exact swept disk/polygon collision geometry are supplied to the shared MPPI planner through optional interfaces. No actuator lag, tracker, covariance adaptation or published robustness guarantee is introduced. Full rollout recording is opt-in; historical compact archives and their original records remain intact. See [Q05 methods](Q05_methods.md) and the [research index](research_index.md) for execution status.
