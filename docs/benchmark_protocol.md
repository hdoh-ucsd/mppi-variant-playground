# Paired experimental protocol

The primary driving protocol now uses seven-state dynamics. The protocol below remains historical four-state evidence; its control coordinates and seed counts do not describe the new campaign. [Seven-state report](seven_state_benchmark_report.md) · [Current research index](research_index.md).

## Experimental questions

This document specifies the existing shared vehicle experiments. Their role in the broader research is to illustrate behavior and provide reproducible diagnostics. The [mathematical investigation plan](research_questions.md) defines the mechanism-specific questions for each variant and the additional probes required to answer them.

Choose the observable according to the mathematical question: weight concentration for a weighting law, update error for importance sampling, covariance geometry for exploration, or nominal/actual error for a tracking architecture. In the vehicle records, report task outcomes alongside these diagnostics. Report disturbance reference-line recovery separately from goal success, and completion times with the number of successful runs to avoid hiding timeouts.

The three budget modes serve different purposes: matched dynamics evaluations are the primary covariance comparison, the calibrated wall-clock target compares behavior with a timing-based sample allocation, and native mode checks the supplied settings. The additional obstacle-field and model-mismatch episodes are single-run environment checks. The existing data do not include a controlled covariance-scale, correlation, horizon or sample-count sweep.

## Fixed conditions

The first milestone compares standard, isotropic, diagonal and full fixed proposal covariances on open space, narrow passage and sudden disturbance. The four methods use one MPPI implementation and differ only in proposal covariance. Every scenario/controller pair runs seeds **42, 43, 44, 45, 46** in that order.

The bicycle uses a 0.15 s control period, 1 m prediction wheelbase, acceleration ±2 m/s², steering ±0.5 rad, and speed [0, 3] m/s. Initial pose is `[0, -0.4, 0, 0]`, target `[8, 0, 0, 0]`, with a 0.22 m disk footprint. Success requires position error ≤0.45 m, wrapped heading error ≤0.35 rad and speed error ≤0.45 m/s with no collision. Episodes end at success, the first swept collision, or 120 steps (18 simulated seconds). These are identical task definitions for all controllers; a close final position alone is not success.

The narrow passage has two circles centered at `(4, ±1.5)`, radius 0.95 m, leaving a 1.10 m geometric gap and 0.66 m footprint-center gap. The disturbance is applied **after execution step 20**, at state time 3.15 s: `[Δx, Δy, Δheading, Δspeed] = [0, 0.9, 0.18, -0.4]`. The prediction model is not informed of its timing or magnitude. Obstacles and goals are shared public task information. Other scenarios implement a nonconvex obstacle field and independent execution wheelbase/steering/acceleration gains; they are runnable but outside the primary comparison.

Two random streams are used: optimizer `default_rng(seed)` and environment `default_rng(SeedSequence([seed,1001]))`. The environment draws four noise coordinates each execution step, regardless of the controller. Identical seeds therefore use paired exogenous noise. After noise or impulse, heading is wrapped and speed is clipped; the resulting state increment can differ near constraints even when the noise draw is identical. Gaussian noise is disabled in the default milestone. The impulse is deterministic and shared.

## Three separate budgets

**A. Matched dynamics evaluations (`dynamics`).** All use $K=384$, $T=28$ (4.2 s), and one update per command. Sampling consumes $KT=10,752$ scalar model transitions per command. An updated-mean plan for diagnostics costs another $T=28$, for **10,780 total planning transitions per MPC step**. Each execution also performs one evaluation-only one-step model prediction for the tracking metric and one environment transition. Both are outside the planning budget; each count equals the number of executed steps, which is saved. Vectorization does not reduce the reported scalar transition count.

**B. Calibrated wall-clock target (`wall_clock`).** For each scenario and controller, time five warmed batches after one warmup at each candidate `K ∈ {64,128,256,384,512,768,1024,1536,3072}`. Use calibration seed 12345 (outside evaluation seeds). Select the largest candidate whose median planning latency is ≤15 ms, freeze it, then evaluate. Calibration curves, target, chosen count and per-step deadline misses are saved. This is offline median-latency matching with observed overruns, **not** a hard wall-clock cap or real-time guarantee. A deadline below the smallest measured batch raises an error. Do not infer equality of achieved latency from equality of requested target.

**C. Native (`native`).** Use the controller YAML sample counts. The delivered native counts happen to equal the matched-evaluation counts (384); this mode records a distinct protocol rather than inventing different settings. A one-seed native check is not additional independent evidence if it duplicates the matched-evaluation numerical trajectory.

Timing uses `perf_counter` around all planning work, including noise, weights and diagnostic plan propagation. Environment integration, copying diagnostics to the runner and artifact writing are excluded from planning latency. Total simulation wall time includes the closed-loop runner and diagnostic copying, excluding final artifact writes. Peak RSS is the Linux **process lifetime high-water mark**, not isolated controller allocation. Run controllers sequentially; pin BLAS threads with the README environment variables. Future nested optimizers must account for every inner pass, tracker computation, linearization and steering solve, not just outer sample counts.

## Saved evidence and metrics

Git dirty status excludes the generated `results/` tree so writing output does not itself mark the implementation dirty. All other repository paths, including untracked source files, are included. Each run saves `config.yaml`, `result.json`, `trajectory.csv`, `controls.csv`, `diagnostics.csv` and `rollouts.npz`. The archive retains the first 32 unweighted rollouts and the updated nominal plan at each step, quantized to float32 for visualization; all computation and CSV state/control data use float64. It is not the complete sample population. Perturbation means/covariances summarize the entire batch in diagnostics. Do not estimate collision rates from the retained sample subset.

Episode metrics include success, final position/heading error, successful completion time, execution duration, evaluation cost, collision, minimum **swept** obstacle clearance, constraint violations, ESS/entropy/degeneracy, terminal sample-position spread, tracking prediction error, reference cross-track RMSE, disturbance recovery, latency, scalar model calls, rollout counts, process RSS and control effort/variation. Control effort is $\Delta t\sum\|D^{-1}u_t\|^2$. Variation is $\sum\|D^{-1}(u_t-u_{t-1})\|^2$ with $u_{-1}=0$. Maximum steering change includes the first change from zero.

Prediction tracking RMSE is the square root of the mean squared components of the actual-next-state minus one-step model prediction. Its units mix position, radians and velocity; it is a toy mismatch diagnostic, not a physical distance. Reference cross-track RMSE is distance to the straight line from initial position to goal and is separate from tracking a nominal controller trajectory. No ancillary tracking architecture is implemented.

Recovery means five consecutive saved states within 0.25 m of that reference line, measured from the first post-impulse state. It is available only when an impulse occurred and recovery was observed. An immediate five-state interval gives zero; a null value means unrecovered/censored or inapplicable. This definition is not settling time to the full pre-disturbance state. Missing time-to-completion, recovery and obstacle clearance values remain null, not zero or fabricated measurements.

`per_seed.csv` preserves pairing. `summary.csv/json` contain mean, sample SD, median and Student-t 95% mean intervals. Counts accompany every metric because success/recovery conditioning changes the denominator. For binary outcomes, the mean is the fraction and the 95% interval uses the Wilson binomial formula within [0,1]. `ci_method` labels the interval in each summary row. Plots show counts for success, seed points and SD for continuous metrics. `paired_differences.json` contains seed-aligned differences against standard MPPI with descriptive Student-t intervals for continuous metric differences. With five seeds, broad intervals, shared tuning and sequential execution, no significance or universal ranking is claimed.

## Reproduction and tuning policy

See README commands. Use new output directories. Regenerating figures reads saved YAML, JSON, CSV and rollout archives without rerunning a controller. Archives and animations are ignored by Git and must be regenerated by executing the experiment if absent. Numerical evidence and presentation figures are tracked; large datasets and videos are not.

The initial seed-42 pilot was inspected for implementation and artifact behavior. No scenario or covariance was retuned after seeing the pilot outcomes. Numerical tests must pass before the final multi-seed comparison. Additional seeds are descriptive validation, not a held-out statistical test of an algorithm selected on seed 42. Keep failures and deadline misses in the report. Use dedicated experiments before claiming sensitivity trends over model mismatch magnitudes.
