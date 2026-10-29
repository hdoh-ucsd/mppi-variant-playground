# Seven-state campaign: pre-implementation audit

Audited starting revision: `07759fbcdc4d76793543e269d16c10de52b2f801` on `research-organization-s-turn`. The new branch is `seven-state-vehicle-benchmark`; main is not merged or rewritten.

| Component | Available and verified scope | Missing for the new campaign |
| --- | --- | --- |
| Controllers | One IT-MPPI implementation with fixed common base, full latent Gaussian likelihood correction, log normalization, optional paired normal draws and stateless plan; standard/isotropic/diagonal/full plus controlled full orientations | Steering/throttle configurations and model-specific controller construction |
| Dynamics | Original four-state acceleration/steering bicycle; separate five-state acceleration/steering-rate bicycle | Seven-state dynamic single-track model, dynamic roll equation, stable integration and independent turning validation |
| Experiments | Exact Gaussian estimation, temperature/dimension, saturation/nonconvex studies; frozen decisions; closed-loop trajectory study; feasible S-turn clearance and budget/rate studies | Execute new independent analytic replications, seven-state Q03–Q05 and budget sensitivity |
| Existing evidence | Manifests and summaries confirm Q01 Gaussian 28,608 trials; Q02 4,544; Q03 7,200 decisions with unresolved reference means; Q04 320 episodes; five-state Q05 480 episodes | New results cannot inherit old outcomes merely by changing a model label |
| Formats | Exact YAML, source/configuration hashes, CSV per-seed summaries, JSON manifests/paths, opt-in NPZ snapshots, PNG/SVG and numeric plot inputs; registry with stable question IDs | Explicit model/input metadata, seven-state traces, lateral/yaw/roll summaries, nested runs/seven_state storage allowances |
| Tests | Baseline numerical, density, replay, frozen-input, geometry, feasibility, registry and saved-data plotting tests | Dynamic-vehicle sign/physics/integration tests and model-specific campaign checks |

Reuse the MPPI planner, Gaussian routines, analytic studies, reference pooling, circle and polygon collision geometry, paired summaries, storage provenance and registry/rendering. Preserve every preexisting tracked result and all original model/controller files. New raw arrays/episodes are local ignored evidence, not a second committed archive.

The 2017 paper, section VI-B, uses the seven requested state coordinates and steering/throttle inputs, but learns the four dynamic update components using system identification (basis functions or a neural network). This campaign supplies its own explicitly documented physics approximation. Source: https://arxiv.org/html/1707.02342v1#S6.SS2 . It does not claim original AutoRally calibration or experimental fidelity.

Execution order: baseline tests → validated dynamic model → controller/cost tests → independent Q01/Q02 replications → new frozen decisions → five seven-state driving scenarios → feasible S-turn pilot and paired evaluation → controlled K checks → saved-data figures, registry and report. Unreliable high-sample means remain unresolved. No fallback to an old vehicle model is allowed.
