# Codebase audit and preservation plan

Audited main at `74ba5d81a16a8c6c01bc7c18eb466caa2d7d0408`, before refactoring.
`uv sync --frozen` succeeded; the existing suite passed **232 tests in 52.19 s**.
The local uv cache must be set to `.uv-cache` in this environment.

The inventory used Python AST imports (including imports inside functions), CLI bodies,
`.github/workflows/ci.yml`, README, docs, YAML/JSON configuration and tracked-result
references. A missing import alone was never treated as evidence of dead code.
No historical result record is proposed for deletion. The baseline has 521 tracked
files totaling 50,673,550 bytes; this is substantially smaller than the ignored local
recovery archive, which is not part of Git.

## Active and shared dependencies

| Paths | Actual callers / purpose | Unique evidence | Action and reproducibility risk |
| --- | --- | --- | --- |
| `src/controllers/{base,mppi,tsallis,dial,tube,robust,covariance,feedback,accounting,registry}.py` | `evaluation.runner.make_controller`, variant/mechanism campaigns and controller tests | Algorithms, fixtures and mathematical choices | Keep; add passive selected-decision snapshot fields. Command/RNG/work invariance must pass. |
| `src/dynamics/dynamic_vehicle.py`, `src/costs/dynamic_vehicle.py`, `src/scenarios/dynamic_scenes.py` | Registry factory, shared seven-state runner, validation and mechanism studies | Common physical/task definition | Keep unchanged. Trace component scoring reuses this cost, without a second cost implementation. |
| `src/sampling/gaussian.py` | Five controller modules, analytic references, frozen studies and audits (22 direct callers) | Shared sampler, normalized weights and density ratio | Keep one implementation; new density observations use its existing ratio identity. |
| `src/controllers/feedback.py` | Tube, Robust, CC, feedback tests | Linearization, wrapped state errors and ancillary gains | Keep shared; do not duplicate trackers in analysis. |
| `src/analysis/dynamic_campaign.py` | Variant and mechanism campaigns, audits and seven-state CLIs | Existing execution order, disturbance streams, contact checks | Keep path; add an optional observer at decision boundaries. Do not create another vehicle runner. |
| `src/analysis/variant_campaign.py` | `experiments/variant_benchmark.py`, mechanism campaign, four controller test modules | Budget allocation, calibration, original studies | Keep path rather than move to `evaluation/`: multiple stable callers and historical reproduction commands depend on it. |
| `src/evaluation/runner.py`, `src/analysis/artifacts.py` | Both model generations and all campaigns | Factory, Git provenance, exclusive output creation | Keep shared. Trace configuration hashing and storage have separate versioned semantics. |
| `src/visualization/figures.py` | Seven plotting/validation modules, video renderer, `generate_figures.py` | Shared scene drawing and historical plots | Keep shared style/scene helpers. New trace plots consume records, not controllers. |
| `src/analysis/{dynamic_plots,formulation_mechanism_plots,variant_audit,experiment_video_records}.py`, `src/visualization/{experiment_video,video_preview}.py` | Active figure, audit and video CLIs; README | Published evidence and source hashes | Keep; no rewrite or re-encoding of V01–V05 evidence. |

## Historical code that remains necessary

| Paths | Actual callers | Unique evidence | Action / risk |
| --- | --- | --- | --- |
| `src/dynamics/bicycle.py` | **17** callers including `dynamic_vehicle`, `feedback`, both costs, runners and scenarios | Four-state model **and** shared `wrap_angle` | Keep. Deleting this as “four-state only” would break the active seven-state model. |
| `src/dynamics/steering_rate.py`, `src/costs/s_turn.py`, `src/scenarios/s_turn.py` | S-turn studies/audits and tests; dynamic scenes import `Polyline` and `Corridor` from `s_turn` | Five-state model plus reused corridor geometry | Keep; module relocation is not a safe cosmetic cleanup. |
| `src/costs/shared.py`, `src/evaluation/{benchmark,metrics}.py`, `src/scenarios/environment.py`, `src/utils/config.py` | Four-state factory, CI smoke and tests; shared interval/collision helpers also serve seven-state work | Historical behavior and reusable primitives | Keep; CI executes `run_single.py` then `generate_figures.py`. |
| `src/analysis/{analytic_gaussian,gaussian_study,covariance_geometry,importance_sampling,nonconvex_study,nonconvex_objectives,saturation,closed_loop_study}.py` | Historical CLIs, plotting and mathematical-reference tests | Q01–Q05 analytic/control studies | Keep import paths; mark historical. Analytic references remain useful to active weight validation. |
| `src/analysis/{trajectory_density,trajectory_statistics,trajectory_plots,rollout_snapshots}.py` | Density study, verifier and tests; seven-state runner imports `first_crossing` | Existing schema-v1 snapshots and route statistics | Keep old schema intact. Do not relabel historical records as full DecisionTrace objects. |
| `src/analysis/{frozen_decision,dynamic_frozen,frozen_decision_audit,frozen_decision_report,frozen_decision_plots}.py` | Frozen CLIs, dynamic audit, mechanism study (`plan_objective`) and tests | Frozen-state reference estimates and provenance | Keep stable paths; new tracing reuses recorded frozen inputs. |
| `experiments/{run_single,run_benchmark,generate_figures,seven_state_campaign,plot_seven_state,time_seven_state,summarize_seven_state,frozen_state_decision,verify_frozen_decision,research_index}.py` | CI, README-derived reports, research registry and vehicle report | Historical reproduction entry points | Leave in place. Not unused merely because they are outside the six-method presentation. |
| Remaining `experiments/verify_*.py`, validation/S-turn CLIs | Existing audit modules, vehicle report or independently runnable CLIs | Replay checks and specialized evidence | Preserve; no deletion justified by this audit. |
| `configs/{controllers,dynamics,benchmarks,investigations}/`, `configs/costs.yaml`, `configs/seven_state/` | CLI defaults, config loader, tests, registry and variant `configuration()` | Exact historical settings, shared seven-state base settings | Keep; standalone configs are entry-point inputs, not dead Python imports. |

## Proposed relocations (no implementation or result deletion)

These eight standalone historical entry points have **no current import, CI, README,
docs, config or tracked-result path caller**. Their CLI bodies still have real users:
they dispatch to the listed shared modules. They are not dead or duplicate algorithms.
Move only these thin entry files to `experiments/legacy/`, with byte-identical contents.
Run from the repository root with the new paths below. Old revisions, original source
hashes and run metadata remain untouched; an exact historical checkout still provides
all original paths. Risk: an untracked personal command using the old path needs the
explicit `legacy/` insertion. No compatibility wrapper or duplicate implementation is added.

| Current path | Actual imports/callers | Unique evidence | Proposed path |
| --- | --- | --- | --- |
| `experiments/analytic_gaussian.py` | CLI → `analysis.cli`, `gaussian_study.run_gaussian`, `plots.generate` | Full Gaussian study dispatch; no embedded results | `experiments/legacy/analytic_gaussian.py` |
| `experiments/covariance_alignment.py` | CLI → same runner, `geometry` / `cost_rotation` subset | Distinct subset selection, not a duplicate experiment | `experiments/legacy/covariance_alignment.py` |
| `experiments/temperature_sweep.py` | CLI → same runner, `temperature` / `dimension` subset | Distinct target diagnostics | `experiments/legacy/temperature_sweep.py` |
| `experiments/actuator_saturation.py` | CLI → `analysis.saturation.run_saturation`, shared parser/plots | Saturation study dispatch | `experiments/legacy/actuator_saturation.py` |
| `experiments/covariance_closed_loop.py` | CLI → `analysis.closed_loop_study.run_closed_loop`, shared parser/plots | Four-state covariance sweeps | `experiments/legacy/covariance_closed_loop.py` |
| `experiments/nonconvex_optimization.py` | CLI → `analysis.nonconvex_study.run_nonconvex`, shared parser/plots | Multimodal analytic landscapes | `experiments/legacy/nonconvex_optimization.py` |
| `experiments/trajectory_density_study.py` | CLI → `analysis.trajectory_density.run_study`, `trajectory_plots.generate` | Historical state-space density recordings | `experiments/legacy/trajectory_density_study.py` |
| `experiments/mechanism_summary.py` | CLI → `analysis.mechanism_summary.collect/render` | Historical five-level evidence map | `experiments/legacy/mechanism_summary.py` |

Each relocated CLI must pass `--help`; the historical numerical tests remain in the
suite. `tests/test_variant_baseline.py` retains its original fixture path. Four/five-state,
Gaussian, density, S-turn and snapshot compatibility tests are not deleted or renamed.

## Outputs, duplication and existing trace inputs

- Keep all tracked `runs/seven_state/` records: largest unique tables are Q01 `trials.csv`
  (7.49 MB), Q03 `per_seed.csv` (4.73 MB), Q03 `figure_inputs.json` (3.19 MB), Q02
  `trials.csv` (2.33 MB), Q02 `cases.json` (2.16 MB). Removing these would discard evidence.
- Keep `results/variants/V01…V06/`, all five mechanism studies, existing media and the
  supplied formulation-map PNG. Historical Q01–Q05 figures move into an expandable
  README supplement, not out of the repository. V06 remains supplementary.
- Existing `runs/archive/02a331f_cleanup/` is an ignored recovery copy, not a second
  active implementation. Its pinned recovery index remains valid.
- `analytic_gaussian`, `covariance_alignment` and `temperature_sweep` are intentionally
  different selections of **one** Gaussian runner. Existing plot modules support
  different numerical schemas; merging them without schema migration would lose semantics.
- The old rollout interchange is float32 display-oriented with execution-required fields;
  DecisionTrace uses float64 arrays, five explicit stages, missingness and frozen decisions.
  Reusing its validation as if the schemas were identical would be incorrect.
- No genuinely dead numerical implementation or unused config was established. Therefore
  **no numerical helper/config/result is deleted**. A single sampler/cost/tracker remains.

Reusable raw inputs: V01 same-sample cost/weight/log-density populations; V02 all four
DIAL inner passes; V03 actual/nominal/correction CSVs and step-9 populations; V04 dual
nominal/actual populations and candidate free energies; V05 planner covariance/gains
and independent fixed-policy executions. Missing original Tube discarded populations,
CC auxiliary histories, per-step collision flags and some prior states must stay missing
in retroactive traces. Selected fresh decisions may fill the live schema, not rewrite
historical traces. A fixed-policy covariance ensemble is explicitly a **related experiment**,
not the execution distribution of the optimizer's updated command.

Full arrays and immutable shared configurations go under ignored `runs/variants/traces/`.
Only the trace index, compact numerical figure inputs, one group manifest, six curated
figures and concise validation/interpretation records belong in `results/variants/traces/`.
