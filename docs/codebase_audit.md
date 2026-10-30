# Codebase audit and preservation plan

The initial audit below is historical. The final section records the subsequent
experiment/result relocation; use the current README and `results/relocations.json`
for current commands and paths.

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


## Completed cleanup and trace validation

All eight proposed launcher moves were byte-identical (`git` detects 100% similarity).
No existing result, configuration, numerical implementation, video or test was removed.
The supplied formulation map is retained; Q01–Q06 are now in an expandable README
supplement. Active reproduction uses `variant_benchmark.py`, the existing mechanism
and video launchers, and the two new trace entry points.

Historical extraction preceded the new simulations. A preliminary adapter check and
the final source-pinned run each used only six frozen decisions and six 12-step
executions; neither reran a historical campaign. The curated group contains 70
historical records plus 78 fresh decisions from source `5863f2a95fa39150585e282a51ca1b16375b7ccc`.
Thirty records are complete for their mode and 118 are partial. A final validation
review corrected the completeness label on 32 non-capture steps: missing stages 3/4
must imply a partial record even when their empty population lists have no individual
missing-field leaves. This changed metadata only; the manifest names all affected
IDs, records source hashes, and preserves the original simulation revision.

Tests cover frozen and closed-loop command/plan/RNG/model-work invariance for all six
controllers, local/external array hashes and round trips, immutable configurations,
units/dimensions, ID stability, meaningful normalized weights, DIAL ordering/work,
Tube's rejected population and active correction, Robust dual populations and
infeasible selection, CC auxiliary states and PSD covariance, numerical-failure
classification in both modes, legacy CLIs, README images and plots-only reconstruction.
The plots-only test forbids controller planning, dynamics steps and random draws and
compares all six PNG hashes. Fresh IT, Tsallis and all four DIAL pass costs/weights and
commands additionally match the original saved mechanism arrays **exactly**.

The README contains the scientific interpretation and reproduction commands. It
separates different targets, one-decision effects, short execution behavior and the
independent fixed-policy covariance ensemble. No universal ranking is inferred.
Full arrays are ignored; compact records and six PNGs are committed. Per-step
recording overhead is not a timing benchmark.

Final required checks: `uv sync --frozen`, `uv run ruff check .`, and
`uv run ruff format --check .` passed; **262 tests passed in 62.11 s**, including
30 new trace tests. The baseline had 232 passing tests.

## Exact changed-file inventory

Relative to the audited main revision. R100 denotes a byte-identical relocation.

| Status | Path |
| --- | --- |
| M | `README.md` |
| A | `configs/variants/traces.yaml` |
| A | `docs/codebase_audit.md` |
| A | `experiments/compare_traces.py` |
| R100 | `experiments/actuator_saturation.py` → `experiments/legacy/actuator_saturation.py` |
| R100 | `experiments/analytic_gaussian.py` → `experiments/legacy/analytic_gaussian.py` |
| R100 | `experiments/covariance_alignment.py` → `experiments/legacy/covariance_alignment.py` |
| R100 | `experiments/covariance_closed_loop.py` → `experiments/legacy/covariance_closed_loop.py` |
| R100 | `experiments/mechanism_summary.py` → `experiments/legacy/mechanism_summary.py` |
| R100 | `experiments/nonconvex_optimization.py` → `experiments/legacy/nonconvex_optimization.py` |
| R100 | `experiments/temperature_sweep.py` → `experiments/legacy/temperature_sweep.py` |
| R100 | `experiments/trajectory_density_study.py` → `experiments/legacy/trajectory_density_study.py` |
| A | `experiments/trace_decision.py` |
| A | `results/variants/traces/figure_provenance.json` |
| A | `results/variants/traces/index.json` |
| A | `results/variants/traces/manifest.json` |
| A | `results/variants/traces/pipeline_overview.png` |
| A | `results/variants/traces/plot_data.json` |
| A | `results/variants/traces/stage_01_dynamics.png` |
| A | `results/variants/traces/stage_02_objective.png` |
| A | `results/variants/traces/stage_03_sampling.png` |
| A | `results/variants/traces/stage_04_update.png` |
| A | `results/variants/traces/stage_05_feedback.png` |
| A | `results/variants/traces/validation.json` |
| A | `src/analysis/decision_trace.py` |
| M | `src/analysis/dynamic_campaign.py` |
| A | `src/analysis/trace_adapters.py` |
| A | `src/analysis/trace_comparison.py` |
| A | `src/analysis/trace_plots.py` |
| M | `src/controllers/covariance.py` |
| M | `src/controllers/dial.py` |
| M | `src/controllers/mppi.py` |
| M | `src/controllers/robust.py` |
| M | `src/controllers/tube.py` |
| M | `src/sampling/gaussian.py` |
| A | `tests/test_decision_trace.py` |


## Experiment and result organization after the README merge

The README organization was merged at `87b738f`. This follow-up audited all 34 CLI
files, their imports, CI commands, report generators, tests and registry references
before moving them. All 34 entry-point bodies remain byte-identical. Shared numerical
modules, controller code, configurations, original raw runs and media remain in place.

| Former location | Current location | Callers updated / evidence treatment |
| --- | --- | --- |
| Root benchmark run/audit/summary CLIs | `experiments/benchmarks/` | README; benchmark figure/video input consumers now read `results/variants/benchmarks/`. |
| Root formulation mechanism CLI | `experiments/mechanisms/` | README, historical Q06 registry and generated research-index commands. Output remains `results/variants/mechanisms/`. |
| Root decision/comparison trace CLIs | `experiments/traces/` | README; output remains `results/variants/traces/`. |
| Root video CLI | `experiments/videos/` | README; existing media and render records stay unchanged. |
| Plant/footprint validation CLIs | `experiments/validation/` | Vehicle report; original validation output paths retained. |
| Four-state and analytic launchers, including the original eight legacy files | `experiments/legacy/covariance/` | CI smoke commands and CLI compatibility tests; same shared implementations. |
| Historical frozen-decision CLIs | `experiments/legacy/frozen/` | Vehicle report and frozen-report generator commands. |
| Historical seven-state run/plot/audit/timing/finalization CLIs | `experiments/legacy/seven_state/` | Vehicle report, research index and question registry. |
| Historical five-state S-turn CLIs | `experiments/legacy/s_turn/` | Historical commands documented in the relocation index. |
| Root historical registry CLI | `experiments/legacy/research_index.py` | Registry report generator and research-index instructions. |
| `results/variants/V01…V06/` | `results/variants/benchmarks/V01…V06/` | All 84 files moved byte-for-byte; original embedded paths and hashes are retained as provenance. |
| `results/showcase/` | `results/historical/showcase/` | All five images moved byte-for-byte; README, docs, registry and publisher paths updated. |
| `results/registry.yaml` | `results/historical/registry.yaml` | Only current navigation/CLI paths changed; questions, findings, source commits and numerical run references retained. |
| `results/archive/` | `results/historical/archive/` | Recovery index moved; only three relative documentation links adjusted for depth. Immutable Git URLs retained. |

`results/relocations.json` lists every entry-point move and result-directory mapping,
with SHA-256 preservation checks for all 148 numerical/figure/provenance results and
all 34 CLI files. The historical registry and recovery index are the only edited
result files, and those edits update navigation. No evidence is deleted or regenerated.
Personal commands using old entry-point paths need the new folder prefix; an original
Git checkout still provides historical paths. Current consumers and CI use new paths,
without duplicate wrappers, symlink aliases or a second experiment framework.


The historical figure audit hashes `dynamic_plots.py`. Its sole edit changes the
publication directory from `results/showcase/` to `results/historical/showcase/`.
The relocation index records both source hashes and that directory mapping. The audit
accepts it only if reversing the declared path replacement reproduces the exact
original source hash; altered plotting logic still fails. Original figure provenance
files are not rewritten to claim a new source or rendering date.

Validation after organization: all 266 tests passed in 71.30 seconds, including
all 34 relocated CLI `--help` checks, preserved-result checksums, original benchmark
provenance resolution and rejection of non-path plotting changes. Ruff lint and
format checks passed (150 Python files); `git diff --check` passed. All 203 local
links in the README, research index, vehicle report and recovery index resolve.
All 210 tracked raw-run and media files retain their original hashes. Git recognizes
125 file relocations, 123 byte-identical and two with navigation-only edits; there
are no unpaired deletions. No research campaigns were rerun or committed figures
regenerated.
