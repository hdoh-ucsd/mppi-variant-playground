# Research index

Stable IDs connect each question to hypotheses, controls, evidence and limitations.

| Question | Status | Finding |
| --- | --- | --- |
| [Q01 — Sampling covariance and estimation accuracy](questions/Q01_sampling_accuracy.md) | EXECUTED | At K=256 in the declared quadratic case, MSE is 0.007657 for isotropic, 0.003699 for +45° anisotropy and 0.049254 for −45°. Geometry helps or hurts depending on alignment. |
| [Q02 — Importance-weight concentration](questions/Q02_weight_degeneracy.md) | EXECUTED | In the isotropic K=512 slice, mean ESS rises from 41.95 at λ=.1 to 453.29 at λ=10 while mean error is nonmonotonic. Separate dimension probes isolate cost and density-ratio concentration. |
| [Q03 — Frozen-state MPPI decision analysis](questions/Q03_frozen_decision.md) | EXECUTED | 7,200 decisions isolate covariance at common frozen inputs. High-sample reference ESS is only 1.10–2.19 from 524,288 rollouts per state; all three reference means remain unresolved. |
| [Q04 — Closed-loop trajectory differences](questions/Q04_closed_loop.md) | EXECUTED | The full setting takes 3 upper and 17 lower station crossings in the two-route task; others take 20 lower. Its aligned mean path has −0.206 m clearance despite collision-free individual executions. |
| [Q05 — S-turn difficulty and controller limitations](questions/Q05_s_turn_difficulty.md) | PLANNED | PLANNED — the feasible reference and final seed-42 pilot pass at all four clearance levels; multi-seed evaluation is pending. |
| [Q06 — Published MPPI formulation comparisons](questions/Q06_variant_formulations.md) | PLANNED | PLANNED — CEM, Tsallis-VI, DIAL, Tube-MPPI, Robust-MPPI and CC-MPPI are not implemented or evaluated. |
## Storage and provenance

- Full new raw records belong in `runs/Qxx/<run_id>/`. Historical run names are relative
  symlinks to original tracked evidence, not copied result trees.
- `results/showcase/` holds one selected PNG per executed question. Other figures stay
  accessible from question documents.
- Raw episodes, arrays, snapshots, animations and redundant plots are ignored. Small
  top-level CSV/JSON summaries, exact YAML and manifests remain versionable under `runs/`.
- [Legacy index](../results/archive/legacy_index.md) preserves original paths and
  provenance. Historical experiments were not rerun for this reorganization.

Validate or rebuild the audience documents:

```bash
uv run python experiments/research_index.py --render
uv run python experiments/research_index.py
```
