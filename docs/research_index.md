# Research index

Stable IDs connect each question to hypotheses, controls, evidence and limitations.

| Question | Status | Finding |
| --- | --- | --- |
| [Q01 — Sampling covariance and estimation accuracy](questions/Q01_sampling_accuracy.md) | EXECUTED | An independent 64-seed replication preserves positive and negative orientation examples: at K=256, MSE is 0.006505 isotropic, 0.003646 at +45° and 0.065524 at −45°. These are estimates of one exact Gaussian target, not vehicle results. |
| [Q02 — Importance-weight concentration](questions/Q02_weight_degeneracy.md) | EXECUTED | At K=512 the isotropic mean ESS grows from 40.56 at λ=.1 to 450.61 at λ=10, while mean error changes from .04284 to .05207 against each changed exact target. Cost-only, ratio-only and combined dimension probes separate concentration mechanisms. |
| [Q03 — Frozen-state MPPI decision analysis](questions/Q03_frozen_decision.md) | EXECUTED | 7,200 seven-state frozen decisions show finite-budget differences. Only the post-disturbance high-sample reference passes the declared diagnostics: standard RMS error drops .1085→.0140 from K=64→4096; sensitivity-aligned .2471→.1597. Open-space and obstacle references remain unresolved. |
| [Q04 — Closed-loop trajectory differences](questions/Q04_closed_loop.md) | EXECUTED | Across five seven-state scenes, compatibility completion is 397/400 and independent holdout completion 391/400, with zero disk-proxy collisions. In holdout two-route trials, full takes 17 upper routes out of 20; standard/isotropic take 5 and diagonal 2. Route frequencies characterize these matrices, not published algorithm families. |
| [Q05 — S-turn difficulty and controller limitations](questions/Q05_s_turn_difficulty.md) | EXECUTED | Seven-state S-turn completion: 0.9 m: standard 20/20, isotropic 19/20, diagonal 20/20, full 20/20; 0.6 m: standard 20/20, isotropic 20/20, diagonal 20/20, full 20/20; 0.35 m: standard 20/20, isotropic 19/20, diagonal 20/20, full 20/20; 0.2 m: standard 20/20, isotropic 16/20, diagonal 20/20, full 19/20. Across 320 primary episodes, 0 disk-proxy collisions; all failed trajectories are retained. The separate state-jump checks have 44 collisions, reported separately. |
| [Q06 — Published MPPI formulation comparisons](questions/Q06_variant_formulations.md) | PLANNED | PLANNED — CEM, Tsallis-VI, DIAL, Tube-MPPI, Robust-MPPI and CC-MPPI are not implemented or evaluated. |
## Storage and provenance

- Full new raw records belong in `runs/seven_state/Qxx/<run_id>/`. Historical run names are relative
  symlinks to original tracked evidence, not copied result trees.
- `results/showcase/` holds one selected PNG per executed question. Other figures stay
  accessible from question documents.
- Raw episodes, arrays, snapshots, animations and redundant plots are ignored. Small
  top-level CSV/JSON summaries, exact YAML and manifests remain versionable under `runs/`.
- [Legacy index](../results/archive/legacy_index.md) preserves original paths and
  provenance. Historical experiments were not rerun for this reorganization.

Validate or rebuild the research index and question documents (the README is maintained separately):

```bash
uv run python experiments/research_index.py --render
uv run python experiments/research_index.py
```
