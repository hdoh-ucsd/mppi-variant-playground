# Q05 — What the S-turn experiment actually shows

**The narrowest tested corridor produces a larger observed completion gap, but the threshold question remains unresolved.** All 480 episodes avoided collision; the 17 failures were timeouts near the corridor end and all missed the terminal speed requirement. These outcomes do not isolate covariance-driven route-discovery failure from terminal recovery, cost and finite-horizon effects.

## Executed evidence

The primary sweep used four specific trace-.40 covariance proposals, clearances .90,.60,.35,.20 m, K=256, T=24, temperature 8, speed target 1 m/s, steering limit .55 rad and rate limit .8 rad/s. Every condition used the same 20 optimizer seeds, 20000–20019. The secondary cells change only K to 128 or the physical rate limit to .4 rad/s at clearance .35 m; they are not pooled with the primary curves.

There are **320 primary + 80 budget + 80 rate episodes = 480**, with **463 completions, 17 timeout-censored runs and zero collisions**. The raw-data audit verified all 480 model traces and 68 selected rollout populations, including Gaussian density ratios and weighted updates. Four requested snapshot times fell after termination and were not fabricated. There were 215,226,408 scalar planning transitions, including updated nominal predictions. The minimum observed executed clearance was only .000115 m; zero model collisions is not a physical safety guarantee.

The clean evaluated source is `ff3df5824856e8252a221bc6a4fdce8105ec298c`. [Manifest](../runs/Q05/s_turn_clearance_v1/manifest.json), [exact configuration](../runs/Q05/s_turn_clearance_v1/config.yaml), [feasibility](../runs/Q05/s_turn_clearance_v1/feasibility.json), [raw-record audit](../runs/Q05/s_turn_clearance_v1/audit.json), [per-seed outcomes](../runs/Q05/s_turn_clearance_v1/per_seed.csv) and [paired differences](../runs/Q05/s_turn_clearance_v1/paired_differences.csv) preserve provenance and all trials. The final seed-42 pilot is separate; its 16 cases all completed without collision.

## Does reducing clearance increase performance differences?

| Nominal clearance [m] | Standard | Isotropic | Diagonal | Full | Largest observed completion gap |
| --- | --- | --- | --- | --- | --- |
| .90 | 20/20 | 18/20 | 19/20 | 20/20 | 10 percentage points |
| .60 | 20/20 | 18/20 | 20/20 | 20/20 | 10 percentage points |
| .35 | 19/20 | 20/20 | 20/20 | 19/20 | 5 percentage points |
| .20 | 17/20 | 18/20 | 20/20 | 20/20 | 15 percentage points |

The gap is larger at .20 m, but it is not monotonic over the full sweep. At .20 m, Wilson 95% completion intervals are approximately [.640,.948] standard, [.699,.972] isotropic, and [.839,1] for diagonal/full. Their paired completion difference from standard is +.15 for diagonal/full, with a descriptive t interval **[−.0215,+.3215]**. This includes zero; it is not a demonstrated covariance-family ranking or an established difficulty threshold. Pointwise intervals have no multiple-comparison adjustment; all-tied seed differences can have degenerate t intervals and do not establish equivalence.

The selected [Q05 composite](../results/showcase/Q05_key.png) shows the entire [0,1] completion scale, common seed counts and uncertainty. Its paths use the predeclared seed 20000 at clearance .35 m, with shared map and zoom limits. All four happen to complete on this seed; it was not selected for success. Failed trials remain in the rates and full records. The zoom is explicitly labeled rather than changing axes between proposals.

## Do some proposals discover feasible controls more often?

At the **common initial state and zero warm start**, clearance .20 m, the mean fraction of collision-free horizon samples is .8828 standard, .8922 isotropic, .8943 diagonal and .8807 full. Mean ESS at that same initial decision is only 1.0773,1.0469,1.0169,1.0761 respectively. Most rollouts are collision-free, yet optimization weights concentrate on roughly one sample. This separates geometric feasibility from cost-and-density weighting.

Across each controller's own complete closed-loop episodes at .20 m, mean feasible-rollout fractions are .2944,.3086,.2908,.2730. Full has the lowest of these averages while completing 20/20 episodes. Thus a larger unweighted feasible population alone is not a completion certificate. These later summaries condition on different states and warm starts and are not frozen-state causal comparisons. See [initial-decision diagnostics](../runs/Q05/s_turn_clearance_v1/initial_decision_summary.csv) and Q03 for the controlled-state distinction.

## Does a larger rollout budget help?

At clearance .35 m and rate .8 rad/s:

| K | Standard | Isotropic | Diagonal | Full |
| --- | --- | --- | --- | --- |
| 128 | 19/20 | 19/20 | 20/20 | 19/20 |
| 256 | 19/20 | 20/20 | 20/20 | 19/20 |

This local near-ceiling slice shows only one additional isotropic completion. It does not establish a general sample-efficiency ordering or an asymptotic convergence rate. Both seeds and tasks remain finite, and estimator accuracy has no exact oracle here.

## Does reducing steering-rate capability reverse an apparent advantage?

At clearance .35 m and K=256, reducing the rate limit from .8 to .4 rad/s changes completions from **19,20,20,19** to **20,19,19,20** in standard/isotropic/diagonal/full order. The ordering reverses, with broad uncertainty. The fixed reference is physically feasible at both limits: its maximum required rate is .3314 rad/s and maximum steering .32175 rad, with a 3 m peak-curvature radius. Every clearance has strictly positive reference swept clearance, including .199475 m at the narrowest level.

This secondary parameter change also renormalizes the physical rate coordinate to its new limit, so the same latent matrices have different physical exploration scales. It is a declared condition-level sensitivity check, not a clean isolation of rate saturation from proposal scaling. Primary comparisons keep that normalization fixed across all four clearances.

## What caused the failures?

The records support diagnostics, not a complete causal decomposition:

- **Mechanical impossibility:** a valid reference exists for all six conditions, so the task as configured is not globally impossible. This does not make every sampled control sequence feasible.
- **Constraint violations:** the audit found all executed commands and steering states within declared limits. No executed collision occurred. Physical actuator saturation remains recorded and can alter future predictions.
- **Insufficient route discovery:** not established. All failures reach at least .950985 of the centerline's clipped progress; they are not uniformly early-turn failures. Most failed planning steps still have some collision-free sampled sequences.
- **Terminal/cost behavior:** all 17 timeouts miss the terminal speed tolerance and 11 also miss heading tolerance. The lowest speed permitted by the completion rule is .3 m/s. This is direct evidence that finish-region recovery materially affects completion statistics; it does not identify a unique cause of that recovery failure.
- **Weighting:** ESS is often close to one, but high ESS is also insufficient. Three standard failures have late median ESS about 145.51–160.46 while late median speed is only .058–.278 m/s. Their last-five-second physical path lengths are .299–1.359 m: zero change in progress clipped at one must not be called standstill.
- **Horizon and optimization:** the 3.6 s prediction horizon is shorter than the full maneuver. There is no horizon sweep, repeated-optimization intervention or reliable target-mean oracle, so horizon myopia, finite-sample error and objective misspecification are not causally separated.

[Failure diagnostics](../runs/Q05/s_turn_clearance_v1/failure_diagnostics.csv) retain all 17 timeouts and separate heading, speed, clipped progress and actual motion. No failure is silently dropped or replaced with a completion time.

The completion region begins .30 m before the terminal arc-length station and also permits recovery in the straight runout. It is not a stopped point-goal or a first-crossing-only test. This definition is common to every proposal and stated before evaluation, but remains an important limit on transfer to other navigation tasks. The earlier stopped-goal pilot was used to identify this confound; its records and prototype-source limitation remain documented separately.

## Scope and next question

The four covariance matrices, one corridor handedness, one speed, one horizon and 20 seeds per condition do not characterize covariance families. A positive full-covariance correlation can have different consequences in a mirrored task; no mirror study was executed. The evidence is consistent with task- and condition-specific finite-budget effects, not a universal benefit of anisotropy.

An expanded seed set, mirrored geometry, explicit horizon intervention and a separately validated terminal task would be needed to locate a robust covariance-sensitive difficulty threshold. These are limitations and future studies, not completed experiments. No published MPPI formulation comparison has been executed.

## Reproduce and regenerate

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run python experiments/s_turn_study.py \
  --config runs/Q05/s_turn_clearance_v1/config.yaml \
  --phase evaluation --output runs/Q05/s_turn_reproduction
# Saved summaries, paths and geometry only; no controller execution.
uv run python experiments/s_turn_study.py --plots-only \
  --output runs/Q05/s_turn_clearance_v1 --showcase results/showcase/Q05_key.png
uv run python experiments/verify_s_turn.py \
  --input runs/Q05/s_turn_clearance_v1 --summaries-only
# Full local raw records are needed for these two commands.
uv run python experiments/verify_s_turn.py --input runs/Q05/s_turn_clearance_v1
uv run python experiments/s_turn_mechanisms.py --input runs/Q05/s_turn_clearance_v1
uv run python experiments/research_index.py --render
```

The raw episode trajectories, controls, diagnostic traces and rollout arrays remain under ignored `runs/Q05/s_turn_clearance_v1/episodes/`. Small summaries, exact YAML, polygon geometry, representative paths, numeric plotting inputs and manifests are versioned. No new bulk archive or redundant per-episode figure tree was committed. Historical evidence remains at its original paths with relative references from Q01–Q04.
