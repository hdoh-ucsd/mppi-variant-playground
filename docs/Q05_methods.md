# Q05 — S-turn methods and interpretation

This is a controlled task-difficulty study of four covariance settings within the shared IT-MPPI implementation. It does not implement Tube-MPPI, Robust-MPPI or CC-MPPI. The clearance range is selected by mechanical feasibility before the comparison; covariance values are declared in advance and are not selected for a preferred winner.

## Model and normalized controls

The separate `SteeringRateBicycle` wraps the original bicycle without replacing it. State is `[px,py,heading,velocity,steering_angle]`; physical input is `[acceleration,steering_rate]`. With explicit Euler, translation and heading use current velocity and current steering, while acceleration and rate update the next speed and steering. Thus a new rate command affects yaw only on subsequent transitions. Steering is clipped to ±.55 rad, rate to ±.8 rad/s in the primary experiment, acceleration to ±2 m/s² and speed to [0,2] m/s. There is no added actuator lag or tire dynamics.

Input normalization is `D=diag(2,.8)` for primary acceleration/rate commands. These are new coordinates: the old acceleration/steering configurations are not reused. The fixed base is diag(.26,.14); proposals are standard diag(.26,.14), isotropic .20 I, diagonal diag(.31,.09), and full [[.26,.07],[.07,.14]]. All traces are .40, with different determinants. The .4 rad/s secondary rate condition renormalizes the rate coordinate to its own physical limit, and is interpreted separately from the clearance sweep. Latent likelihoods use the unchanged full base/proposal correction, including mean and covariance mismatch.

The shared controller now supports a model-declared state dimension and an optional cost-provided transition-clearance method. Original four-state predictions and disk/circle costs use their existing path. No separate MPPI estimator is implemented.

## Corridor and feasible maneuver

The mechanically specified reference accelerates to 1 m/s, traverses an entry straight, executes a positive/negative/positive steering profile, and traverses an exit straight. The peak steering is atan(wheelbase/turn_radius), with turn radius 3 m; smooth sin² lobes have durations proportional to turn separation 3 m. The negative lobe lasts twice as long as each positive lobe. This yields two steering sign changes and near-zero final heading. The reference does not supply commands or warm starts to MPPI.

The corridor is an explicit closed polygon formed by normal offsets of the reference polyline, with straight entry and exit extensions. The existing disk footprint has radius .22 m. Nominal corridor clearance is half-width minus disk radius; primary clearances .90,.60,.35,.20 m give full widths 2.24,1.64,1.14,.84 m. All other primary quantities remain fixed. Before any proposal evaluation, the runner verifies positive swept disk clearance, turning and steering-rate limits, finite goal state, two reversals and a reference duration below the execution horizon for every condition. The .4 rad/s secondary condition also has a feasible reference; its required peak rate is about .3314 rad/s.

Collision checking uses exact point-to-boundary and segment-to-boundary distances and a signed inside-polygon test. A swept disk is unsafe when its centerline transition lies within the footprint radius of a boundary, or an endpoint is outside. Straight transitions match the model's Euler interpolation convention. The KD-tree accelerates candidate selection with certified lower bounds and exhaustive fallback; tests compare it against all-segment distance calculations, including a long chord whose endpoints are inside but whose interior cuts the curved boundary. Segment subdivision preserves geometry exactly. This is not a waypoint-proximity collision surrogate.

The diagnostic projection is the nearest point on the centerline's piecewise linear segments, with deterministic ties. Progress is the segment's cumulative arc length plus projected fraction, clipped to the centerline endpoints; maximum progress is its episode maximum. Cross-track error is Euclidean distance to that projection. The corridor does not impose a centerline tracking constraint, and multiple paths and control sequences remain feasible.

## Objective and finish condition

Running cost includes remaining arc-length squared, desired-speed error, local tangent-heading error, a boundary-clearance margin penalty, collision penalty, normalized effort and input variation. There is no cross-track running penalty. Terminal terms include remaining progress, longitudinal shortfall/lateral goal error, heading and speed. Progress and heading provide navigation guidance but do not require following the centerline exactly. All weights, model parameters and limits are shared within a condition.

An initial exploratory seed-42 pilot with a stopped point-goal produced terminal overshoot/timeouts after the turns. Its raw records remain local at `runs/Q05/s_turn_pilot_42_v2/`; it is not the evaluation dataset. To keep Q05 focused on corridor traversal rather than a separate braking maneuver, the common final design uses a **finish gate at target speed**. No proposal covariance or clearance was changed using that pilot.

Completion requires projected progress within .30 m of the terminal station, heading within .35 rad of the terminal heading, speed within .70 m/s of the 1 m/s target, and no swept collision. A straight 8 m runout beyond the goal avoids penalizing physically safe post-finish prediction. The resolved initial and goal states are stored in each condition's YAML/JSON. Execution stops at first valid completion, first collision or 200 steps (30 s). This differs explicitly from the historical stopped goal-pose criterion.

## Design, evidence and interpretation

The final smoke test uses seed 42, separated from evaluation seeds 20000–20019. Primary evaluation uses 20 paired seeds × four clearances × four proposals at K=256, T=24 (3.6 s), temperature 8 and speed target 1 m/s. Secondary checks at clearance .35 m change only K to 128, or only the physical steering-rate limit to .4 rad/s. They are not pooled with the primary curves. Independent episodes may run concurrently with single-thread NumPy; latency reflects shared-host contention and is descriptive.

Recorded per-seed metrics include success, collision and timeout censoring; maximum projected progress; cross-track RMSE; minimum swept clearance; completion time only for completions; cumulative common evaluation cost; ESS and entropy; feasible-rollout fraction; normalized effort; rate and angle saturation; planning latency and all scalar model transitions including nominal predictions. Goal residuals and steering reversal counts are retained. Small top-level CSV/JSON summaries, polygon geometry, representative paths and manifests are versioned. Full executed states, controls, diagnostic traces and selected rollouts remain local under ignored episode directories. Snapshots are opt-in at steps 15,40,65 for the predeclared representative seed, rather than every decision.

The single composite uses marginal Wilson 95% completion intervals and the same 20 optimizer seeds per proposal. A supporting panel shows seed 20000 at clearance .35 for every proposal, retaining failures and using one map. Paired differences against standard have descriptive pointwise t intervals in their own table. These are exploratory comparisons, not multiplicity-adjusted superiority tests. `key_figure_data.json`, `geometry.json` and `representative_paths.json` support regeneration without simulation or new random draws.

Interpretation must distinguish physical feasibility, finite sampling, horizon limitation, weight concentration, actuator saturation and task-cost effects. A valid reference rules out impossibility of that reference, not feasibility of every sampled command. A high success fraction for one matrix is not a property of its entire covariance family. The secondary budget and rate checks are local sensitivity slices, not complete factorial sweeps or causal decompositions of failure. Zero process noise means seed variation concerns optimizer sampling, not uncertainty in the physical environment.

## Commands

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run python experiments/s_turn_study.py --phase pilot --output runs/Q05/s_turn_pilot_reproduction
uv run python experiments/s_turn_study.py --phase evaluation --output runs/Q05/s_turn_reproduction
uv run python experiments/s_turn_study.py --plots-only --output runs/Q05/s_turn_clearance_v1 --showcase results/showcase/Q05_key.png
uv run python experiments/research_index.py --render
uv run python experiments/research_index.py
```

Completed-run metadata and actual findings are linked from the [Q05 question document](questions/Q05_s_turn_difficulty.md). Earlier historical studies have not been rerun or relocated for this organization work.

The executed final [seed-42 pilot manifest](../runs/Q05/s_turn_finish_gate_pilot_42/manifest.json) records 16/16 finish-gate completions and no collisions. Its [raw-record audit](../runs/Q05/s_turn_finish_gate_pilot_42/audit.json) verifies all 16 episodes. Pilot data are separate from the paired evaluation seeds.
