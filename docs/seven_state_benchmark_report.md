# Seven-state dynamic vehicle campaign

This document distinguishes model validation, analytic checks and vehicle experiments. At this implementation checkpoint, the dynamic model and controller have passed tests; campaign outcomes remain unreported until their immutable run records exist. The [pre-implementation audit](seven_state_audit.md) records the starting scope.

## Motivation and relation to the paper

The new plant adds body lateral velocity, yaw dynamics and a dynamic roll response to expose effects absent from the original kinematic vehicle. It preserves `[px,py,heading,roll,vx,vy,yaw_rate]` and steering/throttle inputs. The original four-state and separate five-state models remain available, and their evidence is retained.

The [2017 paper, §VI-B](https://arxiv.org/html/1707.02342v1#S6.SS2) uses this state/input description with dynamic updates fitted from driving data. Our equations and parameters are an illustrative physics approximation, not its identified basis-function or neural network model, AutoRally calibration, or real-vehicle reproduction.

## Exact implemented equations

Let yaw be ψ, roll φ, forward/lateral velocities u,v, yaw rate r, steering δ and signed normalized throttle a. Body x points forward, y left and z up. Positive steering/yaw turn left. Positive roll raises the left side, corresponding to outward roll during a left turn. All lengths, masses, forces and time use SI; a is dimensionless.

With L=lf+lr, static axle loads Zf=mg lr/L, Zr=mg lf/L, regularization q=√(u²+ε²), and w=u²/(u²+ε²):

$$\alpha_f=\delta-\operatorname{atan2}(v+l_f r,q),\qquad
\alpha_r=-\operatorname{atan2}(v-l_r r,q),$$
$$F_f=w\mu Z_f\tanh(C_f\alpha_f/(\mu Z_f)),\qquad
F_r=w\mu Z_r\tanh(C_r\alpha_r/(\mu Z_r)).$$

Rear-body propulsion/braking is Fd=Fe a for a≥0 and Fb a for a<0. Drag is c1 u+c2 u|u|. The applied forces and yaw moment are

$$F_x=F_d-c_1u-c_2u|u|-F_f\sin\delta,$$
$$F_y=F_f\cos\delta+F_r-m(1-w)v/t_0,$$
$$M_z=l_fF_f\cos\delta-l_rF_r-I_z(1-w)r/t_0.$$

The seven derivatives are

$$\dot p_x=u\cos\psi-v\sin\psi,\quad
\dot p_y=u\sin\psi+v\cos\psi,\quad\dot\psi=r,$$
$$\dot\phi=(hF_y-k_\phi\phi)/c_\phi,\quad
\dot u=F_x/m+vr,\quad\dot v=F_y/m-ur,\quad\dot r=M_z/I_z.$$

Roll is an explicit overdamped torque-balance state, not an algebraic equilibrium substitution. Roll inertia/rate is omitted to retain seven states; its relaxation time is cφ/kφ=.4 s. Tire loads remain static: this model omits load transfer, a friction ellipse, suspension/contact and roll-to-grip feedback. Tire forces saturate smoothly rather than remaining unbounded linear forces. Low-speed blending and viscous lateral/yaw relaxation avoid singular slip divisions and spurious steering forces at rest. These are disclosed regularizations, not an identified low-speed tire law.

Parameters are in [vehicle.yaml](../configs/seven_state/vehicle.yaml): mass 35 kg, yaw inertia 8 kg m², lf=lr=.4 m, Cf=Cr=450 N/rad, μ=.9, Fe=70 N, Fb=105 N, CG height .2 m, net roll restoring stiffness 120 N m/rad and damping 48 N m s/rad. Positive throttle propels; negative throttle brakes. Reverse driving is disabled. A forward-speed governor projects midpoint and accepted speeds to [0,4] m/s; steering is clipped to ±.5 rad and throttle to ±1. The model has no hidden steering servo/rate state or actuator lag.

Integration is explicit midpoint RK2 with dt=.15 s and three substeps, h=.05 s. Predictions and executions use the same integrator, except for explicitly declared physical parameter mismatch. Collision checks sweep the disk over each integration substep's straight segment; continuous curved motion between these substeps is still an approximation.

## Model validation

Validation covers zero equilibrium, straight throttle driving, constant steering, steady turning, left/right symmetry, reversal, braking without reverse, lateral/yaw/roll transients, near-zero perturbations, bounds and refinement convergence. The turning oracle independently solves nonlinear tire/force balances at fixed speed. Fitted XY curvature is compared with that dynamic oracle; kinematic u tan(δ)/L is not treated as exact.

The model must pass `runs/seven_state/validation/dynamics_v1/validation.json` before any campaign driver runs. Nonfinite integration raises; there is no fallback to another vehicle. Preliminary validation passed; formal versioned evidence and actual values are added after execution.

## Protocol

The same shared MPPI planner, full latent p/q correction and base distribution are used within a condition. Control order is **steering, throttle**, with physical transform `clip(diag(.5,1) z, bounds)`. New proposals have normalized trace .40: standard diag(.14,.26), isotropic .20 I, diagonal diag(.09,.31), full [[.14,.07],[.07,.26]]. The fixed base is diag(.14,.26). These are not the old acceleration/steering covariances under new names. Q03 also rotates matched-eigenvalue (.30,.10) proposals using local cost sensitivity, without inferring control directions from XY geometry.

Primary driving uses K=384, T=20, temperature 8 and desired body-forward speed 1.4 m/s. Sampling dimension is **2T=40**, not seven. Optimizer stream entropy is [4400,seed], disturbance entropy [4401,seed], and Q03 paired normals/reference streams use disjoint [101,state_index,seed] / [202,state_index,rep] namespaces. Disturbance impulses are realized once per seed and shared across proposals.

Q04 covers open space, a narrow passage, a two-route circle, a step-20 impulse and execution-model tire stiffness reduced to 70% of planning stiffness. Seeds 42–61 are a compatibility/examined cohort. Separate seeds 31000–31019 are holdouts. Neither cohort is silently pooled into the other's summary.

Q05's corridor is generated around an actually executed seven-state reference steering maneuver (+/−/+ lobes with speed feedback), not a kinematic turning-radius oracle. Offset normals use course direction including lateral slip. The existing exact polygon/segment geometry and disk radius .22 m are reused. Clearances .90,.60,.35,.20 m are declared before comparison. Reference steering/throttle, swept clearance, lateral velocity, yaw and roll are checked for each condition. The reference profile has a rate bound for feasibility, but MPPI steering is an instantaneous angle input; a genuine actuator-rate study requires additional history/augmentation and is not claimed here.

The seed-42 pilot precedes holdouts 32000–32019. If any primary marginal Wilson interval has width >.36, every Q05 condition expands to 40 paired seeds using 32020–32039. This is a predeclared precision rule, not selection for a preferred covariance. Additional secondary studies here vary K=128,384,768,1536 at a declared Q04 obstacle task and Q05 clearance .35; speed, friction, horizon and cost remain fixed. These are equal-rollout-budget, not equal-wall-clock-budget comparisons.

The common finish region uses terminal progress shortfall ≤.30 m, lateral goal error ≤.40 m, heading error ≤.35 rad, vx error ≤.8 m/s, |vy|≤.35 m/s, |r|≤.35 rad/s and |roll|≤.15 rad, without collision. Progress is projected/clipped to the route endpoint, so later recovery in a straight runout is allowed; Euclidean goal error is retained. This is not a stopped-point or first-crossing-only criterion. Timeouts are censored at 160 steps (24 s), and completion time is reported only among completions.

## Question status at implementation checkpoint

- Q01/Q02: existing analytic evidence is preserved; independent-seed replications are pending. These questions do not require vehicle dynamics.
- Q03: new frozen states, 50 paired seeds, eight K values and independent high-sample repetitions are pending. Unreliable references will remain unresolved.
- Q04/Q05 and budget checks: implementation and physical reference validation are available; actual outcomes are pending.
- Q06: planned formulation comparison. CEM, Tsallis-VI, DIAL, Tube-MPPI, RMPPI and CC-MPPI are not implemented by relabeling covariance proposals.

## Storage and reproduction

New immutable run IDs are under `runs/seven_state/Q01`–`Q05`. Exact YAML, small CSV/JSON summaries, selected representative traces/plot inputs and manifests are committed; full episode records and rollout NPZs are ignored local data. Cloning alone does **not** supply those raw arrays. No LFS or external object store is configured, so reproducing raw evidence requires executing the recorded commands. Figures read saved compact inputs or raise if required data is absent; they never launch optimization or resample.

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run python experiments/validate_dynamic_vehicle.py --output runs/seven_state/validation/dynamics_v1
uv run python experiments/seven_state_campaign.py --stage q01 --output runs/seven_state/Q01/seven_state_q01_v1
uv run python experiments/seven_state_campaign.py --stage q02 --output runs/seven_state/Q02/seven_state_q02_v1
uv run python experiments/seven_state_campaign.py --stage q03 --output runs/seven_state/Q03/seven_state_q03_v1
uv run python experiments/seven_state_campaign.py --stage q04 --output runs/seven_state/Q04/seven_state_q04_v1
uv run python experiments/seven_state_campaign.py --stage q05 --output runs/seven_state/Q05/seven_state_q05_v1
uv run python experiments/seven_state_campaign.py --stage budget_q04 --output runs/seven_state/Q04/seven_state_q04_budget_v1
uv run python experiments/seven_state_campaign.py --stage budget_q05 --output runs/seven_state/Q05/seven_state_q05_budget_v1
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Use fresh suffixed IDs for reproductions; completed run directories are never overwritten. Final outcome tables, statistical uncertainty, reference reliability and failure analysis will be appended only from executed records. One primary figure per completed question will be curated after validation, with the old four/five-state findings explicitly labeled and retained.
