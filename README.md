# MPPI Variant Playground

A reproducible research repository for understanding the mathematics and behavior of
**six MPPI formulations on one seven-state vehicle**: IT-MPPI, Tsallis-VI, DIAL-MPC,
Tube-MPPI, Robust-MPPI and CC-MPPI. The original covariance studies remain preserved.

## 1. Unified mathematical framework: where the variants intervene

The six formulations share a planning problem but **not necessarily the same variational objective**. The following notation provides a comparison scaffold, not a claim that every paper optimizes the same equation.

| Symbol | Meaning |
|---|---|
| $x_t,\bar x_t$ | Actual and nominal physical state |
| $u_t,\bar u_t$ | Applied and nominal physical control |
| $Z=(z_0,\ldots,z_{H-1}),\bar Z$ | Sampled and nominal latent control sequences |
| $f,w_t,g$ | Dynamics, process disturbance, actuator transformation |
| $J(Z;\kappa)$ | Predicted finite-horizon cost under a feedback structure |
| $P,p$; $Q_\theta,q_\theta$ | Reference distribution and parameterized sampling distribution |
| $\mathcal R(Q,P),\lambda$ | Regularizer/divergence and its scale |
| $\mathcal A,\mathcal U$ | Sampling/refinement method and update operator |
| $\kappa_t,K_t$ | Ancillary feedback policy and optional linear gain |
| $\Sigma_t^x,\mathcal C_t$ | State covariance and its constraints |
| $N,H$ | Rollout count and prediction horizon |

A generalized **research-level scaffold** is

$$
\begin{aligned}
\inf_{Q,\theta}\quad & \mathbb E_{Z\sim Q,W}\left[J(X(Z,W;\theta),Z)\right]
+\lambda\mathcal R(Q,P),\\
\text{subject to}\quad & X_{t+1}=f\bigl(X_t,g(z_t+\kappa_{\theta,t}(X_t,\bar X_t)),w_t\bigr),\\
&Q\in\mathcal Q,\quad \theta\in\Theta,\quad \mathcal C_t(\mathcal L(X_t))\le 0.
\end{aligned}
$$

This **umbrella is not a theorem asserted by all six papers**. For the IT-MPPI baseline, $\mathcal R=D_{\mathrm{KL}}$ and the ideal latent target is $\pi^*(Z)\propto p(Z)e^{-J(Z)/\lambda}$. Other variants may change that target, the optimization procedure, the feedback architecture, or the state uncertainty constraints.

### Formulation map: shared pipeline and intervention points

```mermaid
flowchart LR
    subgraph SHARED["Shared MPC computational pipeline"]
        direction LR
        A["Dynamics + task<br/>f, g, J"] --> B["Distributional objective<br/>R(Q,P)"]
        B --> C["Sample + refine<br/>q, A"]
        C --> D["Weight + update<br/>ω, U"]
        D --> E["Feedback + execute<br/>κ, u"]
    end
    E -. "next observation" .-> A
    T["Tsallis-VI<br/>divergence + support"] -.-> B
    T -.-> D
    I["DIAL-MPC<br/>iterative annealing"] -.-> C
    U["Tube-MPPI<br/>nominal + tracker"] -.-> E
    R["Robust-MPPI<br/>tracking-aware rollouts"] -.-> A
    R -.-> E
    S["CC-MPPI<br/>covariance steering"] -.-> C
    S -.-> E
    classDef core fill:#eaf2ff,stroke:#3b65a2,color:#18355b
    classDef variant fill:#fff2df,stroke:#b77925,color:#59370c
    class A,B,C,D,E core
    class T,I,U,R,S variant
```

*Solid arrows indicate the planning sequence. Dashed arrows identify the principal mathematical modifications. These dependencies are conceptual: feedback and uncertainty can enter multiple equations.*

| Formulation | Defining mathematical intervention | Main evaluation |
|---|---|---|
| **IT-MPPI** | KL-regularized Gibbs target, Gaussian proposals, importance-corrected weighted mean | Finite-sample accuracy and external task cost |
| **Tsallis-VI** | Generalized divergence, deformed weighting, support restrictions | Target shape, weight concentration and mode selection |
| **DIAL-MPC** | Multiple sampling/refinement passes and annealing schedule | Convergence against total computation |
| **Tube-MPPI** | Nominal plan, ancillary tracking, and nominal reset logic | Tracking under disturbances |
| **Robust-MPPI** | Augmented actual/nominal, tracking-aware sampled dynamics | Robustness and optimization overhead |
| **CC-MPPI** | Feedback-based covariance steering and constraints on state distribution | Predicted/empirical covariance and constraint satisfaction |

Do not conflate the **control proposal covariance** $C$, **predicted state covariance** $\Sigma_t^x=\mathrm{Cov}(X_t)$, and **state covariance bound** $\Sigma_{t,\mathrm{bound}}^x$. Equalizing sampling covariance is not covariance steering. The detailed equations, paper map, and implementation assumptions are included below in this README.

In the implementation equations below, $T=H$ and $K=N$ denote horizon and rollout
count; a subscripted $K_t$ denotes a feedback gain. The scaffold collects possible
interventions: an individual method need not optimize every variable or impose every
constraint. In particular, CC-MPPI uses an auxiliary noise state for its sampling
feedback, as specified in its equations below.

## 2. From mathematical differences to experiments

The question is how a change in **objective, sampling, refinement, feedback, or state
uncertainty** changes a control decision. A covariance setting alone does not constitute
a new published algorithm. These experiments use distinct mathematical mechanisms,
common external task metrics, paired seeds, and separately stated internal objectives.

The implementations are CPU adaptations to this repository's vehicle, not reproductions
of the papers' hardware or headline performance. Tests establish specified numerical
properties; trajectories alone do not establish paper fidelity or theoretical guarantees.

Read each experiment as a question about one intervention in the pipeline above.
The common vehicle exposes its consequences in a control task; the mathematical
tests separately check the defining equations.

| Intervention | Experiment and comparison | Observable difference to inspect | What the existing evidence establishes |
| --- | --- | --- | --- |
| Divergence and support | **V01:** IT-MPPI / Tsallis-VI | Which samples keep weight, ESS, and the resulting progress/cost | This Tsallis transform spreads weight more broadly yet stalls in the two-route task. Weight concentration alone does not explain control quality. |
| Sampling and refinement | **V02:** IT-MPPI / DIAL / fixed-noise DIAL | Updated cost after refinement, including the cost of every inner pass | DIAL has lower observed external cost in these scenes at the shared work ceiling; the fixed-noise ablation separates the schedule from iteration. |
| Nominal planning and tracking | **V03:** IT-MPPI / Tube-MPPI | Nominal–actual error, reset events, and the actual tracking contribution | The tracker is inactive in V03. These runs establish execution of the architecture but do not measure a tracking benefit. |
| Feedback inside sampled dynamics | **V04:** IT-MPPI / Robust-MPPI | Sampled feedback, nominal selection, corrected weights, and computation | Feedback is inactive in V04; candidate/rollout overhead remains visible. V06 S-turn supplies an active-feedback case. |
| State uncertainty | **V05:** IT-MPPI / CC-MPPI | Predicted versus empirical state covariance and terminal PSD residual | Local covariance bounds pass in this cohort; nonlinear prediction error remains and executed-state safety is unproven. |
| Combined consequences | **V06:** all six formulations | Scene-specific completion, external cost, deadline misses, and mechanism activity | Different targets and architectures create different outcomes under each computational protocol; three seeds per condition limit generalization. |

The figures below show external outcomes. The adjacent interpretations and saved
command diagnostics identify the relevant internal mechanism. A completion curve
alone cannot show whether tracking or a covariance constraint was active.

### Shared experimental conditions

| Study | Controlled comparison | Scenes | Executed episodes |
| --- | --- | --- | ---: |
| V01 | IT-MPPI / Tsallis weighting | Open, two-route obstacle | 24 |
| V02 | IT-MPPI / DIAL / non-annealed DIAL | Two-route obstacle, narrow passage | 36 |
| V03 | IT-MPPI / Tube tracking architecture | Disturbance, model mismatch | 24 |
| V04 | IT-MPPI / Robust augmented planning | Disturbance, model mismatch | 24 |
| V05 | IT-MPPI / CC state covariance | Open, narrow passage; shared stochastic actuation | 24 |
| V06 | All six formulations | Open, narrow, two-route, disturbance, mismatch, S-turn | 216 |

Each condition uses paired evaluation seeds **48000–48002**, horizon 20, the same
scene geometry, initial state, finish conditions, external task cost and 160-step episode
limit. S-turn clearance is 0.35 m. Three seeds give wide Wilson intervals; these are
exploratory mechanism studies, not general performance rankings. Repeated seeds and
conditions across stages/protocols are not independent replications.

**Dynamics-work protocol.** Every controller gets a ceiling of **7,700 predicted model
transitions per command**, including candidate searches, nominal rollouts, inner passes
and all finite-difference calls. Tube and DIAL use 7,680 due to integer allocation; the
remaining 20 are not filled with artificial work. Sampling settings are IT/Tsallis 384,
DIAL 95 per pass, Tube 182 per plan, Robust 159 per augmented pair plus candidate probes,
and CC 364. Each transition entails six RHS evaluations. Matrix factorization and solver
work are additionally reflected in planning time, rather than disguised as dynamics calls.

**Measured wall-clock protocol.** Independent calibration seeds 47900–47902 select
settings per algorithm and scenario. After warmup, the largest tested sample count with
90th-percentile time ≤120 ms is selected for a common **150 ms deadline**. V06 scans an
ascending grid until the first overrun; if no setting fits, it uses the minimum and records
violations. Calibration records are retained. This is a common calibrated ceiling, not
identical consumed runtimes or a hard real-time scheduler. V01's original grid ended at
384 and left substantial unused budget. Later grids extend to 3,072 samples.

Wall-clock episodes run serially. Work-budget episodes use eight workers; their timing
reflects concurrent load and is not used to claim comparative speed. The simulation keeps
its fixed control interval even after a deadline miss; delay-induced plant degradation is
not simulated. External metrics include completion, collision, goal error, cost, clearance,
effort, time and work. Internal weights, tracking and covariance metrics retain their
algorithm-specific meanings.

## 3. Experimental observations: what changes in practice

The original comparison figures use this reading order: **top** is completion with Wilson 95%
intervals, **middle** is planning time, and **bottom** is external cost for individual
seeds and their mean. The left column fixes dynamics work; the right uses serial,
calibrated wall-clock settings. Read cost together with completion because timeouts
and completed episodes have different durations. The cost axis is logarithmic.

### Recorded mechanism experiments

The five mechanism figures below add **actual numerical evidence** to the formulation
comparison; the original V01–V06 comparisons remain alongside them. Experiment rules
were committed at `bcfb998` before execution in
[the mechanism configuration](configs/variants/mechanisms.yaml).

Frozen inputs are the **step-8 IT-MPPI observation and warm start** in each named scene,
using source seed 49000. Evaluation seeds are 51000–51002. Every declared sign, seed,
and threshold is retained. Representative panels use the first seed, selected before
seeing outcomes. Shared samples are exact for the KL/Tsallis pair and for Robust's
plain-IT counterfactual; DIAL's different schedules deliberately transform its noise.

| Study | New numerical experiment | Recorded variables |
| --- | --- | --- |
| V01 | Two frozen states × three seeds × two weight laws; independent scalar quadrature and 18 Monte Carlo estimates | Costs, separate base/proposal log densities, normalized weights, ESS, latent updates and first commands |
| V02 | One frozen state × three seeds × three algorithms, each with 7,680 model transitions | Every DIAL population, pre/post-update mean, horizon noise schedule, predicted trajectory and task cost |
| V03 | 27 episodes of 32 commands: zero/positive/negative impulses × three seeds × IT/Tube-450/Tube-0 | Executed states, pre-reset and selected nominal states, requested/applied corrections, activation and reset events |
| V04 | 18 frozen Robust decisions with paired plain-IT counterfactuals | Candidate states/free energies, selection/fallback, nominal/actual sampled states, separate weights, planner and tracker actions |
| V05 | Two fixed CC designs; 2,048 independent 20-step stochastic executions | Per-timestep predicted/empirical covariance, gains, bounds, independent noise and executed state paths |

Full populations and trajectories remain under ignored `runs/variants/mechanisms/`.
Each [compact study directory](results/variants/mechanisms) contains its exact
configuration, frozen inputs, summaries, figure inputs, audit and hashed provenance.
**Optimizer weights, unweighted planning samples, and independently executed state
distributions are separate quantities throughout.** Only V03 and V05 execute the
mechanism interventions through the nonlinear plant; V01, V02 and V04 measure frozen
decisions and their predictions.

### V01 · Divergence and weighting

**Mathematical question.** How does replacing exponential weighting with a finite-support deformed likelihood change which samples contribute to the update? Inspect support and ESS alongside progress in the two-route task.

**Weighting mechanism.** The same samples and raw vehicle costs enter two different
weight laws, followed by the same fixed-covariance mean fit:

$$
w_k^{\rm KL}\propto e^{-J_k/\lambda}\frac{p(Z_k)}{q(Z_k)},\qquad
w_k^{\rm Ts}\propto\left[1-(r-1)\frac{\widehat J_k}{\widetilde\lambda}\right]_+^{1/(r-1)}
\frac{p(Z_k)}{q(Z_k)},\qquad U^+=\sum_k w_k Z_k.
$$

![V01: identical-sample weighting, commands and independent target references](results/variants/mechanisms/V01/V01_weighting_mechanism.png)

At the frozen two-route state, mean ESS across three seeds was **1.48 for KL and
1.66 for Tsallis out of 384 samples**. Tsallis retained 307 samples under its explicit
cutoff, but importance correction still concentrated almost all weight on a few.
Numerical zero weights in KL can arise from floating-point underflow; KL has no
finite-support cutoff. In the first seed, steering changed from **0.369 to 0.215 rad**
and throttle from **0.689 to 0.348**. Mean predicted updated-plan cost was **221.68
versus 1,075.40**. These are one-step effects, not an executed task comparison.

The lower panels separately solve a scalar objective
$J(z)=\tfrac12(z-1)^2$, $p=\mathcal N(0,1)$, with fixed transformed scale 0.8 and
$r=1.5$. Independent quadrature gives target means **0.55556 (KL)** and **0.66508
(Tsallis)**. Monte Carlo estimates are compared to their own reference, not to a
shared optimum. Vehicle Tsallis additionally uses batch normalization and an adaptive
80th-percentile cutoff, so its target changes between decisions.

[Mechanism records](results/variants/mechanisms/V01/summary.csv) ·
[Scalar reference estimates](results/variants/mechanisms/V01/reference_trials.csv) ·
[Audit](results/variants/mechanisms/V01/audit.json)

**Original formulation comparison.**

![V01 · Divergence and weighting](results/variants/V01_tsallis/primary.png)

Both methods completed open space. In the two-route task, IT-MPPI completed 3/3 and this Tsallis setting 0/3 under both protocols, with no collisions. The command-weighted mean ESS in the work-budget two-route runs was 98.80 for IT-MPPI and 197.14 for Tsallis, whose retained support fraction was 0.7995. Higher ESS accompanied stalled progress; it does not establish a better update. Batch normalization and the adaptive cutoff change the objective.

[Configuration](results/variants/V01_tsallis/config.yaml) · [Per-seed records](results/variants/V01_tsallis/per_seed.csv) · [Summary](results/variants/V01_tsallis/summary.csv) · [Command diagnostics](results/variants/V01_tsallis/diagnostics_summary.csv) · [Audit](results/variants/V01_tsallis/audit.json) · [Fidelity and limitations](results/variants/V01_tsallis/findings.json)

### V02 · Refinement and annealing

**Mathematical question.** How does repeated refinement with $C_{j,t}$ change the plan when every pass is counted against the same total work allowance? Compare DIAL with both IT-MPPI and the fixed-noise DIAL ablation.

**Annealing mechanism.** Each inner pass samples a different Gaussian neighborhood
and uses cost-only weights:

$$
C_{j,t}=0.2I_2\exp\!\left[-\frac{j}{\beta_1 I}
-\frac{T-1-t}{\beta_2(T-1)}\right],\qquad
U^{j+1}=U^j+\sum_k\mathrm{softmax}_k(-J/\lambda)\epsilon_k^j.
$$

![V02: recorded per-iteration occupancy, schedules, updates and work](results/variants/mechanisms/V02/V02_annealing_mechanism.png)

With four passes of 95 samples, DIAL and its fixed-noise ablation each use
$4(95+1)20=7{,}680$ transitions; IT-MPPI uses $(383+1)20=7{,}680$. The nominal-plan
rollout is included in each pass. Mean DIAL predicted task cost progressed
**164.34 → 155.13 → 155.09 → 154.19**, versus **157.88** for the fixed-noise final
plan and **221.68** for IT-MPPI. Individual seeds include nonmonotone changes.

The density panels count **unweighted sample-time points** with a shared full
spatial extent and color scale across the four passes. The curves show changed
predicted routes and updates; they do not establish executed behavior. IT-MPPI's
importance-corrected target also differs from DIAL's cost-only smoothing objective,
so the DIAL/fixed-noise pair is the cleaner test of the annealing schedule itself.

[Mechanism records](results/variants/mechanisms/V02/summary.csv) ·
[Audit](results/variants/mechanisms/V02/audit.json)

**Original formulation comparison.**

![V02 · Refinement and annealing](results/variants/V02_dial/primary.png)

All 36 episodes completed without collisions. Under the work ceiling, two-route mean external cost was 260.31 for IT-MPPI, 223.80 for DIAL, and 230.32 for the fixed-noise ablation. This is a small, specific-setting result; all inner work is included.

[Configuration](results/variants/V02_dial/config.yaml) · [Per-seed records](results/variants/V02_dial/per_seed.csv) · [Summary](results/variants/V02_dial/summary.csv) · [Command diagnostics](results/variants/V02_dial/diagnostics_summary.csv) · [Audit](results/variants/V02_dial/audit.json) · [Fidelity and limitations](results/variants/V02_dial/findings.json)

### V03 · Tracking architecture

**Mathematical question.** When does the nominal reset retain a discrepancy that produces a nonzero $K_t(x_t-\bar x_t)$ correction? Similar completion rates are informative about tracking only when that correction is actually used.

**Tracking mechanism.** The reset rule first selects the actual-state plan when
$S(U_x)\le S(U_{\bar x})+\eta$. The selected origin then determines the correction:

$$
e_t=x_t-\bar x_t^{\rm selected},\qquad
u_t=\mathrm{clip}\!\left(u_t^{\rm plan}+D K_t e_t,\,\mathrm{bounds}\right),\qquad
\Delta u_t^{\rm applied}=u_t-u_t^{\rm plan}.
$$

![V03: actual disturbance responses, nominal reset and active tracking](results/variants/mechanisms/V03/V03_tracking_mechanism.png)

The original threshold $\eta=450$ leaves the tracker inactive in **all nine**
Tube episodes. The stricter, predeclared $\eta=0$ activates it for **8 and 9 commands
in two of the three positive-impulse episodes**. It remains inactive for the third
positive seed, all negative impulses, and all zero-impulse controls. Requested and
applied steering corrections differ when the actuator clips.

For the positive impulse, mean integrated absolute lateral position over the window
falls from about **2.50 to 1.45 m·s** with the stricter setting, while mean total task
cost changes from **284.07 to 284.43**. The representative path also overshoots to the
opposite side. This demonstrates an active feedback response and a tradeoff; it does
not establish a general tracking benefit or stability guarantee. Both thresholds use
the same reset equation, and this comparison matches population size, not total work.

[Episode records](results/variants/mechanisms/V03/summary.csv) ·
[State and correction traces](results/variants/mechanisms/V03/steps.csv) ·
[Audit](results/variants/mechanisms/V03/audit.json)

**Original formulation comparison.**

![V03 · Tracking architecture](results/variants/V03_tube/primary.png)

All 24 episodes completed without collisions. The actual-state plan was selected whenever discrepancies arose, and the recorded ancillary correction was zero throughout. The benchmark therefore does not demonstrate a tracking benefit; targeted tests separately exercise the active tracker.

[Configuration](results/variants/V03_tube/config.yaml) · [Per-seed records](results/variants/V03_tube/per_seed.csv) · [Summary](results/variants/V03_tube/summary.csv) · [Command diagnostics](results/variants/V03_tube/diagnostics_summary.csv) · [Audit](results/variants/V03_tube/audit.json) · [Fidelity and limitations](results/variants/V03_tube/findings.json)

### V04 · Feedback-aware robustness

**Mathematical question.** What changes when the optimizer rolls out coupled nominal/actual trajectories and includes feedback in the sampling correction? Inspect sampled feedback and candidate selection before attributing outcomes to robustness.

**Tracking-aware planning mechanism.** Shared perturbations drive coupled nominal
and actual predictions; the actual rollout includes $k_t=K_t(x_t-\bar x_t)$.
At the implemented $\beta=0$, their separate fits use

$$
\widehat J_a=J_a+\frac\lambda2\sum_t k_t^\top C^{-1}k_t,\qquad
S_n=\tfrac12J_n+\tfrac12\max(\min(\widehat J_a,\alpha),J_n)
+\frac\lambda2\sum_tU_t^\top C^{-1}(U_t+2\epsilon_t),
$$
$$
S_a=J_a+\frac\lambda2\sum_t(U_t+k_t)^\top C^{-1}(U_t+k_t+2\epsilon_t),\qquad
w_n\propto e^{-S_n/\lambda},\quad w_a\propto e^{-S_a/\lambda}.
$$

![V04: candidate selection, coupled rollouts, feedback and command decomposition](results/variants/mechanisms/V04/V04_robust_mechanism.png)

At $\alpha=5000$, the signed disturbances select the actual-state candidate and
feedback stays zero. The actual fit matches the paired plain-IT fit to numerical
precision. At $\alpha=100$, **all nine candidate searches are infeasible** and retain
candidate zero. Under the nonzero impulses, that fallback activates sampled feedback:
mean RMS correction is **0.341** for the negative impulse and **0.354** for the
positive impulse in normalized control units. The corresponding actual-plan RMS
differences from same-noise IT are **0.474** and **0.528**.

The figure separates the planner's first action, the clipped tracker increment,
and the resulting command. This demonstrates how the implemented augmented dynamics
and two weight laws change a frozen decision. **It is not evidence for feasible
robust operation:** the active examples are fallback cases, $\alpha$ changes both
selection and the mixed objective, and no physical step is executed here. The
zero-disturbance and inactive original-threshold cases remain in the figure and data.

[Mechanism records](results/variants/mechanisms/V04/summary.csv) ·
[Audit](results/variants/mechanisms/V04/audit.json)

**Original formulation comparison.**

![V04 · Feedback-aware robustness](results/variants/V04_robust/primary.png)

All work-budget episodes completed. In serial timing runs, Robust-MPPI completed 0/3 disturbance and 2/3 mismatch episodes, versus 3/3 for IT-MPPI in each. Candidate-search overhead left only 32 and 16 augmented samples respectively. The default candidate threshold selected the actual state, so these runs do not establish an active-feedback robustness benefit.

[Configuration](results/variants/V04_robust/config.yaml) · [Per-seed records](results/variants/V04_robust/per_seed.csv) · [Summary](results/variants/V04_robust/summary.csv) · [Command diagnostics](results/variants/V04_robust/diagnostics_summary.csv) · [Audit](results/variants/V04_robust/audit.json) · [Fidelity and limitations](results/variants/V04_robust/findings.json)

### V05 · State uncertainty

**Mathematical question.** Does the feedback-shaped sample population satisfy $F_TF_T^\top\preceq\Sigma_f$, and how closely does nonlinear empirical covariance match that prediction? Completion and cost are complementary outcomes, not covariance measurements.

**Covariance mechanism.** A fixed gain acts on the auxiliary noise state, changing
the covariance of the unweighted controlled population:

$$
y_{t+1}=A_ty_t+B_t\epsilon_t,\qquad
z_t=U_t+K_ty_t+\epsilon_t,\qquad
F=(I+\mathcal B K)\mathcal B\bar\Sigma_\epsilon^{1/2},\qquad
\Sigma_T^x=F_TF_T^\top\preceq\Sigma_f.
$$

![V05: independent nonlinear covariance executions, ellipses, gains and PSD residuals](results/variants/mechanisms/V05/V05_covariance_mechanism.png)

Independent executions replay the **fixed warm-start mean and designed gains**, with
paired fresh noise across steered/zero-gain policies. They do not replan, use optimized
sample weights, or include the controller's 20% uncontrolled mixture. Every condition
has 256 independent executions, disjoint from the planning population. Predictions
and empirical covariance use the same noise scale within each comparison.

The normalized terminal trace divides each state by the square root of its declared
terminal variance bound, avoiding sums of incompatible physical units:

| Scene | Noise covariance | Empirical trace: steered / zero gain | Disk-proxy contacts: steered / zero gain |
| --- | --- | ---: | ---: |
| Open | $C_0$ | 0.776 / 3.714 | 0 / 0 |
| Narrow | $C_0$ | 0.767 / 3.264 | 64 / 183 |
| Open | $0.01C_0$ | 0.00706 / 0.05859 | 0 / 0 |
| Narrow | $0.01C_0$ | 0.00768 / 0.05524 | 0 / 22 |

All four steered conditions satisfy the empirical terminal PSD bound; the zero-gain
conditions at $C_0$ violate it in both scenes. The steered normalized prediction error
is about **0.35–0.36** at $C_0$ and **0.00069–0.00126** at $0.01C_0$, showing the
importance of the local-linearization/noise regime. The optimized bound is slack in
these cases; the gain objective still changes dispersion substantially.

**Covariance control is not collision avoidance.** The passing steered covariance
bound coexists with 64 narrow-passage contacts at the larger noise scale. Executions
continue for the full horizon after proxy contact to avoid conditioning covariance on
survival; the plant has no impact dynamics. Position ellipses are Gaussian covariance
summaries in meters, not guaranteed probability coverage or safety regions. Bottom
ellipses are centered to compare shape, while the top panels retain mean displacement.
These results validate a fixed-policy local covariance mechanism, not the full
receding-horizon controller's executed uncertainty distribution.

[Independent-execution summaries](results/variants/mechanisms/V05/summary.csv) ·
[Audit](results/variants/mechanisms/V05/audit.json)

**Original formulation comparison.**

![V05 · State uncertainty](results/variants/V05_covariance_steering/primary.png)

All 24 episodes completed without collisions. The constrained linearized solutions passed their PSD bounds; no empirical controlled-population terminal covariance violations were observed. The nonlinear empirical covariance still differed from the prediction (mean Frobenius discrepancy roughly 0.27–0.29 in mixed state units). This does not certify executed trajectory safety.

[Configuration](results/variants/V05_covariance_steering/config.yaml) · [Per-seed records](results/variants/V05_covariance_steering/per_seed.csv) · [Summary](results/variants/V05_covariance_steering/summary.csv) · [Command diagnostics](results/variants/V05_covariance_steering/diagnostics_summary.csv) · [Audit](results/variants/V05_covariance_steering/audit.json) · [Fidelity and limitations](results/variants/V05_covariance_steering/findings.json)

### V06 · Unified seven-state benchmark

**Mathematical question.** Which interventions become active across all six scenes, and how do their consequences change with the computational protocol? Read the scene-specific records rather than a pooled ranking.

![V06 unified results](results/variants/V06_unified/primary.png)

Completion counts below are out of three seeds per scene. Entries show **work-budget /
wall-clock** results. The two values are distinct protocols, not pooled trials.

| Method | Open | Narrow | Two-route | Disturbance | Mismatch | S-turn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IT-MPPI | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| Tsallis-VI | 3 / 3 | 3 / 3 | 0 / 0 | 3 / 3 | 3 / 3 | 0 / 0 |
| DIAL-MPC | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| Tube-MPPI | 3 / 3 | 3 / 3 | 2 / 3 | 3 / 3 | 3 / 3 | 3 / 2 |
| Robust-MPPI | 3 / 3 | 3 / 1 | 3 / 1 | 3 / 2 | 3 / 3 | 2 / 2 |
| CC-MPPI | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |

V06 recorded **195/216 completions**, **1 disk-proxy collision**,
and **20 timeouts** across its distinct conditions.
These totals describe execution coverage, not a pooled algorithm comparison. The audit
replayed all 216 episodes and checked 251 selected populations.

IT-MPPI, DIAL-MPC and CC-MPPI completed all 18 episodes in each protocol. This
Tsallis setting failed all two-route and S-turn episodes, including the single
collision. Costs include failures and timeouts and must be read with completion;
they are accumulated over episodes of different lengths.

The S-turn activated Tube tracking corrections and Robust sampled feedback under
both protocols. Tube corrections also appeared in serial narrow-passage runs.
Thus V03/V04's inactive tracking does not describe every V06 condition, but these
runs still do not isolate a causal tracking benefit. All CC local PSD bounds passed,
with no observed empirical controlled-population terminal covariance violations.

Serial command deadline misses were IT-MPPI **0/863**, Tsallis **0/1,463**, DIAL
**21/644 (3.26%)**, Tube **3/928 (0.32%)**, Robust **398/1,726 (23.06%)**, and CC
**5/855 (0.58%)**. These command-weighted fractions differ from averaging per-episode
fractions. The minimum calibration setting already exceeded the 120 ms target for
Robust in mismatch and S-turn, and DIAL in S-turn. Equal calibrated deadlines do
not imply equal delivered computation or that the deadline is achievable.

[Configuration](results/variants/V06_unified/config.yaml) ·
[Per-seed results](results/variants/V06_unified/per_seed.csv) ·
[Summary and intervals](results/variants/V06_unified/summary.csv) ·
[Command diagnostics](results/variants/V06_unified/diagnostics_summary.csv) ·
[Calibration](results/variants/V06_unified/calibration.csv) ·
[Audit](results/variants/V06_unified/audit.json) ·
[Findings and limitations](results/variants/V06_unified/findings.json)

## 4. Detailed formulations and paper map

### IT-MPPI foundation and importance correction

The implemented baseline solves a distributional control problem: choose a distribution
$Q$ over latent control sequences that trades expected rollout cost against departure
from a fixed Gaussian base $P$:

$$
\min_Q\;\mathbb E_Q[J(Z)]+\lambda D_{\mathrm{KL}}(Q\Vert P),
\qquad
\pi^*(Z)=\eta^{-1}\exp[-J(Z)/\lambda]p(Z).
$$

Here $Z=(z_0,\ldots,z_{T-1})$, $J$ is the trajectory cost, $\lambda>0$ is temperature,
and $\eta$ normalizes the target. The base and sampling proposal are

$$
p(Z)=\prod_{t=0}^{T-1}\mathcal N(z_t;0,C_0),
\qquad
q_{U,C}(Z)=\prod_{t=0}^{T-1}\mathcal N(z_t;U_t,C).
$$

The base covariance $C_0$ defines part of the objective; the proposal mean $U$ and
covariance $C$ determine where samples are drawn. Projecting the target onto a
Gaussian with fixed covariance gives the ideal mean $U^*=\mathbb E_{\pi^*}[Z]$.
For $K$ samples $Z^{(k)}=U+\epsilon^{(k)}$ from the proposal, MPPI estimates it with

$$
L_k=-J(Z^{(k)})/\lambda+\log p(Z^{(k)})-\log q_{U,C}(Z^{(k)}),
\qquad
w_k=\frac{\exp(L_k-\max_j L_j)}{\sum_i\exp(L_i-\max_j L_j)},
$$

$$
U^+=\sum_{k=1}^{K}w_kZ^{(k)}
=U+\sum_{k=1}^{K}w_k\epsilon^{(k)}.
$$

The full Gaussian density correction is

$$
\log\frac{p(Z^{(k)})}{q_{U,C}(Z^{(k)})}
=-\frac12\sum_t\left[
(z_t^{(k)})^\top C_0^{-1}z_t^{(k)}
-(\epsilon_t^{(k)})^\top C^{-1}\epsilon_t^{(k)}
+\log\det C_0-\log\det C
\right].
$$

When $C=C_0$, this reduces, up to a sample-independent term, to
$-\sum_t U_t^\top C^{-1}\epsilon_t^{(k)}$. The familiar cross-term correction
therefore applies to that equal-covariance case; changing $C$ generally requires the
full ratio. Cost-only weights generally define a different target.


The current IT-MPPI reference retains $C_0=C=\mathrm{diag}(0.14,0.26)$,
$\lambda=8$ and $T=20$. Its latent sequence has dimension $2T=40$.
Normalized inputs map to physical steering and throttle as
$u_t=\mathrm{clip}(\mathrm{diag}(0.5,1)z_t,\text{bounds})$.
Gaussian likelihoods precede clipping. The baseline applies the first updated command,
shifts the latent mean, and appends zero. Its behavior is checked against a saved
pre-implementation oracle from commit `c7bbf86`.

Changing only the proposal preserves the ideal fixed-base target but changes its
finite-sample estimate. The published formulations below can change the target itself.
Effective sample size $1/\sum_k w_k^2$ measures weight concentration; it is not an
accuracy, progress, or safety certificate.

### Equation-level formulation differences

| Method | Objective and sampling | Weight / update | Feedback and required information | Paper anchors |
| --- | --- | --- | --- | --- |
| IT-MPPI | Fixed-base KL objective; Gaussian control proposals. | Exponential cost likelihood with full $p/q$; weighted latent mean. | Nonlinear prediction model; no ancillary tracker. | Williams 2017 (14), (20)–(22). |
| Tsallis-VI | Tsallis divergence; fixed-covariance Gaussian policy class. | Truncated deformed likelihood with importance correction; weighted mean fit. | Nonlinear model; declared transformed scale and finite support. | Wang 2021 (6), (8)–(11), (14), (28), Table II. |
| DIAL-MPC | Iterated Gaussian-smoothed cost optimization. | Cost-only exponential weights at every annealing pass. | Nonlinear model; horizon and iteration noise schedules. | Xue 2024 (1)–(7), Algorithm 1. |
| Tube-MPPI | Separate nominal and observed-state plans. | Two IT-MPPI updates; solution-cost nominal reset rule. | Local trajectory tracker and dynamics Jacobians. | Williams 2018 Algorithms 1–2, §IV-C. |
| Robust-MPPI | Joint nominal/actual rollout objective and nominal-state search. | Separate mixed nominal and feedback-corrected actual weights. | Tracker inside prediction; Jacobians and candidate free-energy estimates. | Gandhi 2021 (7)–(11), Algorithms 1–3. |
| CC-MPPI | Covariance-steering quadratic subproblem and feedback-shaped sampling. | Augmented cost weights; fit shaped control sequences. | Jacobians, input-noise covariance, auxiliary noise state and terminal covariance bound. | Yin 2022 (3), (7)–(21), Algorithm 1. |

### Tsallis-VI: divergence and support

Using the paper's convention,

$$
D_r(Q\Vert P)=\frac{\int q(z)^r p(z)^{1-r}\,dz-1}{r-1},\qquad
\exp_r(x)=[1+(r-1)x]_+^{1/(r-1)},
$$

$$
q_r^*(Z)\propto p(Z)\exp_r[-J(Z)/\tilde\lambda],\qquad
w_k\propto\exp_r[-J_k/\tilde\lambda]\,p(Z_k)/q_{U,C}(Z_k).
$$

The transformed scale $\tilde\lambda$ is **not** the objective's regularization
multiplier. Holding it fixed gives the KL limit as $r\to1$. The executed setting
instead uses $r=1.5$, explicit batch cost normalization to $[0,1]$, and an 80th-percentile
cutoff $\gamma$, with $\tilde\lambda=(r-1)\gamma$. This adaptive scale changes the
decision objective. An empty support raises an error; it never becomes uniform weights.
The fixed-covariance Gaussian projection follows Table II; covariance fitting, mixtures
and Stein policies are not implemented.

### DIAL-MPC: iterative annealing

For inner pass $j=0,\ldots,I-1$ and horizon position $t=0,\ldots,T-1$,

$$
C_{j,t}=0.2I_2\exp\!\left[-\frac{j}{\beta_1 I}
-\frac{T-1-t}{\beta_2(T-1)}\right],\qquad
U\leftarrow U+\sum_k\mathrm{softmax}_k(-J/\lambda)\epsilon^{(k)}.
$$

The implementation uses four passes and $\beta_1=\beta_2=1$, with a separate
non-annealed ablation. Sampling uses the square root of the variance scale. The
cost-only law follows the paper's smoothing interpretation; it does not silently reuse
the baseline fixed-base importance weights. All passes and updated-plan rollouts count
toward work. The shifted warm start uses a zero tail.

### Tube-MPPI: nominal selection and tracking

Run MPPI from both the nominal state $\bar x$ and observation $x$. Select the actual-state
solution when its planned cost satisfies $S(U_x)\le S(U_{\bar x})+450$; otherwise retain
the nominal plan. Apply

$$u=\mathrm{clip}\bigl(u_{\rm nominal}+D K_0(x-\bar x),\text{bounds}\bigr).$$

The ancillary controller is a finite-horizon **local LQR adaptation**, rather than the
paper's iterative iLQG and asynchronous implementation. Central differences linearize
the same discrete plant in normalized input coordinates. A Riccati recursion uses
$Q=\mathrm{diag}(4,8,6,2,2,1,1)$, $R=I_2$, $Q_f=5Q$. No global tube bound or
stability claim is made across clipping kinks, large errors, or model mismatch.

### Robust-MPPI: feedback-aware planning

Nominal and actual rollouts share perturbations; actual rollouts additionally receive
$k_t=K_t(x_t-\bar x_t)$. At the implemented paper parameter $\beta=0$, define

$$
\widehat J_a=J_a+\frac\lambda2\sum_t k_t^\top C^{-1}k_t,
$$
$$
S_n=\tfrac12J_n+\tfrac12\max(\min(\widehat J_a,\alpha),J_n)
+\frac\lambda2\sum_t U_t^\top C^{-1}(U_t+2\epsilon_t),
$$
$$
S_a=J_a+\frac\lambda2\sum_t(U_t+k_t)^\top C^{-1}(U_t+k_t+2\epsilon_t).
$$

Nominal weights update the nominal plan; actual weights determine the executed first
command plus tracking. Five candidates interpolate previous nominal, propagated nominal,
and actual states. Eight probe samples estimate each candidate's free energy; choose the
nearest feasible candidate under $\alpha=5000$. If none qualifies, retain candidate zero
and log that event. The shared local LQR tracker is an adaptation. Contraction, recursive
feasibility and theoretical free-energy bounds are **not established** for this vehicle
and discontinuous task cost. Common physical effort/variation costs remain additional
task preferences beyond the paper's state-only notation.

### CC-MPPI: state covariance, not a full control covariance

For linearized dynamics $A_t,B_t$, auxiliary noise evolves independently of the
feedback-shaped physical trajectory:

$$
y_{t+1}=A_ty_t+B_t\epsilon_t,\quad y_0=0,\qquad
z_t=U_t+K_ty_t+\epsilon_t.
$$

Let $G=\mathcal B\bar\Sigma_\epsilon^{1/2}$ and
$F=(I+\mathcal B K)G$. The gain solve minimizes

$$\|\bar Q^{1/2}F\|_F^2+\|\bar R^{1/2}KG\|_F^2,
\qquad F_TF_T^\top\preceq\Sigma_f.$$

The code uses the feasible unconstrained quadratic solution when possible, otherwise
a Schur-complement SDP through CVXPY/Clarabel. A $10^{-10}$ numerical ridge is declared;
terminal PSD residual tolerance is $10^{-5}$. Infeasibility raises an explicit error.
The seven-state variance bound is $\mathrm{diag}(1,1,.25,.04,1,.5,.5)$.
Twenty percent of samples use zero nominal input, outside the controlled-population
bound. The paper's augmented quadratic cost uses $R=.01I_2$, noise ratio 100; the
weighted update includes feedback-shaped inputs and repeats the final tail input.
The auxiliary initial covariance is zero. A bound below the final additive-noise
floor can be infeasible; no fallback silently relaxes it. The bound concerns the
linearized sampling population, **not** full-body safety or
the executed closed-loop state distribution. Nonlinear empirical covariance is recorded
separately.

### Paper map

Arrows are conceptual reading connections. Blue is the baseline, green denotes
implemented seven-state adaptations, and gray gives historical/tutorial context.

```mermaid
flowchart TB
    P1["[1] Williams et al. · 2016<br/>Original MPPI"]
    P2["[2] Williams et al. · 2017<br/>Information-theoretic formulation<br/>Target, importance weights, mean update"]
    P8["[8] Jordana et al. · 2025<br/>Zero-order optimization tutorial<br/>Random search and smoothing"]

    subgraph SEARCH["Weighting and search"]
        P5["[5] Wang et al. · 2021<br/>Tsallis-VI MPC<br/>Divergence and cost-to-weight transform"]
        P7["[7] Xue et al. · 2024<br/>DIAL-MPC<br/>Iterative refinement and noise annealing"]
    end

    subgraph FEEDBACK["Feedback and uncertainty"]
        P3["[3] Williams et al. · 2018<br/>Tube-MPPI<br/>Nominal planning and tracking"]
        P4["[4] Gandhi et al. · 2021<br/>Robust-MPPI<br/>Tracking-aware rollout optimization"]
        P6["[6] Yin et al. · 2022<br/>CC-MPPI<br/>Feedback sampling and state covariance"]
        P3 -->|Account for tracking in planning| P4
    end

    P1 -->|Information-theoretic derivation| P2
    P8 -.->|Optimization perspective| P2
    P2 -->|Change the weighting law| P5
    P2 -->|Change the search schedule| P7
    P2 -->|Add a tracking architecture| P3
    P2 -->|Control trajectory dispersion| P6

    classDef baseline fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef implemented fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef context fill:#f1f5f9,stroke:#64748b,color:#0f172a
    class P2 baseline
    class P3,P4,P5,P6,P7 implemented
    class P1,P8 context
```

| Ref. | Source used |
| --- | --- |
| [1] | [Williams et al., original MPPI, ICRA 2016](https://doi.org/10.1109/ICRA.2016.7487277). Historical context; no equation-level reproduction claimed. |
| [2] | [Williams et al., IT-MPPI, 1707.02342v1](https://arxiv.org/html/1707.02342v1). |
| [3] | [Williams et al., Tube-MPPI, RSS 2018](https://www.roboticsproceedings.org/rss14/p42.pdf). |
| [4] | [Gandhi et al., Robust-MPPI, 2102.09027v1](https://arxiv.org/html/2102.09027v1). |
| [5] | [Wang et al., Tsallis-VI MPC, 2104.00241v1](https://arxiv.org/html/2104.00241v1). |
| [6] | [Yin et al., CC-MPPI, 2109.12147v3](https://arxiv.org/html/2109.12147v3). |
| [7] | [Xue et al., DIAL-MPC, 2409.15610v1](https://arxiv.org/html/2409.15610v1). |
| [8] | [Jordana et al., zero-order optimization tutorial, 2506.22087v1](https://arxiv.org/html/2506.22087v1). Conceptual context. |

## 5. Shared seven-state dynamics

Every formulation uses the existing validated dynamic single-track model:

$$x=[p_x,p_y,\psi,\phi,v_x,v_y,r],\qquad u=[\delta,\tau].$$

The states are global position, heading, roll, body longitudinal/lateral velocity and
yaw rate; controls are steering angle and signed normalized throttle. Tire forces
saturate smoothly, body velocities are coupled, and roll obeys an overdamped torque
balance. Integration remains midpoint RK2 with three substeps per 0.15 s command interval.
Steering is bounded at ±0.5 rad, throttle at ±1, and forward speed at 0–4 m/s.

No vehicle parameters, state order, goal rules, geometry or cost implementation were
changed to favor a method. The plant omits reverse drive, full load transfer, roll
inertia/feedback and continuous steering-actuator dynamics. It is a physics approximation,
not the learned AutoRally model. The [validated equations and prior campaign](docs/seven_state_benchmark_report.md)
remain available. Collision checks retain the centered 0.22 m disk proxy, which does not
enclose the axles; full-body avoidance is not established.

V01–V04 retain the original deterministic execution plus declared scene disturbances.
V05–V06 add common independent Gaussian normalized actuator noise with covariance
$0.01C_0$, using `SeedSequence([4402, seed])`. All algorithms in those studies receive
the same realized noise stream. Noise is applied before the existing physical clipping;
the commanded and realized controls are recorded separately. CC sampling uses $C_0$,
100 times the execution-noise covariance, so its sampled dispersion is not a direct
prediction of executed closed-loop uncertainty.

## 6. Implementation status and mathematical validation

All six entries are **IMPLEMENTED seven-state adaptations**: defining mechanisms are
coded, mathematical checks pass, and recorded vehicle runs exist. The exact paper
hardware, tuning, theoretical guarantees and alternative policy classes are not claimed.

| Registry name | Implementation | Independent validation |
| --- | --- | --- |
| `it_mppi` | [Baseline](src/controllers/mppi.py) | Pre-change three-decision oracles for open, disturbance and S-turn scenes; full density correction. |
| `tsallis_vi` | [Tsallis](src/controllers/tsallis.py) | Analytically truncated weights; independently optimized discrete divergence problem; fixed-scale KL limit and reproducibility. |
| `dial_mpc` | [DIAL](src/controllers/dial.py) | Exact schedule; Gaussian tilted-mean integral; cost-only update and all-pass count. `dial_no_anneal` is an ablation. |
| `tube_mppi` | [Tube](src/controllers/tube.py) | Riccati gains versus independent batch least squares; plant derivatives; reset rejection and nonzero tracking test. |
| `robust_mppi` | [Robust](src/controllers/robust.py) | Feedback correction versus SciPy conditional Gaussian densities; mixed-cost clamp; feasible/infeasible candidate selection. |
| `cc_mppi` | [Covariance steering](src/controllers/covariance.py) | Analytic binding/inactive constrained optima; infeasible final-noise floor; Monte Carlo moments, symmetry/PSD, shaped mean fit. |

The [registry](src/controllers/registry.py) exposes one interface:
`reset(initial_state, seed)`, `command(observed_state)`, `get_diagnostics()`.
Controllers share the existing dynamics, costs, collision checker, episode recorder and
figure publisher. A wrapper counts actual calls to the plant; it does not implement a
second dynamics engine. Covariance and tracking states remain inside controllers.

## 7. Preserved covariance studies

Q01–Q05 below are the earlier fixed-proposal studies, not published-formulation
comparisons. Their original configurations, manifests, figures and raw paths are retained.
The [historical registry](results/registry.yaml) and [question index](docs/research_index.md)
continue to describe that evidence.

### Q01 · Does covariance geometry improve finite-sample estimation?

![Q01: sampling covariance and estimation accuracy](results/showcase/Q01_key.png)

For a **fixed, analytically solvable Gaussian target**, proposal orientation can help *or* hurt. At $K=256$ with 64 independent replication seeds, observed mean squared error was **0.003646** for the +45° full proposal, **0.006505** for isotropic, and **0.065524** for the −45° full proposal. These results concern estimation, **not vehicle performance**. [Experiment Q01](docs/research_index.md#q01).

### Q02 · What makes importance weights concentrate?

![Q02: importance-weight concentration](results/showcase/Q02_key.png)

Temperature and high-dimensional proposal mismatch can concentrate weights by different mechanisms. Effective sample size (ESS) is an overlap diagnostic, **not a guarantee of estimator accuracy**; changing temperature also changes the target. [Experiment Q02](docs/research_index.md#q02).

### Q03 · How does sampling geometry change one MPPI decision?

![Q03: frozen-state sampling, weighting and updated plan](results/showcase/Q03_key.png)

At an identical seven-state vehicle observation and warm start, different proposal covariances produce different sampled trajectories, corrected weights, and updated controls. The high-sample target reference is **unresolved** for the open-space and pre-obstacle decisions; only the post-disturbance reference passes the stated diagnostic screens. The pre-obstacle reference's pooled ESS is just **1.51 despite 524,288 samples**. [Experiment Q03](docs/research_index.md#q03).

### Q04 · Do decision differences persist in closed-loop driving?

![Q04: seven-state vehicle closed-loop comparisons](results/showcase/Q04_key.png)

The seven-state study recorded **391 completions in 400 independent holdout episodes** across five scenarios. Completion is often nearly saturated, but route selection can differ: in the two-route task, the full proposal selected the upper route in **17/20** holdout runs, versus **5/20** for standard and isotropic and **2/20** for diagonal. These are observations about specific covariance matrices, **not general rankings of MPPI families**. [Experiment Q04](docs/research_index.md#q04).

### Q05 · When does a more demanding S-turn expose limitations?

![Q05: S-turn success, representative paths and lateral dynamics](results/showcase/Q05_key.png)

In the primary seven-state S-turn study ($K=384$, 20 paired holdout seeds per setting), **313/320 episodes completed**. At the tightest 0.20 m clearance, completions were 20/20 standard, 16/20 isotropic, 20/20 diagonal, and 19/20 full. All seven primary failures were terminal-state timeouts; there were zero disk-proxy collisions. Separate studies investigate command-slew limits, speed preference, state-jump disturbances and sample budgets; they are **not pooled** into the primary clearance results. [Experiment Q05](docs/research_index.md#q05).

### Q06 · Original published-formulation plan

The original Q06 plan is preserved as historical context. The V01–V06 studies
above implement and evaluate six formulations; CEM remains planned. The historical
registry does not relabel covariance settings as published algorithms.

## 8. Reproduction instructions

Linux, Python 3.11+, `uv`; CPU NumPy/SciPy and CVXPY/Clarabel. No GPU is required.

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=.mpl-cache
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Run the stages individually, in order, with **fresh output and raw directories**:

```bash
uv run python experiments/variant_benchmark.py --study V01_tsallis \
  --output results/variants/V01_tsallis_reproduction --raw runs/variants/V01_tsallis_reproduction
uv run python experiments/variant_benchmark.py --study V02_dial \
  --output results/variants/V02_dial_reproduction --raw runs/variants/V02_dial_reproduction
uv run python experiments/variant_benchmark.py --study V03_tube \
  --output results/variants/V03_tube_reproduction --raw runs/variants/V03_tube_reproduction
uv run python experiments/variant_benchmark.py --study V04_robust \
  --output results/variants/V04_robust_reproduction --raw runs/variants/V04_robust_reproduction
uv run python experiments/variant_benchmark.py --study V05_covariance_steering \
  --output results/variants/V05_covariance_steering_reproduction --raw runs/variants/V05_covariance_steering_reproduction
uv run python experiments/variant_benchmark.py --study V06_unified \
  --output results/variants/V06_unified_reproduction --raw runs/variants/V06_unified_reproduction
```

The current config includes the final calibration policy. For exact historical protocol
reproduction, pass `--config results/variants/<study>/config.yaml`; the saved configuration
contains the original seeds, candidate grid and algorithm settings. Wall-clock calibration
is machine/load dependent and may select different sample counts on another machine.

Reproduce the mechanism experiments using fresh directories (the official records
are protected from overwrite):

```bash
uv run python experiments/formulation_mechanisms.py \
  --config configs/variants/mechanisms.yaml \
  --output results/variants/mechanisms_reproduction \
  --raw runs/variants/mechanisms_reproduction
uv run python experiments/formulation_mechanisms.py --plot-only \
  --output results/variants/mechanisms_reproduction
uv run python experiments/formulation_mechanisms.py --audit-only \
  --output results/variants/mechanisms_reproduction
```

Rebuild all five committed mechanism figures **from compact saved records alone**:

```bash
uv run python experiments/formulation_mechanisms.py --plot-only \
  --output results/variants/mechanisms
```

The mechanism audit requires the ignored raw files; plot regeneration does not. It
checks raw hashes, weight/update identities, all-pass work, recomputed tracker gains,
and covariance against an independent linear impulse-response construction. Tests
also compare recording enabled/disabled commands and RNG state, and prohibit sampling
or dynamics calls during saved-record figure regeneration.

Regenerate a figure from committed compact data, with no controller execution:

```bash
uv run python experiments/variant_benchmark.py --plot-only --output results/variants/V06_unified
```

Replay a completed run's ignored raw trajectories and selected populations:

```bash
uv run python experiments/audit_variants.py results/variants/V06_unified
uv run python experiments/summarize_variants.py results/variants/V06_unified
# After a fresh reproduction, substitute its output directory above.
```

Each study commits its config, manifest, per-seed and summary tables, primary figure,
interpretation and audit. Complete executed trajectories, command diagnostics and selected
rollout populations live under ignored `runs/variants/` directories. **A clone does not
include these raw files**; reproduce the simulations before running raw replay audits.
Every population is not stored at every decision; recorded seeds and configs permit
regeneration. No new result overwrites the historical Q01–Q05 evidence.

The maintained documentation has two roles: this README contains formulations and
experimental interpretation; `docs/` contains the detailed vehicle report and one
consolidated historical evidence index. Superseded reports and bulk earlier outputs
are recoverable through the [historical recovery index](results/archive/legacy_index.md).
The local ignored archive is `runs/archive/02a331f_cleanup/`; it is not part of a clone.

```text
docs/                   Seven-state report, consolidated Q01–Q06 index, three key supplements
results/showcase/       Five primary Q01–Q05 historical figures
results/archive/        One index linking superseded evidence at an immutable Git revision
src/controllers/        Explicit formulations, registry and shared feedback/accounting
src/dynamics/           Existing validated physical plant
src/costs/              Existing common external task objectives
src/analysis/           Shared runner, artifacts, figures and variant audit
configs/variants/       Frozen staged experiment settings
results/variants/       Compact V01–V06 evidence and one primary image per study
runs/variants/          Ignored executed trajectories and selected populations
tests/                  Mathematical references, invariants and baseline regression
```

## 9. Limitations and future work

- The original comparison has three seeds per condition and one tuning per method.
  The mechanism extension declares additional threshold sensitivities and fixed-policy
  covariance trials; neither establishes general rankings.
  Wider cohorts, parameter sensitivity, stronger tracking stress and additional covariance
  constraints are future experiments. Failed runs and inactive mechanisms are retained.
- Shared external scores do not imply identical distributional objectives. Tsallis changes
  the cost transform, DIAL changes smoothing, and feedback/covariance formulations add
  internal optimization. These differences are explicit, not treated as estimator errors
  against one shared target.
- Local LQR and finite-difference models have no proven global contraction or tube bound.
  Nominal candidate feasibility uses a small Monte Carlo estimate. Solver feasibility is
  a numerical property of the local CC subproblem, not a nonlinear safety certificate.
- Timing calibration is not a hard real-time guarantee. Solver/matrix work, candidate
  searches and extra passes can dominate. Missed deadlines are measured, but controller
  delay is not fed back into the simulated plant.
- Disk collision geometry, simplified roll, instantaneous steering, no reverse, terminal
  runout recovery and one S-turn handedness remain limitations of the shared benchmark.
- CEM, adaptive covariance/policy classes, contraction-metric trackers and hardware
  reproductions remain unimplemented. The formulations and paper map are embedded
  in this README; superseded source notes are available through the
  [historical recovery index](results/archive/legacy_index.md).
