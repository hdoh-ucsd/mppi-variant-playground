# Mathematical formulation and comparison

The primary Q03–Q05 campaign now uses seven-state single-track dynamics and steering/throttle controls. The Gaussian latent-control formulation is unchanged; the new transform is `clip(diag(.5,1) z, bounds)`, with densities evaluated before clipping. [Seven-state report](seven_state_benchmark_report.md) · [Current research index](research_index.md).

## Implemented target and projection

This section derives the **specific finite-dimensional problem implemented here**. It is an information-theoretic control-sequence formulation, rather than a claim to reproduce continuous-time stochastic vehicle dynamics.

Let physical control scales be $D=\operatorname{diag}(a_{max},\delta_{max})$. Latent controls $z_t$ are dimensionless, applied controls are $a_t=g(Dz_t)$, where $g$ clamps actuator limits. States propagate through the configurable bicycle model $x_{t+1}=f(x_t,a_t)$. Independent time steps have Gaussian base density and proposal density

$$p(Z)=\prod_{t=0}^{T-1}\mathcal N(z_t;0,C_0),\quad q_U(Z)=\prod_{t=0}^{T-1}\mathcal N(z_t;U_t,C).$$

The base $C_0=\operatorname{diag}(0.28,0.16)$ is fixed across comparisons. Proposal $C$ is fixed within each run; the four configurations only change $C$. There is no covariance adaptation or temporal correlation.

The unconstrained distributional problem is

$$\min_Q\{\mathbb E_Q[J(Z)] + \lambda D_{KL}(Q\Vert P)\},\qquad
q^*(Z)=\eta^{-1} e^{-J(Z)/\lambda}p(Z).$$

Both physical-control effort and variation are included in $J$ as explicit, common toy task preferences. They are **additional to** the implicit base-distribution regularization. A control-dependent trajectory functional is valid for this finite-dimensional Gibbs construction: alternatively augment the state with the previous applied control. This is an implementation choice beyond the state-only $S$ notation in the baseline prompt.

For a fixed SPD proposal covariance, minimizing $D_{KL}(Q^*\Vert Q_U)$ with respect to the mean gives $U_t^*=\mathbb E_{Q^*}[z_t]$. Using proposal samples $Z^{(k)}=U+\epsilon^{(k)}$, define

$$\ell_k=\log p(Z^{(k)})-\log q_U(Z^{(k)}),\qquad
\widetilde J_k=J_k-\lambda\ell_k.$$

The implementation retains the entire Gaussian log ratio:

$$\ell_k=-\tfrac12\sum_t\left[
(z_t^{(k)})^TC_0^{-1}z_t^{(k)}-
(\epsilon_t^{(k)})^TC^{-1}\epsilon_t^{(k)}+
\log|C_0|-\log|C|\right].$$

The determinant terms are constant across a batch and cancel in normalized weights, but are retained to make density-ratio tests exact. With $\rho=\min_k J_k$, compute

$$L_k=-(J_k-\rho)/\lambda+\ell_k,\quad
w_k=\frac{e^{L_k-\max_jL_j}}{\sum_j e^{L_j-\max_iL_i}},\quad
U_t^+=U_t+\sum_k w_k\epsilon_t^{(k)}.$$

This is equivalent to subtracting the minimum augmented cost. Subtracting the physical trajectory-cost baseline first also avoids unnecessary scale in the logits. The ratios of the weights are unchanged.

When $C=C_0$, the ratio reduces to

$$\ell_k=-\sum_t U_t^TC^{-1}\epsilon_t^{(k)}-\tfrac12\sum_tU_t^TC^{-1}U_t.$$

The final term is sample-independent. Thus the familiar cross-term augmented cost $J_k+\lambda\sum_t U_t^TC^{-1}\epsilon_t^{(k)}$ is sufficient **only for this equal-covariance case**. Cost-only weighting has a different target unless the base/proposal correction is constant. Our tests compare the general ratio against independent SciPy Gaussian log densities and check the equal-covariance reduction.

We saturate the first updated mean for execution, shift the **latent** mean, and append a zero tail. We do not clip noise before likelihood computation, replace latent noise by effective clipped perturbations, project the warm-start mean onto the box, or smooth the updated mean. Saturation changes the induced physical distribution and its perturbation energy, which is logged through the saturation fraction.

## Assumptions and numerical behavior

Gaussian covariances must be finite, symmetric and positive definite. The base and proposal then have common full support. The normalizing integral and first moments must exist; here the nonnegative cost gives a bounded Boltzmann factor relative to the Gaussian base. The observation and prediction model are available at planning time. Prediction rollouts are deterministic conditional on latent controls; environment noise is not leaked into planning. The sampling covariance is constant in time and finite-dimensional control sequences are independent across samples.

Finite self-normalized importance sampling introduces finite-sample bias and may have severe variance. No claim of an unbiased finite-$K$ update, global convergence, or monotonic closed-loop improvement is made. Low ESS is detected (`ESS < 2`), logged and retained; a valid single-sample weight distribution remains a legitimate update. Nonfinite costs or ratios raise an exception. A silent uniform-weight fallback would conceal an invalid experiment.

ESS is $1/\sum_k w_k^2$. Weight entropy is $-\sum_{k:w_k>0}w_k\log w_k$. Covariance is validated by Cholesky factorization; samples use standard normals times its transpose. The latent control update is tested with hand-computed values.

## Optimization and evaluation costs

Let $d=\|p-p_g\|$, $h=\operatorname{wrap}(\theta-\theta_g)$, $v_r=\min(v_{cruise},d)$ and $s(x)$ be minimum signed clearance from the disk footprint to circle obstacles. The implemented running cost is

$$\Delta t\,[w_p d^2+w_h h^2+w_v(v-v_r)^2+
 w_o[\max(m-s(x),0)]^2+w_c\mathbf1_{s(x)\le0}+
 w_u\|D^{-1}a\|^2+w_{\Delta u}\|D^{-1}(a-a_{prev})\|^2].$$

A further $\Delta t\,w_c$ is added for each transition whose swept line segment intersects an inflated obstacle. State collision and transition collision penalties are separate terms; sampled paths can incur both when they stay inside an obstacle. The swept check prevents missed collisions between integration endpoints. The terminal cost weights squared position, wrapped heading and speed errors. State costs are evaluated at $x_t$, terminal cost at $x_T$.

`configs/costs.yaml` contains separate optimization and evaluation weight sets. Evaluation is applied to the actual executed transitions with no importance correction. All controllers receive the same optimization cost and all runs are scored with the same evaluation cost. These values are task objectives with mixed physical units; they are not a universal performance scale.

## Covariance fairness

The acceleration and steering scales are 2 m/s² and 0.5 rad. Physical covariance before saturation is $DCD^T$. Calling a covariance isotropic in raw acceleration/steering coordinates would compare incompatible units.

| Name | $C_{11}$ | $C_{22}$ | $C_{12}$ | trace | determinant |
|---|---:|---:|---:|---:|---:|
| Standard | .28 | .16 | 0 | .44 | .0448 |
| Isotropic | .22 | .22 | 0 | .44 | .0484 |
| Diagonal | .34 | .10 | 0 | .44 | .0340 |
| Full | .28 | .16 | .08 | .44 | .0384 |

Matched trace is matched **expected squared latent normalized perturbation energy per time step**, since $\mathbb E\|\epsilon\|^2=\operatorname{tr}C$. It does not match Gaussian volume (determinant), physical perturbation energy, entropy, or realized energy after clipping. The original vehicle benchmark implements matched trace only. The separate [covariance mechanism investigation](covariance_mechanisms.md) adds matched-determinant experiments and explicit saturation measurements; matched post-saturation energy remains future work. Correlation between acceleration and steering is not a global alignment with a spatial obstacle passage.

With a fixed target and infinite samples the mean projection does not depend on proposal covariance. Differences in this experiment concern finite-budget estimation, exploration, saturation and subsequent closed-loop warm starts. If each controller instead used $C_0=C$, it would change the target regularizer as well as the exploration distribution.

## Algorithm comparison and implementation status

The following is a **schematic map**, not a full derivation of the planned controllers. Verified original equation numbers are in the paper map. Implementing any planned algorithm requires its own derivation, tests and closed-loop benchmark first.

| Algorithm | Optimization / sampling / weights / update | Covariance and robustness | Cost and computational work | Original setting → our setting |
|---|---|---|---|---|
| IT-MPPI (implemented) | KL-regularized Gibbs target; Gaussian proposals; exponential weights with $p/q$; weighted mean | Fixed SPD covariance; replanning only | Shared $J$; $O(KT)$ dynamics, $O(KTm^2)$ density/sampling work | Autonomous dirt-track driving → deterministic kinematic goal reaching |
| Isotropic, diagonal, full MPPI (implemented) | Same problem, update and cost as above | Restricted proposal structure; fixed common base | Same counts, distinct Gaussian geometry | Controlled ablations in our toy setting |
| CEM MPC (planned) | Quantile/elite selection, refit mean and covariance | Elite-fitted covariance; no intrinsic tracking architecture | Repeated iterations and elite fitting must be counted | Sampling trajectory optimization → shared bicycle tasks |
| Tsallis-VI (planned) | Tsallis-regularized distribution objective; deformed cost likelihood; weighted policy fit | Policy covariance depends on parameterization; no automatic tracking guarantee | Generalized cost transform and repeated policy fitting | Multiple robot systems and policy classes → bicycle tests |
| DIAL-MPC (planned) | Iterative sampling and exponential weighted updates over decreasing noise stages | Trajectory-wise and horizon-wise annealing; optimization coverage mechanism | Sum every inner-pass rollout; different cost-only smoothing target must be explicit | Torque-level legged locomotion → limited low-dimensional annealing study |
| Tube-MPPI (planned) | Nominal sampling planner plus ancillary trajectory tracker and nominal reset logic | Tracking architecture separate from planner | Nominal/actual rollout and tracker work; tracking cost distinct | Simulated tasks and AutoRally → disturbed bicycle with explicit tracker |
| Robust-MPPI (planned) | Tracking-aware augmented nominal/actual sampling, mixed costs and nominal selection | Assumptions on tracker contraction underlie analysis | Both dynamics and tracker evaluations, nominal candidate search | Off-road driving → toy disturbance/mismatch architecture |
| CC-MPPI (planned) | Linearize dynamics, solve covariance-steering feedback, sample under feedback, weighted update | Sampling noise, predicted state covariance and steering target are distinct | Covariance-steering solve plus linearization plus nonlinear rollouts | Ground-vehicle navigation → stochastic bicycle with explicit noise model |

For Tube/Robust architectures a schematic applied command is $u_{actual}=u_{nominal}+k(x_{actual},x_{nominal})$. This equation alone does not specify either algorithm. RMPPI incorporates the tracker inside prediction and its importance law; a disturbance in the environment alone does not make standard MPPI robust MPPI. CC-MPPI requires a designed feedback gain and a state-distribution constraint, not just a full control covariance.
