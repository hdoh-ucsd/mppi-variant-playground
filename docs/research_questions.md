# Mathematical investigation of MPPI variants

The purpose is to explain the characteristics of each variant through its equations and suitable experiments. Every study should connect **mathematical change → predicted effect → isolated numerical observation → control behavior**, including assumptions and cases where that prediction ceases to apply.

The baseline covariance probes below are now implemented in the [covariance mechanism study](covariance_mechanisms.md), with [execution details](mathematical_investigations.md) and [observed findings](experimental_findings.md). The broader algorithm-family studies remain planned.

## Questions to answer for each family

1. What objective or target distribution is defined, and under which assumptions?
2. What distribution generates samples, and how are sampling and target distributions related?
3. How are trajectory costs transformed into weights? Which likelihood corrections are required?
4. Which parameters are updated: mean, covariance, feedback gains, or a nominal trajectory?
5. What does the population update predict? What changes under finite sampling, clipping or model error?
6. Which small experiment can reveal that mechanism without mixing several changes?

Compare mechanisms using these questions. A family can require its own dynamics, noise model or diagnostic to make its defining feature observable.

## Current implementation: fixed Gaussian proposal geometry

The implemented baseline uses a fixed Gaussian base distribution and a Gaussian proposal over latent normalized control sequences:

$$p(Z)=\prod_t\mathcal N(z_t;0,C_0),\qquad q_U(Z)=\prod_t\mathcal N(z_t;U_t,C).$$

Its target and sampled mean update are

$$q^*(Z)\propto e^{-J(Z)/\lambda}p(Z),\qquad
w_k\propto e^{-J(Z^{(k)})/\lambda}\frac{p(Z^{(k)})}{q_U(Z^{(k)})},\qquad
U^+=\sum_k w_kZ^{(k)}.$$

With fixed $J,p,\lambda$ and frozen $U$, changing proposal $C$ preserves $q^*$. Under the integrability and support assumptions in the [derivation](formulations.md), the population mean projection is unchanged. Finite-sample error, weight concentration and subsequent closed-loop evolution can change. Changing the base $C_0$ instead changes the target itself. This distinction is central to interpreting covariance experiments. See [Williams et al., sections III-A–III-C](https://arxiv.org/html/1707.02342).

| Setting | Mathematical characteristic | Observable for an isolated study |
|---|---|---|
| Standard configuration | Here, $C=C_0$ and both are diagonal; the log-density correction reduces to a mean/noise cross term plus a batch constant. | Numerical agreement of the full ratio and reduced weights. “Standard” identifies this repository's baseline setting. |
| Isotropic | $C=\sigma^2 I$: equal variance in all normalized input directions. | Circular latent contours and equal marginal variance; physical scaling and clipping can break that symmetry. |
| Diagonal | $C=\operatorname{diag}(\sigma_a^2,\sigma_\delta^2)$: independent Gaussian input perturbations with adjustable marginal variances. | Axis-aligned latent contours and changes in per-input sampling spread as the variance ratio changes. |
| Full | $C\succ0$ permits off-diagonal terms, rotating latent contours and coupling input perturbations. | Eigenvectors, eigenvalues and $\mathbb E[\epsilon_\delta\mid\epsilon_a]=(C_{21}/C_{11})\epsilon_a$. Compare positive, zero and negative correlation. |

“Full” here describes the two-input covariance at each step. Sampling remains independent across horizon steps. It does not introduce temporal correlation, adaptive covariance, or covariance steering of the state.

All four chosen matrices have trace 0.44, fixing $\mathbb E\|\epsilon_t\|^2$. Their determinants, anisotropy and post-saturation behavior differ. A single matrix of each form cannot establish a general property of that form.

## Proposed numerical probes for the implemented baseline

These designs are now implemented for the fixed-proposal baseline in the [covariance mechanism study](covariance_mechanisms.md). Its [findings](experimental_findings.md) report executed sweeps and figures. The broader algorithm-family experiments in the following section remain proposed.

| Probe | Controlled setup | Quantity to explain |
|---|---|---|
| Sampling geometry | Freeze the mean; sample without dynamics or costs. Sweep the diagonal variance ratio and correlation while declaring the chosen scale constraint. | Empirical covariance versus the specified matrix, eigenstructure and conditional input coupling. |
| Importance correction | Use a quadratic cost with an analytic Gaussian target; freeze the proposal mean and compare several proposals. | Error of the estimated target mean, repeat-to-repeat variability and ESS as sample count changes. Omitting the density ratio generally changes the target; include that update as a control and check the special case where proposal equals base. |
| Weight concentration | Reuse a fixed sample batch and costs; vary temperature. Separately vary horizon dimension on a separable analytic problem. | Maximum weight, entropy, ESS and target-moment error. Temperature changes the target as well as the weights; dimension is a separate intervention. |
| Saturation | Apply physical scaling and clipping to known latent samples; vary mean and covariance. | Marginal boundary mass, applied covariance and latent/applied update differences. Compute likelihoods on latent samples. |
| Control illustration | Choose a state and goal that exercise the studied input directions; inspect a frozen planning update before a closed-loop run. | How sampling geometry, weights and the updated mean produce the selected command and predicted trajectory. |

For the quadratic probe, let $Z\in\mathbb R^d$, $p=\mathcal N(0,P)$, and $J(Z)=\tfrac12(Z-b)^TH(Z-b)$ with $H\succeq0$. Completing the square gives

$$\Sigma_*=(P^{-1}+H/\lambda)^{-1},\qquad
\mu_*=\Sigma_*Hb/\lambda.$$

These moments provide an independent reference. Rotate $H$ to study alignment with the proposal, and repeat batches to measure estimation error. High ESS alone is insufficient if a finite batch misses an important region of the target.

## Broader families: defining questions and required studies

The entries below are proposed studies grounded in the cited formulations. These controllers are not implemented in this repository. CEM is a related sampling optimizer used to understand the weighting/update distinction.

| Family | Mechanism to derive | Experiment designed to reveal it | Source |
|---|---|---|---|
| IT-MPPI | Gibbs target, importance sampling and Gaussian mean projection; temperature controls cost concentration. | Known costs and analytic target moments: plot weights and update error while varying temperature and proposal overlap. | [Williams et al., III](https://arxiv.org/html/1707.02342) |
| CEM MPC | Elite selection and distribution refitting. | Fix samples and costs, vary elite fraction, and examine changes in selected support, fitted mean and covariance. | [Williams et al., V-A](https://arxiv.org/html/1707.02342) |
| Tsallis-VI MPC | The chosen Tsallis divergence defines a deformed cost transform and policy updates. | Derive the paper's parameter/scale convention, plot the resulting weight curve and support, and check the exponential limit before a control example. | [Wang et al., II–IV](https://arxiv.org/html/2104.00241v1) |
| DIAL-MPC | Iterative sampling with trajectory-wise and action-wise annealing. | On a visible multimodal cost landscape, track sample coverage, mean motion and covariance through the noise stages; include fixed-noise controls and count every pass. | [Xue et al., III–IV-A](https://arxiv.org/html/2409.15610v1) |
| Tube-MPPI | Nominal sampling planner coupled to a trajectory tracker. | Apply an impulse and plot nominal versus actual state, tracking error and corrective control; isolate tracker and nominal-reset effects. | [Williams et al., RSS 2018](https://www.roboticsproceedings.org/rss14/p42.pdf) |
| Robust-MPPI | Tracking-aware augmented rollouts, nominal-state selection and the assumptions supporting performance bounds. | Inspect nominal/actual predictions and realized tracking errors under controlled perturbations; distinguish empirical observations from assumptions needed for a bound. | [Gandhi et al., III–IV](https://arxiv.org/html/2102.09027v1) |
| CC-MPPI | Feedback sampling and a terminal state-covariance constraint under a linearized prediction model. | Start with a linear stochastic system, compare predicted and empirical state covariance against the prescribed bound, then examine nonlinear-model deviations. | [Yin et al., IV–V](https://arxiv.org/html/2109.12147v3) |

Sampling control covariance, uncertainty in future states, and a designed state-covariance constraint are distinct quantities. Likewise, ordinary replanning, an ancillary tracker, and tracking-aware rollout optimization require different implementations and evidence.

## How to use the existing vehicle results

The existing results provide reproducible examples for the fixed-proposal baseline: sampled geometry, concentrated weights, saturation, goal approach and post-disturbance behavior. Use those records to motivate an isolated probe and revisit the vehicle example after establishing the mechanism.

A success count cannot establish a weighting law, an annealing mechanism or a tracking guarantee. The current records contain no experiments for the six planned families. The [implementation fidelity report](implementation_fidelity.md) and [paper map](paper_map.md) specify the implemented scope and source references.
