# Covariance mechanisms: derivation and testable claims

This milestone studies Gaussian proposal geometry within the existing IT-MPPI formulation. The mathematical target, sampling proposal and physically applied control distribution are separate objects. The baseline derivation and reference notes remain in [formulations.md](formulations.md) and [paper_map.md](paper_map.md).

The control target and mean projection follow [Williams et al., sections III-A–III-C](https://arxiv.org/html/1707.02342v1). For the statistical distinction between self-normalized estimators, integrand-dependent asymptotic variance and weight diagnostics, see [Owen, *Monte Carlo theory, methods and examples*, sections 9.2–9.3](https://artowen.su.domains/mc/Ch-var-is.pdf). The quadratic-target and Gaussian-overlap formulas below are derived for this study and checked numerically; paper guarantees are not transferred to the vehicle implementation.

## Exact Gaussian target

Let $p(z)=\mathcal N(0,P)$, $P\succ0$, and $J(z)=\frac12(z-b)^TH(z-b)$, $H\succeq0$. For $\lambda>0$, the target is $\pi(z)=\eta^{-1}\exp(-J(z)/\lambda)p(z)$. Expanding its exponent gives

$$\log\pi(z)=\mathrm{constant}-\tfrac12z^TAz+c^Tz,
\qquad A=P^{-1}+H/\lambda\succ0,\quad c=Hb/\lambda.$$

Completing the square yields

$$C_*=A^{-1},\qquad \mu_*=A^{-1}c,\qquad
\eta=\frac{|C_*|^{1/2}}{|P|^{1/2}}
\exp\left[-\frac{b^THb}{2\lambda}+\frac{\mu_*^TA\mu_*}{2}\right].$$

**Proved identity:** these moments depend on $P,H,b,\lambda$, and are independent of the sampling covariance. Independent two-dimensional adaptive quadrature verifies the evidence, first moments and second moments in the tests. The zero-cost and scalar-factorized limits provide additional checks.

## Finite-sample mean estimation

For $q=\mathcal N(m,\Sigma)$, draw $z_k\sim q$ and set

$$\ell_k=-J(z_k)/\lambda+\log p(z_k)-\log q(z_k),\quad
w_k=\frac{e^{\ell_k-\max_j\ell_j}}{\sum_j e^{\ell_j-\max_i\ell_i}},\quad
\hat\mu_K=\sum_k w_kz_k.$$

The code reuses the controller's stable normalization and weighted-mean update. The generic log density is independently checked against SciPy in dimensions 2, 8 and 56. Every covariance case records a target fingerprint, the frozen proposal mean, covariance eigenvalues/angle, scale rule and actual cost-evaluation count.

**Proved distinction:** self-normalization generally introduces finite-$K$ bias. Under the usual finite-moment conditions the estimator is consistent; this does not imply monotonic improvement on every seed. For independent trial estimates $\hat\mu_r$, the report measures empirical bias, sample variance trace, MSE, and error quantiles. Their exact empirical decomposition is

$$\widehat{\mathrm{MSE}}=\|\overline{\hat\mu}-\mu_*\|^2+
\frac{R-1}{R}\operatorname{tr}\widehat{\operatorname{Cov}}(\hat\mu).$$

Here covariance uses denominator $R-1$, while empirical MSE uses $R$. Bias is estimated from a finite number of trials and has uncertainty too.

## Alignment, overlap and estimator accuracy

The target's long axis is an eigenvector of $(P^{-1}+H/\lambda)^{-1}$. It need not coincide with an eigenvector of $H$ when the base is anisotropic. The study records major-axis angle (undefined for an isotropic covariance), covariance-only mismatch and full $D_{KL}(\pi\Vert q)$, which also accounts for mean displacement.

Let $r(z)=\pi(z)/q(z)$. When $E_q[r^2]<\infty$, the large-$K$ ESS fraction tends to $1/E_q[r^2]$. For Gaussian target and proposal, this second moment is finite exactly when

$$2C_*^{-1}-\Sigma^{-1}\succ0.$$

The implementation evaluates this Gaussian integral by completing the square and records a null log-second-moment when it diverges. A finite batch still has finite ESS in that case. Its apparent ESS can be misleading if it misses rare, high-weight regions.

Under conditions for the self-normalized importance-sampling central limit theorem, the covariance of $\sqrt K(\hat\mu_K-\mu_*)$ is

$$V=E_q\left[r(z)^2(z-\mu_*)(z-\mu_*)^T\right].$$

**Proved distinction:** ESS concerns weights, whereas mean error also depends on the weighted integrand. Even choosing $q=\pi$ does not minimize every importance-sampling estimation variance. For Euclidean mean MSE, an unconstrained proposal minimizing $\operatorname{tr}V$ has density proportional to $\pi(z)\|z-\mu_*\|$, provided normalization and the required moments exist. It is generally non-Gaussian. Consequently, neither Hessian alignment, target-covariance matching nor a high ESS is an automatic certificate of minimum mean-estimation error.

**Hypothesis to test:** some aligned proposals reduce finite-sample MSE relative to isotropic proposals under a declared scale constraint. Misaligned or overly narrow proposals can reverse that comparison. The sweeps include both favorable and unfavorable angles and scales without selecting a winner in advance.

## Two distinct scale constraints

For eigenvalue ratio $a\ge1$, rotation $R_\theta$, and scale parameter $t>0$:

$$\Sigma_{\mathrm{trace}}=R_\theta\operatorname{diag}\left(\frac{ta}{a+1},\frac{t}{a+1}\right)R_\theta^T,$$

$$\Sigma_{\mathrm{det}}=R_\theta\operatorname{diag}\left(\frac{t\sqrt a}{2},\frac{t}{2\sqrt a}\right)R_\theta^T.$$

The first has trace $t$; the second has determinant $(t/2)^2$. Their isotropic members agree. For $a>1$, their total variance differs. The code labels and reports the experiments separately. Rotation affects off-diagonal correlation; eigenvalue ratio affects anisotropy. Changing proposal mean is an additional, explicitly recorded intervention.

## Temperature and dimension

Changing $\lambda$ changes both $C_*$ and $\mu_*$. Temperature-specific errors always use the corresponding exact target. A separate factorial diagnostic varies dimension and temperature for (a) proposal equal to base, so density ratios vanish; (b) zero cost, so only density ratios remain; and (c) both effects together. This separates mathematical sources of concentration. It does not identify the dominant cause of low ESS in a particular vehicle episode.

## Limits of transfer

The analytic study freezes the proposal mean. Iterative optimization adapts it using earlier batches; closed-loop MPC additionally changes state, horizon warm start and actuator saturation. Gaussian likelihoods remain defined on latent controls. The applied clipped distribution has boundary atoms and is not Gaussian.

Experiments at these three levels are separately reported. A static estimation improvement does not establish improved navigation or a robustness guarantee. See [the execution protocol](mathematical_investigations.md) and [recorded findings](experimental_findings.md).
