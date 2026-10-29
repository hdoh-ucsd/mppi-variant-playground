# Frozen-state MPPI decision analysis

The question is whether changing a Gaussian proposal, with the state and optimization target held fixed, changes the finite-sample MPPI update. The experiment connects **same state and target → proposal geometry → sampled controls and trajectories → corrected weights → updated controls → open-loop predicted consequences**. It does not compare complete navigation episodes or introduce a new published controller.

The implementation calls `MPPI.plan`, the same optimization path used by `MPPI.command`. Planning accepts optional shared standard normals and returns the command candidate and unshifted latent mean. Only `command` shifts the warm start, increments the planning step and records an executed previous command. The sampler, log density ratio, stable weight normalization, weighted update, clipped bicycle dynamics and shared cost remain those derived in [formulations.md](formulations.md). Earlier configurations and results are preserved.

## Frozen inputs and selection

One standard-controller reference prefix, optimizer seed **31415**, supplies each state's observed state, latent warm start and previously applied physical command. Environment seed entropy is `[31415,1001]`. No proposal controller chooses its own state. `frozen_states/*.json` stores full resolved scenario, model, cost, temperature, horizon, base, event metadata and inputs; adjacent NPZ files preserve all states and commands before capture. Reference capture is separate from the decision experiment, which never advances the environment.

- **Open space:** observed planning step 8, while moving toward the goal.
- **Pre-obstacle:** first observed state with x in `[0.7,1.0]` m for which a deterministic pulse-control grid reaches both sides of the obstacle's x station with strictly positive swept clearance. The original center is `(4,0)` m, obstacle radius 0.9 m and vehicle radius 0.22 m. Grid candidates use constant normalized acceleration in `{0,.25,.5}`, steering amplitude in `{.4,.7,1}`, signed pulse duration fractions in `{.2,.25,.4,.55}` and continuation multipliers in `{0,-.25,-.5,-1}`. The lowest common task cost among feasible candidates on each side is stored as a witness. These checks establish finite-horizon route availability, not that random samples will find both routes. The geometric selection was checked before evaluating the proposals; the initial later-state selection failed the two-route feasibility check and was discarded before the main study.
- **Post-disturbance:** observed planning step 21, immediately after the step-20 impulse `[0,.9,.18,-.4]`. The planner predicts without additional impulses. The lateral displacement and heading/speed change are common inputs for every proposal.

The horizon is **28 × 0.15 = 4.2 s**, temperature **5**, and the Gaussian base covariance in normalized latent coordinates is **diag(.28,.16)** at every horizon step. Previous physical control is frozen too, because the cost includes control variation. A SHA-256 fingerprint includes all frozen inputs. Changing K creates a separate budget comparison, with identical K across its six proposals.

## Proposals and paired sampling

The four existing fixed settings are standard, isotropic, diagonal anisotropic and full covariance. Their normalized trace is .44. Two controlled orientations use eigenvalues `.34,.10`, with identical trace and determinant. The aligned major axis follows local deterministic cost descent under a constant two-channel offset: `g = ∂J(U + 1_T d)/∂d` at zero. Central differences at h=.001 and .0005 report sensitivity stability. The misaligned major axis is perpendicular. This is a documented local heuristic; it is neither a target-covariance estimate nor an assertion of optimal importance sampling. It uses vehicle dynamics and the task cost, rather than inferring input directions from obstacle XY orientation. Every matrix, eigenpair, determinant and empirical perturbation statistic is recorded.

For each state index and evaluation seed 10000–10049, PCG64 uses `SeedSequence([101,state_index,seed])` to generate one maximum-budget standard-normal array. Every proposal uses `epsilon = Z @ chol(Sigma).T`; budgets use prefixes of this same array. Thus proposals and budgets are paired. The intervention transforms samples rather than making controls identical.

Weights use `exp(-J/lambda + log p - log q)` in log space. The base remains fixed when proposal covariance changes. Raw minimum cost and maximum corrected weight are recorded independently. Positive definite covariances, finite costs/ratios and normalized weights are checked; there is no uniform fallback.

## Reference and uncertainty

Each frozen state has **eight independent 65,536-rollout reference batches**, centered on the same frozen warm start with the standard covariance. These use the disjoint namespace `SeedSequence([202,state_index,repetition])`. The total 524,288-rollout reference is separate from the evaluation budget.

Batch normalized means are combined using their **unnormalized log weight masses**, rather than equally averaging self-normalized estimates. Saved arrays include batch means/masses, the pooled estimate, delete-one-batch jackknife standard errors and pointwise 95% t intervals. These intervals describe variability among independent blocks; they cannot bound unknown importance-sampling bias or certify missing-mode coverage. Reference ESS, maximum weight, every batch's diagnostics and source streams are saved.

A reference is labeled `unresolved` if any batch ESS is below 100, the pooled largest weight exceeds .05, or RMS pointwise CI halfwidth exceeds .05 latent units. Passing these declared screens is only `diagnostically_usable`, never proof of ground truth. When unresolved, the plots call the measured quantity a **distance to the reference**, not validated estimation error. Independent paired distances to standard and deterministic mean-plan objectives remain interpretable without a reliable target mean.

## Recorded evidence and figures

The primary decision budget is K=384, repeated over 50 paired seeds. Every proposal/seed at this budget stores its **entire** rollout population: trajectories, latent and applied controls, costs, corrected weights, density ratios, updated latent mean, predicted mean trajectory and clipped first command. Frozen snapshots have no fictitious executed-next-state. Trajectories/applied controls use float32 for display; latent controls, updates, weights, costs and nominal predictions remain float64.

The budget study uses K=64,128,256,384,512,1024,2048,4096 over the same 50 seeds. All budget trials retain scalar diagnostics and full updated latent means. Supplemental full populations at every K are retained for the predeclared representative seed 10000; other non-primary budgets do not archive every rollout trajectory. Regenerating the specified figures needs no optimization or new samples. Recreating unarchived populations requires deterministic replay through the experiment.

`per_seed.csv` records ESS, weight entropy, saturation, raw minimum and weighted cost, updated-plan objective, update magnitude, physical/normalized first command, paired command differences, normalized control-mean and command distances, minimum predicted swept clearance, dynamics/objective counts and latency. `updated_means.npz` is aligned by CSV row index. Mean objectives score the existing predicted paths using precisely the shared sampled-trajectory cost. Sample dynamics counts are `(K+1)T`, including one updated nominal prediction; reference, state selection, route witnesses, local sensitivity, cost scoring and disk/plot work are separate. Planning latency excludes controller construction, extra objective scoring and serialization, and includes selected snapshot collection. It is descriptive CPU timing, with snapshot overhead at the primary budget and representative seed.

Each frozen state produces these six stems under `results/investigations/frozen_decision_v1/figures/`, with PNG, SVG and numeric `_data.npz` files:

| Stem | Question and interpretation |
| --- | --- |
| `{state}_A_decision_microscope` | How do the unweighted population, weighted population and updated nominal prediction differ? All six proposals use the same spatial axes and common weight scale. Weighted paths also use opacity `.025+.925*w/common_max` and width `.3+2*w/common_max` so near-zero-weight paths remain faint; these are display rules, not additional normalization. Lowest raw cost and highest weight are distinct paths. |
| `{state}_B_first_command_difference` | What is the paired change from standard? All seeds are shown with mean ± pointwise 95% t interval. Comparisons use normalized clipped commands; secondary axes show m/s² and rad. |
| `{state}_C_weighted_endpoint_density` | Where is empirical weighted rollout endpoint mass? Identical bins, domain and linear color scale, no smoothing. This is not a physical uncertainty probability density. |
| `{state}_D_budget_estimation` | Do distances to the independent reference or paired standard diminish with K? Mean ± pointwise 95% t intervals across paired seeds; unresolved references remain prominently labeled. ESS and deterministic objective are separate. |
| `{state}_E_decision_dashboard` | Which mechanisms accompany the changed decision? ESS, update size, first-command difference, nominal cost, latency and dynamics budget are shown for all seeds. |
| `{state}_F_counterfactual_consequence` | What would each updated plan predict from the same state? Representative seed, common full view and declared common zoom, plus clearance or lateral position. This is an open-loop prediction, not a closed-loop guarantee. |

Pointwise t intervals are descriptive, assume independent optimizer seeds, and are neither simultaneous intervals nor adjusted for many comparisons. Seeds are paired across proposals; rows within a seed or across nested K are not independent repetitions. `repetition_sensitivity.csv` checks the first 10,25,50 preordered seeds rather than selecting subsets by outcome. Saturating the target mean differs from averaging saturated target controls. Closeness to a Gibbs mean also differs from minimizing a nonlinear deterministic mean-plan cost.

## Reproduce

```bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python experiments/frozen_state_decision.py \
  --output results/investigations/frozen_decision_reproduction

# Existing evidence only: no optimization or random draws.
uv run python experiments/frozen_state_decision.py \
  --output results/investigations/frozen_decision_v1 --plots-only
uv run python experiments/verify_frozen_decision.py \
  --input results/investigations/frozen_decision_v1
```

The runner refuses to overwrite a study or snapshot. The manifest records source revision, dirty status, source/config hashes, packages, platform and thread settings. Saved-data audit checks frozen fingerprints, reference stream namespaces, covariance transforms, Gaussian ratios, normalized weights, control updates, mean-plan objectives and density normalization. `interpretation.md` in the executed output gives actual findings and reference reliability. No earlier result directory is rewritten.
