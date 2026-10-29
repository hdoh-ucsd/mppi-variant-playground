"""Mechanism experiments using existing controllers, seven-state runner and artifacts.

Only orchestration and numerical reductions live here. Plotting reads compact saved
records. Full populations and independent executions stay in ignored run directories.
"""

import hashlib
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from analysis.artifacts import begin
from analysis.dynamic_campaign import STATE_FIELDS, assert_validated, run_episode
from analysis.frozen_decision import plan_objective
from analysis.variant_campaign import case_config
from controllers.feedback import state_error
from controllers.tsallis import tsallis_weights
from evaluation.runner import make_controller, write_json
from sampling.gaussian import importance_weights
from utils.config import read_yaml


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def archive(path, snapshot):
    """Use the same NumPy array interchange as the existing episode recorder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, **{k: v for k, v in snapshot.items() if isinstance(v, np.ndarray)}
        )


def frozen_inputs(spec, options, raw):
    result = {}
    for scene in spec["scenes"]:
        c, _ = case_config(options, scene, "it_mppi", 384)
        result[scene] = run_episode(
            c,
            "it_mppi",
            spec["source_seed"],
            raw / "sources" / scene,
            freeze_rule=lambda candidate: candidate["step"] == spec["frozen_step"],
        )
    return result


def controller_at(frozen, options, algorithm, samples, seed, **changes):
    c = deepcopy(frozen["config"])
    c["controller"].update(options["algorithms"][algorithm])
    c["controller"].update(algorithm=algorithm, samples=samples, snapshot_steps=[0], **changes)
    ctrl = make_controller(c)
    ctrl.reset(np.array(frozen["state"]), seed)
    ctrl.mean = np.array(frozen["warm_start"])
    ctrl.previous_control = np.array(frozen["previous_control"])
    return ctrl


def finish_study(root, manifest, rows, picture, raw, notes):
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    write_json(root / "plot_data.json", native(picture))
    hashes = {
        str(p.relative_to(raw)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(raw.rglob("*"))
        if p.is_file()
    }
    write_json(raw / "sha256.json", hashes)
    manifest.update(
        complete=True,
        finished_at_utc=datetime.now(UTC).isoformat(),
        raw_storage=str(raw),
        records=len(rows),
        raw_file_count=len(hashes),
        raw_hash_index=str(raw / "sha256.json"),
        raw_index_sha256=hashlib.sha256((raw / "sha256.json").read_bytes()).hexdigest(),
        frozen_inputs_sha256=hashlib.sha256((root / "frozen_inputs.json").read_bytes()).hexdigest(),
    )
    write_json(root / "manifest.json", manifest)
    write_json(root / "findings.json", notes)


def weighting(root, raw, spec, options, frozen, manifest):
    cfg = spec["V01"]
    rows, pictures = [], []
    for scene in cfg["scenes"]:
        f = frozen[scene]
        for seed in spec["seeds"]:
            normals = np.random.default_rng(seed).standard_normal((cfg["samples"], 20, 2))
            pair = {}
            for algorithm in ["it_mppi", "tsallis_vi"]:
                c = controller_at(f, options, algorithm, cfg["samples"], seed)
                command, _ = c.plan(np.array(f["state"]), standard_normals=normals)
                snap = c.get_rollout_snapshot()
                z = snap["latent_controls"]
                snap["log_base_density"] = multivariate_normal.logpdf(
                    z, mean=np.zeros(2), cov=c.base_covariance
                ).sum(axis=1)
                snap["log_proposal_density"] = multivariate_normal.logpdf(
                    z - snap["proposal_mean"], mean=np.zeros(2), cov=c.covariance
                ).sum(axis=1)
                np.testing.assert_allclose(
                    snap["log_base_density"] - snap["log_proposal_density"],
                    snap["log_base_over_proposal"],
                    atol=1e-12,
                )
                snap["command"] = command
                archive(raw / scene / f"{seed}_{algorithm}.npz", snap)
                pair[algorithm] = snap
                row = dict(
                    scene=scene,
                    seed=seed,
                    algorithm=algorithm,
                    ess=float(1 / np.sum(snap["weights"] ** 2)),
                    support=int(np.count_nonzero(snap["weights"])),
                    steering=float(command[0]),
                    throttle=float(command[1]),
                    planned_cost=plan_objective(
                        c, snap["nominal_states"], snap["updated_mean"], f["previous_control"]
                    ),
                )
                if algorithm == "tsallis_vi":
                    row.update(c.weight_details)
                rows.append(row)
            for key in ["costs", "latent_controls", "sampled_states", "log_base_over_proposal"]:
                np.testing.assert_array_equal(pair["it_mppi"][key], pair["tsallis_vi"][key])
            if seed == spec["seeds"][0]:
                pictures.append(
                    dict(
                        scene=scene,
                        costs=pair["it_mppi"]["costs"],
                        kl_weights=pair["it_mppi"]["weights"],
                        tsallis_weights=pair["tsallis_vi"]["weights"],
                        commands=np.array([pair[a]["command"] for a in pair])
                        / c.model.control_scale,
                    )
                )
    # Independent quadrature reference, not the controller weight helper.
    scale, r, center = cfg["reference_scale"], cfg["reference_r"], cfg["reference_cost_center"]

    def cost(z):
        return 0.5 * (z - center) ** 2

    kernels = {
        "KL": lambda z: norm.pdf(z) * np.exp(-cost(z) / scale),
        "Tsallis": lambda z: norm.pdf(z) * max(1 - (r - 1) * cost(z) / scale, 0) ** (1 / (r - 1)),
    }
    bounds = {
        "KL": (-np.inf, np.inf),
        "Tsallis": (center - np.sqrt(2 * scale / (r - 1)), center + np.sqrt(2 * scale / (r - 1))),
    }
    grid = np.linspace(-4, 4, 801)
    refs, trials = {}, []
    for name, kernel in kernels.items():
        lo, hi = bounds[name]
        normalizer = quad(kernel, lo, hi, epsabs=1e-11)[0]
        mean = quad(lambda z, kernel=kernel: z * kernel(z), lo, hi, epsabs=1e-11)[0] / normalizer
        refs[name] = dict(mean=mean, density=[kernel(z) / normalizer for z in grid])
        for seed in cfg["reference_seeds"]:
            values = np.random.default_rng(seed).normal(
                0.3, 1.2, max(cfg["reference_sample_counts"])
            )
            for n in cfg["reference_sample_counts"]:
                z = values[:n]
                ratio = norm.logpdf(z) - norm.logpdf(z, 0.3, 1.2)
                w = (
                    importance_weights(cost(z), scale, ratio)
                    if name == "KL"
                    else tsallis_weights(cost(z), scale, r, ratio)
                )
                estimate = float(w @ z)
                trials.append(
                    dict(
                        target=name,
                        seed=seed,
                        samples=n,
                        estimate=estimate,
                        reference=mean,
                        absolute_error=abs(estimate - mean),
                    )
                )
    pd.DataFrame(trials).to_csv(root / "reference_trials.csv", index=False)
    finish_study(
        root,
        manifest,
        rows,
        dict(vehicle=pictures, reference=refs, grid=grid),
        raw,
        dict(
            design=(
                "Identical costs, latent samples, reference/proposal densities "
                "and warm start per pair."
            ),
            targets=(
                "Vehicle Tsallis uses adaptive normalization/cutoff; KL uses "
                "lambda=8. Scalar reference uses fixed transformed scale 0.8 for "
                "distinct targets."
            ),
            inference=(
                "One-step commands and predicted costs, not executed driving or "
                "physical uncertainty."
            ),
        ),
    )


def occupancy(clouds, plans):
    """Full observed extent, shared across passes; no sample/time points discarded."""
    points = np.concatenate([np.asarray(x).reshape(-1, 2) for x in [*clouds, *plans]])
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = np.maximum(0.05 * (hi - lo), 0.01)
    edges = (
        np.linspace(lo[0] - pad[0], hi[0] + pad[0], 81),
        np.linspace(lo[1] - pad[1], hi[1] + pad[1], 65),
    )
    hist = [np.histogram2d(x[..., 0].ravel(), x[..., 1].ravel(), bins=edges)[0] for x in clouds]
    for count, cloud in zip(hist, clouds, strict=True):
        assert int(count.sum()) == np.prod(cloud.shape[:2])
    return hist, edges


def annealing(root, raw, spec, options, frozen, manifest):
    cfg = spec["V02"]
    f = frozen[cfg["scene"]]
    rows, pictures = [], {}
    for seed in spec["seeds"]:
        for alg in ["it_mppi", "dial_mpc", "dial_no_anneal"]:
            n = cfg["it_samples"] if alg == "it_mppi" else cfg["samples_per_iteration"]
            c = controller_at(f, options, alg, n, seed)
            command, updated = c.plan(np.array(f["state"]))
            snap = c.get_rollout_snapshot()
            snap["command"] = command
            archive(raw / f"{seed}_{alg}.npz", snap)
            assert c.model.transitions == cfg["transitions_per_decision"]
            count = 1 if alg == "it_mppi" else cfg["iterations"]
            plans, clouds, scales = [], [], []
            for i in range(count):
                states = snap["nominal_states"] if count == 1 else snap["inner_nominal_states"][i]
                mean = updated if count == 1 else snap["inner_updated_mean"][i]
                if count != 1:
                    c.diagnostics["planned_substeps"] = snap["inner_planned_substeps"][i]
                score = plan_objective(c, states, mean, f["previous_control"])
                costs = snap["costs"] if count == 1 else snap["inner_costs"][i]
                rows.append(
                    dict(
                        seed=seed,
                        algorithm=alg,
                        iteration=i + 1,
                        transitions=(i + 1) * (n + 1) * c.horizon,
                        planned_cost=score,
                        mean_sample_cost=float(costs.mean()),
                        minimum_sample_cost=float(costs.min()),
                        update_rms=float(np.sqrt(np.mean((mean - np.array(f["warm_start"])) ** 2))),
                    )
                )
                plans.append(states[:, :2])
                cloud = snap["sampled_states"] if count == 1 else snap["inner_sampled_states"][i]
                clouds.append(cloud[:, :, :2])
                if count > 1:
                    scales.append(snap["inner_variance_scales"][i])
            if seed == spec["seeds"][0]:
                # Histogram occupancy over all unweighted sample/time points.
                hist, edges = occupancy(clouds, plans)
                pictures[alg] = dict(
                    plans=plans,
                    histograms=hist,
                    xedges=edges[0],
                    yedges=edges[1],
                    variance_scales=scales,
                )
    pictures["obstacles"] = f["config"]["scenario"]["obstacles"]
    finish_study(
        root,
        manifest,
        rows,
        pictures,
        raw,
        dict(
            fixed=(
                "Frozen state, initial nominal sequence, task/model and 7680 "
                "transitions per decision."
            ),
            pairing=(
                "DIAL annealed/fixed use identical standard normals before their "
                "different scales; IT uses one larger population with its own "
                "fixed-base target."
            ),
            density=(
                "Unweighted occupancy pooled over prediction time, not a physical "
                "state probability distribution."
            ),
            inference=(
                "Refinement costs need not decrease monotonically. No closed-loop "
                "behavior is inferred."
            ),
        ),
    )


def tracking(root, raw, spec, options, manifest):
    cfg = spec["V03"]
    rows, steps = [], []
    for sign in cfg["disturbance_multipliers"]:
        for seed in spec["seeds"]:
            for threshold in [None, *cfg["reset_thresholds"]]:
                alg = "it_mppi" if threshold is None else "tube_mppi"
                label = alg if threshold is None else f"tube_{threshold:g}"
                c, _ = case_config(options, cfg["scene"], alg, cfg["samples"])
                c["execution"]["max_steps"] = cfg["steps"]
                c["execution"]["extra_snapshot_seeds"] = spec["seeds"]
                c["scenario"]["disturbance"] = dict(
                    step=cfg["disturbance_step"],
                    delta=(sign * np.array(cfg["disturbance_delta"])).tolist(),
                )
                if threshold is not None:
                    c["controller"]["reset_threshold"] = threshold
                folder = raw / f"impulse_{sign}" / label / str(seed)
                result, trajectory, diag = run_episode(
                    c, alg, seed, folder, capture_steps=[cfg["disturbance_step"] + 1]
                )
                row = dict(
                    impulse=sign,
                    seed=seed,
                    algorithm=label,
                    tracking_active_commands=int(
                        (diag.get("tracking_control_norm", pd.Series([0])) > 1e-10).sum()
                    ),
                    executed_steps=len(diag),
                    collision=result["collision"],
                    cost=result["cumulative_evaluation_cost"],
                    resets=int(diag.get("reset_event", pd.Series([0])).sum()),
                )
                rows.append(row)
                # Compact command-level records join existing recorder outputs.
                for i, command in diag.iterrows():
                    record = dict(
                        impulse=sign,
                        seed=seed,
                        algorithm=label,
                        step=int(i),
                        px=float(trajectory.px.iloc[i]),
                        py=float(trajectory.py.iloc[i]),
                        heading=float(trajectory.heading.iloc[i]),
                        executed_next_py=float(trajectory.py.iloc[i + 1]),
                    )
                    keys = [
                        k
                        for k in diag
                        if k.startswith(
                            ("nominal_", "tracking_", "planner_control_", "applied_correction_")
                        )
                    ]
                    for key in [
                        *keys,
                        "selected_actual",
                        "reset_event",
                        "commanded_steering",
                        "commanded_throttle",
                    ]:
                        if key in command and pd.notna(command[key]):
                            record[key] = float(command[key])
                    steps.append(record)
    pd.DataFrame(steps).to_csv(root / "steps.csv", index=False)
    finish_study(
        root,
        manifest,
        rows,
        dict(
            event_step=cfg["disturbance_step"],
            dt=options.get("dt", 0.15),
            representative_seed=spec["seeds"][0],
        ),
        raw,
        dict(
            fixed=(
                "Three paired seeds, zero/+/- impulse at step 8; all variants use "
                "64 samples per planning population. This is not an equal-work "
                "comparison."
            ),
            thresholds=(
                "Original reset threshold 450 and declared stricter threshold 0; "
                "both retain the original reset equation. No forced reset "
                "rejection."
            ),
            impulse=(
                "Existing runner multiplies declared impulse by 1+0.1*N(0,1), "
                "paired across controllers; exact deltas are in raw result.json."
            ),
            inference=(
                "Short disturbance-response windows, not goal-completion trials; "
                "separate pre-reset discrepancy, selected-reference error, "
                "requested and applied corrections."
            ),
        ),
    )


def robust(root, raw, spec, options, frozen, manifest):
    cfg = spec["V04"]
    f = frozen[cfg["scene"]]
    rows, pictures = [], []
    nominal = np.array(f["state"])
    for sign in cfg["disturbance_multipliers"]:
        observed = nominal + sign * np.array(cfg["disturbance_delta"])
        for threshold in cfg["free_energy_thresholds"]:
            for seed in spec["seeds"]:
                c = controller_at(
                    f, options, "robust_mppi", cfg["samples"], seed, free_energy_threshold=threshold
                )
                c.previous_plan = c.mean.copy()
                command, _ = c.plan(observed)
                z = c.get_rollout_snapshot()
                # Same final candidate perturbations and mean for a plain-IT counterfactual.
                it = controller_at(f, options, "it_mppi", cfg["samples"], seed)
                it.mean = z["proposal_mean"].copy()
                noise = z["latent_controls"] - z["proposal_mean"]
                normals = noise @ np.linalg.inv(np.linalg.cholesky(c.covariance)).T
                it_command, _ = it.plan(observed, standard_normals=normals)
                baseline = it.get_rollout_snapshot()
                z.update(
                    command=command,
                    baseline_command=it_command,
                    baseline_states=baseline["sampled_states"],
                    baseline_weights=baseline["weights"],
                    baseline_updated_mean=baseline["updated_mean"],
                    baseline_costs=baseline["costs"],
                )
                archive(raw / f"{sign}_{threshold:g}_{seed}.npz", z)
                rows.append(
                    dict(
                        impulse=sign,
                        threshold=threshold,
                        seed=seed,
                        selected_candidate=c.diagnostics["nominal_selection"],
                        infeasible=c.diagnostics["nominal_selection_infeasible"],
                        feedback_rms=c.diagnostics["sampled_feedback_rms"],
                        planner_steering=float(c.diagnostics["planner_control"][0]),
                        requested_tracking_steering=float(
                            c.diagnostics["tracking_correction"][0] * c.model.control_scale[0]
                        ),
                        applied_tracking_steering=float(c.diagnostics["applied_correction"][0]),
                        command_steering=float(command[0]),
                        it_command_steering=float(it_command[0]),
                        plan_difference_rms=float(
                            np.sqrt(
                                np.mean((z["real_updated_mean"] - baseline["updated_mean"]) ** 2)
                            )
                        ),
                        model_transitions=c.model.transitions,
                    )
                )
                if seed == spec["seeds"][0]:
                    pictures.append(
                        dict(
                            impulse=sign,
                            threshold=threshold,
                            energies=z["candidate_free_energies"],
                            candidates=z["nominal_candidates"],
                            selected=c.diagnostics["nominal_selection"],
                            nominal_mean=z["nominal_samples"].mean(axis=0),
                            actual_mean=z["sampled_states"].mean(axis=0),
                            it_mean=baseline["sampled_states"].mean(axis=0),
                            nominal_paths=z["nominal_samples"][:12, :, :2],
                            actual_paths=z["sampled_states"][:12, :, :2],
                            feedback_rms=np.sqrt(np.mean(z["feedback"] ** 2, axis=(0, 2))),
                            nominal_weights=z["nominal_weights"],
                            actual_weights=z["real_weights"],
                        )
                    )
    finish_study(
        root,
        manifest,
        rows,
        dict(cases=pictures),
        raw,
        dict(
            fixed=(
                "Same frozen nominal/history, warm start, cost and paired "
                "perturbations for Robust and plain IT; actual observed state "
                "differs only by declared signed impulse."
            ),
            thresholds=(
                "Original alpha=5000 and alpha=100 sensitivity; alpha changes "
                "nominal feasibility AND the mixed objective. Infeasible searches "
                "retain candidate zero and are explicitly reported."
            ),
            pairing=(
                "IT counterfactual reuses Robust's selected candidate mean and "
                "exact perturbations, so initial feedback-induced rollout "
                "differences can be isolated; this is not an equal-work "
                "comparison."
            ),
            inference=(
                "Only frozen decisions and sampled predictions. Planner command, "
                "requested tracker and clipped applied correction are separate; "
                "no executed robustness guarantee."
            ),
        ),
    )


class FrozenCovarianceExecution:
    """Replay a fixed CC auxiliary-noise policy through the existing episode runner.

    This is an experimental execution policy, not a new optimizer. Each replicate
    has independent noise; no importance weights or candidate selection are used.
    """

    def __init__(self, config, mean, a, b, gains, noise_covariance):
        shell = make_controller(config)
        self.model, self.cost = shell.model, shell.cost
        self.mean, self.a, self.b, self.gains = mean, a, b, gains
        self.chol = np.linalg.cholesky(noise_covariance)

    def reset(self, state, seed):
        self.rng = np.random.default_rng(seed)
        self.aux = np.zeros(7)
        self.index = 0
        self.noise_history = []
        self.diagnostics = {}

    def command(self, state):
        t = self.index
        noise = self.rng.standard_normal(2) @ self.chol.T
        control = self.model.clip_control(
            (self.mean[t] + self.gains[t] @ self.aux + noise) * self.model.control_scale
        )
        self.diagnostics = dict(execution_noise=noise.copy(), auxiliary_state=self.aux.copy())
        self.aux = self.a[t] @ self.aux + self.b[t] @ noise
        self.noise_history.append(noise)
        self.index += 1
        return control

    def get_rollout_snapshot(self):
        return None


def covariance(root, raw, spec, options, frozen, manifest):
    cfg = spec["V05"]
    rows, pictures = [], []
    for scene in cfg["scenes"]:
        f = frozen[scene]
        c = controller_at(f, options, "cc_mppi", cfg["planner_samples"], cfg["planner_seed"])
        c.plan(np.array(f["state"]))
        snap = c.get_rollout_snapshot()
        bound = c.diagnostics["terminal_covariance_bound"]
        reference, _ = c.predict(np.array(f["state"]), c.mean)
        snap.update(terminal_bound=bound, reference_states=reference)
        archive(raw / scene / "planner.npz", snap)
        normalization = np.sqrt(np.diag(bound))
        execution_config = deepcopy(f["config"])
        execution_config["scenario"]["initial_state"] = f["state"]
        execution_config["execution"]["max_steps"] = c.horizon
        execution_config["execution"]["stop_on_terminal"] = False
        for fraction in cfg["noise_fractions"]:
            for mode in cfg["feedback_modes"]:
                gains = c.gains if mode == "steered" else np.zeros_like(c.gains)
                paths, noises, outcomes = [], [], []
                for index in range(cfg["execution_replicates"]):
                    made = []

                    def factory(config, c=c, gains=gains, fraction=fraction, made=made):
                        policy = FrozenCovarianceExecution(
                            config, c.mean, c.a, c.b, gains, fraction * c.covariance
                        )
                        made.append(policy)
                        return policy

                    folder = raw / scene / f"{fraction:g}_{mode}" / str(index)
                    outcome, trajectory, _ = run_episode(
                        execution_config,
                        "fixed_cc_policy",
                        cfg["execution_seed_start"] + index,
                        folder,
                        controller_factory=factory,
                    )
                    x = trajectory[STATE_FIELDS].to_numpy()
                    # Never condition a covariance estimate on surviving paths.
                    if len(x) != c.horizon + 1:
                        raise RuntimeError(
                            "Early-terminated execution would censor covariance estimates"
                        )
                    paths.append(x)
                    noises.append(made[0].noise_history)
                    outcomes.append(outcome)
                paths, noises = np.array(paths), np.array(noises)
                errors = state_error(paths, reference[None])
                empirical = np.array(
                    [np.cov(errors[:, t], rowvar=False) for t in range(c.horizon + 1)]
                )
                if mode == "steered":
                    predicted = fraction * c.predicted_covariances
                else:
                    predicted = np.zeros_like(c.predicted_covariances)
                    for t in range(c.horizon):
                        predicted[t + 1] = (
                            c.a[t] @ predicted[t] @ c.a[t].T
                            + fraction * c.b[t] @ c.covariance @ c.b[t].T
                        )
                pred_norm = predicted / normalization[:, None] / normalization[None, :]
                emp_norm = empirical / normalization[:, None] / normalization[None, :]
                predicted_residual = float(np.linalg.eigvalsh(pred_norm[-1] - np.eye(7)).max())
                empirical_residual = float(np.linalg.eigvalsh(emp_norm[-1] - np.eye(7)).max())
                archive(
                    raw / scene / f"{fraction:g}_{mode}.npz",
                    dict(
                        executed_states=paths,
                        sampled_noise=noises,
                        predicted_covariances=predicted,
                        empirical_covariances=empirical,
                        gains=gains,
                        terminal_bound=bound,
                    ),
                )
                rows.append(
                    dict(
                        scene=scene,
                        noise_fraction=fraction,
                        mode=mode,
                        replicates=len(paths),
                        collisions=sum(o["collision"] for o in outcomes),
                        predicted_normalized_terminal_trace=float(np.trace(pred_norm[-1])),
                        empirical_normalized_terminal_trace=float(np.trace(emp_norm[-1])),
                        normalized_prediction_error=float(
                            np.linalg.norm(pred_norm[-1] - emp_norm[-1])
                        ),
                        predicted_normalized_psd_residual=predicted_residual,
                        empirical_normalized_psd_residual=empirical_residual,
                        empirical_constraint_violation=empirical_residual > 1e-5,
                    )
                )
                pictures.append(
                    dict(
                        scene=scene,
                        fraction=fraction,
                        mode=mode,
                        predicted_covariances=predicted,
                        empirical_covariances=empirical,
                        reference=reference,
                        empirical_mean=paths.mean(axis=0),
                        paths=paths[:16, :, :2],
                        gains=gains,
                        predicted_trace=np.trace(pred_norm, axis1=1, axis2=2),
                        empirical_trace=np.trace(emp_norm, axis1=1, axis2=2),
                        predicted_residual=predicted_residual,
                        empirical_residual=empirical_residual,
                    )
                )
                print(
                    f"V05 {scene} noise={fraction} {mode}: {len(paths)} independent executions",
                    flush=True,
                )
    finish_study(
        root,
        manifest,
        rows,
        dict(cases=pictures),
        raw,
        dict(
            execution=(
                "Independent nonlinear seven-state executions of a fixed "
                "auxiliary-noise policy; no replanning, optimizer weights, or "
                "uncontrolled mixture."
            ),
            noise=(
                "Fractions 1 and .01 of planning C0; gains and nominal warm start "
                "fixed. Independent seeds disjoint from planning samples; paired "
                "noise across steered and zero-gain arms."
            ),
            population=(
                "Planner sample covariance, independent executed covariance, and "
                "linearized predictions are distinct. The original controller's "
                "20% uncontrolled mixture is excluded from this controlled-policy "
                "validation."
            ),
            normalization=(
                "Divide each state by sqrt(diag(terminal bound)); full 7D PSD "
                "residual uses normalized coordinates. Position ellipses use "
                "meters and 95% Gaussian contours (not distribution-free "
                "coverage)."
            ),
            limits=(
                "A loose bound may be inactive. Small-noise agreement does not "
                "imply nonlinear safety, model-mismatch robustness, or covariance "
                "of receding-horizon MPPI execution."
            ),
        ),
    )


def run(output, raw, config_path="configs/variants/mechanisms.yaml"):
    assert_validated()
    spec = read_yaml(config_path)
    options = read_yaml(spec["benchmark_config"])
    output, raw = Path(output), Path(raw)
    if output.exists() or raw.exists():
        raise FileExistsError("Use fresh mechanism output and raw directories")
    raw.mkdir(parents=True)
    frozen = frozen_inputs(spec, options, raw)
    for study, function in [
        ("V01", weighting),
        ("V02", annealing),
        ("V03", tracking),
        ("V04", robust),
        ("V05", covariance),
    ]:
        cfg = deepcopy(spec) | dict(study=study, benchmark=options)
        root, manifest = begin(output / study, cfg, "formulation_mechanism")
        write_json(root / "frozen_inputs.json", frozen)
        directory = raw / study
        directory.mkdir()
        args = (root, directory, spec, options)
        if study == "V03":
            function(*args, manifest)
        else:
            function(*args, frozen, manifest)
        print(f"{study} complete", flush=True)


def audit(output):
    """Audit hashes and numerical identities in recorded populations/executions."""
    import json

    from controllers.feedback import tracking_design

    results = {}
    for study in ["V01", "V02", "V03", "V04", "V05"]:
        root = Path(output) / study
        manifest = json.loads((root / "manifest.json").read_text())
        raw = Path(manifest["raw_storage"])
        index = raw / "sha256.json"
        assert hashlib.sha256(index.read_bytes()).hexdigest() == manifest["raw_index_sha256"]
        for relative, expected in json.loads(index.read_text()).items():
            assert hashlib.sha256((raw / relative).read_bytes()).hexdigest() == expected
        assert (
            hashlib.sha256((root / "frozen_inputs.json").read_bytes()).hexdigest()
            == manifest["frozen_inputs_sha256"]
        )
        summary = pd.read_csv(root / "summary.csv")
        checked = 0
        if study == "V01":
            for row in summary.itertuples():
                with np.load(raw / row.scene / f"{row.seed}_{row.algorithm}.npz") as z:
                    ratio = z["log_base_density"] - z["log_proposal_density"]
                    np.testing.assert_allclose(ratio, z["log_base_over_proposal"], atol=1e-12)
                    if row.algorithm == "it_mppi":
                        logits = -z["costs"] / 8 + ratio
                    else:
                        normalized = (z["costs"] - row.cost_offset) / row.cost_span
                        support = 1 - (row.tsallis_r - 1) * normalized / row.transform_scale
                        logits = np.full(len(support), -np.inf)
                        valid = support > 0
                        logits[valid] = np.log(support[valid]) / (row.tsallis_r - 1) + ratio[valid]
                    weights = np.exp(logits - logits.max())
                    weights /= weights.sum()
                    np.testing.assert_allclose(weights, z["weights"], atol=1e-12)
                    np.testing.assert_allclose(
                        np.einsum("k,kti->ti", weights, z["latent_controls"]),
                        z["updated_mean"],
                        atol=1e-12,
                    )
                    checked += 1
        elif study == "V02":
            for path in raw.glob("*.npz"):
                with np.load(path) as z:
                    if "inner_weights" in z:
                        np.testing.assert_allclose(z["inner_weights"].sum(axis=1), 1)
                        np.testing.assert_allclose(
                            np.einsum(
                                "ik,iktd->itd", z["inner_weights"], z["inner_latent_controls"]
                            ),
                            z["inner_updated_mean"],
                            atol=1e-12,
                        )
                        for costs, weights in zip(
                            z["inner_costs"], z["inner_weights"], strict=True
                        ):
                            np.testing.assert_allclose(importance_weights(costs, 8), weights)
                        assert z["inner_weights"].shape == (4, 95)
                    else:
                        assert len(z["weights"]) == 383
                    checked += 1
            assert (summary.groupby(["seed", "algorithm"]).transitions.max() == 7680).all()
        elif study == "V03":
            for folder in sorted(raw.glob("impulse_*/*/*")):
                diag = pd.read_csv(folder / "diagnostics.csv")
                trajectory = pd.read_csv(folder / "trajectory.csv")
                if "tracking_correction_0" not in diag:
                    continue
                correction = diag[["tracking_correction_0", "tracking_correction_1"]].to_numpy()
                np.testing.assert_allclose(
                    np.linalg.norm(correction, axis=1), diag.tracking_control_norm, atol=1e-12
                )
                planner = diag[["planner_control_0", "planner_control_1"]].to_numpy()
                command = diag[["commanded_steering", "commanded_throttle"]].to_numpy()
                np.testing.assert_allclose(
                    command - planner,
                    diag[["applied_correction_0", "applied_correction_1"]],
                    atol=1e-12,
                )
                with np.load(folder / "snapshots/step_0009.npz") as z:
                    config = read_yaml(folder / "config.yaml")
                    controller = make_controller(config)
                    gains, _, _ = tracking_design(
                        controller.model,
                        z["nominal_states"],
                        controller.project_controls(z["updated_mean"]),
                        config["controller"],
                    )
                    observed = trajectory.loc[9, STATE_FIELDS].to_numpy(dtype=float)
                    nominal = diag.loc[9, [f"nominal_state_{j}" for j in range(7)]].to_numpy(
                        dtype=float
                    )
                    np.testing.assert_allclose(
                        gains[0] @ state_error(observed, nominal), correction[9], atol=1e-10
                    )
                    checked += 1
        elif study == "V04":
            for path in raw.glob("*.npz"):
                with np.load(path) as z:
                    for key, costs in [
                        ("nominal_weights", "nominal_augmented_costs"),
                        ("real_weights", "real_augmented_costs"),
                    ]:
                        np.testing.assert_allclose(
                            importance_weights(z[costs], 8), z[key], atol=1e-12
                        )
                    np.testing.assert_allclose(
                        np.einsum("k,kti->ti", z["real_weights"], z["latent_controls"]),
                        z["real_updated_mean"],
                        atol=1e-12,
                    )
                    command = np.clip(
                        (z["real_updated_mean"][0] + z["feedback"][0, 0]) * [0.5, 1],
                        [-0.5, -1],
                        [0.5, 1],
                    )
                    np.testing.assert_allclose(command, z["command"], atol=1e-12)
                    checked += 1
        else:
            for row in summary.itertuples():
                with (
                    np.load(raw / row.scene / "planner.npz") as plan,
                    np.load(raw / row.scene / f"{row.noise_fraction:g}_{row.mode}.npz") as z,
                ):
                    covariance = z["predicted_covariances"]
                    np.testing.assert_allclose(
                        covariance, covariance.transpose(0, 2, 1), atol=1e-12
                    )
                    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
                    errors = state_error(z["executed_states"], plan["reference_states"][None])
                    empirical = np.array([np.cov(errors[:, t], rowvar=False) for t in range(21)])
                    np.testing.assert_allclose(empirical, z["empirical_covariances"], atol=1e-12)
                    # Independently excite each whitened noise basis of the LTV policy.
                    noise = (
                        np.eye(40).reshape(40, 20, 2)
                        @ np.linalg.cholesky(row.noise_fraction * np.diag([0.14, 0.26])).T
                    )
                    state, auxiliary = np.zeros((40, 7)), np.zeros((40, 7))
                    for t in range(20):
                        a, b, gain = plan["linearized_a"][t], plan["linearized_b"][t], z["gains"][t]
                        state = state @ a.T + (auxiliary @ gain.T + noise[:, t]) @ b.T
                        auxiliary = auxiliary @ a.T + noise[:, t] @ b.T
                        np.testing.assert_allclose(state.T @ state, covariance[t + 1], atol=1e-10)
                    scale = np.sqrt(np.diag(z["terminal_bound"]))
                    normalized = empirical[-1] / scale[:, None] / scale[None, :]
                    np.testing.assert_allclose(
                        np.trace(normalized), row.empirical_normalized_terminal_trace
                    )
                    np.testing.assert_allclose(
                        np.linalg.eigvalsh(normalized - np.eye(7)).max(),
                        row.empirical_normalized_psd_residual,
                    )
                    assert z["executed_states"].shape == (256, 21, 7)
                    checked += 1
        result = dict(
            passed=True,
            checked_populations_or_conditions=checked,
            raw_files=manifest["raw_file_count"],
            method=(
                "Recorded-data identities, raw hashes, independent covariance impulse response; "
                "no optimization rerun."
            ),
        )
        write_json(root / "audit.json", result)
        results[study] = result
    return results
