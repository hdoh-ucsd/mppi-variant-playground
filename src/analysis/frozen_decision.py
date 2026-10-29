"""Frozen-input decisions through the shared MPPI planner; no evaluation execution."""

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t

from analysis.artifacts import begin, fingerprint
from evaluation.runner import make_controller, write_json
from sampling.gaussian import validate_covariance
from scenarios.environment import Environment, swept_clearance
from utils.config import load_experiment

PROPOSALS = ("standard", "isotropic", "diagonal", "full", "aligned", "misaligned")
STREAMS = {"evaluation": 101, "reference": 202}


def normal_draws(state_index, seed, count, horizon, stream="evaluation"):
    """Disjoint, recorded SeedSequence namespaces for selection, evaluation and reference."""
    sequence = np.random.SeedSequence([STREAMS[stream], state_index, seed])
    return np.random.default_rng(sequence).standard_normal((count, horizon, 2))


def inputs_signature(frozen):
    return fingerprint(
        {key: frozen[key] for key in ("config", "state", "warm_start", "previous_control")}
    )


def controller_at(frozen, covariance, count, capture=True):
    config = deepcopy(frozen["config"])
    config["controller"].update(
        kind="full",
        covariance=np.asarray(covariance).tolist(),
        samples=int(count),
        snapshot_steps=[0] if capture else [],
        saved_rollouts=0,
    )
    controller = make_controller(config)
    controller.mean = np.array(frozen["warm_start"], dtype=float)
    controller.previous_control = np.array(frozen["previous_control"], dtype=float)
    return controller


def plan_objective(controller, states, latent, previous):
    """Score a saved plan with the shared rollout objective, without new dynamics."""
    controls = controller.model.clip_control(latent * controller.model.control_scale)
    value = 0.0
    for i, control in enumerate(controls):
        value += controller.cost.running(states[i], control, previous)
        if getattr(controller.cost, "substep_sweep", False):
            gap = controller.cost.path_clearance(controller.diagnostics["planned_substeps"][i])
        elif hasattr(controller.cost, "transition_clearance"):
            gap = controller.cost.transition_clearance(states[i], states[i + 1])
        else:
            gap = swept_clearance(
                states[i], states[i + 1], controller.cost.obstacles, controller.cost.vehicle_radius
            )
        value += controller.cost.dt * controller.cost.weights["collision"] * (gap <= 0)
        previous = control
    return float(value + controller.cost.terminal(states[-1]))


def predict(frozen, latent):
    """Existing vehicle and cost; used only for sensitivity and route feasibility checks."""
    controller = controller_at(frozen, frozen["config"]["controller"]["covariance"], 2, False)
    states = np.empty((len(latent) + 1, controller.state_dimension))
    states[0] = frozen["state"]
    detailed = getattr(controller.cost, "substep_sweep", False)
    paths = []
    for i, control in enumerate(latent):
        if detailed:
            states[i + 1], path = controller.model.step(
                states[i], control * controller.model.control_scale, return_path=True
            )
            paths.append(path)
        else:
            states[i + 1] = controller.model.step(
                states[i], control * controller.model.control_scale
            )
    if detailed:
        controller.diagnostics["planned_substeps"] = np.array(paths)
    return states, plan_objective(controller, states, latent, frozen["previous_control"])


def decision(frozen, covariance, z, capture=True):
    """One unchanged MPPI update. Frozen inputs and environment are never mutated."""
    started = perf_counter()
    controller = controller_at(frozen, covariance, len(z), capture)
    command, updated = controller.plan(np.array(frozen["state"]), standard_normals=z)
    diagnostics = controller.diagnostics
    nominal = diagnostics["planned_trajectory"]
    objective = plan_objective(controller, nominal, updated, frozen["previous_control"])
    gap = (
        controller.cost.path_clearance(diagnostics["planned_substeps"])
        if getattr(controller.cost, "substep_sweep", False)
        else swept_clearance(
            nominal[:-1], nominal[1:], controller.cost.obstacles, controller.cost.vehicle_radius
        )
    )
    mean = np.array(frozen["warm_start"])
    stats = {
        key: diagnostics[key]
        for key in (
            "ess",
            "weight_entropy",
            "max_weight",
            "minimum_sample_cost",
            "weighted_sample_cost",
            "saturation_fraction",
            "planning_latency_s",
            "dynamics_evaluations",
            "rollout_evaluations",
        )
    }
    stats.update(
        nominal_objective=objective,
        update_rms=float(np.sqrt(np.mean((updated - mean) ** 2))),
        first_acceleration=float(command[0]),
        first_steering=float(command[1]),
        first_normalized_acceleration=float(command[0] / controller.model.control_scale[0]),
        first_normalized_steering=float(command[1] / controller.model.control_scale[1]),
        latent_first_acceleration=float(updated[0, 0]),
        latent_first_steering=float(updated[0, 1]),
        min_predicted_clearance=float(np.min(gap)) if np.isfinite(gap).any() else None,
        perturbation_mean_acceleration=float(diagnostics["perturbation_mean"][0]),
        perturbation_mean_steering=float(diagnostics["perturbation_mean"][1]),
        perturbation_variance_acceleration=float(diagnostics["perturbation_covariance"][0, 0]),
        perturbation_variance_steering=float(diagnostics["perturbation_covariance"][1, 1]),
        perturbation_cross_covariance=float(diagnostics["perturbation_covariance"][0, 1]),
        decision_and_scoring_latency_s=perf_counter() - started,
    )
    if frozen["config"].get("model_kind") == "dynamic7":
        for key in list(stats):
            if "acceleration" in key or "steering" in key:
                stats.pop(key)
        for i, name in enumerate(("steering", "throttle")):
            stats[f"first_{name}"] = float(command[i])
            stats[f"first_normalized_{name}"] = float(
                command[i] / controller.model.control_scale[i]
            )
            stats[f"latent_first_{name}"] = float(updated[0, i])
            stats[f"perturbation_mean_{name}"] = float(diagnostics["perturbation_mean"][i])
            stats[f"perturbation_variance_{name}"] = float(
                diagnostics["perturbation_covariance"][i, i]
            )
        stats["rhs_evaluations"] = diagnostics["rhs_evaluations"]
    snapshot = controller.get_rollout_snapshot() if capture else None
    if snapshot is not None:
        stats["best_is_highest_weight"] = (
            snapshot["best_rollout_index"] == snapshot["highest_weight_index"]
        )
        snapshot.update(command=command, updated_mean=updated)
    return updated, stats, snapshot


def route_witnesses(frozen):
    """Deterministic control grid verifies two feasible station-crossing routes."""
    horizon = frozen["config"]["controller"]["horizon"]
    witnesses = {}
    for acceleration in (0.0, 0.25, 0.5):
        for amplitude in (0.4, 0.7, 1.0):
            for side in (-1, 1):
                for fraction in (0.2, 0.25, 0.4, 0.55):
                    for second in (0.0, -0.25, -0.5, -1.0):
                        latent = np.zeros((horizon, 2))
                        latent[:, 0] = acceleration
                        n = int(horizon * fraction)
                        latent[:n, 1] = side * amplitude
                        latent[n:, 1] = second * side * amplitude
                        states, objective = predict(frozen, latent)
                        scenario = frozen["config"]["scenario"]
                        gap = swept_clearance(
                            states[:-1],
                            states[1:],
                            scenario["obstacles"],
                            scenario["vehicle_radius"],
                        )
                        station = scenario["obstacles"][0][0]
                        crossings = np.flatnonzero(
                            (states[:-1, 0] < station) & (states[1:, 0] >= station)
                        )
                        if np.min(gap) > 0 and len(crossings):
                            i = crossings[0]
                            alpha = (station - states[i, 0]) / (states[i + 1, 0] - states[i, 0])
                            y = states[i, 1] + alpha * (states[i + 1, 1] - states[i, 1])
                            label = "upper" if y > 0 else "lower"
                            if label not in witnesses or objective < witnesses[label]["objective"]:
                                witnesses[label] = dict(
                                    latent=latent.tolist(),
                                    states=states.tolist(),
                                    clearance=float(np.min(gap)),
                                    station_y=float(y),
                                    objective=objective,
                                )
    return witnesses


def capture_states(config, output):
    """Stop standard reference prefixes at geometric/event rules before evaluation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen_states = []
    for name, source in (
        ("open_space", "open_space"),
        ("pre_obstacle", "two_route_obstacle"),
        ("post_disturbance", "disturbance"),
    ):
        resolved = load_experiment("standard", source)
        resolved["controller"].update(horizon=config["horizon"], samples=config["primary_samples"])
        seed = config["reference_episode_seed"]
        env = Environment(
            resolved["scenario"], resolved["dynamics"], np.random.SeedSequence([seed, 1001])
        )
        controller = make_controller(resolved)
        controller.reset(env.state, seed)
        states, controls, frozen = [env.state.copy()], [], None
        for step in range(resolved["scenario"]["max_steps"]):
            if name == "pre_obstacle":
                eligible = (
                    config["selection"]["pre_obstacle_min_x"]
                    <= env.state[0]
                    <= config["selection"]["pre_obstacle_max_x"]
                )
            else:
                eligible = step == config["selection"][f"{name}_step"]
            if eligible:
                candidate = dict(
                    name=name,
                    config=deepcopy(resolved),
                    state=env.state.tolist(),
                    warm_start=controller.mean.tolist(),
                    previous_control=controller.previous_control.tolist(),
                    step=step,
                    source_seed=seed,
                    source_scenario=source,
                    disturbance=deepcopy(resolved["scenario"].get("disturbance")),
                    event_applied=bool(name == "post_disturbance" and step == 21),
                )
                witnesses = route_witnesses(candidate) if name == "pre_obstacle" else {}
                if name != "pre_obstacle" or set(witnesses) == {"upper", "lower"}:
                    candidate["route_witnesses"] = witnesses
                    frozen = candidate
                    break
            control = controller.command(env.state.copy())
            controls.append(control)
            states.append(env.step(control))
        if frozen is None:
            raise RuntimeError(f"Predeclared selection rule found no eligible state for {name}")
        frozen["selection_rule"] = (
            "First state in declared x interval with two collision-free grid route witnesses"
            if name == "pre_obstacle"
            else f"Observed reference state at planning step {step}"
        )
        frozen["inputs_sha256"] = inputs_signature(frozen)
        write_json(output / f"{name}.json", frozen)
        np.savez_compressed(
            output / f"{name}_source_episode.npz",
            states=np.array(states),
            controls=np.array(controls),
        )
        frozen_states.append(frozen)
        print(f"Frozen {name}: step={step}, state={frozen['state']}", flush=True)
    return frozen_states


def proposals(frozen, config):
    """Major axis along local deterministic cost descent under a constant latent offset.

    Central differences differentiate J(U + 1_T d), not obstacle XY orientation.
    The orthogonal direction uses the same eigenvalues. This is a heuristic, not
    an optimal importance covariance or an inferred Gibbs target covariance.
    """
    mean = np.array(frozen["warm_start"])
    gradients = []
    for h in (config["sensitivity_step"], config["sensitivity_step"] / 2):
        gradient = []
        for axis in range(2):
            offset = np.zeros(2)
            offset[axis] = h
            gradient.append(
                (predict(frozen, mean + offset)[1] - predict(frozen, mean - offset)[1]) / (2 * h)
            )
        gradients.append(np.array(gradient))
    gradient = gradients[-1]
    if np.linalg.norm(gradient) < 1e-10:
        raise ValueError("No resolved local cost descent direction")
    useful = -gradient / np.linalg.norm(gradient)
    orthogonal = np.array([-useful[1], useful[0]])
    rotation = np.column_stack([useful, orthogonal])
    eigenvalues = np.array(config["orientation_eigenvalues"])
    matrices = deepcopy(
        config.get(
            "proposal_matrices",
            {name: load_experiment(name)["controller"]["covariance"] for name in PROPOSALS[:4]},
        )
    )
    required_trace = (
        float(np.trace(frozen["config"]["controller"]["base_covariance"]))
        if frozen["config"].get("model_kind") == "dynamic7"
        else 0.44
    )
    matrices["aligned"] = rotation @ np.diag(eigenvalues) @ rotation.T
    matrices["misaligned"] = rotation @ np.diag(eigenvalues[::-1]) @ rotation.T
    result = {}
    for name, matrix in matrices.items():
        cov = validate_covariance(matrix)
        if not np.isclose(np.trace(cov), required_trace):
            raise ValueError("Primary covariance trace must match declared intervention")
        values, vectors = np.linalg.eigh(cov)
        result[name] = dict(
            covariance=cov.tolist(),
            eigenvalues=values.tolist(),
            eigenvectors=vectors.tolist(),
            determinant=float(np.linalg.det(cov)),
            trace=float(np.trace(cov)),
            major_axis_alignment_abs=(
                None if np.isclose(values[0], values[1]) else float(abs(vectors[:, -1] @ useful))
            ),
        )
    return result, dict(
        direction=useful.tolist(),
        gradients=[g.tolist() for g in gradients],
        finite_difference_steps=[config["sensitivity_step"], config["sensitivity_step"] / 2],
        relative_gradient_change=float(
            np.linalg.norm(gradients[1] - gradients[0]) / np.linalg.norm(gradient)
        ),
        additional_dynamics_evaluations=8 * config["horizon"],
    )


def save_decision(path, snapshot):
    """Frozen schema: no fictitious executed-next-state; exact controls/weights stay float64."""
    weights = np.asarray(snapshot["weights"])
    if (weights < 0).any() or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid weights in decision snapshot")
    data = {
        key: snapshot[key]
        for key in (
            "sampled_states",
            "sampled_controls",
            "latent_controls",
            "costs",
            "weights",
            "log_base_over_proposal",
            "current_state",
            "nominal_states",
            "updated_mean",
            "command",
        )
    }
    data.update(
        schema_version=np.array(1),
        best_rollout_index=np.array(snapshot["best_rollout_index"]),
        highest_weight_index=np.array(snapshot["highest_weight_index"]),
    )
    for key in ("sampled_states", "sampled_controls"):
        data[key] = np.asarray(data[key], dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **data)


def load_decision(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    if int(result["schema_version"]) != 1:
        raise ValueError("Unsupported decision snapshot schema")
    weights = result["weights"]
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid snapshot weights")
    if int(result["best_rollout_index"]) != int(result["costs"].argmin()):
        raise ValueError("Invalid minimum-cost index")
    if int(result["highest_weight_index"]) != int(weights.argmax()):
        raise ValueError("Invalid maximum-weight index")
    return result


def reference_estimate(frozen, state_index, config, output):
    """Independent high-K SNIS repetitions with pooled log-mass weighting and block uncertainty."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    means, rows, log_masses = [], [], []
    covariance = frozen["config"]["controller"]["covariance"]
    for repetition in range(config["reference_repetitions"]):
        z = normal_draws(
            state_index, repetition, config["reference_samples"], config["horizon"], "reference"
        )
        # Full arrays are needed only transiently for stable unnormalized masses.
        mean, stats, snap = decision(frozen, covariance, z)
        logits = (
            -snap["costs"] / frozen["config"]["controller"]["temperature"]
            + snap["log_base_over_proposal"]
        )
        mass = float(logsumexp(logits))
        means.append(mean)
        log_masses.append(mass)
        rows.append(
            {
                "repetition": repetition,
                "seed_namespace": STREAMS["reference"],
                **stats,
                "log_weight_mass": mass,
            }
        )
        del snap, z
    means, log_masses = np.array(means), np.array(log_masses)
    masses = np.exp(log_masses - logsumexp(log_masses))
    pooled = np.einsum("r,rti->ti", masses, means)
    # Delete-one-block jackknife of the pooled ratio estimator; not an accuracy guarantee.
    leave_one = np.array(
        [
            np.einsum(
                "r,rti->ti",
                np.exp(np.delete(log_masses, i) - logsumexp(np.delete(log_masses, i))),
                np.delete(means, i, axis=0),
            )
            for i in range(len(means))
        ]
    )
    se = np.sqrt(
        (len(means) - 1) / len(means) * np.sum((leave_one - leave_one.mean(axis=0)) ** 2, axis=0)
    )
    ci_half = student_t.ppf(0.975, len(means) - 1) * se
    pooled_ess = 1 / sum(masses[i] ** 2 / rows[i]["ess"] for i in range(len(means)))
    max_weight = max(masses[i] * rows[i]["max_weight"] for i in range(len(means)))
    reliable = bool(
        min(r["ess"] for r in rows) >= config["reference_min_ess"]
        and max_weight <= config["reference_max_weight"]
        and np.sqrt(np.mean(ci_half**2)) <= config["reference_max_rms_ci_halfwidth"]
    )
    status = dict(
        status="diagnostically_usable" if reliable else "unresolved",
        note=(
            "Finite-sample diagnostics cannot certify target coverage. "
            "Intervals describe block variation, not unknown SNIS bias."
        ),
        pooled_ess=float(pooled_ess),
        pooled_max_weight=float(max_weight),
        min_repetition_ess=float(min(r["ess"] for r in rows)),
        rms_ci95_halfwidth=float(np.sqrt(np.mean(ci_half**2))),
        samples_per_repetition=config["reference_samples"],
        repetitions=len(means),
        total_rollouts=config["reference_samples"] * len(means),
        proposal="standard, centered on the identical frozen warm start",
        seed_sequences=[[STREAMS["reference"], state_index, i] for i in range(len(means))],
    )
    np.savez_compressed(
        output / "estimate.npz",
        mean=pooled,
        repetition_means=means,
        log_masses=log_masses,
        block_masses=masses,
        standard_error=se,
        ci95_low=pooled - ci_half,
        ci95_high=pooled + ci_half,
    )
    pd.DataFrame(rows).to_csv(output / "repetitions.csv", index=False)
    write_json(output / "status.json", status)
    print(f"Reference {frozen['name']}: {status['status']}, ESS={pooled_ess:.2f}", flush=True)
    return pooled, status


def summarize(root, frame, config):
    metrics = [
        "reference_distance_rms",
        "first_command_reference_distance",
        "paired_mean_distance_rms",
        "paired_first_command_distance",
        "ess",
        "nominal_objective",
        "planning_latency_s",
        "update_rms",
        "saturation_fraction",
        "delta_acceleration",
        "delta_steering",
        "delta_normalized_acceleration",
        "delta_normalized_steering",
    ]
    rows = []
    for (state, proposal, count), group in frame.groupby(
        ["state", "proposal", "samples"], sort=False
    ):
        row = dict(
            state=state,
            proposal=proposal,
            samples=int(count),
            n=len(group),
            reference_status=group.reference_status.iloc[0],
        )
        for metric in metrics:
            values = group[metric].to_numpy()
            row.update(
                {
                    f"{metric}_{key}": float(value)
                    for key, value in {
                        "mean": values.mean(),
                        "sd": values.std(ddof=1),
                        "q05": np.quantile(values, 0.05),
                        "q50": np.median(values),
                        "q95": np.quantile(values, 0.95),
                        "ci95_halfwidth": student_t.ppf(0.975, len(values) - 1)
                        * values.std(ddof=1)
                        / np.sqrt(len(values)),
                    }.items()
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    sensitivity = []
    for n in config["repetition_counts"]:
        for (state, proposal), group in frame[frame.samples == config["primary_samples"]].groupby(
            ["state", "proposal"]
        ):
            part = group.sort_values("seed").head(n)
            for metric in (
                "delta_normalized_acceleration",
                "delta_normalized_steering",
                "reference_distance_rms",
                "ess",
            ):
                values = part[metric].to_numpy()
                sensitivity.append(
                    dict(
                        state=state,
                        proposal=proposal,
                        repetitions=len(values),
                        metric=metric,
                        mean=float(values.mean()),
                        ci95_halfwidth=float(
                            student_t.ppf(0.975, len(values) - 1)
                            * values.std(ddof=1)
                            / np.sqrt(len(values))
                        ),
                    )
                )
    pd.DataFrame(sensitivity).to_csv(root / "repetition_sensitivity.csv", index=False)


def run_study(config, output):
    seeds = list(
        range(config["seeds"]["start"], config["seeds"]["start"] + config["seeds"]["count"])
    )
    if len(seeds) < 2 or config["reference_repetitions"] < 3:
        raise ValueError(
            "At least two evaluation seeds and three independent reference batches required"
        )
    if (
        config["representative_seed"] not in seeds
        or config["primary_samples"] not in config["sample_counts"]
    ):
        raise ValueError("Representative seed and primary budget must be evaluated")
    root, manifest = begin(output, config, "frozen_decision")
    manifest.update(
        rng=(
            "SeedSequence([101,state_index,seed]) evaluation; "
            "[202,state_index,repetition] reference; PCG64; prefix-nested K"
        ),
        snapshot_note=(
            "All primary-K seed/proposal populations; all K metrics and mean plans. "
            "Budget clouds saved for representative seed only."
        ),
        experiment_note=(
            "Reference capture executes one standard prefix per state. "
            "Evaluation performs one frozen optimization, never Environment.step."
        ),
        timing_note=(
            "planning_latency_s excludes construction, nominal objective scoring and IO; "
            "captures add observation overhead; no performance guarantee"
        ),
    )
    write_json(root / "manifest.json", manifest)
    frozen_states = capture_states(config, root / "frozen_states")
    all_rows, all_means, proposal_records, reference_statuses = [], [], {}, {}
    for index, frozen in enumerate(frozen_states):
        name = frozen["name"]
        candidates, sensitivity = proposals(frozen, config)
        proposal_records[name] = dict(
            proposals=candidates, sensitivity=sensitivity, inputs_sha256=frozen["inputs_sha256"]
        )
        reference, status = reference_estimate(
            frozen, index, config, root / "reference_estimates" / name
        )
        reference_statuses[name] = status
        scale = make_controller(frozen["config"]).model.control_scale
        ref_command = np.clip(reference[0], -1, 1)
        for seed in seeds:
            z = normal_draws(index, seed, max(config["sample_counts"]), config["horizon"])
            for count in config["sample_counts"]:
                standard_stats = None
                for proposal in PROPOSALS:
                    capture = (
                        count == config["primary_samples"] or seed == config["representative_seed"]
                    )
                    mean, stats, snapshot = decision(
                        frozen, candidates[proposal]["covariance"], z[:count], capture
                    )
                    if proposal == "standard":
                        standard_stats = stats
                        standard_mean = mean
                    row = dict(
                        state=name,
                        proposal=proposal,
                        seed=seed,
                        samples=count,
                        inputs_sha256=frozen["inputs_sha256"],
                        reference_status=status["status"],
                        reference_distance_rms=float(np.sqrt(np.mean((mean - reference) ** 2))),
                        paired_mean_distance_rms=float(
                            np.sqrt(np.mean((mean - standard_mean) ** 2))
                        ),
                        paired_first_command_distance=float(
                            np.linalg.norm(
                                np.clip(mean[0], -1, 1) - np.clip(standard_mean[0], -1, 1)
                            )
                        ),
                        first_command_reference_distance=float(
                            np.linalg.norm(np.clip(mean[0], -1, 1) - ref_command)
                        ),
                        **stats,
                    )
                    for field in ("acceleration", "steering"):
                        row[f"delta_{field}"] = (
                            stats[f"first_{field}"] - standard_stats[f"first_{field}"]
                        )
                        row[f"delta_normalized_{field}"] = (
                            row[f"delta_{field}"] / scale[0 if field == "acceleration" else 1]
                        )
                    if capture:
                        relative = (
                            Path("snapshots") / name / f"K{count}" / proposal / f"seed_{seed}.npz"
                        )
                        save_decision(root / relative, snapshot)
                        row["snapshot_path"] = str(relative)
                    all_rows.append(row)
                    all_means.append(mean)
            if (seed - seeds[0] + 1) % 10 == 0:
                print(
                    f"Completed {name}: {seed - seeds[0] + 1}/{len(seeds)} paired seeds", flush=True
                )
        if inputs_signature(frozen) != frozen["inputs_sha256"]:
            raise RuntimeError("Frozen input mutated during evaluation")
    frame = pd.DataFrame(all_rows)
    frame.to_csv(root / "per_seed.csv", index=False)
    np.savez_compressed(
        root / "updated_means.npz", means=np.array(all_means), row_index=np.arange(len(frame))
    )
    summarize(root, frame, config)
    write_json(root / "proposals.json", proposal_records)
    write_json(root / "reference_status.json", reference_statuses)
    manifest.update(
        complete=True,
        decisions=len(frame),
        frozen_states=len(frozen_states),
        evaluation_rollouts=int(frame.rollout_evaluations.sum()),
        evaluation_dynamics_evaluations=int(frame.dynamics_evaluations.sum()),
        reference_rollouts=sum(v["total_rollouts"] for v in reference_statuses.values()),
        snapshots=int(frame.snapshot_path.notna().sum()),
    )
    write_json(root / "manifest.json", manifest)
    return frame
