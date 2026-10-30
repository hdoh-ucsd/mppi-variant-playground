"""Passive adapters: preserve mathematical distinctions while sharing five-stage records."""

from copy import deepcopy

import numpy as np

from analysis.decision_trace import (
    CONTROL_UNITS,
    STATE_UNITS,
    StageRecord,
    missing,
    native,
    new_trace,
    not_applicable,
)
from costs.dynamic_vehicle import VehicleCost
from sampling.gaussian import log_sequence_density

ALGORITHMS = ("it_mppi", "tsallis_vi", "dial_mpc", "tube_mppi", "robust_mppi", "cc_mppi")
REGULARIZERS = {
    "it_mppi": "KL fixed-base Gibbs projection",
    "tsallis_vi": "Tsallis deformed target with explicit transformed scale",
    "dial_mpc": "Cost-only Gaussian smoothing and iterated mean refinement",
    "tube_mppi": "Two KL planners, cost-based nominal reset, separate local LQR tracker",
    "robust_mppi": "Coupled nominal/actual augmented objectives; feedback-dependent AIS terms",
    "cc_mppi": "Auxiliary-noise covariance steering plus augmented-cost shaped mean fit",
}


def capture_inputs(controller, state, previous_executed=None):
    inputs = dict(
        observed_state=np.asarray(state).copy(),
        warm_start=controller.mean.copy(),
        previous_control=controller.previous_control.copy(),
        rng_before=deepcopy(controller.rng.bit_generator.state),
    )
    if previous_executed is not None:
        inputs["previous_executed_control"] = np.asarray(previous_executed).copy()
    for key in ("nominal_state", "nominal_previous", "previous_nominal", "previous_plan"):
        if hasattr(controller, key):
            inputs[key] = np.asarray(getattr(controller, key)).copy()
    return inputs


def cost_components(cost, states, controls, previous, contacts=None):
    """Evaluate recorded states only with the existing task cost, one coefficient at a time.

    No dynamics or duplicated cost formula. Swept contact is included only when its
    original substep outcome was recorded; state-only collision is never substituted.
    """
    if states is None or controls is None or previous is None:
        return None
    previous_controls = np.concatenate(
        [np.broadcast_to(previous, (len(states), 1, 2)), controls[:, :-1]], axis=1
    )
    weights = cost.weights
    coefficients = [k for k in weights if k != "safety_margin"]
    terms = {}
    for key in coefficients:
        masked = {k: (v if k in (key, "safety_margin") else 0.0) for k, v in weights.items()}
        single = VehicleCost(cost.geometry, cost.model, masked, cost.speed)
        if key.startswith("terminal_"):
            terms[key] = single.terminal(states[:, -1])
        else:
            terms[key] = single.running(states[:, :-1], controls, previous_controls).sum(axis=1)
    if contacts is not None:
        terms["swept_contact"] = cost.dt * weights["collision"] * contacts.sum(axis=1)
    return terms


def build_trace(
    config,
    inputs,
    snapshot,
    diagnostics,
    command,
    *,
    experiment_id,
    seed,
    planning_step,
    mode,
    source_commit,
    execution=None,
    provenance=None,
    cost=None,
):
    """Same adapter for live and retroactive records. Missing fields stay explicit."""
    snap, diag = snapshot or {}, diagnostics or {}
    trace = new_trace(
        config,
        experiment_id=experiment_id,
        seed=seed,
        planning_step=planning_step,
        mode=mode,
        inputs=inputs,
        source_commit=source_commit,
        provenance=provenance,
    )
    alg = config["controller"]["algorithm"]
    ctrl, h = config["controller"], config["controller"]["horizon"]
    scale = np.array([config["vehicle"]["steering_limit"], config["vehicle"]["throttle_limit"]])

    def take(mapping, key, units="dimensionless"):
        value = mapping.get(key)
        return trace.put(value, units) if value is not None else missing(f"{key} was not recorded")

    def scalar(key):
        value = diag.get(key)
        return (
            native(value) if value is not None and np.isfinite(value) else missing(f"{key} absent")
        )

    shared = {
        "configuration_hash": trace.metadata["configuration_hash"],
        "fields": [
            "scenario",
            "vehicle",
            "cost",
            "speed_target",
            "execution",
            "execution_control_noise_covariance",
        ],
        "planning_model": "DynamicVehicle(config.vehicle)",
        "execution_model": "DynamicVehicle(vehicle | scenario.execution_overrides)",
        "collision_geometry": "centered disk; recorded substep swept contacts",
        "actuator_units": CONTROL_UNITS,
    }
    stage1 = dict(
        observed_state=take(inputs, "observed_state", STATE_UNITS),
        nominal_initial_state=take(inputs, "nominal_state", STATE_UNITS)
        if alg in ("tube_mppi", "robust_mppi")
        else not_applicable("No nominal-state memory"),
        initial_warm_start=take(inputs, "warm_start", "normalized controls"),
        previous_planner_control=take(inputs, "previous_control", CONTROL_UNITS),
        previous_executed_control=take(inputs, "previous_executed_control", CONTROL_UNITS),
        shared_configuration=shared,
    )
    for key in ("nominal_previous", "previous_nominal", "previous_plan"):
        if key in inputs:
            stage1[key] = take(
                inputs,
                key,
                STATE_UNITS
                if key == "previous_nominal"
                else CONTROL_UNITS
                if key == "nominal_previous"
                else "normalized controls",
            )
    trace.stages["dynamics_task"] = StageRecord("available", stage1)
    objective = dict(
        regularizer=REGULARIZERS[alg],
        temperature=ctrl["temperature"],
        base_covariance=ctrl["base_covariance"],
        proposal_family="Gaussian latent noise",
        reference_note="Different variants can define different targets/objectives",
    )
    if alg == "tsallis_vi":
        offset, span = diag.get("cost_offset"), diag.get("cost_span")
        r, transformed = ctrl.get("tsallis_r", 1.5), diag.get("transform_scale")
        # Recover only deterministic normalization from the saved full costs, not from a plot.
        if "costs" in snap and transformed is None:
            raw = snap["costs"]
            elite = ctrl.get("elite_fraction", 0.8)
            offset, span = (
                (float(raw.min()), float(np.ptp(raw))) if elite is not None else (0.0, 1.0)
            )
            normalized = (raw - offset) / span if span > 0 else np.zeros_like(raw)
            transformed = (
                (r - 1) * max(float(np.quantile(normalized, elite)), np.finfo(float).eps)
                if elite is not None
                else ctrl["temperature"]
            )
        objective["tsallis"] = dict(
            parameter=r,
            transformed_scale=transformed,
            cost_offset=offset,
            cost_span=span,
            elite_fraction=ctrl.get("elite_fraction", 0.8),
            normalized_cutoff=(transformed / (r - 1) if r > 1 and transformed else None),
            provenance="Recorded or deterministic reduction of saved raw costs",
        )
    if alg == "dial_mpc":
        objective["dial"] = dict(
            iterations=ctrl.get("iterations", 4),
            beta_trajectory=ctrl.get("beta_trajectory", 1.0),
            beta_horizon=ctrl.get("beta_horizon", 1.0),
            anneal=ctrl.get("anneal", True),
            score="-raw cost / temperature; no base/proposal correction",
            interpretation="Repeated Gaussian smoothing at each pass covariance",
        )
    if alg in ("tube_mppi", "robust_mppi"):
        objective["tracking"] = {
            k: ctrl.get(k) for k in ["tracking_q", "tracking_r", "tracking_terminal_multiplier"]
        }
        objective["tracking"]["role"] = (
            "Ancillary tracking after two KL solutions; no feedback in rollouts"
            if alg == "tube_mppi"
            else "Feedback enters actual rollouts and both augmented objectives"
        )
        objective["selection"] = (
            dict(reset_threshold=ctrl.get("reset_threshold", 450.0))
            if alg == "tube_mppi"
            else dict(free_energy_threshold=ctrl.get("free_energy_threshold", 5000.0))
        )
    if alg == "cc_mppi":
        objective["cc"] = {
            k: ctrl.get(k)
            for k in [
                "terminal_covariance_bound",
                "covariance_q",
                "covariance_r",
                "cc_control_r",
                "noise_ratio",
                "uncontrolled_fraction",
            ]
        }
        objective["cc"].update(
            solver=diag.get("solver", missing("Solver status not saved")),
            residual=scalar("covariance_constraint_residual"),
            objective=scalar("covariance_objective"),
            scope="Linearized controlled population; uncontrolled fraction excluded",
        )
    trace.stages["distributional_objective"] = StageRecord("available", objective)
    populations, updates = [], []

    def population(
        label,
        record,
        *,
        role,
        iteration=None,
        covariance=None,
        weight_key="weights",
        cost_key="costs",
        update_key="updated_mean",
        plan_key="nominal_states",
        path_key="planned_substeps",
        correction=True,
    ):
        if "sampled_states" not in record:
            return
        k = len(record["sampled_states"])
        covariance = np.asarray(ctrl["covariance"] if covariance is None else covariance)
        pop = dict(
            population_id=label,
            role=role,
            samples=k,
            latent_controls=take(record, "latent_controls", "normalized controls"),
            physical_controls=take(record, "sampled_controls", CONTROL_UNITS),
            states=take(record, "sampled_states", STATE_UNITS),
            proposal_mean=take(record, "proposal_mean", "normalized controls"),
            proposal_covariance=trace.put(covariance, "normalized control squared"),
            feasible=take(record, "feasible", "boolean"),
            swept_contacts=take(record, "swept_contacts", "boolean"),
        )
        if iteration is not None:
            pop["iteration"] = iteration
            pop["model_transitions"] = (k + 1) * h
        populations.append(pop)
        costs, weights = record.get(cost_key), record.get(weight_key)
        latent, mean = record.get("latent_controls"), record.get("proposal_mean")
        log_base = log_proposal = ratio = scores = None
        if latent is not None and mean is not None and alg != "cc_mppi":
            log_proposal = log_sequence_density(latent, mean, covariance)
            if correction:
                log_base = log_sequence_density(
                    latent, np.zeros_like(mean), ctrl["base_covariance"]
                )
                ratio = record.get("log_base_over_proposal", log_base - log_proposal)
        if costs is not None:
            effective = costs
            if alg == "robust_mppi":
                effective = record.get(
                    "nominal_augmented_costs" if role == "nominal" else "real_augmented_costs"
                )
            if alg == "cc_mppi":
                effective = record.get("augmented_costs")
            if effective is not None:
                scores = -effective / ctrl["temperature"]
                if correction:
                    scores = scores + ratio if ratio is not None else None
                if alg == "tsallis_vi":
                    ts = objective["tsallis"]
                    normalized = (
                        (costs - ts["cost_offset"]) / ts["cost_span"]
                        if ts["cost_span"] > 0
                        else np.zeros_like(costs)
                    )
                    r, lam = ts["parameter"], ts["transformed_scale"]
                    argument = 1 - (r - 1) * normalized / lam
                    scores = np.full(k, -np.inf)
                    support = argument > 0
                    if r == 1:
                        scores = -normalized / lam
                    else:
                        scores[support] = np.log(argument[support]) / (r - 1)
                    if ratio is not None:
                        scores += ratio
                    objective["tsallis"]["support"] = trace.put(support, "boolean")

        def optional(value, units, reason):
            return trace.put(value, units) if value is not None else missing(reason)

        weight_metrics = dict(ess=None, maximum_weight=None, entropy=None)
        if weights is not None:
            positive = weights > 0
            weight_metrics = dict(
                ess=float(1 / np.sum(weights**2)),
                maximum_weight=float(weights.max()),
                entropy=float(-np.sum(weights[positive] * np.log(weights[positive]))),
            )
        update = dict(
            population_id=label,
            role=role,
            raw_costs=take(record, cost_key, "task cost"),
            weights=take(record, weight_key, "normalized optimizer mass"),
            updated_latent=take(record, update_key, "normalized controls"),
            nominal_prediction=take(record, plan_key, STATE_UNITS),
            log_scores=optional(scores, "log score", "Unrecorded effective objective"),
            log_base_density=optional(log_base, "latent log density", "Not recorded"),
            log_proposal_density=optional(log_proposal, "latent log density", "Not recorded"),
            density_correction=optional(ratio, "log density ratio", "Not recorded"),
            lowest_raw_cost_index=int(np.argmin(costs)) if costs is not None else None,
            highest_weight_index=int(np.argmax(weights)) if weights is not None else None,
            **weight_metrics,
        )
        if not correction:
            update["log_base_density"] = not_applicable(
                "No separate base density in this weight law"
            )
            update["density_correction"] = not_applicable(
                "No additional ratio applied to these scores"
            )
        if alg == "cc_mppi":
            update["log_proposal_density"] = not_applicable(
                "Shaped physical population is not a plain Gaussian"
            )
            if "sampled_noise" in record:
                update["log_noise_density"] = trace.put(
                    log_sequence_density(record["sampled_noise"], np.zeros((h, 2)), covariance),
                    "normalized-noise log density",
                )
        components = None
        if cost is not None:
            components = cost_components(
                cost,
                record.get("sampled_states"),
                record.get("sampled_controls"),
                record.get("previous_control", inputs.get("previous_control")),
                record.get("swept_contacts"),
            )
        update["cost_components"] = (
            {key: trace.put(value, "task cost") for key, value in components.items()}
            if components is not None
            else missing("Per-term task costs unavailable")
        )
        if components is not None and "swept_contact" in components and costs is not None:
            np.testing.assert_allclose(sum(components.values()), costs, atol=1e-8, rtol=1e-11)
        plan, updated = record.get(plan_key), record.get(update_key)
        update["updated_plan_external_cost"] = missing(
            "Updated plan/substeps or cost evaluator unavailable"
        )
        if cost is not None and plan is not None and updated is not None and path_key in record:
            physical = cost.model.clip_control(updated * scale)
            contacts = np.array([cost.path_clearance(p) <= 0 for p in record[path_key]])[None]
            terms = cost_components(
                cost,
                plan[None],
                physical[None],
                record.get("previous_control", inputs.get("previous_control")),
                contacts,
            )
            update["updated_plan_external_cost"] = float(sum(value[0] for value in terms.values()))
        if updated is not None:
            update["first_mean_command"] = trace.put(
                np.clip(updated[0] * scale, -scale, scale), CONTROL_UNITS
            )
        updates.append(update)

    if alg == "dial_mpc":
        if "inner_sampled_states" in snap:
            for i in range(len(snap["inner_sampled_states"])):
                record = {
                    key[6:]: value[i] for key, value in snap.items() if key.startswith("inner_")
                }
                record["previous_control"] = inputs.get("previous_control")
                schedule = record["variance_scales"][:, None, None] * np.asarray(ctrl["covariance"])
                population(
                    f"pass_{i}",
                    record,
                    role="refinement",
                    iteration=i,
                    covariance=schedule,
                    correction=False,
                )
        # Never relabel a saved final-only cloud as every internal pass.
    elif alg == "tube_mppi" and "tube_nominal_sampled_states" in snap:
        for role in ("nominal", "actual"):
            prefix = "tube_" + role + "_"
            population(
                role,
                {k[len(prefix) :]: v for k, v in snap.items() if k.startswith(prefix)},
                role=role,
            )
    elif alg == "robust_mppi" and "nominal_samples" in snap:
        nominal = dict(
            snap,
            sampled_states=snap["nominal_samples"],
            costs=snap["nominal_costs"],
            weights=snap["nominal_weights"],
        )
        for key in ("sampled_controls", "latent_controls", "feasible", "swept_contacts"):
            nominal[key] = snap.get("robust_nominal_" + key)
        population("nominal", nominal, role="nominal", correction=False)
        actual = dict(
            snap,
            updated_mean=snap.get("real_updated_mean"),
            nominal_states=snap.get("real_updated_states"),
        )
        if "real_planned_substeps" in snap:
            actual["planned_substeps"] = snap["real_planned_substeps"]
        population("actual", actual, role="actual", correction=False)
    else:
        population(
            "selected" if alg == "tube_mppi" else "main",
            snap,
            role="selected (discarded population missing)" if alg == "tube_mppi" else "main",
            correction=alg in ("it_mppi", "tsallis_vi", "tube_mppi"),
        )

    sampling = dict(
        populations=populations,
        stream=inputs.get("rng_before", missing("Original RNG state not saved")),
        stream_note=(provenance or {}).get("random_pairing", "See recorded seed/stream"),
        model_transitions=scalar("dynamics_evaluations"),
    )
    if alg == "robust_mppi":
        sampling["robust"] = dict(
            candidate_states=take(snap, "nominal_candidates", STATE_UNITS),
            free_energies=take(snap, "candidate_free_energies", "task cost"),
            candidate_feasible=trace.put(
                snap["candidate_free_energies"] <= ctrl.get("free_energy_threshold", 5000.0),
                "boolean",
            )
            if "candidate_free_energies" in snap
            else missing("Candidate energies not recorded"),
            feedback=take(snap, "feedback", "normalized controls"),
            gains=take(snap, "feedback_gains", "normalized control / state units"),
        )
    if alg == "cc_mppi":
        sampling["cc"] = {
            key: take(snap, key, units)
            for key, units in [
                ("auxiliary_states", STATE_UNITS),
                ("feedback_gains", "normalized control / state units"),
                ("linearized_a", "state / state"),
                ("linearized_b", "state / normalized control"),
                ("predicted_covariances", "outer product of state units"),
                ("terminal_bound", "outer product of state units"),
                ("sampled_noise", "normalized controls"),
                ("control_means", "normalized controls"),
            ]
        }
        sampling["cc"]["controlled_samples"] = diag.get(
            "controlled_samples",
            max(2, int((1 - ctrl.get("uncontrolled_fraction", 0.2)) * ctrl["samples"])),
        )
        sampling["cc"]["sampled_terminal_covariance"] = take(
            snap, "empirical_terminal_covariance", "outer product of state units"
        )
    trace.stages["sample_refine"] = StageRecord("available" if populations else "missing", sampling)
    trace.stages["weight_update"] = StageRecord(
        "available" if updates else "missing",
        dict(
            updates=updates,
            first_command_candidate=(
                trace.put(command, CONTROL_UNITS)
                if command is not None
                else missing("Command not saved")
            ),
        ),
    )
    feedback = dict(
        decision_id=trace.decision_id,
        observed_state=stage1["observed_state"],
        next_executed_state=not_applicable("Frozen decision: no execution")
        if mode == "FROZEN_DECISION"
        else missing("Execution not recorded"),
        command_candidate=trace.put(command, CONTROL_UNITS)
        if command is not None
        else missing("Command not saved"),
        independently_executed_covariance=not_applicable(
            "One trajectory does not estimate covariance"
        ),
        computation_time_s=scalar("planning_latency_s"),
    )
    if alg in ("tube_mppi", "robust_mppi"):
        feedback.update(
            {
                key: take(diag, key, units)
                for key, units in [
                    ("nominal_state", STATE_UNITS),
                    ("tracking_error_vector", STATE_UNITS),
                    ("planner_control", CONTROL_UNITS),
                    ("tracking_correction", "normalized controls"),
                    ("applied_correction", CONTROL_UNITS),
                ]
            }
        )
        feedback["nominal_before_reset"] = (
            take(diag, "nominal_before_reset", STATE_UNITS)
            if alg == "tube_mppi"
            else not_applicable("Only Tube resets")
        )
        feedback.update(
            reset_event=diag.get(
                "reset_event",
                not_applicable("Robust candidate selection replaces Tube reset")
                if alg == "robust_mppi"
                else missing("Reset event not saved"),
            ),
            selected_actual=diag.get(
                "selected_actual", not_applicable("Only Tube reset uses selected_actual")
            ),
            selected_candidate=diag.get(
                "nominal_selection", not_applicable("Only Robust searches candidates")
            ),
            selection_infeasible=diag.get(
                "nominal_selection_infeasible", not_applicable("Only Robust searches candidates")
            ),
        )
        if "tracking_correction" in diag:
            feedback["requested_correction_physical"] = trace.put(
                diag["tracking_correction"] * scale, CONTROL_UNITS
            )
    else:
        feedback["ancillary_tracking"] = not_applicable(
            "CC auxiliary feedback shapes proposals, not execution"
            if alg == "cc_mppi"
            else "No ancillary tracker"
        )
    if execution is not None:
        for key, value in execution.items():
            if key in (
                "next_executed_state",
                "predicted_next_state",
                "prediction_error",
                "state_impulse",
            ):
                feedback[key] = trace.put(value, STATE_UNITS)
            elif key in ("executed_control", "final_clipped_command", "actuator_noise"):
                feedback[key] = trace.put(value, CONTROL_UNITS)
            else:
                feedback[key] = native(value)
    inactive = "applied_correction" in diag and np.linalg.norm(diag["applied_correction"]) <= 1e-12
    status = (
        "not_applicable"
        if mode == "FROZEN_DECISION"
        else "mechanism_inactive"
        if inactive
        else "available"
        if execution is not None
        else "missing"
    )
    trace.stages["feedback_execute"] = StageRecord(
        status,
        feedback,
        notes=["Candidate control/feedback may be recorded without execution in frozen mode."],
    )
    for stage in trace.stages.values():

        def collect(value, path="", stage=stage):
            if isinstance(value, dict):
                if value.get("status") == "missing":
                    stage.missing_fields.append(path)
                else:
                    for key, val in value.items():
                        collect(val, f"{path}.{key}".strip("."))
            elif isinstance(value, list):
                for i, val in enumerate(value):
                    collect(val, f"{path}[{i}]")

        collect(stage.data)
    trace.metadata["record_completeness"] = (
        "partial" if any(s.missing_fields for s in trace.stages.values()) else "complete_for_mode"
    )
    return trace.validate()


class TraceRecorder:
    """Observer passed to the existing runner; never calls a controller or RNG itself."""

    def __init__(self, store, experiment_id, source_commit, steps):
        self.store, self.experiment_id, self.source_commit = store, experiment_id, source_commit
        self.steps = frozenset(steps)
        self.traces = []

    def before(self, controller, state, previous):
        return capture_inputs(controller, state, previous)

    def after(self, controller, config, seed, step, inputs, command, execution):
        trace = build_trace(
            config,
            inputs,
            controller.get_rollout_snapshot(),
            controller.get_diagnostics(),
            command,
            experiment_id=self.experiment_id,
            seed=seed,
            planning_step=step,
            mode="CLOSED_LOOP",
            source_commit=self.source_commit,
            execution=execution,
            cost=controller.cost,
            provenance={
                "origin": "live runner observer",
                "rng_after": deepcopy(controller.rng.bit_generator.state),
            },
        )
        self.store.save(trace, config)
        self.traces.append(trace)
        return trace

    def failed(self, config, seed, step, inputs, error):
        trace = build_trace(
            config,
            inputs,
            {},
            {},
            None,
            experiment_id=self.experiment_id,
            seed=seed,
            planning_step=step,
            mode="CLOSED_LOOP",
            source_commit=self.source_commit,
            provenance={"origin": "live numerical failure; no fallback"},
        )
        trace.stages["distributional_objective"].status = "numerical_failure"
        trace.stages["distributional_objective"].data["exception"] = str(error)
        trace.metadata["failure_type"] = type(error).__name__
        self.store.save(trace, config)
        self.traces.append(trace)
