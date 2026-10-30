"""Select existing evidence first, then trace minimal decisions through the existing runner."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.decision_trace import (
    STATE_UNITS,
    TraceStore,
    fingerprint,
    native,
)
from analysis.dynamic_campaign import STATE_FIELDS, run_episode
from analysis.trace_adapters import TraceRecorder, build_trace, capture_inputs
from analysis.variant_campaign import equal_work_samples
from evaluation.runner import git_info, make_controller, write_json
from utils.config import read_yaml


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def source_provenance(path, study):
    return dict(
        origin="retroactive; unrecorded quantities remain missing",
        source_path=str(path),
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        study=study,
        random_pairing="See original mechanism configuration; no new draws",
    )


def frozen_config(frozen, options, algorithm, samples, **changes):
    c = deepcopy(frozen["config"])
    c["controller"].update(options["algorithms"][algorithm])
    c["controller"].update(algorithm=algorithm, samples=samples, snapshot_steps=[0], **changes)
    return c


def historical_inputs(frozen):
    return dict(
        observed_state=np.array(frozen["state"]),
        warm_start=np.array(frozen["warm_start"]),
        previous_control=np.array(frozen["previous_control"]),
        previous_executed_control=np.array(frozen["previous_control"]),
    )


def extract_historical(store, *, source="runs/variants/mechanisms", seed=51000):
    source = Path(source)
    frozen = read("results/variants/mechanisms/V01/frozen_inputs.json")
    options = read_yaml("configs/variants/benchmark.yaml")
    traces = []
    source_commit = read("results/variants/mechanisms/V01/manifest.json")["git"]["commit"]

    def save(
        config,
        inputs,
        snap,
        diag,
        command,
        path,
        study,
        step=0,
        mode="FROZEN_DECISION",
        execution=None,
        suffix="",
        enrich=None,
        reuse=(),
        seed_override=None,
    ):
        t = build_trace(
            config,
            inputs,
            snap,
            diag,
            command,
            experiment_id="retro-" + study + suffix,
            seed=seed if seed_override is None else seed_override,
            planning_step=step,
            mode=mode,
            source_commit=source_commit,
            execution=execution,
            provenance=source_provenance(path, study),
        )
        if enrich:
            enrich(t)
        store.save(t, config, reuse_sources=[path, *reuse] if path.suffix == ".npz" else reuse)
        traces.append(t)
        return t

    for algorithm in ["it_mppi", "tsallis_vi"]:
        f = frozen["two_route_obstacle"]
        c = frozen_config(f, options, algorithm, 384)
        path = source / "V01/two_route_obstacle" / f"{seed}_{algorithm}.npz"
        z = arrays(path)
        frame = pd.read_csv("results/variants/mechanisms/V01/summary.csv")
        row = frame[
            (frame.seed == seed)
            & (frame.scene == "two_route_obstacle")
            & (frame.algorithm == algorithm)
        ].iloc[0]
        diag = {
            k: v for k, v in row.to_dict().items() if isinstance(v, (float, int)) and np.isfinite(v)
        }
        diag["dynamics_evaluations"] = (384 + 1) * 20

        def reference(t):
            d = read("results/variants/mechanisms/V01/plot_data.json")
            t.stages["distributional_objective"].data["related_scalar_reference"] = dict(
                objective="J(z)=0.5*(z-1)^2; prior N(0,1); scale 0.8; Tsallis r=1.5",
                scope="Separate tractable objective, not a reference optimum for vehicle controls",
                grid=t.put(d["grid"]),
                targets={
                    k: dict(density=t.put(v["density"]), mean=v["mean"])
                    for k, v in d["reference"].items()
                },
            )

        save(
            c,
            historical_inputs(f),
            z,
            diag,
            z["command"],
            path,
            "V01",
            step=f["step"],
            enrich=reference,
        )
    f = frozen["two_route_obstacle"]
    c = frozen_config(f, options, "dial_mpc", 95)
    path = source / "V02" / f"{seed}_dial_mpc.npz"
    z = arrays(path)
    save(
        c,
        historical_inputs(f),
        z,
        {"dynamics_evaluations": 7680},
        z["command"],
        path,
        "V02",
        step=f["step"],
    )

    for threshold in (0, 450):
        folder = source / f"V03/impulse_1/tube_{threshold}/{seed}"
        c = read_yaml(folder / "config.yaml")
        states = pd.read_csv(folder / "trajectory.csv")[STATE_FIELDS].to_numpy()
        controls = pd.read_csv(folder / "controls.csv").to_numpy()
        diags = pd.read_csv(folder / "diagnostics.csv")
        result = read(folder / "result.json")
        for step, row in diags.iterrows():
            diag = {k: v for k, v in row.items() if np.isscalar(v) and np.isfinite(v)}
            for key, size in [
                ("nominal_state", 7),
                ("nominal_before_reset", 7),
                ("tracking_error_vector", 7),
                ("tracking_correction", 2),
                ("planner_control", 2),
                ("applied_correction", 2),
            ]:
                if key + "_0" in row:
                    diag[key] = np.array([row[f"{key}_{i}"] for i in range(size)])
            path = folder / f"snapshots/step_{step:04d}.npz"
            z = arrays(path) if path.exists() else {}
            inputs = dict(
                observed_state=states[step],
                previous_executed_control=controls[step - 1] if step else np.zeros(2),
            )
            if "proposal_mean" in z:
                inputs["warm_start"] = z["proposal_mean"]
            if "nominal_before_reset" in diag:
                inputs["nominal_state"] = diag["nominal_before_reset"]
            delta = np.array(result["disturbance_delta"]) if step == 8 else np.zeros(7)
            execution = dict(
                next_executed_state=states[step + 1],
                executed_control=controls[step],
                final_clipped_command=controls[step],
                state_impulse=delta,
                disturbance_applied=step == 8,
                collision=bool(row.clearance <= 0),
                time_s=step * 0.15,
                next_time_s=(step + 1) * 0.15,
            )
            save(
                c,
                inputs,
                z,
                diag,
                controls[step],
                path if path.exists() else folder / "diagnostics.csv",
                "V03",
                step=step,
                mode="CLOSED_LOOP",
                execution=execution,
                suffix=f"-tube{threshold}",
            )

    f = frozen["open_space"]
    for threshold in (100, 5000):
        c = frozen_config(f, options, "robust_mppi", 128, free_energy_threshold=threshold)
        path = source / "V04" / f"1_{threshold}_{seed}.npz"
        z = arrays(path)
        frame = pd.read_csv("results/variants/mechanisms/V04/summary.csv")
        row = frame[
            (frame.seed == seed) & (frame.threshold == threshold) & (frame.impulse == 1)
        ].iloc[0]
        inputs = historical_inputs(f)
        inputs.update(
            observed_state=z["current_state"],
            nominal_state=np.array(f["state"]),
            previous_nominal=np.array(f["state"]),
            previous_plan=np.array(f["warm_start"]),
        )
        # Recover only saved fields; throttle correction is absent.
        diag = dict(
            nominal_selection=int(row.selected_candidate),
            nominal_selection_infeasible=bool(row.infeasible),
            nominal_state=z["nominal_candidates"][int(row.selected_candidate)],
            sampled_feedback_rms=float(row.feedback_rms),
            dynamics_evaluations=int(row.model_transitions),
        )

        def robust_extra(t, row=row, z=z, threshold=threshold):
            t.stages["feedback_execute"].data["recorded_steering_decomposition"] = {
                key: float(row[key])
                for key in [
                    "planner_steering",
                    "requested_tracking_steering",
                    "applied_tracking_steering",
                    "command_steering",
                ]
            }
            t.stages["sample_refine"].data["robust"]["candidate_feasible"] = t.put(
                z["candidate_free_energies"] <= threshold, "boolean"
            )

        save(
            c,
            inputs,
            z,
            diag,
            z["command"],
            path,
            "V04",
            step=f["step"],
            suffix=f"-alpha{threshold}",
            enrich=robust_extra,
        )

    f = frozen["narrow_passage"]
    c = frozen_config(f, options, "cc_mppi", 384)
    path = source / "V05/narrow_passage/planner.npz"
    z = arrays(path)
    ensemble_path = path.parent / "1_steered.npz"
    ensemble = arrays(ensemble_path)
    frame = pd.read_csv("results/variants/mechanisms/V05/summary.csv")
    row = frame[
        (frame.scene == "narrow_passage")
        & (frame.noise_fraction == 1)
        & (frame["mode"] == "steered")
    ].iloc[0]

    def independent(t):
        states = ensemble["executed_states"]
        reference = z["reference_states"]
        error = states - reference
        error[..., 2] = (error[..., 2] + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(
            np.stack([np.cov(error[:, i], rowvar=False) for i in range(21)]),
            ensemble["empirical_covariances"],
            atol=1e-12,
        )
        t.stages["feedback_execute"].data["related_fixed_policy_experiment"] = dict(
            source=str(ensemble_path),
            source_sha256=hashlib.sha256(ensemble_path.read_bytes()).hexdigest(),
            policy="Frozen warm-start mean and gains; NOT execution of the optimized mean",
            seeds=list(range(56000, 56256)),
            noise_fraction=1.0,
            collisions=int(row.collisions),
            states=t.put(states, STATE_UNITS),
            reference=t.put(reference, STATE_UNITS),
            predicted_covariances=t.put(
                ensemble["predicted_covariances"], "outer product of state units"
            ),
            empirical_covariances=t.put(
                ensemble["empirical_covariances"], "outer product of state units"
            ),
            terminal_bound=t.put(ensemble["terminal_bound"], "outer product of state units"),
            time=t.put(np.arange(21) * c["vehicle"]["dt"], "s"),
        )

    # The original planner archive did not retain its command or solver status.
    save(
        c,
        historical_inputs(f),
        z,
        {"controlled_samples": int(0.8 * 384)},
        None,
        path,
        "V05",
        step=f["step"],
        enrich=independent,
        reuse=[ensemble_path],
        seed_override=55000,
    )
    return traces


def trace_frozen(
    config, inputs, seed, store, experiment_id, *, standard_normals=None, source_commit=None
):
    c = deepcopy(config)
    c["controller"]["snapshot_steps"] = [0]
    ctrl = make_controller(c)
    state = np.array(inputs["observed_state"])
    ctrl.reset(state, seed)
    ctrl.mean = np.array(inputs["warm_start"]).copy()
    ctrl.previous_control = np.array(inputs["previous_control"]).copy()
    for key in ("nominal_state", "previous_nominal", "previous_plan", "nominal_previous"):
        if key in inputs and hasattr(ctrl, key):
            setattr(ctrl, key, np.array(inputs[key]).copy())
    before = capture_inputs(ctrl, state, inputs.get("previous_executed_control"))
    command, _ = ctrl.plan(state, standard_normals=standard_normals)
    t = build_trace(
        c,
        before,
        ctrl.get_rollout_snapshot(),
        ctrl.get_diagnostics(),
        command,
        experiment_id=experiment_id,
        seed=seed,
        planning_step=0,
        mode="FROZEN_DECISION",
        source_commit=source_commit or git_info()["commit"],
        cost=ctrl.cost,
        provenance={
            "origin": "selected live frozen decision",
            "rng_after": deepcopy(ctrl.rng.bit_generator.state),
            "random_pairing": "Shared IT/Tsallis standard normals; others own internal streams",
            "shared_normals_hash": fingerprint(standard_normals)
            if standard_normals is not None
            else None,
        },
    )
    store.save(t, c)
    return t


def minimal_live(store, spec, raw):
    options = read_yaml("configs/variants/benchmark.yaml")
    options["transition_budget"] = spec["frozen_transition_cap"]
    f = read(spec["frozen_source"])[spec["frozen_scene"]]
    if f["step"] != spec["frozen_step"] or f["source_seed"] != spec["source_seed"]:
        raise ValueError("Frozen source selection differs from predeclared rule")
    count = equal_work_samples(options, "it_mppi")
    normals = np.random.default_rng(spec["seed"]).standard_normal((count, 20, 2))
    traces = []
    for alg in spec["algorithms"]:
        c = frozen_config(f, options, alg, equal_work_samples(options, alg))
        c["trace_transition_cap"] = spec["frozen_transition_cap"]
        t = trace_frozen(
            c,
            historical_inputs(f),
            spec["seed"],
            store,
            spec["experiment_id"] + "-frozen",
            standard_normals=normals if alg in ("it_mppi", "tsallis_vi") else None,
        )
        traces.append(t)
    # A short window of the already-defined positive-impulse mechanism task, not a new benchmark.
    base = read_yaml(spec["closed_loop"]["source_config"])
    base["execution"]["max_steps"] = spec["closed_loop"]["steps"]
    for alg in spec["algorithms"]:
        c = deepcopy(base)
        common = deepcopy(f["config"]["controller"])
        common.update(options["algorithms"][alg])
        common.update(algorithm=alg, samples=spec["closed_loop"]["samples_per_population"])
        if alg == "tube_mppi":
            common["reset_threshold"] = spec["closed_loop"]["tube_reset_threshold"]
        c["controller"] = common
        recorder = TraceRecorder(
            store,
            spec["experiment_id"] + "-closed",
            git_info()["commit"],
            spec["closed_loop"]["capture_steps"],
        )
        run_episode(c, alg, spec["seed"], Path(raw) / "episodes" / alg, trace_recorder=recorder)
        traces.extend(recorder.traces)
    return traces


def brief(trace):
    return dict(
        trace.metadata,
        stages={
            k: dict(status=s.status, missing_fields=s.missing_fields)
            for k, s in trace.stages.items()
        },
        arrays=len(trace.arrays),
        raw_trace=f"{trace.decision_id}/trace.json",
    )


def compact(trace):
    """Only small numerical reductions/selected paths; no full sampled clouds in Git."""
    get = trace.get
    m = trace.metadata
    one = dict(
        decision_id=trace.decision_id,
        configuration_hash=m["configuration_hash"],
        algorithm=m["algorithm"],
        mode=m["mode"],
        experiment_id=m["experiment_id"],
        step=m["planning_step"],
        seed=m["seed"],
        observed=get(trace.stages["dynamics_task"].data["observed_state"]),
        objective=deepcopy(trace.stages["distributional_objective"].data),
        stage_status={k: s.status for k, s in trace.stages.items()},
        stages_missing={k: s.missing_fields for k, s in trace.stages.items()},
        populations=[],
        updates=[],
    )

    # Resolve small objective references (support masks and tractable scalar densities).
    def resolve(value):
        if isinstance(value, dict):
            if "array" in value:
                return get(value)
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    one["objective"] = resolve(one["objective"])
    sample = trace.stages["sample_refine"].data
    one["model_transitions"] = sample["model_transitions"]
    for p in sample["populations"]:
        states, latent = get(p["states"]), get(p["latent_controls"])
        record = dict(
            population_id=p["population_id"],
            role=p["role"],
            samples=p["samples"],
            iteration=p.get("iteration"),
            mean_xy=states[:, :, :2].mean(axis=0),
            selected_xy=states[:8, :, :2],
            first_latent=latent[:, 0] if latent is not None else None,
            model_transitions=p.get("model_transitions"),
        )
        one["populations"].append(record)
    for u in trace.stages["weight_update"].data["updates"]:
        record = {
            k: resolve(u.get(k))
            for k in [
                "population_id",
                "role",
                "raw_costs",
                "weights",
                "ess",
                "maximum_weight",
                "entropy",
                "updated_latent",
                "nominal_prediction",
                "first_mean_command",
                "updated_plan_external_cost",
                "log_scores",
                "density_correction",
                "cost_components",
            ]
        }
        # JSON cannot represent -inf; unsupported scores remain explicitly null with support masks.
        if isinstance(record["log_scores"], np.ndarray):
            record["log_scores"] = [
                float(x) if np.isfinite(x) else None for x in record["log_scores"]
            ]
        comp = record["cost_components"]
        if isinstance(comp, dict) and "status" not in comp:
            record["cost_components"] = {k: float(np.mean(v)) for k, v in comp.items()}
        one["updates"].append(record)
    one["command"] = get(trace.stages["weight_update"].data["first_command_candidate"])
    feedback = resolve(trace.stages["feedback_execute"].data)
    related = feedback.get("related_fixed_policy_experiment")
    if related:
        states = np.asarray(related.pop("states"))
        related["mean_states"] = states.mean(axis=0)
        related["selected_paths"] = states[:16, :, :2]
        related["replicates"] = len(states)
    one["feedback"] = feedback
    if "robust" in sample:
        one["robust"] = resolve(sample["robust"])
    if "robust" in one:
        feedback_samples = one["robust"].pop("feedback")
        if isinstance(feedback_samples, np.ndarray):
            one["robust"]["sampled_feedback_rms"] = float(np.sqrt(np.mean(feedback_samples**2)))
    if "cc" in sample:
        one["cc"] = {
            k: resolve(v)
            for k, v in sample["cc"].items()
            if k in ["predicted_covariances", "terminal_bound", "controlled_samples"]
        }
    return native(one)


def run(
    output="results/variants/traces",
    raw="runs/variants/traces",
    *,
    live=True,
    config="configs/variants/traces.yaml",
):
    output, raw = Path(output), Path(raw)
    output.mkdir(parents=True, exist_ok=False)
    if raw.exists():
        raise FileExistsError("Use a new trace group directory; do not overwrite traces")
    store = TraceStore(raw)
    spec = read_yaml(config)
    manifest = dict(
        schema_version="1.0",
        git=git_info(),
        configuration=spec,
        raw=str(raw),
        baseline_tests="232 passed before refactoring",
        complete=False,
        selection="Fixed first seed and source step; extraction precedes any new simulation",
    )
    write_json(output / "manifest.json", manifest)
    traces = extract_historical(store, source=spec["retroactive"]["source"], seed=spec["seed"])
    print(f"Extracted {len(traces)} historical records before new simulation.", flush=True)
    if live:
        traces += minimal_live(store, spec, raw)
    # Round-trip verification also checks external source/config/NPZ hashes.
    for t in traces:
        loaded = store.load(t.decision_id)
        assert loaded.document() == t.document()
    write_json(
        output / "index.json",
        dict(schema_version="1.0", raw=str(raw), traces=[brief(t) for t in traces]),
    )
    write_json(
        output / "plot_data.json",
        dict(
            schema_version="1.0",
            traces=[compact(t) for t in traces],
            scenes={
                t.metadata["configuration_hash"]: read(
                    store.root / "configurations" / (t.metadata["configuration_hash"] + ".json")
                )["scenario"]
                for t in traces
            },
        ),
    )
    manifest.update(
        complete=True,
        traces=len(traces),
        retroactive=sum(t.metadata["experiment_id"].startswith("retro-") for t in traces),
        fresh_frozen=sum(t.metadata["experiment_id"].endswith("-frozen") for t in traces),
        fresh_closed_decisions=sum(t.metadata["experiment_id"].endswith("-closed") for t in traces),
    )
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "validation.json",
        dict(
            schema_version="1.0",
            traces_validated=len(traces),
            array_roundtrips=True,
            configuration_hashes=True,
            weight_normalization=True,
            covariance_psd=True,
            stage_classification=True,
            limitations="Missing history retained; instrumented timing is not a speed ranking.",
        ),
    )
    from analysis.trace_plots import plot_all

    if live:
        plot_all(output)
    return manifest
