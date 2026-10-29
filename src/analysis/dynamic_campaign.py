"""Seven-state execution adapter using the existing planner and artifact framework."""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.artifacts import begin, fingerprint
from analysis.trajectory_statistics import first_crossing
from dynamics.bicycle import wrap_angle
from dynamics.dynamic_vehicle import DynamicVehicle
from evaluation.metrics import binomial_interval, mean_interval
from evaluation.runner import make_controller, write_json
from scenarios.dynamic_scenes import make_s_turn, make_scene
from utils.config import read_yaml

STATE_FIELDS = ["px", "py", "heading", "roll", "vx", "vy", "yaw_rate"]
CONTROL_FIELDS = ["steering", "throttle"]
PROPOSALS = ["standard", "isotropic", "diagonal", "full"]


def configuration():
    c = read_yaml("configs/seven_state/campaign.yaml")
    c["vehicle"] = read_yaml("configs/seven_state/vehicle.yaml")
    c["control_fields"] = CONTROL_FIELDS
    return c


def assert_validated():
    import json

    path = Path("runs/seven_state/validation/dynamics_v1/validation.json")
    if not path.exists() or not json.loads(path.read_text())["passed"]:
        raise RuntimeError("Seven-state model must pass formal validation before MPC evaluation")


def resolved_case(config, name, samples=None, clearance=None):
    resolved = deepcopy(config)
    if samples is not None:
        resolved["controller"]["samples"] = int(samples)
    if name == "s_turn":
        scenario, geometry, states, controls, feasible = make_s_turn(resolved, clearance)
        if not feasible["feasible"]:
            raise RuntimeError("S-turn physical feasibility unresolved")
    else:
        scenario, geometry = make_scene(resolved, name)
        states = controls = None
        feasible = None
    resolved["scenario"] = scenario
    return resolved, geometry, states, controls, feasible


def reached(state, geometry, config):
    _, progress, _ = geometry.project(state)
    goal = geometry.goal
    tol = config["execution"]
    # Finish region is explicit; projection is clipped, so recovery in the runout is allowed.
    return bool(
        progress >= geometry.center.arc[-1] - tol["position_tolerance"]
        and abs(wrap_angle(state[2] - goal[2])) <= tol["heading_tolerance"]
        and abs(state[4] - goal[4]) <= tol["speed_tolerance"]
        and abs(state[1] - goal[1]) <= tol["goal_lateral_tolerance"]
        and abs(state[5]) <= tol["lateral_velocity_tolerance"]
        and abs(state[6]) <= tol["yaw_rate_tolerance"]
        and abs(state[3]) <= tol["roll_tolerance"]
    )


def run_episode(config, proposal, seed, output, *, capture_steps=None, freeze_rule=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    resolved = deepcopy(config)
    resolved["controller"].update(
        kind=proposal,
        covariance=config["proposals"][proposal],
        saved_rollouts=0,
        snapshot_steps=(capture_steps or [])
        if seed in {42, 31000, 32000, config["frozen"]["seed"]}
        else [],
    )
    controller = make_controller(resolved)
    geometry = controller.cost.geometry
    model = controller.model
    execution = DynamicVehicle(
        **(config["vehicle"] | config["scenario"].get("execution_overrides", {}))
    )
    state = np.array(config["scenario"]["initial_state"])
    controller.reset(state, np.random.SeedSequence([4400, seed]))
    disturbance_rng = np.random.default_rng(np.random.SeedSequence([4401, seed]))
    factor = float(1 + 0.1 * disturbance_rng.normal())
    event = config["scenario"].get("disturbance")
    event_delta = np.array(event["delta"]) * factor if event else np.zeros(7)
    states, controls, logs = [state.copy()], [], []
    previous = np.zeros(2)
    total = 0.0
    frozen = None
    for step in range(config["execution"]["max_steps"]):
        if freeze_rule is not None:
            candidate = dict(
                name=config["scenario"]["name"],
                config=deepcopy(resolved),
                state=state.tolist(),
                warm_start=controller.mean.tolist(),
                previous_control=controller.previous_control.tolist(),
                step=step,
                source_seed=seed,
                optimizer_entropy=[4400, seed],
                disturbance_entropy=[4401, seed],
                disturbance_delta=event_delta.tolist(),
                disturbance_applied=bool(event and step > event["step"]),
            )
            if freeze_rule(candidate):
                frozen = candidate
                break
        control = controller.command(state.copy())
        diag = controller.diagnostics
        nxt, path = execution.step(state, control, return_path=True)
        if event and step == event["step"]:
            nxt += event_delta
            nxt[2] = wrap_angle(nxt[2])
            nxt[4] = np.clip(nxt[4], 0, execution.velocity_max)
            path = np.concatenate([path, nxt[None]], axis=0)
        gap = float(controller.cost.path_clearance(path))
        expected = model.step(state, control)
        error = nxt - expected
        error[2] = wrap_angle(error[2])
        total += float(
            controller.cost.running(state, control, previous)
        ) + model.dt * controller.cost.weights["collision"] * (gap <= 0)
        logs.append(
            dict(
                step=step,
                ess=diag["ess"],
                weight_entropy=diag["weight_entropy"],
                max_weight=diag["max_weight"],
                feasible_rollout_fraction=diag["feasible_rollout_fraction"],
                saturation_fraction=diag["saturation_fraction"],
                minimum_sample_cost=diag["minimum_sample_cost"],
                planning_latency_s=diag["planning_latency_s"],
                dynamics_evaluations=diag["dynamics_evaluations"],
                rhs_evaluations=diag["rhs_evaluations"],
                clearance=gap,
                cumulative_cost=total,
                prediction_error_norm=float(np.linalg.norm(error)),
            )
        )
        snap = controller.get_rollout_snapshot()
        if snap is not None:
            folder = output / "snapshots"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(
                folder / f"step_{step:04d}.npz",
                **{
                    k: snap[k]
                    for k in (
                        "sampled_states",
                        "sampled_controls",
                        "latent_controls",
                        "costs",
                        "weights",
                        "proposal_mean",
                        "updated_mean",
                        "nominal_states",
                        "current_state",
                        "log_base_over_proposal",
                    )
                },
            )
        states.append(nxt)
        controls.append(control)
        state = nxt
        previous = control
        if gap <= 0 or reached(state, geometry, config):
            break
    states, controls = np.array(states), np.array(controls)
    trajectory = pd.DataFrame(states, columns=STATE_FIELDS)
    trajectory.insert(0, "time_s", np.arange(len(states)) * model.dt)
    distance, progress, _ = geometry.project(states)
    trajectory["progress_m"] = progress
    trajectory["cross_track_m"] = distance
    trajectory.to_csv(output / "trajectory.csv", index=False)
    pd.DataFrame(controls, columns=CONTROL_FIELDS).to_csv(output / "controls.csv", index=False)
    diag = pd.DataFrame(logs)
    diag.to_csv(output / "diagnostics.csv", index=False)
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    if frozen is not None:
        frozen["source_episode"] = str(output)
        write_json(output / "frozen.json", frozen)
        return frozen
    if freeze_rule is not None:
        raise RuntimeError("Predefined frozen selection found no state")
    total += float(controller.cost.terminal(state))
    collision = bool((diag.clearance <= 0).any())
    success = reached(state, geometry, config) and not collision
    station = (
        first_crossing(states[:, :2], 4.0)
        if config["scenario"]["name"] == "two_route_obstacle"
        else None
    )
    result = dict(
        station_crossing_y_m=station,
        route_class=("not_reached" if station is None else ("upper" if station > 0 else "lower"))
        if config["scenario"]["name"] == "two_route_obstacle"
        else "not_assessed",
        success=success,
        collision=collision,
        termination="completed" if success else ("collision" if collision else "censored_timeout"),
        steps=len(controls),
        max_progress_fraction=float(progress.max() / geometry.center.arc[-1]),
        cross_track_rmse_m=float(np.sqrt(np.mean(distance**2))),
        goal_position_error_m=float(np.linalg.norm(state[:2] - geometry.goal[:2])),
        goal_heading_error_rad=float(abs(wrap_angle(state[2] - geometry.goal[2]))),
        goal_speed_m_s=float(state[4]),
        minimum_clearance_m=float(diag.clearance.min())
        if np.isfinite(diag.clearance).any()
        else None,
        time_to_goal_s=len(controls) * model.dt if success else None,
        executed_duration_s=len(controls) * model.dt,
        cumulative_evaluation_cost=total,
        mean_ess=float(diag.ess.mean()),
        mean_weight_entropy=float(diag.weight_entropy.mean()),
        mean_feasible_rollout_fraction=float(diag.feasible_rollout_fraction.mean()),
        steering_saturation_fraction=float(
            np.mean(abs(controls[:, 0]) >= model.steering_limit - 1e-9)
        ),
        throttle_saturation_fraction=float(
            np.mean(abs(controls[:, 1]) >= model.throttle_limit - 1e-9)
        ),
        control_effort=float(model.dt * np.sum((controls / model.control_scale) ** 2)),
        max_abs_vy=float(abs(states[:, 5]).max()),
        max_abs_yaw_rate=float(abs(states[:, 6]).max()),
        max_abs_roll=float(abs(states[:, 3]).max()),
        mean_planning_latency_s=float(diag.planning_latency_s.mean()),
        model_transitions=int(diag.dynamics_evaluations.sum()),
        rhs_evaluations=int(diag.rhs_evaluations.sum()),
        execution_transitions=len(controls),
        tracking_check_transitions=len(controls),
        prediction_error_rmse=float(np.sqrt(np.mean(diag.prediction_error_norm**2))),
        optimizer_entropy=[4400, seed],
        disturbance_entropy=[4401, seed],
        disturbance_delta=event_delta.tolist(),
        planning_parameters=config["vehicle"],
        execution_parameters=config["vehicle"] | config["scenario"].get("execution_overrides", {}),
    )
    write_json(output / "result.json", result)
    return result, trajectory, diag


def summary(frame, output, groups):
    rows = []
    pairs = []
    metrics = [
        "max_progress_fraction",
        "cross_track_rmse_m",
        "minimum_clearance_m",
        "time_to_goal_s",
        "cumulative_evaluation_cost",
        "mean_ess",
        "mean_weight_entropy",
        "mean_feasible_rollout_fraction",
        "control_effort",
        "steering_saturation_fraction",
        "throttle_saturation_fraction",
        "max_abs_vy",
        "max_abs_yaw_rate",
        "max_abs_roll",
        "mean_planning_latency_s",
        "model_transitions",
        "rhs_evaluations",
        "prediction_error_rmse",
    ]
    for keys, group in frame.groupby(groups, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(groups, keys, strict=True))
        ref = group[group.proposal == "standard"].set_index("seed")
        for proposal, sub in group.groupby("proposal", sort=False):
            row = base | dict(
                proposal=proposal,
                n=len(sub),
                completions=int(sub.success.sum()),
                collisions=int(sub.collision.sum()),
                censored=int((sub.termination == "censored_timeout").sum()),
            )
            for key in ["success", "collision", *metrics]:
                stat = (binomial_interval if key in ["success", "collision"] else mean_interval)(
                    sub[key].to_numpy(dtype=float)
                )
                row.update({f"{key}_{k}": v for k, v in stat.items()})
            rows.append(row)
            other = sub.set_index("seed")
            for key in [
                "success",
                "collision",
                "cumulative_evaluation_cost",
                "max_progress_fraction",
            ]:
                delta = other[key].astype(float) - ref[key].astype(float)
                if delta.isna().any():
                    raise ValueError("Seed pairs missing")
                pairs.append(base | dict(proposal=proposal, metric=key, **mean_interval(delta)))
    table = pd.DataFrame(rows)
    table.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "paired_differences.csv", index=False)
    return table


def job(args):
    config, case, proposal, seed, cohort, output, capture = args
    result, path, diag = run_episode(
        config,
        proposal,
        seed,
        Path(output) / "episodes" / case / proposal / f"seed_{seed}",
        capture_steps=capture,
    )
    for key in [
        "optimizer_entropy",
        "disturbance_entropy",
        "disturbance_delta",
        "planning_parameters",
        "execution_parameters",
    ]:
        result.pop(key)
    row = dict(
        case=case,
        scenario=config["scenario"]["name"],
        clearance=config["scenario"].get("clearance", -1.0),
        samples=config["controller"]["samples"],
        horizon=config["controller"]["horizon"],
        proposal=proposal,
        seed=seed,
        cohort=cohort,
        reused=False,
        episode_path=str(Path(output) / "episodes" / case / proposal / f"seed_{seed}"),
        shared_condition_sha256=fingerprint(config),
        **result,
    )
    clean = diag.replace([np.inf, -np.inf], np.nan)
    representative = (
        (path.to_dict("list"), clean.astype(object).where(pd.notna(clean), None).to_dict("list"))
        if seed in [42, 31000, 32000]
        else None
    )
    return row, representative


def run_closed(config, output, *, stage="q04", pilot=False, additional_seeds=None):
    assert_validated()
    config = deepcopy(config)
    config["phase"] = "pilot" if pilot else "evaluation"
    root, manifest = begin(output, config, "seven_state_" + stage)
    core = hashlib.sha256()
    for file in [
        "src/controllers/mppi.py",
        "src/dynamics/dynamic_vehicle.py",
        "src/costs/dynamic_vehicle.py",
        "src/scenarios/dynamic_scenes.py",
        "src/sampling/gaussian.py",
    ]:
        core.update(Path(file).read_bytes())
    manifest["engine_source_sha256"] = core.hexdigest()
    write_json(root / "manifest.json", manifest)
    cases = []
    if stage == "q04":
        for name in config["scenarios"]:
            c, g, *_ = resolved_case(config, name)
            cases.append((name, c, g, None))
    elif stage == "q05":
        for value in config["s_turn"]["clearance_levels"]:
            c, g, states, controls, check = resolved_case(config, "s_turn", clearance=value)
            cases.append((f"clearance_{value:g}", c, g, check))
    elif stage in ["budget_q04", "budget_q05"]:
        for k in config["budget_counts"]:
            name = config["budget_q04_scenario"] if stage == "budget_q04" else "s_turn"
            c, g, states, controls, check = resolved_case(
                config, name, samples=k, clearance=config["budget_q05_clearance"]
            )
            cases.append((f"K{k}", c, g, check))
    else:
        raise ValueError("Unknown campaign phase")
    write_json(
        root / "conditions.json",
        [dict(case=name, config=c, feasibility=f) for name, c, g, f in cases],
    )
    write_json(
        root / "geometry.json",
        {
            name: dict(
                obstacles=getattr(g, "obstacles", []),
                upper=getattr(g, "upper", np.empty((0, 2))).tolist(),
                lower=getattr(g, "lower", np.empty((0, 2))).tolist(),
                centerline=g.center.points.tolist(),
                goal=g.goal.tolist(),
                vehicle_radius=g.vehicle_radius,
            )
            for name, c, g, f in cases
        },
    )
    parent = None
    parent_frame = None
    seed_options = config["holdout_seeds"] if stage == "budget_q04" else config["s_turn_seeds"]
    reuse_case = None
    if stage.startswith("budget_"):
        q = "Q04" if stage == "budget_q04" else "Q05"
        parent = Path("runs/seven_state") / q / f"seven_state_{q.lower()}_v1"
        parent_manifest = json.loads((parent / "manifest.json").read_text())
        if not parent_manifest["complete"]:
            raise RuntimeError("Budget parent is incomplete")
        if parent_manifest["engine_source_sha256"] != manifest["engine_source_sha256"]:
            raise RuntimeError("Budget reference used a different model/controller implementation")
        parent_frame = pd.read_csv(parent / "per_seed.csv")
        if stage == "budget_q05":
            additional_seeds = parent_manifest["cohorts"]["holdout"]
        reuse_case = f"K{config['controller']['samples']}"
    cohorts = (
        {"pilot": [42]}
        if pilot
        else (
            {
                "compatibility": list(range(42, 62)),
                "holdout": list(
                    range(
                        config["holdout_seeds"]["start"],
                        config["holdout_seeds"]["start"] + config["holdout_seeds"]["count"],
                    )
                ),
            }
            if stage == "q04"
            else {
                "holdout": additional_seeds
                or list(
                    range(
                        seed_options["start"],
                        seed_options["start"] + seed_options["count"],
                    )
                )
            }
        )
    )
    args = [
        (c, name, p, seed, cohort, str(root), config["execution"]["snapshot_steps"])
        for name, c, g, f in cases
        if name != reuse_case
        for cohort, seeds in cohorts.items()
        for p in PROPOSALS
        for seed in seeds
    ]
    rows = []
    paths = {}
    diagnostics = {}
    if parent_frame is not None:
        selected = parent_frame[
            (parent_frame.samples == config["controller"]["samples"])
            & (parent_frame.cohort == "holdout")
        ]
        if stage == "budget_q04":
            selected = selected[selected.scenario == config["budget_q04_scenario"]]
        else:
            selected = selected[selected.clearance == config["budget_q05_clearance"]]
        expected = cohorts["holdout"]
        selected = selected[selected.seed.isin(expected)].copy()
        c = next(c for name, c, g, f in cases if name == reuse_case)
        if selected.shared_condition_sha256.nunique() != 1 or selected.shared_condition_sha256.iloc[
            0
        ] != fingerprint(c):
            raise RuntimeError("Cannot reuse a different physical/controller condition")
        for _proposal, g in selected.groupby("proposal"):
            if sorted(g.seed) != expected:
                raise RuntimeError("Budget parent seeds are incomplete")
        old_case = selected.case.iloc[0]
        selected["case"] = reuse_case
        selected["reused"] = True
        rows = selected.to_dict("records")
        for filename, target in [
            ("representative_paths.json", paths),
            ("representative_diagnostics.json", diagnostics),
        ]:
            original = json.loads((parent / filename).read_text())
            for key, value in original.items():
                if key.startswith(old_case + "__") and key.endswith("__holdout"):
                    target[reuse_case + key[len(old_case) :]] = value
        manifest["reused_primary_run"] = str(parent)
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        for row, record in pool.map(job, args):
            row["reused"] = False
            rows.append(row)
            if record is not None:
                key = f"{row['case']}__{row['proposal']}__{row['cohort']}"
                paths[key], diagnostics[key] = record
            if len(rows) % 20 == 0:
                pd.DataFrame(rows).to_csv(root / "per_seed.csv", index=False)
                print(f"{stage}: {len(rows)}/{len(args)} episodes", flush=True)
    if stage == "q05" and not pilot:
        preliminary = pd.DataFrame(rows)
        widths = [
            binomial_interval(g.success.astype(float))
            for _, g in preliminary.groupby(["case", "proposal"])
        ]
        expand = any(
            w["ci95_high"] - w["ci95_low"] > config["s_turn_precision_width"] for w in widths
        )
        if expand:
            old = cohorts["holdout"]
            extra = list(
                range(old[-1] + 1, config["s_turn_seeds"]["start"] + config["s_turn_max_seeds"])
            )
            more = [
                (c, name, p, seed, "holdout", str(root), config["execution"]["snapshot_steps"])
                for name, c, g, f in cases
                for p in PROPOSALS
                for seed in extra
            ]
            with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
                for row, _record in pool.map(job, more):
                    rows.append(row)
                    if len(rows) % 20 == 0:
                        pd.DataFrame(rows).to_csv(root / "per_seed.csv", index=False)
                        print(f"q05 precision expansion: {len(rows)} episodes", flush=True)
            cohorts["holdout"] += extra
            manifest["precision_expansion"] = dict(
                trigger="any marginal Wilson width > .36",
                initial_n=len(old) - len(extra),
                final_n=len(cohorts["holdout"]),
            )
    return finish_records(root, config, rows, paths, diagnostics, manifest, cohorts)


def finish_records(root, config, rows, paths, diagnostics, manifest, cohorts):
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "per_seed.csv", index=False)
    summary(frame, root, ["case", "scenario", "clearance", "samples", "cohort"])
    write_json(root / "representative_paths.json", paths)
    write_json(root / "representative_diagnostics.json", diagnostics)
    manifest.update(
        complete=True,
        run_id=root.name,
        dynamics_model="dynamic_single_track_7",
        state_fields=STATE_FIELDS,
        control_fields=CONTROL_FIELDS,
        sampling_dimension=2 * config["controller"]["horizon"],
        episodes=len(frame),
        cohorts=cohorts,
        episodes_executed_new=int(
            (
                ~frame.get("reused", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
            ).sum()
        ),
        rhs_per_transition=2 * config["vehicle"]["substeps"],
        finished_at_utc=datetime.now(UTC).isoformat(),
        disturbance_stream="SeedSequence([4401,seed]); one common realized impulse per seed",
        optimizer_stream="SeedSequence([4400,seed])",
        collision_check=(
            "Swept disk on each midpoint integration substep; "
            "same cost and geometry in planning/execution"
        ),
        raw_storage=(
            "Local ignored episodes and snapshots; cloning alone does not contain full raw evidence"
        ),
    )
    write_json(root / "manifest.json", manifest)
    return frame
