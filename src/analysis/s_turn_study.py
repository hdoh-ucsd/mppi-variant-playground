"""Separate feasible S-turn design, seed-42 pilot, and paired-seed evaluation."""

import subprocess
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.artifacts import begin, fingerprint
from controllers.mppi import MPPI
from costs.s_turn import STurnCost
from dynamics.bicycle import wrap_angle
from dynamics.steering_rate import make_model
from evaluation.metrics import binomial_interval, mean_interval
from evaluation.runner import git_info, write_json
from scenarios.s_turn import feasibility


def reached(state, corridor, config):
    limits = config["execution"]
    goal = corridor.goal
    progress = corridor.project(state)[1]
    return bool(
        progress >= corridor.center.arc[-1] - limits["position_tolerance"]
        and abs(wrap_angle(state[2] - goal[2])) <= limits["heading_tolerance"]
        and abs(state[3] - goal[3]) <= limits["speed_tolerance"]
    )


def condition(config, clearance, samples, rate):
    resolved = deepcopy(config)
    resolved["vehicle"]["steering_rate_limit"] = rate
    resolved["controller"]["samples"] = samples
    resolved["corridor"]["clearance"] = clearance
    corridor, states, controls, check = feasibility(resolved, clearance)
    resolved["initial_state"] = states[0].tolist()
    resolved["goal_state"] = states[-1].tolist()
    return resolved, corridor, states, controls, check


def run_episode(config, corridor, proposal, seed, output):
    """Use the shared MPPI; every variant sees identical condition-dependent inputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = make_model(config["vehicle"])
    cost = STurnCost(corridor, model, config["cost"], config["corridor"]["speed_target"])
    c = deepcopy(config["controller"])
    c.update(
        kind=proposal,
        covariance=config["proposals"][proposal],
        saved_rollouts=0,
        snapshot_steps=config["execution"]["snapshot_steps"]
        if seed == config["execution"]["representative_seed"]
        else [],
    )
    controller = MPPI(model, cost, c)
    state = np.array(config["initial_state"])
    controller.reset(state, seed)
    states, controls, logs = [state.copy()], [], []
    previous = np.zeros(2)
    total = 0.0
    collision = False
    for step in range(config["execution"]["max_steps"]):
        control = controller.command(state.copy())
        diag = controller.diagnostics
        next_state = model.step(state, control)
        gap = float(corridor.transition_clearance(state, next_state))
        total += float(cost.running(state, control, previous)) + model.dt * cost.weights[
            "collision"
        ] * (gap <= 0)
        logs.append(
            dict(
                step=step,
                ess=diag["ess"],
                weight_entropy=diag["weight_entropy"],
                max_weight=diag["max_weight"],
                feasible_rollout_fraction=diag["feasible_rollout_fraction"],
                sampling_saturation_fraction=diag["saturation_fraction"],
                minimum_sample_cost=diag["minimum_sample_cost"],
                planning_latency_s=diag["planning_latency_s"],
                dynamics_evaluations=diag["dynamics_evaluations"],
                clearance=gap,
                cumulative_running_cost=total,
                acceleration=control[0],
                steering_rate=control[1],
            )
        )
        snapshot = controller.get_rollout_snapshot()
        if snapshot is not None:
            folder = output / "snapshots"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(
                folder / f"step_{step:04d}.npz",
                **{
                    k: snapshot[k]
                    for k in (
                        "sampled_states",
                        "costs",
                        "weights",
                        "latent_controls",
                        "sampled_controls",
                        "proposal_mean",
                        "updated_mean",
                        "nominal_states",
                        "current_state",
                        "log_base_over_proposal",
                    )
                },
            )
        states.append(next_state)
        controls.append(control)
        previous = control
        state = next_state
        collision = gap <= 0
        if collision or reached(state, corridor, config):
            break
    states, controls = np.array(states), np.array(controls)
    frame = pd.DataFrame(logs)
    total += float(cost.terminal(states[-1]))
    distance, progress, _ = corridor.project(states)
    success = reached(state, corridor, config) and not collision
    signs = np.sign(states[:, 4][abs(states[:, 4]) > 0.02])
    result = dict(
        goal_position_error_m=float(np.linalg.norm(state[:2] - corridor.goal[:2])),
        goal_heading_error_rad=float(abs(wrap_angle(state[2] - corridor.goal[2]))),
        goal_speed_m_s=float(state[3]),
        success=success,
        collision=bool(collision),
        termination="completed" if success else ("collision" if collision else "censored_timeout"),
        max_progress_m=float(progress.max()),
        max_progress_fraction=float(progress.max() / corridor.center.arc[-1]),
        cross_track_rmse_m=float(np.sqrt(np.mean(distance**2))),
        minimum_clearance_m=float(frame.clearance.min()),
        time_to_goal_s=len(controls) * model.dt if success else None,
        executed_duration_s=len(controls) * model.dt,
        cumulative_evaluation_cost=total,
        mean_ess=float(frame.ess.mean()),
        mean_weight_entropy=float(frame.weight_entropy.mean()),
        mean_feasible_rollout_fraction=float(frame.feasible_rollout_fraction.mean()),
        zero_feasible_rollout_fraction=float((frame.feasible_rollout_fraction == 0).mean()),
        control_effort=float(model.dt * np.sum((controls / model.control_scale) ** 2)),
        steering_rate_saturation_fraction=float(
            np.mean(abs(controls[:, 1]) >= model.steering_rate_limit - 1e-8)
        ),
        steering_angle_saturation_fraction=float(
            np.mean(abs(states[:, 4]) >= model.bicycle.steering_limit - 1e-8)
        ),
        steering_reversals=int(np.count_nonzero(np.diff(signs))),
        mean_planning_latency_s=float(frame.planning_latency_s.mean()),
        model_transitions=int(frame.dynamics_evaluations.sum()),
        steps=len(controls),
    )
    trajectory = pd.DataFrame(states, columns=["px", "py", "heading", "velocity", "steering_angle"])
    trajectory.insert(0, "time_s", np.arange(len(states)) * model.dt)
    trajectory["progress_m"] = progress
    trajectory["cross_track_m"] = distance
    trajectory.to_csv(output / "trajectory.csv", index=False)
    pd.DataFrame(controls, columns=["acceleration", "steering_rate"]).to_csv(
        output / "controls.csv", index=False
    )
    frame.to_csv(output / "diagnostics.csv", index=False)
    write_json(output / "result.json", result)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return result, trajectory


def summarize(frame, root):
    rows = []
    for (clearance, samples, rate, proposal), g in frame.groupby(
        ["clearance", "samples", "steering_rate_limit", "proposal"], sort=False
    ):
        row = dict(
            clearance=clearance,
            samples=int(samples),
            steering_rate_limit=rate,
            proposal=proposal,
            n=len(g),
            completions=int(g.success.sum()),
            collisions=int(g.collision.sum()),
            censored=int((g.termination == "censored_timeout").sum()),
        )
        for metric in ("success", "collision"):
            stats = binomial_interval(g[metric].astype(float))
            for k in ("mean", "ci95_low", "ci95_high"):
                row[f"{metric}_{k}"] = stats[k]
        for metric in (
            "max_progress_fraction",
            "cross_track_rmse_m",
            "minimum_clearance_m",
            "cumulative_evaluation_cost",
            "time_to_goal_s",
            "mean_ess",
            "mean_weight_entropy",
            "control_effort",
            "steering_rate_saturation_fraction",
            "mean_planning_latency_s",
            "model_transitions",
            "mean_feasible_rollout_fraction",
        ):
            for k, v in mean_interval(g[metric].to_numpy(dtype=float)).items():
                row[f"{metric}_{k}"] = v
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)
    paired = []
    for (clearance, samples, rate), g in frame.groupby(
        ["clearance", "samples", "steering_rate_limit"], sort=False
    ):
        ref = g[g.proposal == "standard"].set_index("seed")
        for proposal, h in g.groupby("proposal", sort=False):
            h = h.set_index("seed")
            if not h.index.equals(ref.index):
                raise ValueError("Unpaired seeds")
            for metric in (
                "success",
                "collision",
                "max_progress_fraction",
                "cumulative_evaluation_cost",
            ):
                stats = mean_interval(h[metric].astype(float) - ref[metric].astype(float))
                paired.append(
                    dict(
                        clearance=clearance,
                        samples=int(samples),
                        steering_rate_limit=rate,
                        proposal=proposal,
                        metric=metric,
                        **stats,
                    )
                )
    pd.DataFrame(paired).to_csv(root / "paired_differences.csv", index=False)
    return summary


def execute_job(job):
    cell, resolved, corridor, suite, proposal, seed, root, phase = job
    relative = Path("episodes") / cell / proposal / f"seed_{seed}"
    result, trajectory = run_episode(resolved, corridor, proposal, seed, Path(root) / relative)
    row = dict(
        cell=cell,
        suite=suite,
        clearance=resolved["corridor"]["clearance"],
        samples=resolved["controller"]["samples"],
        steering_rate_limit=resolved["vehicle"]["steering_rate_limit"],
        proposal=proposal,
        seed=seed,
        shared_condition_sha256=fingerprint(resolved),
        episode_path=str(relative),
        **result,
    )
    representative = (
        trajectory.to_dict("list")
        if (seed == resolved["execution"]["representative_seed"] or phase == "pilot")
        else None
    )
    return row, representative


def run_study(config, output, phase="evaluation", *, conditions=None):
    config = deepcopy(config)
    config["phase"] = phase
    source = git_info()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)runs"],
        capture_output=True,
        text=True,
        check=True,
    )
    source["dirty"] = bool(status.stdout.strip())
    source["status_scope"] = "repository excluding generated results/ and runs/"
    root, manifest = begin(output, config, "s_turn")
    manifest["git"] = source
    model = config["vehicle"]
    primary = config["controller"]["samples"]
    base_rate = model["steering_rate_limit"]
    cells = conditions or [
        (c, primary, base_rate, "primary") for c in config["corridor"]["clearance_levels"]
    ]
    if phase == "evaluation" and conditions is None:
        cells += [
            (config["secondary_clearance"], k, base_rate, "sample_budget")
            for k in config["secondary_samples"]
        ]
        cells += [
            (config["secondary_clearance"], primary, r, "steering_rate")
            for r in config["secondary_rate_limits"]
        ]
    seeds = (
        [config["pilot_seed"]]
        if phase == "pilot"
        else list(
            range(config["seeds"]["start"], config["seeds"]["start"] + config["seeds"]["count"])
        )
    )
    feasibility_rows = []
    resolved_cells = []
    for clearance, samples, rate, suite in cells:
        resolved, corridor, states, controls, check = condition(config, clearance, samples, rate)
        cell = f"c{clearance:g}_K{samples}_r{rate:g}"
        check.update(
            cell=cell, clearance=clearance, samples=samples, steering_rate_limit=rate, suite=suite
        )
        feasibility_rows.append(check)
        if not check["feasible"]:
            raise RuntimeError(f"Unverified mechanical feasibility: {cell}")
        folder = root / "feasibility"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(
            folder / f"{cell}.npz",
            states=states,
            controls=controls,
            upper=corridor.upper,
            lower=corridor.lower,
        )
        resolved_cells.append((cell, resolved, corridor, suite))
    write_json(root / "feasibility.json", feasibility_rows)
    write_json(
        root / "geometry.json",
        {
            cell: dict(
                upper=corridor.upper.tolist(),
                lower=corridor.lower.tolist(),
                centerline=corridor.center.points.tolist(),
                goal=corridor.goal.tolist(),
            )
            for cell, _, corridor, _ in resolved_cells
        },
    )
    print(f"Validated {len(cells)} conditions before evaluating proposals", flush=True)
    jobs = [
        (cell, resolved, corridor, suite, proposal, seed, str(root), phase)
        for cell, resolved, corridor, suite in resolved_cells
        for proposal in config["proposals"]
        for seed in seeds
    ]
    trials, representatives = [], {}
    workers = config.get("workers", 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for row, representative in executor.map(execute_job, jobs):
                trials.append(row)
                if representative is not None:
                    representatives[f"{row['cell']}__{row['proposal']}"] = representative
                if len(trials) % len(seeds) == 0:
                    print(
                        f"Completed {row['cell']} / {row['proposal']}: {len(seeds)} seeds",
                        flush=True,
                    )
                    pd.DataFrame(trials).to_csv(root / "per_seed.csv", index=False)
    else:
        for job in jobs:
            row, representative = execute_job(job)
            trials.append(row)
            if representative is not None:
                representatives[f"{row['cell']}__{row['proposal']}"] = representative
            if len(trials) % len(seeds) == 0:
                print(
                    f"Completed {row['cell']} / {row['proposal']}: {len(seeds)} seeds", flush=True
                )
    pd.DataFrame(trials).to_csv(root / "per_seed.csv", index=False)
    frame = pd.DataFrame(trials)
    summarize(frame, root)
    write_json(root / "representative_paths.json", representatives)
    write_json(
        root / "conditions.json", [dict(cell=a, suite=d, config=b) for a, b, c, d in resolved_cells]
    )
    manifest.update(
        complete=True,
        episodes=len(frame),
        phase=phase,
        run_id=config["run_id"] if phase == "evaluation" else "s_turn_pilot_42",
        paired_seeds=seeds,
        source_covariance_note=(
            "New normalized acceleration/rate covariances, trace .40; base fixed at diag(.26,.14)"
        ),
        collision_geometry=(
            "Exact segment/polygon boundary distance for swept disk; Euler straight transitions"
        ),
        reference_note=(
            "Mechanical feasibility verified before proposal evaluation; no reference "
            "controls supplied to MPPI"
        ),
        raw_records=(
            "episodes/ and feasibility arrays are local and ignored; summaries and "
            "representative XY evidence versioned"
        ),
        primary_axis=(
            "corridor clearance = nominal half-width minus disk radius; speed, horizon, "
            "limits and K fixed"
        ),
    )
    write_json(root / "manifest.json", manifest)
    return frame
