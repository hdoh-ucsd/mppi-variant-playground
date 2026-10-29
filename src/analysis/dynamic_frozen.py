"""Seven-state adapter for the existing frozen-decision estimator and reference pooling."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.artifacts import begin
from analysis.dynamic_campaign import assert_validated, resolved_case, run_episode
from analysis.frozen_decision import (
    PROPOSALS,
    decision,
    inputs_signature,
    normal_draws,
    proposals,
    reference_estimate,
    save_decision,
)
from evaluation.metrics import mean_interval
from evaluation.runner import make_controller, write_json


def route_witnesses(frozen):
    controller = make_controller(frozen["config"])
    model, cost = controller.model, controller.cost
    horizon = controller.horizon
    specifications = [
        (throttle, amplitude, side, fraction, second)
        for throttle in (0.2, 0.4, 0.6)
        for amplitude in (0.2, 0.35, 0.5)
        for side in (-1, 1)
        for fraction in (0.2, 0.35, 0.5)
        for second in (0.0, -0.5, -1.0)
    ]
    k = len(specifications)
    physical = np.zeros((k, horizon, 2))
    for i, (throttle, amplitude, side, fraction, second) in enumerate(specifications):
        n = int(horizon * fraction)
        physical[i, :, 1] = throttle
        physical[i, :n, 0] = side * amplitude
        physical[i, n:, 0] = second * side * amplitude
    states = np.broadcast_to(frozen["state"], (k, 7)).copy()
    history = [states.copy()]
    previous = np.broadcast_to(frozen["previous_control"], (k, 2))
    costs = np.zeros(k)
    gaps = np.full(k, np.inf)
    for t in range(horizon):
        costs += cost.running(states, physical[:, t], previous)
        nxt, path = model.step(states, physical[:, t], return_path=True)
        gap = cost.path_clearance(path)
        gaps = np.minimum(gaps, gap)
        costs += model.dt * cost.weights["collision"] * (gap <= 0)
        states = nxt
        previous = physical[:, t]
        history.append(states.copy())
    costs += cost.terminal(states)
    history = np.stack(history, axis=1)
    station = frozen["config"]["scenario"]["obstacles"][0][0]
    result = {}
    for i in np.argsort(costs):
        x = history[i, :, 0]
        cross = np.flatnonzero((x[:-1] < station) & (x[1:] >= station))
        if gaps[i] > 0 and len(cross) and x[-1] > station + 0.5 and abs(history[i, -1, 2]) < 1.2:
            j = cross[0]
            f = (station - x[j]) / (x[j + 1] - x[j])
            y = history[i, j, 1] + f * (history[i, j + 1, 1] - history[i, j, 1])
            label = "upper" if y > 0 else "lower"
            if label not in result:
                result[label] = dict(
                    states=history[i].tolist(),
                    physical_controls=physical[i].tolist(),
                    clearance=float(gaps[i]),
                    station_y=float(y),
                    objective=float(costs[i]),
                )
    return result


def capture(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    options = config["frozen"]
    result = []
    for label, scenario in [
        ("open_space", "open_space"),
        ("pre_obstacle", "two_route_obstacle"),
        ("post_disturbance", "disturbance"),
    ]:
        resolved, _, *_ = resolved_case(config, scenario, samples=options["samples"])
        resolved["controller"]["horizon"] = options["horizon"]

        def rule(candidate, label=label):
            if label == "open_space":
                return candidate["step"] == options["open_step"]
            if label == "post_disturbance":
                return candidate["step"] == options["disturbance_step"]
            if options["obstacle_min_x"] <= candidate["state"][0] <= options["obstacle_max_x"]:
                witnesses = route_witnesses(candidate)
                if set(witnesses) == {"upper", "lower"}:
                    candidate["route_witnesses"] = witnesses
                    return True
            return False

        value = run_episode(
            resolved,
            "standard",
            options["seed"],
            output.parent / "episodes" / f"source_{label}",
            freeze_rule=rule,
        )
        value["name"] = label
        value["inputs_sha256"] = inputs_signature(value)
        value["selection_rule"] = (
            "Fixed step 8 / first x interval with two feasible routes / "
            "first observation after step-20 impulse"
        )
        write_json(output / f"{label}.json", value)
        result.append(value)
        print(
            f"Captured seven-state {label}: step {value['step']}, x={value['state'][0]:.3f}",
            flush=True,
        )
    return result


def display_snapshot(snap):
    weights = snap["weights"]
    indices = np.unique(np.r_[np.arange(min(48, len(weights))), np.argsort(weights)[-16:]])
    return dict(
        indices=indices.tolist(),
        unweighted_indices=list(range(min(48, len(weights)))),
        paths_xy=snap["sampled_states"][indices, :, :2].tolist(),
        endpoint_xy=snap["sampled_states"][:, -1, :2].tolist(),
        weights=weights.tolist(),
        costs=snap["costs"].tolist(),
        nominal_states=snap["nominal_states"].tolist(),
        command=snap["command"].tolist(),
        updated_mean=snap["updated_mean"].tolist(),
        best_raw_index=int(snap["best_rollout_index"]),
        max_weight_index=int(snap["highest_weight_index"]),
        population_bounds=[
            snap["sampled_states"][:, :, :2].min(axis=(0, 1)).tolist(),
            snap["sampled_states"][:, :, :2].max(axis=(0, 1)).tolist(),
        ],
    )


def state_job(args):
    index, frozen, config, root = args
    root = Path(root)
    options = deepcopy(config["frozen"])
    options["proposal_matrices"] = config["proposals"]
    candidates, sensitivity = proposals(frozen, options)
    reference, status = reference_estimate(
        frozen, index, options, root / "reference_estimates" / frozen["name"]
    )
    with np.load(root / "reference_estimates" / frozen["name"] / "estimate.npz") as data:
        reference_data = {key: data[key].tolist() for key in data.files}
    rows = []
    means = []
    pictures = {}
    seeds = range(options["seeds"]["start"], options["seeds"]["start"] + options["seeds"]["count"])
    for seed in seeds:
        z = normal_draws(index, seed, max(options["sample_counts"]), options["horizon"])
        for count in options["sample_counts"]:
            standard_stats = standard_mean = None
            for name in PROPOSALS:
                record = count == options["samples"] or seed == options["representative_seed"]
                mean, stats, snap = decision(
                    frozen, candidates[name]["covariance"], z[:count], record
                )
                if name == "standard":
                    standard_stats, standard_mean = stats, mean
                row = dict(
                    state=frozen["name"],
                    proposal=name,
                    samples=count,
                    seed=seed,
                    inputs_sha256=frozen["inputs_sha256"],
                    reference_status=status["status"],
                    reference_distance_rms=float(np.sqrt(np.mean((mean - reference) ** 2))),
                    first_command_reference_distance=float(
                        np.linalg.norm(np.clip(mean[0], -1, 1) - np.clip(reference[0], -1, 1))
                    ),
                    paired_mean_distance_rms=float(np.sqrt(np.mean((mean - standard_mean) ** 2))),
                    paired_command_distance=float(
                        np.linalg.norm(np.clip(mean[0], -1, 1) - np.clip(standard_mean[0], -1, 1))
                    ),
                    **stats,
                )
                for control in ("steering", "throttle"):
                    row[f"delta_{control}"] = (
                        stats[f"first_{control}"] - standard_stats[f"first_{control}"]
                    )
                if record:
                    path = (
                        root
                        / "snapshots"
                        / frozen["name"]
                        / f"K{count}"
                        / name
                        / f"seed_{seed}.npz"
                    )
                    save_decision(path, snap)
                if seed == options["representative_seed"] and count == options["samples"]:
                    pictures[name] = display_snapshot(snap)
                rows.append(row)
                means.append(mean)
        if (seed - options["seeds"]["start"] + 1) % 10 == 0:
            print(
                f"Q03 {frozen['name']}: {seed - options['seeds']['start'] + 1} paired seeds",
                flush=True,
            )
    return (
        rows,
        np.array(means),
        pictures,
        dict(
            proposals=candidates,
            sensitivity=sensitivity,
            reference_status=status,
            reference=reference_data,
        ),
    )


def run_frozen(config, output):
    assert_validated()
    root, manifest = begin(output, config, "seven_state_frozen")
    frozen = capture(config, root / "frozen_states")
    rows = []
    means = []
    pictures = {}
    details = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for values, estimates, images, info in pool.map(
            state_job, [(i, f, config, str(root)) for i, f in enumerate(frozen)]
        ):
            name = values[0]["state"]
            rows.extend(values)
            means.extend(estimates)
            pictures[name] = images
            details[name] = info
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "per_seed.csv", index=False)
    summaries = []
    metrics = [
        "ess",
        "weight_entropy",
        "reference_distance_rms",
        "first_command_reference_distance",
        "nominal_objective",
        "paired_mean_distance_rms",
        "paired_command_distance",
        "delta_steering",
        "delta_throttle",
        "planning_latency_s",
        "saturation_fraction",
    ]
    for keys, group in frame.groupby(["state", "proposal", "samples"], sort=False):
        row = dict(
            zip(["state", "proposal", "samples"], keys, strict=True),
            n=len(group),
            reference_status=group.reference_status.iloc[0],
        )
        for metric in metrics:
            row.update({f"{metric}_{k}": v for k, v in mean_interval(group[metric]).items()})
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    np.savez_compressed(root / "updated_means.npz", means=np.array(means))
    write_json(root / "figure_inputs.json", pictures)
    write_json(root / "proposal_and_reference.json", details)
    manifest.update(
        complete=True,
        run_id=root.name,
        decisions=len(rows),
        dynamics_model="dynamic_single_track_7",
        control_fields=["steering", "throttle"],
        sampling_dimension=2 * config["frozen"]["horizon"],
        evaluation_rollouts=int(frame.rollout_evaluations.sum()),
        rhs_evaluations=int(frame.rhs_evaluations.sum()),
        evaluation_stream="SeedSequence([101,state_index,seed])",
        reference_stream="SeedSequence([202,state_index,rep])",
        raw_storage=(
            "Full primary and representative-budget populations are local ignored arrays; "
            "curated inputs contain XY display subsets, full endpoint weights and costs"
        ),
    )
    write_json(root / "manifest.json", manifest)
    return frame
