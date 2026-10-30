"""Paired behavior studies and saved-data interchange for present and future controllers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.artifacts import begin, fingerprint, finish
from analysis.rollout_snapshots import save_snapshot, snapshot_statistics
from analysis.trajectory_statistics import first_crossing, occupancy, progress_dispersion
from evaluation.runner import make_controller, run_episode, write_json
from utils.config import load_experiment

ALIASES = {"sudden_disturbance": "disturbance"}
VARIANTS = ("standard", "isotropic", "diagonal", "full")


class EpisodeRecorder:
    """Observe the existing runner without changing actions, RNG streams or termination."""

    def __init__(self, directory, expected_steps):
        self.directory = Path(directory)
        self.expected_steps = set(expected_steps)
        self.events = []
        self.snapshots = []

    def __call__(self, event):
        self.events.append(
            {
                "cumulative_running_cost": event["cumulative_running_cost"],
                "collision": event["collision"],
            }
        )
        if event["step"] in self.expected_steps:
            snapshot = event["snapshot"]
            if snapshot is None:
                raise ValueError("Controller did not supply the requested generic rollout snapshot")
            snapshot = snapshot | {
                "executed_next_state": event["next_state"],
                "chosen_control": event["control"],
            }
            relative = Path("snapshots") / f"step_{event['step']:04d}.npz"
            save_snapshot(self.directory / relative, snapshot)
            self.snapshots.append(
                {"step": event["step"], "path": str(relative), **snapshot_statistics(snapshot)}
            )

    def finalize(self, metadata):
        trajectory = pd.read_csv(self.directory / "trajectory.csv").rename(
            columns={"theta": "heading"}
        )
        controls = pd.read_csv(self.directory / "controls.csv")
        diagnostics = pd.read_csv(self.directory / "diagnostics.csv")
        n = len(controls)
        if len(self.events) != n or len(trajectory) != n + 1:
            raise ValueError("Observer and executed-trajectory lengths do not agree")
        metrics = metadata["metrics"]
        trajectory.insert(0, "time_index", np.arange(n + 1))
        for field in controls:
            # Controls apply on the outgoing interval; the terminal state has no next action.
            trajectory[field] = np.r_[controls[field], np.nan]
        trajectory["terminal"] = np.arange(n + 1) == n
        trajectory["success"] = trajectory.terminal & metrics["success"]
        trajectory["failure"] = trajectory.terminal & (not metrics["success"])
        trajectory["collision"] = np.r_[
            False, np.maximum.accumulate([e["collision"] for e in self.events])
        ]
        cumulative = np.r_[0.0, [e["cumulative_running_cost"] for e in self.events]]
        cumulative[-1] = metrics["cumulative_evaluation_cost"]
        trajectory["cumulative_evaluation_cost"] = cumulative
        trajectory.to_csv(self.directory / "executed_trajectory.csv", index=False)
        diagnostics["chosen_acceleration"] = controls.acceleration
        diagnostics["chosen_steering_angle"] = controls.steering_angle
        diagnostics.to_csv(self.directory / "mpc_diagnostics.csv", index=False)
        pd.DataFrame(self.snapshots).to_csv(self.directory / "snapshot_index.csv", index=False)
        return trajectory, diagnostics


def validate_config(config, factories):
    seeds = config["seeds"]
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("At least two distinct nonnegative paired seeds required")
    if not config["variants"] or len(set(config["variants"])) != len(config["variants"]):
        raise ValueError("Distinct variants required")
    if any(name not in factories for name in config["variants"]):
        raise ValueError("A controller factory must be registered for every variant")
    if config["samples"] < 2 or config["horizon"] < 1:
        raise ValueError("Sample count >= 2 and positive horizon required")
    if not config["snapshot_steps"] or any(
        not isinstance(s, int) or s < 0 for s in config["snapshot_steps"]
    ):
        raise ValueError("Snapshot steps must be nonnegative integers")
    if (
        config["representative_seed"] not in seeds
        or config["representative_step"] not in config["snapshot_steps"]
    ):
        raise ValueError("Representative seed/step must be included in the recorded plan")
    if config["occupancy_weighting"] not in {"equal_seed", "pooled_time"}:
        raise ValueError("Unsupported occupancy weighting")


def run_study(config, output, *, controller_factories=None):
    """Factories accept resolved configs; plotting depends only on recorded schemas.

    Future controllers must implement the existing Controller API and optionally provide
    get_rollout_snapshot() using rollout_snapshots.py's schema. No future algorithm is
    emulated by relabeling one of the current four covariance settings.
    """
    factories = {name: make_controller for name in VARIANTS}
    factories.update(controller_factories or {})
    validate_config(config, factories)
    output, manifest = begin(output, config, "trajectory_behavior")
    manifest.update(
        {
            "rng": "Optimizer PCG64(seed); paired environment SeedSequence([seed,1001])",
            "state_fields": ["px", "py", "heading", "velocity"],
            "state_units": ["m", "m", "rad", "m/s"],
            "control_fields": ["acceleration", "steering_angle"],
            "control_units": ["m/s^2", "rad"],
            "snapshot_schema_version": 1,
            "timing_note": "Planning includes snapshot array collection; disk IO is excluded",
            "seed_note": "Descriptive expanded seed set; 42–46 have been inspected in prior work",
        }
    )
    write_json(output / "manifest.json", manifest)
    cases, trials = [], []
    for scenario in config["scenarios"]:
        source = ALIASES.get(scenario, scenario)
        for variant in config["variants"]:
            base_name = variant if variant in VARIANTS else "standard"
            resolved = load_experiment(base_name, source)
            resolved["controller"].update(
                samples=config["samples"],
                horizon=config["horizon"],
                snapshot_steps=config["snapshot_steps"],
            )
            resolved["controller"].update(config.get("variant_overrides", {}).get(variant, {}))
            if "max_steps" in config:
                resolved["scenario"]["max_steps"] = config["max_steps"]
            shared = {key: resolved[key] for key in ("dynamics", "scenario", "costs")}
            shared.update(
                {
                    "samples": resolved["controller"]["samples"],
                    "horizon": resolved["controller"]["horizon"],
                }
            )
            signature = fingerprint(shared)
            previous = [c["shared_setup_sha256"] for c in cases if c["scenario"] == scenario]
            if previous and signature != previous[0]:
                raise ValueError(
                    "Variants must share dynamics, task, cost, horizon and rollout budget"
                )
            case_id = len(cases)
            cases.append(
                {
                    "case_id": case_id,
                    "scenario": scenario,
                    "source_scenario": source,
                    "variant": variant,
                    "proposal_name": variant,
                    "shared_setup_sha256": signature,
                    "controller": resolved["controller"],
                }
            )
            for seed in config["seeds"]:
                directory = output / "episodes" / scenario / variant / f"seed_{seed}"
                recorder = EpisodeRecorder(directory, config["snapshot_steps"])
                result = run_episode(
                    resolved,
                    seed,
                    directory,
                    mode="trajectory_behavior",
                    observer=recorder,
                    controller_factory=factories[variant],
                )
                trajectory, diagnostics = recorder.finalize(result)
                position = trajectory[["px", "py"]].to_numpy()
                crossing = first_crossing(position, config["route_station_x"])
                route = (
                    "not_reached" if crossing is None else ("upper" if crossing > 0 else "lower")
                )
                trials.append(
                    {
                        "case_id": case_id,
                        "comparison_id": scenario,
                        "proposal_name": variant,
                        "scenario": scenario,
                        "variant": variant,
                        "seed": seed,
                        **result["metrics"],
                        "ess": result["metrics"]["mean_ess"],
                        "max_weight": float(diagnostics.max_weight.mean()),
                        "weight_entropy": float(diagnostics.weight_entropy.mean()),
                        "saturation_fraction": float(diagnostics.saturation_fraction.mean()),
                        "episode_path": str(directory.relative_to(output)),
                        "snapshots": len(recorder.snapshots),
                        "station_crossing_y_m": crossing,
                        "route_class": route
                        if scenario == "two_route_obstacle"
                        else "not_assessed",
                        "cost_evaluations": result["metrics"]["total_rollout_evaluations"],
                    }
                )
            print(f"Completed {scenario} / {variant}: {len(config['seeds'])} seeds", flush=True)
    finish(
        output,
        manifest,
        cases,
        trials,
        "# Trajectory behavior study\n\n"
        "Executed bundles and progress bands describe inter-seed behavior. Rollout clouds describe "
        "one intra-step population and its optimizer weights. Occupancy pools recorded states with "
        "explicit seed/time weights. It is neither proposal covariance nor a predictive state PDF. "
        "Failures remain included. Paired seed/time snapshots can have different current states "
        "and warm starts. Pointwise 50/90% bands are descriptive sample quantiles. "
        "See results/historical/archive/legacy_index.md for interpretation and inference limits.",
    )
    manifest["complete"] = False
    write_json(output / "manifest.json", manifest)
    compute_distributions(output)
    manifest.update({"complete": True, "snapshots": sum(r["snapshots"] for r in trials)})
    write_json(output / "manifest.json", manifest)
    return pd.DataFrame(trials)


def load_runs(root):
    root = Path(root)
    cases = json.loads((root / "cases.json").read_text())
    trials = pd.read_csv(root / "trials.csv")
    records = []
    for trial in trials.to_dict("records"):
        directory = root / trial["episode_path"]
        records.append(
            {
                **trial,
                "case": cases[int(trial["case_id"])],
                "directory": directory,
                "trajectory": pd.read_csv(directory / "executed_trajectory.csv"),
                "diagnostics": pd.read_csv(directory / "mpc_diagnostics.csv"),
                "snapshot_index": pd.read_csv(directory / "snapshot_index.csv"),
            }
        )
    return records


def scene_bounds(records, scenario):
    points = np.concatenate([r["trajectory"][["px", "py"]].to_numpy() for r in records])
    anchors = [scenario["initial_state"][:2], scenario["goal_state"][:2]]
    for x, y, radius in scenario["obstacles"]:
        r = radius + scenario["vehicle_radius"]
        anchors.extend([[x - r, y - r], [x + r, y + r]])
    points = np.vstack([points, anchors])
    low, high = points.min(axis=0) - 0.5, points.max(axis=0) + 0.5
    return low, high


def compute_distributions(root):
    """Save exact plot inputs, including aligned paths, densities and seed selection."""
    from utils.config import read_yaml

    root = Path(root)
    config = read_yaml(root / "config.yaml")
    records = load_runs(root)
    derived = root / "distributions"
    derived.mkdir(exist_ok=True)
    index, behavior, snapshot_rows = [], [], []
    for record in records:
        snapshot_rows.extend(
            {
                "scenario": record["scenario"],
                "variant": record["variant"],
                "seed": record["seed"],
                **row,
            }
            for row in record["snapshot_index"].to_dict("records")
        )
    for scenario in config["scenarios"]:
        selected = [r for r in records if r["scenario"] == scenario]
        setup = read_yaml(selected[0]["directory"] / "config.yaml")["scenario"]
        low, high = scene_bounds(selected, setup)
        x_edges = np.linspace(low[0], high[0], config["occupancy_bins"][0] + 1)
        y_edges = np.linspace(low[1], high[1], config["occupancy_bins"][1] + 1)
        for variant in config["variants"]:
            group = [r for r in selected if r["variant"] == variant]
            paths = [r["trajectory"][["px", "py"]].to_numpy() for r in group]
            density = occupancy(
                paths,
                x_edges,
                y_edges,
                weighting=config["occupancy_weighting"],
                smoothing=config["occupancy_smoothing_sigma_bins"],
            )
            dispersion = progress_dispersion(paths, config["progress_points"])
            # Select an actual path nearest median cost; ties break on seed, not success.
            median_cost = float(np.median([r["cumulative_evaluation_cost"] for r in group]))
            representative = min(
                group, key=lambda r: (abs(r["cumulative_evaluation_cost"] - median_cost), r["seed"])
            )
            filename = f"{scenario}__{variant}.npz"
            np.savez_compressed(
                derived / filename,
                **{f"occupancy_{k}": v for k, v in density.items()},
                **dispersion,
                seeds=np.array([r["seed"] for r in group]),
            )
            index.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "path": filename,
                    "representative_seed": representative["seed"],
                    "median_cost": median_cost,
                    "occupancy_weighting": config["occupancy_weighting"],
                }
            )
            crossing = np.array(
                [r["station_crossing_y_m"] for r in group if pd.notna(r["station_crossing_y_m"])]
            )
            behavior.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "n": len(group),
                    "successes": sum(r["success"] for r in group),
                    "collisions": sum(r["collision"] for r in group),
                    "mean_evaluation_cost": float(
                        np.mean([r["cumulative_evaluation_cost"] for r in group])
                    ),
                    "mean_ess": float(np.mean([r["ess"] for r in group])),
                    "mean_lateral_q90_width_m": float(
                        np.mean(dispersion["q95_xy"][:, 1] - dispersion["q05_xy"][:, 1])
                    ),
                    "station_crossed_n": len(crossing),
                    "station_crossing_y_sd_m": float(crossing.std(ddof=1))
                    if len(crossing) > 1
                    else None,
                    "route_upper_n": sum(r["route_class"] == "upper" for r in group),
                    "route_lower_n": sum(r["route_class"] == "lower" for r in group),
                    "route_censored_n": sum(r["route_class"] == "not_reached" for r in group),
                }
            )
    write_json(derived / "index.json", index)
    pd.DataFrame(behavior).to_csv(root / "behavior_summary.csv", index=False)
    pd.DataFrame(snapshot_rows).to_csv(root / "snapshot_summary.csv", index=False)
