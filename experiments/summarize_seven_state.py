"""Derive terminal-failure and selected weight diagnostics from existing episode records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import mean_interval
from sampling.gaussian import importance_weights
from utils.config import read_yaml


def summarize(root):
    root = Path(root)
    if not json.loads((root / "manifest.json").read_text())["complete"]:
        raise ValueError("Requires a complete run")
    frame = pd.read_csv(root / "per_seed.csv")
    geometries = json.loads((root / "geometry.json").read_text())
    lengths = {
        name: np.linalg.norm(np.diff(value["centerline"], axis=0), axis=1).sum()
        for name, value in geometries.items()
    }
    failures = []
    components = []
    pushforwards = []
    goals = []
    for _, row in frame.iterrows():
        episode = Path(row.episode_path)
        config = read_yaml(episode / "config.yaml")
        goal = np.asarray(config["scenario"]["goal_state"])
        tol = config["execution"]
        goals.append(
            dict(
                case=row["case"],
                cohort=row.cohort,
                proposal=row.proposal,
                seed=row.seed,
                position_error_m=row.goal_position_error_m,
                heading_error_rad=row.goal_heading_error_rad,
                speed_error_m_s=abs(row.goal_speed_m_s - goal[4]),
            )
        )
        if not row.success:
            last = pd.read_csv(episode / "trajectory.csv").iloc[-1]
            flags = dict(
                collision=bool(row.collision),
                progress_shortfall=last.progress_m
                < lengths[row["case"]] - tol["position_tolerance"],
                lateral_goal=abs(last.py - goal[1]) > tol["goal_lateral_tolerance"],
                heading=row.goal_heading_error_rad > tol["heading_tolerance"],
                speed=abs(last.vx - goal[4]) > tol["speed_tolerance"],
                lateral_velocity=abs(last.vy) > tol["lateral_velocity_tolerance"],
                yaw_rate=abs(last.yaw_rate) > tol["yaw_rate_tolerance"],
                roll=abs(last["roll"]) > tol["roll_tolerance"],
            )
            diag = pd.read_csv(episode / "diagnostics.csv")
            hits = np.flatnonzero(diag.clearance.to_numpy() <= 0)
            collision_step = int(hits[0]) if len(hits) else None
            event = config["scenario"].get("disturbance")
            collision_phase = (
                "at_imposed_impulse"
                if event and collision_step == event["step"]
                else "after_impulse"
                if event and collision_step is not None and collision_step > event["step"]
                else "collision"
                if collision_step is not None
                else "no_collision"
            )
            failures.append(
                dict(
                    case=row["case"],
                    proposal=row.proposal,
                    seed=row.seed,
                    cohort=row.cohort,
                    termination=row.termination,
                    collision_step=collision_step,
                    collision_phase=collision_phase,
                    violated_checks=";".join(k for k, v in flags.items() if v),
                    final_x=last.px,
                    final_y=last.py,
                    final_vx=last.vx,
                    final_vy=last.vy,
                    final_yaw_rate=last.yaw_rate,
                    final_roll=last["roll"],
                    heading_error=row.goal_heading_error_rad,
                    **flags,
                )
            )
        for path in sorted((episode / "snapshots").glob("*.npz")):
            with np.load(path) as data:
                rate = config["controller"].get("steering_rate_limit")
                if rate is not None and path.stem == "step_0000":
                    scale = np.array(
                        [config["vehicle"]["steering_limit"], config["vehicle"]["throttle_limit"]]
                    )
                    record = dict(
                        case=row["case"],
                        proposal=row.proposal,
                        seed=row.seed,
                        rate_limit=rate,
                        samples=len(data["weights"]),
                        source=str(path),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    for label, values in [
                        ("latent", data["latent_controls"][:, 0]),
                        ("applied_normalized", data["sampled_controls"][:, 0] / scale),
                    ]:
                        covariance = np.cov(values.T)
                        record.update(
                            {
                                label + "_steering_variance": covariance[0, 0],
                                label + "_throttle_variance": covariance[1, 1],
                                label + "_correlation": covariance[0, 1]
                                / np.sqrt(covariance[0, 0] * covariance[1, 1]),
                            }
                        )
                    record["first_sample_rate_saturation_fraction"] = float(
                        np.mean(
                            abs(data["sampled_controls"][:, 0, 0])
                            >= rate * config["vehicle"]["dt"] - 1e-12
                        )
                    )
                    pushforwards.append(record)
                for component, costs, ratio in [
                    ("cost_only", data["costs"], None),
                    ("ratio_only", np.zeros_like(data["costs"]), data["log_base_over_proposal"]),
                    ("combined", data["costs"], data["log_base_over_proposal"]),
                ]:
                    weights = importance_weights(costs, config["controller"]["temperature"], ratio)
                    components.append(
                        dict(
                            case=row["case"],
                            cohort=row.cohort,
                            proposal=row.proposal,
                            seed=row.seed,
                            snapshot=path.name,
                            component=component,
                            sampling_dimension=2 * config["controller"]["horizon"],
                            samples=len(weights),
                            ess=1 / np.sum(weights**2),
                            max_weight=float(weights.max()),
                        )
                    )
    failures_frame = (
        pd.DataFrame(failures)
        if failures
        else pd.DataFrame(
            columns=[
                "case",
                "proposal",
                "seed",
                "cohort",
                "termination",
                "violated_checks",
                "collision_step",
                "collision_phase",
            ]
        )
    )
    failures_frame.to_csv(root / "failure_modes.csv", index=False)
    pd.DataFrame(
        components,
        columns=[
            "case",
            "cohort",
            "proposal",
            "seed",
            "snapshot",
            "component",
            "sampling_dimension",
            "samples",
            "ess",
            "max_weight",
        ],
    ).to_csv(root / "selected_weight_components.csv", index=False)
    if pushforwards:
        pd.DataFrame(pushforwards).to_csv(root / "proposal_pushforward.csv", index=False)
    goals = pd.DataFrame(goals)
    records = []
    for keys, g in goals.groupby(["case", "cohort", "proposal"], sort=False):
        record = dict(zip(["case", "cohort", "proposal"], keys, strict=True))
        for metric in ["position_error_m", "heading_error_rad", "speed_error_m_s"]:
            record.update(
                {metric + "_" + key: value for key, value in mean_interval(g[metric]).items()}
            )
        records.append(record)
    pd.DataFrame(records).to_csv(root / "goal_summary.csv", index=False)
    print(f"{root}: {len(failures)} failures; {len(components)} selected weight components")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    summarize(p.parse_args().input)
