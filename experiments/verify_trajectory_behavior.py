"""Audit full rollout records, occupancy mass and overlapping baseline trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rollout_snapshots import load_snapshot
from analysis.trajectory_density import load_runs
from evaluation.runner import git_info, write_json
from scenarios.environment import swept_clearance
from utils.config import read_yaml


def audit(root, baseline):
    root, baseline = Path(root), Path(baseline)
    config = read_yaml(root / "config.yaml")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["complete"]
    records = load_runs(root)
    snapshots, matched, revisions = 0, 0, set()
    for record in records:
        directory = record["directory"]
        result = json.loads((directory / "result.json").read_text())
        assert not result["git"]["dirty"]
        revisions.add(result["git"]["commit"])
        for row in record["snapshot_index"].itertuples():
            snapshot = load_snapshot(directory / row.path)
            assert snapshot["sample_count"] == config["samples"]
            assert snapshot["horizon"] == config["horizon"]
            trajectory = record["trajectory"]
            fields = ["px", "py", "heading", "velocity"]
            np.testing.assert_allclose(
                snapshot["current_state"], trajectory[fields].iloc[row.step], atol=1e-12
            )
            np.testing.assert_allclose(
                snapshot["executed_next_state"], trajectory[fields].iloc[row.step + 1], atol=1e-12
            )
            np.testing.assert_allclose(
                snapshot["chosen_control"],
                trajectory[["acceleration", "steering_angle"]].iloc[row.step],
                atol=1e-12,
            )
            snapshots += 1
        source = record["case"]["source_scenario"]
        original = baseline / source / record["variant"] / f"seed_{record['seed']}"
        if (original / "trajectory.csv").exists():
            for filename in ("trajectory.csv", "controls.csv"):
                pd.testing.assert_frame_equal(
                    pd.read_csv(directory / filename),
                    pd.read_csv(original / filename),
                    check_exact=True,
                )
            matched += 1
    assert snapshots == manifest["snapshots"] and len(revisions) == 1
    clearances = []
    index = json.loads((root / "distributions/index.json").read_text())
    for entry in index:
        with np.load(root / "distributions" / entry["path"], allow_pickle=False) as data:
            assert np.isclose(data["occupancy_mass"].sum(), 1, atol=1e-12)
            area = (
                np.diff(data["occupancy_y_edges"])[:, None]
                * np.diff(data["occupancy_x_edges"])[None, :]
            )
            assert np.isclose(np.sum(data["occupancy_density"] * area), 1, atol=1e-12)
            mean = data["mean_xy"]
        record = next(
            r
            for r in records
            if r["scenario"] == entry["scenario"] and r["variant"] == entry["variant"]
        )
        setup = read_yaml(record["directory"] / "config.yaml")["scenario"]
        clearance = swept_clearance(
            mean[:-1], mean[1:], setup["obstacles"], setup["vehicle_radius"]
        )
        clearances.append(
            {
                "scenario": entry["scenario"],
                "variant": entry["variant"],
                "mean_path_min_swept_clearance_m": float(clearance.min())
                if np.isfinite(clearance).any()
                else None,
            }
        )
    return {
        "episodes": len(records),
        "full_population_snapshots": snapshots,
        "normalized_occupancy_maps": len(index),
        "matched_baseline_episodes": matched,
        "source_revisions": sorted(revisions),
        "audit_git": git_info(),
    }, clearances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline", default="runs/legacy/covariance_comparison")
    parser.add_argument(
        "--output", help="Optional new directory for the audit and derived clearance table"
    )
    args = parser.parse_args()
    result, clearances = audit(args.input, args.baseline)
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "verification.json", result)
        pd.DataFrame(clearances).to_csv(output / "mean_path_clearance.csv", index=False)
    print(json.dumps(result, indent=2))
