"""Summarize recorded per-command diagnostics without repeating any optimization."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from evaluation.runner import write_json


def summarize(root):
    root = Path(root)
    records = pd.read_csv(root / "per_seed.csv")
    frames = []
    hashes = {}
    for row in records.itertuples():
        path = Path(row.raw_episode) / "diagnostics.csv"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        frame["protocol"] = row.protocol
        frame["scenario"] = row.scenario
        frame["algorithm"] = row.algorithm
        frames.append(frame)
    all_steps = pd.concat(frames, ignore_index=True)
    rows = []
    for (protocol, scenario, algorithm), group in all_steps.groupby(
        ["protocol", "scenario", "algorithm"]
    ):
        row = dict(
            protocol=protocol,
            scenario=scenario,
            algorithm=algorithm,
            commands=len(group),
            missed_deadlines=int(group.deadline_violation.sum()),
            deadline_violation_fraction=float(group.deadline_violation.mean()),
            planning_ms_mean=float(1000 * group.planning_latency_s.mean()),
            planning_ms_p95=float(1000 * group.planning_latency_s.quantile(0.95)),
            model_transitions_per_command=int(group.dynamics_evaluations.max()),
        )
        for metric in [
            "tracking_error",
            "tracking_control_norm",
            "sampled_feedback_rms",
            "nominal_selection_infeasible",
            "covariance_violation",
            "covariance_constraint_residual",
            "empirical_covariance_residual",
            "covariance_prediction_error",
            "solver_seconds",
            "ess",
            "support_fraction",
        ]:
            if metric in group and group[metric].notna().any():
                values = group[metric].dropna().astype(float)
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_max"] = float(values.max())
        for metric in ["tracking_control_norm", "sampled_feedback_rms"]:
            if metric in group and group[metric].notna().any():
                row[metric + "_active_fraction"] = float(
                    (group[metric].dropna().abs() > 1e-10).mean()
                )
        if "nominal_selection" in group and group.nominal_selection.notna().any():
            row["actual_candidate_fraction"] = float((group.nominal_selection.dropna() == 4).mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(root / "diagnostics_summary.csv", index=False)
    write_json(
        root / "diagnostics_provenance.json",
        dict(inputs_sha256=hashes, method="Saved scalar diagnostics only; no simulation."),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    args = parser.parse_args()
    print(summarize(args.input).to_string(index=False))
