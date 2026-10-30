"""Summarize recorded diagnostic evidence; no simulations or new random draws."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.metrics import mean_interval


def summarize(root):
    root = Path(root)
    config = yaml.safe_load((root / "config.yaml").read_text())
    frame = pd.read_csv(root / "per_seed.csv")
    first = []
    failures = []
    for row in frame.itertuples():
        folder = root / row.episode_path
        diag = pd.read_csv(folder / "diagnostics.csv")
        initial = diag.iloc[0]
        first.append(
            dict(
                cell=row.cell,
                proposal=row.proposal,
                seed=row.seed,
                clearance=row.clearance,
                samples=row.samples,
                steering_rate_limit=row.steering_rate_limit,
                feasible_rollout_fraction=initial.feasible_rollout_fraction,
                ess=initial.ess,
                weight_entropy=initial.weight_entropy,
            )
        )
        if not row.success:
            trajectory = pd.read_csv(folder / "trajectory.csv")
            n = int(round(5 / config["vehicle"]["dt"]))
            late = diag.tail(n)
            tail = trajectory.tail(n + 1)
            heading_bad = row.goal_heading_error_rad > config["execution"]["heading_tolerance"]
            speed_bad = (
                abs(row.goal_speed_m_s - config["corridor"]["speed_target"])
                > config["execution"]["speed_tolerance"]
            )
            failures.append(
                dict(
                    cell=row.cell,
                    proposal=row.proposal,
                    seed=row.seed,
                    termination=row.termination,
                    max_progress_fraction=row.max_progress_fraction,
                    terminal_heading_error_rad=row.goal_heading_error_rad,
                    terminal_speed_m_s=row.goal_speed_m_s,
                    terminal_heading_outside_tolerance=heading_bad,
                    terminal_speed_outside_tolerance=speed_bad,
                    progress_last_5s_m=float(tail.progress_m.iloc[-1] - tail.progress_m.iloc[0]),
                    path_length_last_5s_m=float(
                        np.linalg.norm(np.diff(tail[["px", "py"]].to_numpy(), axis=0), axis=1).sum()
                    ),
                    late_median_ess=float(late.ess.median()),
                    late_median_speed_m_s=float(tail.velocity.median()),
                    late_zero_feasible_fraction=float((late.feasible_rollout_fraction == 0).mean()),
                    steering_angle_saturation_fraction=row.steering_angle_saturation_fraction,
                )
            )
    first = pd.DataFrame(first)
    rows = []
    for (cell, proposal), group in first.groupby(["cell", "proposal"], sort=False):
        record = dict(cell=cell, proposal=proposal, n=len(group))
        for metric in ("feasible_rollout_fraction", "ess", "weight_entropy"):
            record.update(
                {f"{metric}_{key}": value for key, value in mean_interval(group[metric]).items()}
            )
        rows.append(record)
    pd.DataFrame(rows).to_csv(root / "initial_decision_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(root / "failure_diagnostics.csv", index=False)
    print(
        f"Summarized {len(first)} initial decisions and {len(failures)} failures from raw records"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    summarize(parser.parse_args().input)
