"""Measure per-plan latency in one process at fixed saved campaign inputs (no driving)."""

import argparse
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.artifacts import begin
from analysis.dynamic_campaign import configuration, resolved_case
from evaluation.metrics import mean_interval
from evaluation.runner import make_controller, write_json


def measure(output, scenario):
    config = configuration()
    protocol = dict(
        phase="timing_calibration",
        scenario=scenario,
        seed_start=36000,
        warmups=1,
        repetitions=10,
        sample_counts=config["budget_counts"],
        methodology="One process; no other campaign stage running; per-plan timer",
        campaign=config,
    )
    root, manifest = begin(output, protocol, "seven_state_latency")
    rows = []
    for k in protocol["sample_counts"]:
        case, *_ = resolved_case(config, scenario, samples=k, clearance=0.35)
        state = np.asarray(case["scenario"]["initial_state"])
        for proposal in config["proposals"]:
            case["controller"].update(kind=proposal, covariance=config["proposals"][proposal])
            controller = make_controller(case)
            for repeat in range(-protocol["warmups"], protocol["repetitions"]):
                controller.reset(state.copy(), np.random.SeedSequence([5500, 36000 + repeat]))
                controller.plan(state.copy())
                if repeat >= 0:
                    rows.append(
                        dict(
                            proposal=proposal,
                            samples=k,
                            repeat=repeat,
                            seed=36000 + repeat,
                            latency_s=controller.diagnostics["planning_latency_s"],
                            model_transitions=controller.diagnostics["dynamics_evaluations"],
                            rhs_evaluations=controller.diagnostics["rhs_evaluations"],
                        )
                    )
    d = pd.DataFrame(rows)
    d.to_csv(root / "per_seed.csv", index=False)
    summary = []
    for (k, p), g in d.groupby(["samples", "proposal"]):
        summary.append(dict(samples=k, proposal=p, **mean_interval(g.latency_s)))
    pd.DataFrame(summary).to_csv(root / "summary.csv", index=False)
    manifest.update(
        complete=True,
        run_id=root.name,
        plans_measured=len(d),
        finished_at_utc=datetime.now(UTC).isoformat(),
        dynamics_model="dynamic_single_track_7",
        sampling_dimension=40,
    )
    write_json(root / "manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["two_route_obstacle", "s_turn"], required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    measure(args.output, args.scenario)
