"""Trace one frozen decision or a short episode using the existing vehicle runner."""

import argparse
from pathlib import Path

from analysis.decision_trace import TraceStore
from analysis.dynamic_campaign import run_episode
from analysis.trace_adapters import ALGORITHMS, TraceRecorder
from analysis.trace_comparison import frozen_config, historical_inputs, read, trace_frozen
from analysis.variant_campaign import equal_work_samples
from evaluation.runner import git_info
from utils.config import read_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["FROZEN_DECISION", "CLOSED_LOOP"], required=True)
    parser.add_argument("--config", help="Resolved seven-state episode YAML")
    parser.add_argument("--algorithm", choices=ALGORITHMS, help="Frozen mode without --config")
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--output", required=True, help="New ignored raw directory")
    parser.add_argument("--experiment-id", default="individual-trace")
    parser.add_argument(
        "--frozen-inputs", default="results/variants/mechanisms/V01/frozen_inputs.json"
    )
    parser.add_argument("--scene", default="two_route_obstacle")
    parser.add_argument("--capture-steps", type=int, nargs="+", default=[0, 8, 9, 11])
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        parser.error("--output must be a new directory")
    if args.config and args.algorithm:
        parser.error("Use either an exact --config or --algorithm with the saved frozen task")
    if args.mode == "CLOSED_LOOP" and not args.config:
        parser.error("CLOSED_LOOP requires a resolved episode --config")
    config = read_yaml(args.config) if args.config else None
    store = TraceStore(root)
    if args.mode == "FROZEN_DECISION":
        saved = read(args.frozen_inputs)[args.scene]
        if config is None:
            options = read_yaml("configs/variants/benchmark.yaml")
            algorithm = args.algorithm or "it_mppi"
            config = frozen_config(
                saved, options, algorithm, equal_work_samples(options, algorithm)
            )
        if config["scenario"]["name"] != saved["config"]["scenario"]["name"]:
            parser.error("The resolved config must match the saved frozen scene")
        inputs = historical_inputs(saved)
        trace = trace_frozen(config, inputs, args.seed, store, args.experiment_id)
        print(trace.decision_id)
    else:
        recorder = TraceRecorder(
            store, args.experiment_id, git_info()["commit"], args.capture_steps
        )
        run_episode(
            config,
            config["controller"]["algorithm"],
            args.seed,
            root / "episode",
            trace_recorder=recorder,
        )
        print(f"Recorded {len(recorder.traces)} decisions in {root}")
