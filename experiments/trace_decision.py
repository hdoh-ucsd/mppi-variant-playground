"""Trace one frozen decision or a short episode using the existing vehicle runner."""

import argparse
from pathlib import Path

from analysis.decision_trace import TraceStore
from analysis.dynamic_campaign import run_episode
from analysis.trace_adapters import TraceRecorder
from analysis.trace_comparison import historical_inputs, read, trace_frozen
from evaluation.runner import git_info
from utils.config import read_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["FROZEN_DECISION", "CLOSED_LOOP"], required=True)
    parser.add_argument("--config", required=True, help="Resolved seven-state episode YAML")
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
    config = read_yaml(args.config)
    store = TraceStore(root)
    if args.mode == "FROZEN_DECISION":
        inputs = historical_inputs(read(args.frozen_inputs)[args.scene])
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
