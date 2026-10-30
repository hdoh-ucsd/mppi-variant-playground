"""Record and visualize paired state-space trajectory distributions."""

import argparse

from analysis.trajectory_density import run_study
from analysis.trajectory_plots import generate
from utils.config import read_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/investigations/trajectory_behavior.yaml")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument(
        "--scenarios", nargs="+", help="Optional scenario subset, or add two_route_obstacle"
    )
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    config = read_yaml(args.config)
    if args.scenarios:
        config["scenarios"] = args.scenarios
    if not args.plots_only:
        run_study(config, args.output)
    generate(args.output)
