"""Evaluate frozen-state MPPI decisions or regenerate figures from saved evidence."""

import argparse

from analysis.frozen_decision import run_study
from analysis.frozen_decision_plots import generate
from analysis.frozen_decision_report import generate_report
from utils.config import read_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/investigations/frozen_decision.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if not args.plots_only:
        run_study(read_yaml(args.config), args.output)
    generate(args.output)
    generate_report(args.output)
