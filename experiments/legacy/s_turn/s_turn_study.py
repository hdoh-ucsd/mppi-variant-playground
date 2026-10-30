"""Feasible S-turn pilot/evaluation or saved-data-only figure reconstruction."""

import argparse

from analysis.s_turn_study import run_study
from utils.config import read_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/investigations/s_turn.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["pilot", "evaluation"], default="evaluation")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--showcase", help="Optional curated PNG destination")
    args = parser.parse_args()
    if not args.plots_only:
        run_study(read_yaml(args.config), args.output, args.phase)
    if args.plots_only or args.phase == "evaluation":
        from analysis.s_turn_plots import generate

        generate(args.output, args.showcase)
