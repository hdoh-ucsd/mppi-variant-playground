"""Validate the seven-state model before MPC experiments, or rebuild saved validation plots."""

import argparse

from analysis.dynamic_validation import figures, run_validation
from utils.config import read_yaml

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--plots-only", action="store_true")
    args = p.parse_args()
    if not args.plots_only:
        run_validation(
            {"phase": "validation", "vehicle": read_yaml("configs/seven_state/vehicle.yaml")},
            args.output,
        )
    figures(args.output)
