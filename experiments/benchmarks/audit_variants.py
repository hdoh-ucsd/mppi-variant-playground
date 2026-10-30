"""Replay compact results against ignored raw trajectories and selected populations."""

import argparse
import json

from analysis.variant_audit import audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    args = parser.parse_args()
    print(json.dumps(audit(args.input), indent=2))
