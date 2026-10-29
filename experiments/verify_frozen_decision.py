"""Verify recorded frozen decisions and their fixed-target/common-random-number invariants."""

import argparse
import json

from analysis.frozen_decision_audit import audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input), indent=2))
