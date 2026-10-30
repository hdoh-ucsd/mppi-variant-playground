"""Audit new-model frozen or driving evidence; raw audit never runs the controller."""

import argparse
import json

from analysis.dynamic_audit import driving_audit, frozen_audit

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--summaries-only", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            frozen_audit(args.input)
            if args.frozen
            else driving_audit(args.input, not args.summaries_only),
            indent=2,
        )
    )
