"""Audit small S-turn evidence or include locally retained raw episode records."""

import argparse
import json

from analysis.s_turn_audit import audit

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--summaries-only", action="store_true")
    args = p.parse_args()
    print(json.dumps(audit(args.input, raw=not args.summaries_only), indent=2))
