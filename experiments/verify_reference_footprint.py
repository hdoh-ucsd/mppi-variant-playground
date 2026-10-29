"""Check an axle-enclosing capsule along the existing physical reference (no optimization)."""

import argparse

from analysis.dynamic_campaign import configuration
from analysis.footprint_validation import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(configuration(), p.parse_args().output)
