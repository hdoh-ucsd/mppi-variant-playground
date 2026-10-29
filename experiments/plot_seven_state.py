"""Rebuild curated and supporting campaign figures using saved records only."""

import argparse

from analysis.dynamic_plots import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--stage", required=True, choices=["Q01", "Q02", "Q03", "Q04", "Q05", "budget"]
    )
    args = parser.parse_args()
    generate(args.input, args.stage)
